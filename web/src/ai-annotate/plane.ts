// From a planned plane to labels: read the raw, build the request, put the
// mask on the write box, label it. The port of
// cellmap_flow/ai_annotate/pipeline.py (read_plane, _resize, build_request,
// mask_to_write_shape, labels_for_box, raw_extent_nm) for the browser, where
// the raw comes from the engine's reader (Inference.readRaw) over the
// DatasetInfo the engine opened, and PIL's resampling is done by hand.
//
// Where voxel 0 is, on the raw side. A DatasetInfo level's `origin` is the
// OME translation as chunkmirage reads it (web/chunkmirage/web/src/ome.ts
// openImage composes each level's coordinateTransformations and takes the
// shift as the origin, converted to nm by infer/geometry.ts datasetInfoFrom),
// and an OME translation is voxel 0's CENTRE: that is how the Python reads a
// translation (io/ome.py ome_corner takes half a voxel off it to get the
// corner, which is what ImageDataInterface.offset, the `corner` the server's
// read_plane passes raw_box, holds), how infer/geometry.ts outputLevel
// treats it (the served level's origin is the raw corner, `origin - voxel/2`,
// plus half an output voxel) and so how finetune/volume.ts planVolume gets a
// volume's dataset_offset_nm, which fill.ts volumeCornerNm reads as a centre.
// So the raw grid rawBox is given here starts at `origin - voxel / 2`, and
// the plane read and the write box cover the same physical area (the test
// in test/unit/ai-annotate-plane.test.ts checks that). viewer/layers.ts
// centrePosition's comment calls a level's origin the corner of its voxel
// 0; it is not, by the code above (the comment only shifts the initial view
// centre by half a voxel, which is why nothing shows it).
//
// The resampling is PIL's (libImaging/Resample.c, Pillow 12: precompute_coeffs
// and the two 32-bit float passes; Geometry.c's ImagingScaleAffine for
// NEAREST), followed operation by operation so the image sent and the mask's
// voxels are the server's: a box filter when the plane shrinks to the image
// (membranes fade rather than alias), bilinear when it grows (the coarse
// axis of an anisotropic plane), each pass accumulated in double and stored
// as float32 as PIL stores its "F" images; nearest neighbour, pixel centre
// to voxel centre, for the mask's objects.
import type { DatasetInfo, Inference, PaintVolume } from "../contracts";
import { padded } from "../../chunkmirage/web/src/stages";
import { chooseRawLevel, clipBox, type Level } from "../infer/geometry";
import { clickPx, fromScreen, type Plane, planeAxes, rawBox, screenShape, toScreen, writeShape } from "./geometry";
import { label2d, seedLabels } from "./labels";
import { greyToRgba, normalizeToUint8 } from "./mask";
import { buildRecolorPrompt } from "./prompts";
import type { Labels, OrganelleProfile, PlanePlan, SegmentRequest } from "./types";

/** A raw plane as readPlane returns it: `plan.image_shape` rows × cols, laid
 * out as on screen, float32 (the dataset's values; a float64 dataset loses
 * precision as it does through PIL); `isUint8` when the dataset is uint8
 * and the plane was not resized, so normalizeToUint8 passes the values
 * through as the Python's does with a uint8 plane. */
export interface RawPlane extends Plane<Float32Array> {
  isUint8: boolean;
}

/** The lower corner of a raw level's voxel 0, in nm: its origin, the OME
 * translation, is voxel 0's centre (the module header says why). */
export function levelCornerNm(level: Pick<Level, "origin" | "voxel">): number[] {
  return level.origin.map((o, a) => o - level.voxel[a] / 2);
}

/** `[lo, hi]`, world nm z, y, x: where the volume's raw data is, read at
 * its input voxel size (pipeline.raw_extent_nm: planPlane's `dataBoxNm`).
 * The level is the one chooseRawLevel picks for `volume.input_voxel_size`,
 * which planVolume set to a level's own voxel size. */
export function rawExtentNm(dataset: Pick<DatasetInfo, "levels">, volume: Pick<PaintVolume, "input_voxel_size">): [number[], number[]] {
  const level = dataset.levels[chooseRawLevel(dataset.levels, volume.input_voxel_size)];
  const lo = levelCornerNm(level);
  return [lo, lo.map((l, a) => l + level.shape[a] * level.voxel[a])];
}

/** The raw plane `plan` reads, as `plan.image_shape` laid out as on screen
 * (pipeline.read_plane).
 *
 * Read as stored (no input normalization: the image is scaled to 8 bits for
 * the model on its own) from the level of `dataset` at the volume's input
 * voxel size (chooseRawLevel), over the box rawBox plans on that level's
 * grid, clipped to the level and zero past it (chunkmirage's `padded`, as
 * finetune/samples.ts pads a training block); the one plane along the depth
 * axis is taken, laid out as the panel draws it (toScreen) and resampled to
 * the plan's image (resizePlane). */
export async function readPlane(infer: Pick<Inference, "readRaw">, dataset: DatasetInfo, volume: Pick<PaintVolume, "input_voxel_size">, plan: PlanePlan): Promise<RawPlane> {
  if (dataset.channels > 1) throw new Error(`${dataset.url} has ${dataset.channels} channels; the browser mode annotates single-channel volumes`);
  const levelIndex = chooseRawLevel(dataset.levels, volume.input_voxel_size);
  const level = dataset.levels[levelIndex];
  const { begin, count } = rawBox(plan, level.voxel, levelCornerNm(level));
  const end = begin.map((b, a) => b + count[a]);
  const inside = clipBox(begin, count, level.shape);
  const data = inside ? await infer.readRaw(levelIndex, inside.lo, inside.hi) : null;
  const block = padded(data, dataset.dtype, 1, inside?.lo ?? begin, inside?.hi ?? begin, begin, end, "zero");

  // The first plane along the depth axis (np.take(data, 0, axis)), in z, y, x order.
  const [rowAxis, colAxis] = planeAxes(plan);
  const rows = count[rowAxis], cols = count[colAxis];
  const plane = new Float32Array(rows * cols);
  const stride = [count[1] * count[2], count[2], 1];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const value = block.data[r * stride[rowAxis] + c * stride[colAxis]];
      plane[r * cols + c] = Number(value);
    }
  }
  const onScreen = toScreen(plane, rows, cols, plan.depth_axis);
  const [imageRows, imageCols] = plan.image_shape;
  const resized = onScreen.rows !== imageRows || onScreen.cols !== imageCols;
  const image = resized ? resizePlane(onScreen, imageRows, imageCols) : onScreen;
  return { ...image, isUint8: dataset.dtype === "uint8" && !resized };
}

// ---------------------------------------------------------------------------
// PIL's resampling (pipeline._resize and mask_to_write_shape's NEAREST)
// ---------------------------------------------------------------------------

interface Filter {
  support: number;
  weight(x: number): number;
}

/** PIL's box filter: Image.BOX. */
const BOX: Filter = { support: 0.5, weight: (x) => (x > -0.5 && x <= 0.5 ? 1 : 0) };
/** PIL's triangle filter: Image.BILINEAR. */
const BILINEAR: Filter = {
  support: 1,
  weight: (x) => {
    x = Math.abs(x);
    return x < 1 ? 1 - x : 0;
  },
};

/** PIL's precompute_coeffs for one axis of `inSize` samples resampled to
 * `outSize`: for each output sample, the first input sample it reads
 * (`bounds[2i]`), how many (`bounds[2i + 1]`) and their normalized weights
 * (`ksize` per output sample). The filter's support is scaled by the
 * shrink factor when shrinking, so every input sample is weighed. */
function coefficients(inSize: number, outSize: number, filter: Filter): { bounds: Int32Array; weights: Float64Array; ksize: number } {
  const scale = inSize / outSize;
  const filterscale = scale < 1 ? 1 : scale;
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2), weights = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);  // "Round the value": C's (int) truncates
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    const k = xx * ksize;
    for (let x = 0; x < xmax; x++) {
      const w = filter.weight((x + xmin - center + 0.5) * ss);
      weights[k + x] = w;
      ww += w;
    }
    if (ww !== 0) for (let x = 0; x < xmax; x++) weights[k + x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, weights, ksize };
}

/** `plane` (float32 values as PIL's "F" image holds them) resampled to
 * `rows × cols` (pipeline._resize): itself when it already is that. Shrinking
 * (neither side grows) averages each pixel's footprint (PIL's box filter), so
 * fine membranes fade rather than alias; otherwise (the coarse axis of an
 * anisotropic plane, or a mixed case) interpolates linearly, PIL's BILINEAR.
 * The horizontal pass runs first, the vertical over its float32 result, as
 * PIL's ImagingResampleInner does, each sum accumulated in double. */
export function resizePlane(plane: Plane<Float32Array>, rows: number, cols: number): Plane<Float32Array> {
  if (plane.rows === rows && plane.cols === cols) return plane;
  if (rows < 1 || cols < 1) throw new Error(`cannot resize a plane to ${rows}×${cols}`);
  const filter = plane.rows >= rows && plane.cols >= cols ? BOX : BILINEAR;
  let data = plane.data, inRows = plane.rows, inCols = plane.cols;
  if (inCols !== cols) {
    const { bounds, weights, ksize } = coefficients(inCols, cols, filter);
    const out = new Float32Array(inRows * cols);
    for (let yy = 0; yy < inRows; yy++) {
      for (let xx = 0; xx < cols; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k = xx * ksize;
        let ss = 0;
        for (let x = 0; x < xmax; x++) ss += data[yy * inCols + x + xmin] * weights[k + x];
        out[yy * cols + xx] = ss;  // stored as float32, as PIL's "F" image is
      }
    }
    data = out;
    inCols = cols;
  }
  if (inRows !== rows) {
    const { bounds, weights, ksize } = coefficients(inRows, rows, filter);
    const out = new Float32Array(rows * inCols);
    for (let yy = 0; yy < rows; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], k = yy * ksize;
      for (let xx = 0; xx < inCols; xx++) {
        let ss = 0;
        for (let y = 0; y < ymax; y++) ss += data[(y + ymin) * inCols + xx] * weights[k + y];
        out[yy * inCols + xx] = ss;
      }
    }
    data = out;
    inRows = rows;
  }
  return { rows, cols, data };
}

/** The input index each of `outSize` output samples takes from `inSize`
 * inputs under PIL's NEAREST (Geometry.c ImagingScaleAffine): the input
 * sample under the output sample's centre, the position accumulated in
 * double as PIL accumulates it, so a centre that lands within rounding of a
 * boundary takes the side PIL takes. */
function nearestIndices(inSize: number, outSize: number): Int32Array {
  const step = inSize / outSize;
  const indices = new Int32Array(outSize);
  let position = step * 0.5;
  for (let i = 0; i < outSize; i++) {
    indices[i] = Math.min(Math.max(Math.trunc(position), 0), inSize - 1);
    position += step;
  }
  return indices;
}

// ---------------------------------------------------------------------------
// The request and the mask
// ---------------------------------------------------------------------------

/** The SegmentRequest for `plane` (pipeline.build_request): its grey RGBA
 * image (normalizeToUint8, greyToRgba), the prompt and the colour. The
 * prompt states `plan.resolution_nm`, the pixel size of the image actually
 * sent; `promptOverride` replaces the catalog's description
 * (prompts.buildRecolorPrompt). A plane that is not the plan's image is an
 * Error (the Python's ValueError). */
export function buildRequest(plane: RawPlane, plan: PlanePlan, profile: OrganelleProfile, promptOverride: string | null = null): SegmentRequest {
  const [rows, cols] = plan.image_shape;
  if (plane.rows !== rows || plane.cols !== cols) throw new Error(`The plane is (${plane.rows}, ${plane.cols}), the plan's image (${rows}, ${cols})`);
  const image = greyToRgba(normalizeToUint8(plane.data, plane.isUint8), rows, cols);
  return {
    image,
    prompt: buildRecolorPrompt(profile, plan.resolution_nm, promptOverride),
    target_rgb: [profile.rgb[0], profile.rgb[1], profile.rgb[2]],
    click_px: clickPx(plan),
  };
}

/** The objects of `mask` (one byte per pixel of the plan's image, rows
 * first, as on screen, nonzero = set) on the write box's in-plane voxels, in
 * z, y, x order (writeShape; pipeline.mask_to_write_shape): 0 for
 * background and 1..n for the objects, numbered in the box's raster order
 * as label2d numbers a mask.
 *
 * Objects are the mask's 4-connected pieces (label2d, so pieces touching
 * only at a corner stay apart, as the model was asked to leave them), found
 * at the image's own resolution, where the model drew the lines between
 * them: on voxels coarser than the pixels a line can fall between the
 * pixels the voxels take, and the objects either side of it still keep ids
 * of their own. Then nearest neighbour, pixel centre to voxel centre: each
 * annotation voxel takes the object of the pixel at its centre, so the mask
 * keeps hard edges and lands where it was drawn whatever the two sizes
 * are. An object no voxel centre falls in is lost. */
export function maskToWriteShape(mask: Uint8Array, plan: PlanePlan): Int32Array {
  const [imageRows, imageCols] = plan.image_shape;
  if (mask.length !== imageRows * imageCols) throw new Error(`the mask has ${mask.length} pixels, the plan's image ${imageRows}×${imageCols}`);
  const { objects, count } = label2d(mask, imageRows, imageCols);
  const [rows, cols] = screenShape(writeShape(plan), plan.depth_axis);
  let onScreen = objects;
  if (rows !== imageRows || cols !== imageCols) {
    const rowOf = nearestIndices(imageRows, rows), colOf = nearestIndices(imageCols, cols);
    onScreen = new Int32Array(rows * cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) onScreen[r * cols + c] = objects[rowOf[r] * imageCols + colOf[c]];
  }
  const write = fromScreen(onScreen, rows, cols, plan.depth_axis).data;
  // Renumbered by first voxel, so fresh ids go to the objects in the order
  // labelling the box itself would give them, whatever the screen layout.
  const number = new Int32Array(count + 1);
  let next = 0;
  for (let i = 0; i < write.length; i++) {
    const o = write[i];
    if (o === 0) continue;
    if (number[o] === 0) number[o] = ++next;
    write[i] = number[o];
  }
  return write;
}

/** The labels for the write box [writeLo, writeHi): an id per object of
 * `objects`, 1 elsewhere (pipeline.labels_for_box).
 *
 * `objects` is the in-plane objects (maskToWriteShape), 0 for background,
 * `existing` the one-voxel-thick box as the volume holds it (fill.readBox,
 * C order), which for a box one voxel thick along `depthAxis` is laid out
 * exactly as the plane is. The objects are labelled by seedLabels: an
 * object the user has already painted part of keeps that id, the rest get
 * fresh ones, counting up for an instance target's `countUp`.
 *
 * `beside` is the planes on either side of the box along the depth axis, as
 * the volume holds them (each laid out like `existing`, or null at the
 * volume's edge; the accept route reads them with fill.readBox). Planes are
 * annotated one at a time, so an object usually continues one in the plane
 * next to it: where this plane is unpainted, the ids beside it count as
 * painted here (the first plane given first), and an object takes the id it
 * overlaps most, rather than a fresh one that an instance target would read
 * as a boundary between the two planes. Fresh ids skip those planes' ids
 * too, so a new object does not take the id of a different object right
 * next to it, which an instance target would read as one object. */
export function labelsForBox(
  objects: ArrayLike<number>,
  existing: Labels,
  writeLo: readonly number[],
  writeHi: readonly number[],
  depthAxis: PlanePlan["depth_axis"],
  countUp: boolean,
  beside: readonly (Labels | null)[] = [],
): Labels {
  const [rowAxis, colAxis] = planeAxes({ depth_axis: depthAxis });
  const rows = writeHi[rowAxis] - writeLo[rowAxis], cols = writeHi[colAxis] - writeLo[colAxis];
  if (writeHi[depthAxis] - writeLo[depthAxis] !== 1) throw new Error(`The box ${writeLo.join(",")}..${writeHi.join(",")} is not one voxel thick along axis ${depthAxis}`);
  if (objects.length !== rows * cols) throw new Error(`The mask makes a box of ${rows * cols} voxels; the box holds ${objects.length}`);
  if (existing.length !== rows * cols) throw new Error(`The mask makes a box of ${rows * cols} voxels; the box holds ${existing.length}`);
  const planes = beside.filter((p): p is Labels => p !== null);
  if (planes.some((p) => p.length !== existing.length)) throw new Error(`The planes beside the box must hold ${existing.length} voxels`);
  if (!planes.length) return seedLabels(objects, existing, countUp);

  const n = existing.length;
  const seen = existing.slice();
  for (const plane of planes) for (let i = 0; i < n; i++) if (seen[i] === 0 && plane[i] >= 2) seen[i] = plane[i];
  // The planes beside are stacked after the box with no objects in them:
  // seedLabels labels only the box's objects, but its fresh ids see every
  // id in the stack.
  const stackExisting: Labels = existing instanceof Uint16Array ? new Uint16Array(n * (1 + planes.length)) : new Uint8Array(n * (1 + planes.length));
  (stackExisting as Uint16Array).set(seen as Uint16Array, 0);
  planes.forEach((plane, k) => (stackExisting as Uint16Array).set(plane as Uint16Array, n * (k + 1)));
  const stackObjects = new Int32Array(n * (1 + planes.length));
  stackObjects.set(objects, 0);
  return seedLabels(stackObjects, stackExisting, countUp).slice(0, n);
}
