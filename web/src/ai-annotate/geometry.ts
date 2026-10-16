// Which plane of the data an AI annotation reads, and which voxels it
// writes: the port of cellmap_flow/ai_annotate/geometry.py (screen_shape,
// to_screen, from_screen, PlanePlan's properties, plan_plane, raw_box) for
// the browser. The viewer side of that module, depth_axis_from_layout and
// depth_axis_for_view, is viewer/state.ts's (depthAxisFromLayout), which
// reads the page's Neuroglancer; the `volume` here is a PaintVolume record
// (finetune/volume.ts) and the raw grid readPlane (plane.ts) passes rawBox is
// a DatasetInfo level's.
//
// A click (or the view centre) picks a point in world nm and the plane the
// user is looking at picks the depth axis. planPlane turns those into a
// PlanePlan: a square field of view around the point, snapped to the
// annotation volume's voxels and clipped to the volume (the write box, one
// annotation voxel thick along the depth axis), and the raw read that covers
// the same physical area at the model's input voxel size (one input voxel
// thick, at the point's depth). So whatever the model paints on the image
// lands on the annotation voxels under it.
//
// Where voxel 0 is (the conclusion the plane read rests on; plane.ts has the
// raw side): the volume's `dataset_offset_nm` is voxel 0's CENTRE, the OME
// translation, which is how volume.py reads it (volume_corner_nm, fill.ts
// here) and how finetune/volume.ts planVolume sets it (the served output
// level's origin, outputLevel: the raw level's corner plus half an output
// voxel, itself derived from the raw origin chunkmirage reads from the OME
// translation, ome.ts openImage). A voxel is in a box when its centre is
// (fill.boxVoxels). The raw is read on its own grid from the voxel that
// holds the first pixel's centre (rawBox).
//
// Axes are z, y, x throughout, as everywhere in cellmap-flow, except in the
// image sent: that is laid out as the viewer's panel shows the plane
// (toScreen), so the model, and the user reviewing its answer, see what was
// on screen. The XZ and YZ planes are transposed from z, y, x order, and the
// XY plane runs x leftwards, as Neuroglancer draws it.
//
// Arithmetic follows the Python operation by operation (snapIntegral where
// it snaps, numpy's round-half-to-even where it rounds), so a plan made here
// is the plan the server would make for the same volume and point; the
// parity fixture (test/fixtures/ai_annotate.json) checks that.
import type { PaintVolume } from "../contracts";
import { boxVoxels, volumeCornerNm } from "./fill";
import { AIAnnotateError, type DepthAxis, MAX_IMAGE_SIDE, PLANE_NAMES, type PlanePlan } from "./types";

export const AXIS_NAMES: readonly ["z", "y", "x"] = ["z", "y", "x"];

/** The volume fields a plan reads. */
export type PlanVolume = Pick<PaintVolume, "output_voxel_size" | "input_voxel_size" | "dataset_offset_nm">;

// ---------------------------------------------------------------------------
// numpy's arithmetic
// ---------------------------------------------------------------------------

/** metadata.snap_integral's tolerance: a value this close (relative, at
 * least absolute) to a whole number is that number. */
const INTEGRAL_TOLERANCE = 1e-6;

/** `value` with a near-whole number made exactly whole (io/metadata.py
 * snap_integral): voxel arithmetic in nm leaves 511.99999999 where 512 is
 * meant, and a floor or a round after it must not see the difference. */
export function snapIntegral(value: number): number {
  const rounded = Math.round(value);
  return Math.abs(value - rounded) <= INTEGRAL_TOLERANCE * Math.max(1, Math.abs(value)) ? rounded : value;
}

/** numpy's np.round: to the nearest whole number, halves to the even one
 * (Math.round takes them up). */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Python's `format(value, "g")` with `precision` significant digits (6, as
 * `f"{x:g}"`): the shortest of fixed and scientific notation, trailing zeros
 * dropped, the exponent at least two digits; the messages here and the
 * prompt's resolution sentence spell numbers with it. A value exactly
 * halfway at the last digit may round the other way than Python (which
 * rounds half to even where JavaScript's toExponential rounds half up). */
export function formatG(value: number, precision = 6): string {
  if (Number.isNaN(value)) return "nan";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  if (value === 0) return Object.is(value, -0) ? "-0" : "0";
  const p = Math.max(precision, 1);
  const sign = value < 0 ? "-" : "";
  const magnitude = Math.abs(value);
  const scientific = magnitude.toExponential(p - 1);  // "d.ddddde+N", rounded to p digits
  const [mantissa, exponentText] = scientific.split("e");
  const exponent = Number(exponentText);
  if (exponent < -4 || exponent >= p) {
    const digits = stripZeros(mantissa);
    const expSign = exponent < 0 ? "-" : "+";
    return `${sign}${digits}e${expSign}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  return sign + stripZeros(magnitude.toFixed(p - 1 - exponent));
}

/** A decimal string without its trailing zeros after the point, and without the point when nothing follows it. */
function stripZeros(text: string): string {
  return text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text;
}

// ---------------------------------------------------------------------------
// Screen layout (geometry.py _SCREEN, screen_shape, to_screen, from_screen)
// ---------------------------------------------------------------------------

/** How the panel looking along each depth axis draws its plane, with
 * cellmap-flow's display dimensions (z, y, x): the axis running down it, the
 * axis running across, and whether across runs backwards. Neuroglancer's "xy"
 * panel draws display dimension 0 across and 1 down; "xz" turns it a quarter
 * about the first, so 0 across and 2 down; "yz" a quarter about the second,
 * so 1 down and 2 across, increasing to the left (data_panel_layout.ts's
 * AXES_RELATIVE_ORIENTATION quaternions). */
const SCREEN: Record<DepthAxis, { down: number; across: number; backwards: boolean }> = {
  2: { down: 1, across: 0, backwards: false },
  1: { down: 2, across: 0, backwards: false },
  0: { down: 1, across: 2, backwards: true },
};

function screen(depthAxis: DepthAxis): { transposed: boolean; backwards: boolean } {
  const { down, across, backwards } = SCREEN[checkDepthAxis(depthAxis)];
  return { transposed: down > across, backwards };
}

function checkDepthAxis(depthAxis: number): DepthAxis {
  if (depthAxis !== 0 && depthAxis !== 1 && depthAxis !== 2) throw new Error(`depth_axis must be 0, 1 or 2 (z, y, x), got ${JSON.stringify(depthAxis)}`);
  return depthAxis;
}

/** A plane's (rows, cols) in z, y, x order, as the panel draws it. */
export function screenShape(shape: readonly number[], depthAxis: DepthAxis): [number, number] {
  const { transposed } = screen(depthAxis);
  const [rows, cols] = shape;
  return transposed ? [cols, rows] : [rows, cols];
}

/** The typed arrays a plane may come as. */
export type PlaneArray = Uint8Array | Uint16Array | Uint32Array | Int32Array | Float32Array | Float64Array;

/** A 2D array, rows first (C order): `data` has `rows × cols` values. */
export interface Plane<T extends PlaneArray = PlaneArray> {
  rows: number;
  cols: number;
  data: T;
}

function sameKind<T extends PlaneArray>(like: T, n: number): T {
  return new (like.constructor as new (n: number) => T)(n);
}

/** `plane` (rows × cols, its two axes in z, y, x order) as the panel looking
 * along `depthAxis` draws it (geometry.to_screen): transposed when the axis
 * drawn down is the later one in z, y, x order, then mirrored across when
 * across runs backwards. */
export function toScreen<T extends PlaneArray>(plane: T, rows: number, cols: number, depthAxis: DepthAxis): Plane<T> {
  const { transposed, backwards } = screen(depthAxis);
  if (plane.length !== rows * cols) throw new Error(`a ${rows}×${cols} plane has ${rows * cols} values, not ${plane.length}`);
  const outRows = transposed ? cols : rows, outCols = transposed ? rows : cols;
  const out = sameKind(plane, plane.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const row = transposed ? c : r;
      let col = transposed ? r : c;
      if (backwards) col = outCols - 1 - col;
      out[row * outCols + col] = plane[r * cols + c];
    }
  }
  return { rows: outRows, cols: outCols, data: out };
}

/** toScreen undone (geometry.from_screen): an image as the panel draws it
 * (`rows × cols` as on screen), in z, y, x order. */
export function fromScreen<T extends PlaneArray>(image: T, rows: number, cols: number, depthAxis: DepthAxis): Plane<T> {
  const { transposed, backwards } = screen(depthAxis);
  if (image.length !== rows * cols) throw new Error(`a ${rows}×${cols} image has ${rows * cols} pixels, not ${image.length}`);
  const outRows = transposed ? cols : rows, outCols = transposed ? rows : cols;
  const out = sameKind(image, image.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const col = backwards ? cols - 1 - c : c;
      const index = transposed ? col * outCols + r : r * outCols + col;
      out[index] = image[r * cols + c];
    }
  }
  return { rows: outRows, cols: outCols, data: out };
}

// ---------------------------------------------------------------------------
// A plan's properties (PlanePlan.plane_axes, plane_name, write_shape, click_px)
// ---------------------------------------------------------------------------

/** The two in-plane axes, in z, y, x order. */
export function planeAxes(plan: Pick<PlanePlan, "depth_axis">): [number, number] {
  return [0, 1, 2].filter((a) => a !== plan.depth_axis) as [number, number];
}

/** "XY", "XZ" or "YZ": the data axes the plane shows. */
export function planeName(plan: Pick<PlanePlan, "depth_axis">): string {
  return PLANE_NAMES[plan.depth_axis];
}

/** The write box's in-plane shape in annotation voxels, its planeAxes in z, y, x order. */
export function writeShape(plan: Pick<PlanePlan, "depth_axis" | "write_lo" | "write_hi">): [number, number] {
  return planeAxes(plan).map((a) => plan.write_hi[a] - plan.write_lo[a]) as [number, number];
}

/** (row, col) of the point in the image sent (PlanePlan.click_px): the
 * pixel the point's fraction of the raw box falls in along the axis drawn
 * down and the axis drawn across, clipped to the image; mirrored when across
 * runs backwards, counted from the near end first so a point on a boundary
 * takes the pixel it takes in z, y, x order. */
export function clickPx(plan: PlanePlan): [number, number] {
  const { down, across, backwards } = SCREEN[checkDepthAxis(plan.depth_axis)];
  const position = [down, across].map((a, i) => {
    const side = plan.image_shape[i];
    const fraction = (plan.point_nm[a] - plan.raw_offset_nm[a]) / plan.raw_shape_nm[a];
    return Math.min(Math.max(Math.floor(fraction * side), 0), side - 1);
  });
  if (backwards) position[1] = plan.image_shape[1] - 1 - position[1];
  return [position[0], position[1]];
}

// ---------------------------------------------------------------------------
// plan_plane and raw_box
// ---------------------------------------------------------------------------

/** `values` as Python's "a, b, c" of `:g` numbers. */
const joinG = (values: readonly number[]) => values.map((v) => formatG(v)).join(", ");

/** The plane through `pointNm` (world nm, z, y, x) across `depthAxis`
 * (geometry.plan_plane).
 *
 * `volume` is the annotation volume's record (output_voxel_size,
 * input_voxel_size, dataset_offset_nm), `volumeShape` its labels' shape
 * (dataset_shape_voxels). The field of view is square in nm, `cropSizePx`
 * annotation voxels along the finer in-plane axis (both, for isotropic
 * voxels), centred on the point; its write box is the annotation voxels
 * whose centres lie in it, clipped to the volume, in the annotation voxel
 * that holds the point along the depth axis.
 *
 * `dataBoxNm` is the raw data's extent, [lo, hi] world nm z, y, x
 * (plane.ts rawExtentNm): the field is clipped to it too. The volume is the
 * data rounded up to whole chunks of the model's output, which for Cellpose
 * at 64 nm is up to 33 um past the data's edge, and there the plane is only
 * the read's zero padding: an image of it was labelled as background, and
 * the cells beside it smeared into it.
 *
 * The raw is read over the clipped box's extent, so the image and the box
 * cover the same physical area. The image's pixels are the finer in-plane
 * input voxel size, square; when that makes a side longer than
 * MAX_IMAGE_SIDE the pixels are made larger to fit.
 *
 * Throws AIAnnotateError("refused", ..., 400) when the point is outside the
 * volume or the data: a field clipped to an edge it does not reach is not
 * what was aimed at. A bad depth axis, crop size or point is a plain Error
 * (the Python's ValueError). */
export function planPlane(
  volume: PlanVolume,
  volumeShape: readonly number[],
  pointNm: readonly number[],
  depthAxis: DepthAxis,
  cropSizePx: number,
  dataBoxNm: readonly [readonly number[], readonly number[]] | null = null,
): PlanePlan {
  checkDepthAxis(depthAxis);
  if (!Number.isFinite(cropSizePx) || Math.trunc(cropSizePx) < 1) throw new Error(`crop_size_px must be at least 1, got ${JSON.stringify(cropSizePx)}`);
  if (pointNm.length !== 3 || pointNm.some((p) => typeof p !== "number" || !Number.isFinite(p))) {
    throw new Error(`point_nm must be three finite numbers (z, y, x), got ${JSON.stringify(pointNm)}`);
  }
  const output = volume.output_voxel_size.map(Number), input = volume.input_voxel_size.map(Number);
  const shape = volumeShape.map((n) => Math.trunc(n));
  const corner = volumeCornerNm(volume.dataset_offset_nm, output);
  const point = pointNm.map(Number);

  const voxel = point.map((p, a) => Math.floor(snapIntegral((p - corner[a]) / output[a])));
  if (voxel.some((v, a) => v < 0 || v >= shape[a])) {
    const extent = corner.map((c, a) => `${formatG(c)}..${formatG(c + shape[a] * output[a])}`);
    throw new AIAnnotateError(
      "refused",
      `The point (${joinG(point)}) nm is outside the annotation volume (z, y, x ${extent.join(", ")} nm): move the view onto the data and try again.`,
      400,
    );
  }

  const plane = planeAxes({ depth_axis: depthAxis });
  const sideNm = Math.trunc(cropSizePx) * Math.min(...plane.map((a) => output[a]));
  const offset = point.map((p) => p - sideNm / 2);
  const size = [sideNm, sideNm, sideNm];
  if (dataBoxNm) {
    const [dataLo, dataHi] = dataBoxNm.map((v) => v.map(Number));
    if (point.some((p, a) => p < dataLo[a] || p >= dataHi[a])) {
      const extent = dataLo.map((lo, a) => `${formatG(lo)}..${formatG(dataHi[a])}`);
      throw new AIAnnotateError(
        "refused",
        `The point (${joinG(point)}) nm is outside the data (z, y, x ${extent.join(", ")} nm): move the view onto the data and try again.`,
        400,
      );
    }
    for (const a of plane) {
      const top = Math.min(offset[a] + size[a], dataHi[a]);
      offset[a] = Math.max(offset[a], dataLo[a]);
      size[a] = top - offset[a];
    }
  }
  // Along the depth axis, exactly the voxel that holds the point: a box one
  // voxel long whose only centre is that voxel's.
  offset[depthAxis] = corner[depthAxis] + voxel[depthAxis] * output[depthAxis];
  size[depthAxis] = output[depthAxis];
  const box = boxVoxels(volume, offset, size, shape);
  if (!box) throw new AIAnnotateError("refused", "No annotation voxels lie around the point.", 400);  // cannot happen with the point inside, but never write an empty box
  const { lo, hi } = box;

  const rawOffset = lo.map((l, a) => corner[a] + l * output[a]);
  const rawShape = hi.map((h, a) => (h - lo[a]) * output[a]);
  rawOffset[depthAxis] = point[depthAxis] - input[depthAxis] / 2;
  rawShape[depthAxis] = input[depthAxis];

  const extent = plane.map((a) => rawShape[a]);
  const resolution = Math.max(Math.min(...plane.map((a) => input[a])), Math.max(...extent) / MAX_IMAGE_SIDE);
  const pixels = extent.map((e) => Math.max(roundHalfEven(snapIntegral(e / resolution)), 1));
  return {
    depth_axis: depthAxis,
    point_nm: point,
    raw_offset_nm: rawOffset.map(snapIntegral),
    raw_shape_nm: rawShape.map(snapIntegral),
    image_shape: screenShape(pixels, depthAxis),
    input_voxel_size: input,
    write_lo: lo,
    write_hi: hi,
    resolution_nm: snapIntegral(resolution),
  };
}

/** The raw voxels to read for `plan`, on a raw grid of `voxelSize` from
 * `corner` (the lower corner of its voxel 0, in nm; geometry.raw_box):
 * `begin` and `count` per axis (io/geometry.py's Box).
 *
 * Each axis starts at the voxel that holds the first sample's centre (half
 * a voxel in from the box's corner) and runs for as many voxels as fit in
 * the box, at least one: so a box on the grid reads exactly its voxels, and
 * the depth axis reads the voxel that holds the point. */
export function rawBox(plan: Pick<PlanePlan, "raw_offset_nm" | "raw_shape_nm">, voxelSize: readonly number[], corner: readonly number[]): { begin: number[]; count: number[] } {
  const begin = plan.raw_offset_nm.map((o, a) => Math.floor(snapIntegral((o + voxelSize[a] / 2 - corner[a]) / voxelSize[a])));
  const count = plan.raw_shape_nm.map((s, a) => Math.max(roundHalfEven(snapIntegral(s / voxelSize[a])), 1));
  return { begin, count };
}
