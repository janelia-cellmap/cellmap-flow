// Which plane an AI annotation reads, and the image the model is sent, in the
// browser: the raw read here, from the engine's reader (Inference.readRaw)
// over the DatasetInfo the engine opened, and everything decided about it by
// the Python both modes run (cellmap_flow/portable/ai_annotate.py, in the
// engine's Pyodide worker through portable/ai_annotate_calls.py): the plan
// (plan_plane: the field around the point, its write box and its raw box),
// the plane laid out as the viewer's panel draws it, resampled to the image
// and scaled to 8-bit grey. So the plane planned and the image sent are the
// server's for the same volume and point, by construction.
//
// Where voxel 0 is, on the raw side. A DatasetInfo level's `origin` is the
// OME translation as chunkmirage reads it (web/chunkmirage/web/src/ome.ts
// openImage composes each level's coordinateTransformations and takes the
// shift as the origin, converted to nm by infer/geometry.ts datasetInfoFrom),
// and an OME translation is voxel 0's CENTRE: that is how the Python reads a
// translation (worldarray takes half a voxel off it for its `corner`, the
// `corner` the server's read_plane passes raw_box), how infer/geometry.ts outputLevel
// treats it (the served level's origin is the raw corner, `origin - voxel/2`,
// plus half an output voxel) and so how finetune/volume.ts planVolume gets a
// volume's dataset_offset_nm, which the Python reads as a centre. So the raw
// grid the plan's raw box is planned on starts at `origin - voxel / 2`
// (levelCornerNm), and the plane read and the write box cover the same
// physical area (test/unit/ai-annotate-python.test.ts checks that).
// viewer/layers.ts centrePosition's comment calls a level's origin the
// corner of its voxel 0; it is not, by the code above (the comment only
// shifts the initial view centre by half a voxel, which is why nothing
// shows it).
import { padded } from "../../chunkmirage/web/src/stages";
import type { DatasetInfo, Inference, PaintVolume } from "../contracts";
import { chooseRawLevel, clipBox, type Level } from "../infer/geometry";
import { buildRecolorPrompt } from "./prompts";
import { AIAnnotateError, type DepthAxis, type OrganelleProfile, type PlanePlan, type RgbaImage, type SegmentRequest } from "./types";

/** The AI annotation's entry points in the portable Python, by module path. */
export const AI_PYTHON = "cellmap_flow.portable.ai_annotate_calls";

/** What the Python answers for a call the user is to be told no to: a point
 * outside the volume or the data, ids running out (ai_annotate_calls.py). */
export interface PythonRefusal {
  refused: string;
}

/** A plane planned, as the Python's plan call answers it: the plan
 * (PlanePlan.to_dict), the point's pixel in the image, the write box's
 * in-plane shape, and the raw box read on the level chosen (its index here). */
export interface PlannedPlane {
  plan: PlanePlan;
  click_px: [number, number];
  write_shape: [number, number];
  raw_box: { begin: number[]; count: number[] };
  level: number;
}

/** The lower corner of a raw level's voxel 0, in nm: its origin, the OME
 * translation, is voxel 0's centre (the module header says why). */
export function levelCornerNm(level: Pick<Level, "origin" | "voxel">): number[] {
  return level.origin.map((o, a) => o - level.voxel[a] / 2);
}

export function levelExtentNm(level: Pick<Level, "origin" | "voxel" | "shape">): { offset: number[]; size: number[] } {
  return { offset: levelCornerNm(level), size: level.shape.map((n, a) => n * level.voxel[a]) };
}

/** `[lo, hi]`, world nm z, y, x: where the volume's raw data is, read at
 * its input voxel size (pipeline.raw_extent_nm: plan_plane's `data_box_nm`).
 * The level is the one chooseRawLevel picks for `volume.input_voxel_size`,
 * which planVolume set to a level's own voxel size. */
export function rawExtentNm(dataset: Pick<DatasetInfo, "levels">, volume: Pick<PaintVolume, "input_voxel_size">): [number[], number[]] {
  const { offset, size } = levelExtentNm(dataset.levels[chooseRawLevel(dataset.levels, volume.input_voxel_size)]);
  return [offset, offset.map((o, a) => o + size[a])];
}

/** The plane through `pointNm` (world nm, z, y, x) across `depthAxis`, planned
 * by the Python (portable/ai_annotate.py plan_plane, whose docstring says
 * what the plane is): over `volume`'s grid (its dataset_shape_voxels), the
 * field `cropSizePx` annotation voxels a side, clipped to the raw data's
 * extent (rawExtentNm), laid out for the viewer's `displayDimensions` (z, y,
 * x when null), with its raw box on the level of `dataset` at the volume's
 * input voxel size (chooseRawLevel). Throws AIAnnotateError("refused", 400)
 * when the point is outside the volume or the data. */
export async function planPlane(
  infer: Pick<Inference, "python">,
  dataset: DatasetInfo,
  volume: Pick<PaintVolume, "output_voxel_size" | "input_voxel_size" | "dataset_offset_nm" | "dataset_shape_voxels">,
  pointNm: readonly number[],
  depthAxis: DepthAxis,
  cropSizePx: number,
  displayDimensions: readonly string[] | null = null,
): Promise<PlannedPlane> {
  const level = chooseRawLevel(dataset.levels, volume.input_voxel_size);
  const raw = dataset.levels[level];
  const answer = await infer.python<Omit<PlannedPlane, "level"> | PythonRefusal>(`${AI_PYTHON}:plan`, {
    volume: { output_voxel_size: volume.output_voxel_size, input_voxel_size: volume.input_voxel_size, dataset_offset_nm: volume.dataset_offset_nm },
    volume_shape: volume.dataset_shape_voxels, point_nm: pointNm, depth_axis: depthAxis, crop_size_px: cropSizePx,
    data_box_nm: rawExtentNm(dataset, volume), display_dimensions: displayDimensions,
    raw: { voxel: raw.voxel, corner: levelCornerNm(raw) },
  });
  if ("refused" in answer) throw new AIAnnotateError("refused", answer.refused, 400);
  return { ...answer, level };
}

/** The image sent for `planned`'s plane (pipeline.read_plane and
 * build_request's image): its raw box read as stored (no input
 * normalization: the image is scaled to 8 bits for the model on its own)
 * from channel `channel` of its level (the one the session's chain picks,
 * as the server's RawInput reads it with normalize=False), clipped to the
 * level and zero past it (chunkmirage's
 * `padded`, as finetune/samples.ts pads a training block), and handed to the
 * Python, which takes the plane along the depth axis, lays it out as the
 * panel draws it, resamples it to the plan's image and scales it to grey.
 * RGBA, the grey in every channel. */
export async function readImage(infer: Pick<Inference, "readRaw" | "python">, dataset: DatasetInfo, planned: PlannedPlane, channel: number): Promise<RgbaImage> {
  const { begin, count } = planned.raw_box;
  const end = begin.map((b, a) => b + count[a]);
  const inside = clipBox(begin, count, dataset.levels[planned.level].shape);
  const data = inside ? await infer.readRaw(planned.level, inside.lo, inside.hi, channel) : null;
  const block = padded(data, dataset.dtype, 1, inside?.lo ?? begin, inside?.hi ?? begin, begin, end, "zero").data;
  const bytes = block.buffer.slice(block.byteOffset, block.byteOffset + block.byteLength) as ArrayBuffer;
  const rgba = await infer.python<ArrayBuffer>(`${AI_PYTHON}:request_image`, { plan: planned.plan, dtype: dataset.dtype, count }, [bytes]);
  const [rows, cols] = planned.plan.image_shape;
  return { width: cols, height: rows, data: new Uint8ClampedArray(rgba) };
}

/** The SegmentRequest for `image` (pipeline.build_request): the image, the
 * prompt, the colour and the point's pixel. The prompt states
 * `plan.resolution_nm`, the pixel size of the image actually sent;
 * `promptOverride` replaces the catalog's description
 * (prompts.buildRecolorPrompt). */
export function buildRequest(image: RgbaImage, planned: PlannedPlane, profile: OrganelleProfile, promptOverride: string | null = null): SegmentRequest {
  return {
    image,
    prompt: buildRecolorPrompt(profile, planned.plan.resolution_nm, promptOverride),
    target_rgb: [profile.rgb[0], profile.rgb[1], profile.rgb[2]],
    click_px: planned.click_px,
  };
}
