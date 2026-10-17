// Run on a region: where the region lies on each served model's grid, and
// the viewer's layer the region is drawn in. The region is a box in nm
// (z, y, x), the viewer's coordinates: the panel takes it from the latest
// box drawn in the run_region layer, from the whole volume, or from its
// fields, which show either and can be edited. Each model is written over
// the voxels of the level it is served at that the box touches (any part of
// a voxel counts, so the box is rounded out, never in). That box is what is
// saved, an array of exactly its voxels; chunkmirage's Engine.materialize
// computes the model's chunks (one output block each) that the box reaches
// and stitches the saved array's chunks from them. Pure functions over
// the dataset, the served models and the layer's JSON, so all of it is
// unit-tested; the run is run.ts's, the DOM panel.ts's.
//
// Where the grids are: a served level is placed by its OME translation,
// voxel 0's centre (infer/geometry.ts outputLevel; label-it/prediction.ts
// places a seed on the same grid), so voxel v of a level spans
// [corner + v * voxel, corner + (v + 1) * voxel) with the corner half a
// voxel before the origin. The level is the one infer/engine.ts serves the
// model's view at (servedLevel: the raw level chosen for the model's input
// voxel size, read as if it were at that size), in nm whatever the source's
// unit, as the viewer's coordinates are; voxel indices need no unit.
import { levelCornerNm, levelExtentNm } from "../ai-annotate/plane";
import { blobOf } from "../backend/serving";
import type { Chain, DatasetInfo, LayerSpec, ServedModel } from "../contracts";
import { steady } from "../finetune/good-regions";
import { TYPED } from "../infer/dtype";
import { geometryOf, type Level, outputShape } from "../infer/geometry";
import { outputChannels, outputDtype } from "../infer/post";
import { servedLevel } from "../label-it/prediction";
import { decodeArgs } from "../shared/args";
import { count, formatBytes, formatDuration } from "../shared/format";
import { goodRegionsLayer } from "../viewer/layers";
import { coordsFromNm, pointerNm } from "../viewer/state";

/** A box in the viewer's coordinates, nm, z, y, x: its lower corner and its
 * size along each axis. */
export interface RegionNm {
  offset: number[];
  size: number[];
}

/** More chunks than this in one run is said before it starts: at a few
 * hundred milliseconds a chunk on a laptop's GPU, a quarter of an hour or
 * more, and the files to match. */
export const LARGE_RUN_CHUNKS = 2000;

/** The whole volume as a region: the finest level's extent, from its
 * voxel 0's corner (the level's origin is that voxel's centre) to the far
 * corner. Every model's level lies within it (each is read from a level of
 * the same pyramid and floor-divided into output voxels), so this maps to
 * the whole of each model's level. */
export function wholeVolumeNm(dataset: Pick<DatasetInfo, "levels">): RegionNm {
  return levelExtentNm(dataset.levels[0]);
}

export function chunkStart(v: number, chunk: number): number {
  return Math.floor(v / chunk) * chunk;
}

/** `region` with its corner moved back along `axis` (0 z, 1 y, 2 x) to the
 * start of the model chunk it is in, its far side kept: a box whose corner
 * is on the model's chunk edges along an axis is saved without holding a
 * slab of stitched chunks across it (Inference.materializePlan). The region
 * grows by less than a chunk. The new corner is rounded up to a whole nm,
 * as the fields hold whole nm and round a corner down: rounded down, a
 * voxel of 10.48 nm put it a fraction of a voxel before the edge, in the
 * chunk before. */
export function snappedRegion(region: RegionNm, plan: Pick<ModelPlan, "lo" | "chunk" | "level">, axis: number): RegionNm {
  const { level } = plan;
  const start = levelCornerNm(level)[axis] + chunkStart(plan.lo[axis], plan.chunk[axis]) * level.voxel[axis];
  const offset = [...region.offset], size = [...region.size];
  const at = Math.ceil(steady(start));
  size[axis] = steady(region.offset[axis] + region.size[axis] - at);
  offset[axis] = at;
  return { offset, size };
}

/** Why a region's numbers cannot be run, or null: three finite offsets and
 * three sizes, none negative. A size of 0 is a box drawn without scrolling
 * between its two clicks (flat along the axis the view looks along), which
 * levelBox gives the one voxel it lies in, so it is allowed. */
export function regionError(region: RegionNm): string | null {
  const triple = (v: unknown) => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n));
  if (!triple(region.offset)) return "The region's offset must be three numbers (z, y, x, in nm).";
  if (!triple(region.size)) return "The region's size must be three numbers (z, y, x, in nm).";
  if (region.size.some((s) => s < 0)) return "The region's size must not be negative.";
  return null;
}

/** The voxels [lo, hi) of `level` (z, y, x) that the region touches,
 * clipped to the level; null when it touches none. A side of the box on a
 * voxel boundary does not take the voxel beyond it; a box of no extent
 * along an axis (a flat box) takes the one voxel it lies in. */
export function levelBox(region: RegionNm, level: Level): { lo: number[]; hi: number[] } | null {
  const lo: number[] = [], hi: number[] = [];
  const corner = levelCornerNm(level);
  for (let a = 0; a < 3; a++) {
    const start = steady((region.offset[a] - corner[a]) / level.voxel[a]);
    const end = steady((region.offset[a] + region.size[a] - corner[a]) / level.voxel[a]);
    let l = Math.floor(start), h = Math.ceil(end);
    if (h <= l) h = l + 1;
    l = Math.max(l, 0);
    h = Math.min(h, level.shape[a]);
    if (h <= l) return null;
    lo.push(l);
    hi.push(h);
  }
  return { lo, hi };
}

/** The chunks of a grid chunked by `chunk` that voxels [lo, hi) reach: the
 * number along each axis and in all, as materialize computes them. */
export function chunkGrid(lo: readonly number[], hi: readonly number[], chunk: readonly number[]): { grid: number[]; count: number } {
  const grid = lo.map((l, a) => Math.ceil(hi[a] / chunk[a]) - Math.floor(l / chunk[a]));
  return { grid, count: grid.reduce((n, g) => n * g, 1) };
}

/** The chain a served model's view runs, from the blob in its name: the
 * served view is named by the chain it was served through, which the
 * session's own chain may have moved past (a Submit All that has not yet
 * served the models again). Empty when the name carries no blob. */
export function chainOf(served: ServedModel): Chain {
  const blob = blobOf(served);
  return blob ? decodeArgs(blob).chain : { input_norm: [], postprocess: [] };
}

/** One model of a run: what is written, and where in its grid. */
export interface ModelPlan {
  /** The model's name: its layer's, and its zarr's (`<name>.zarr`). */
  name: string;
  served: ServedModel;
  chain: Chain;
  /** The level its view is served at, in nm. */
  level: Level;
  /** One output block of the model's served variant, in voxels: a chunk. */
  chunk: number[];
  /** The box on the level, in voxels: what is saved, an array of hi − lo
   * voxels whose voxel 0 is the level's voxel lo. */
  lo: number[];
  hi: number[];
  /** The model's chunks the box reaches, along each axis and in all: what
   * a run computes. */
  grid: number[];
  chunks: number;
  /** The saved array's chunks, counted from its corner (the last ones
   * reaching past the box, stored whole as zarr wants). */
  savedChunks: number;
  /** What a voxel holds: the view's channels (1 without a channel axis) and dtype. */
  channels: number;
  dtype: string;
  /** The saved chunks' bytes before compression: the most a run writes,
   * empty chunks skipped and gzip being what makes it less. */
  bytes: number;
}

/** The plan for `served` over `region`, or why there is none: the region
 * outside what the model predicts. */
export function planModel(served: ServedModel, dataset: DatasetInfo, region: RegionNm): ModelPlan | { name: string; error: string } {
  const name = served.entry.name;
  const level = servedLevel(dataset, served.entry);
  const box = levelBox(region, level);
  if (!box) return { name, error: `${name}: the region is outside what it predicts (${level.shape.join("×")} voxels at ${level.voxel.join("×")} nm)` };
  const chunk = outputShape(geometryOf(served.entry));
  const { grid, count: chunks } = chunkGrid(box.lo, box.hi, chunk);
  const chain = chainOf(served);
  const channels = served.info.has_channel ? outputChannels(chain.postprocess, served.info.output_channels) : 1;
  const dtype = outputDtype(chain.postprocess, "float32");
  const perValue = TYPED[dtype]?.BYTES_PER_ELEMENT ?? 4;
  const savedChunks = box.hi.reduce((n, h, a) => n * Math.ceil((h - box.lo[a]) / chunk[a]), 1);
  const bytes = savedChunks * chunk.reduce((n, c) => n * c, 1) * channels * perValue;
  return { name, served, chain, level, chunk, lo: box.lo, hi: box.hi, grid, chunks, savedChunks, channels, dtype, bytes };
}

/** Whether `plan` is one (not a refusal). */
export function isPlan(plan: ModelPlan | { name: string; error: string }): plan is ModelPlan {
  return !("error" in plan);
}

/** A region's numbers rounded out to whole nm (the corner down, the far
 * corner up), for the fields: a box drawn with the mouse lands anywhere
 * within a voxel, and rounding out never loses a voxel it touched. */
export function roundedOut(region: RegionNm): RegionNm {
  const offset = region.offset.map((o) => Math.floor(steady(o)));
  const end = region.offset.map((o, a) => Math.ceil(steady(o + region.size[a])));
  return { offset, size: end.map((e, a) => e - offset[a]) };
}

// ---------------------------------------------------------------------------
// The words the panel shows
// ---------------------------------------------------------------------------

/** "a×b×c", a size's axes joined as the dashboard writes them. */
const dims = (values: readonly number[]) => values.join("×");

/** One model's line in the region's summary: the box in voxels at the
 * level's voxel size, the chunks it takes and their size, and the most it
 * writes. */
export function describePlan(plan: ModelPlan): string {
  const voxels = plan.hi.map((h, a) => h - plan.lo[a]);
  return `${plan.name}: ${dims(voxels)} voxels at ${dims(plan.level.voxel)} nm, saved as that box (at most ${formatBytes(plan.bytes)} before compression); computes ${dims(plan.grid)} = ${count(plan.chunks)} chunk${plan.chunks === 1 ? "" : "s"} of ${dims(plan.chunk)}`;
}

/** What is said of a run of `chunks` chunks before it starts, when there
 * are more than LARGE_RUN_CHUNKS: how many, and how long they take at
 * `msPerChunk` (the last measured time of a forward pass, when there is
 * one: InferenceStats.avgMs); null for a run that is not large. */
export function largeRunWarning(chunks: number, msPerChunk: number | null): string | null {
  if (chunks <= LARGE_RUN_CHUNKS) return null;
  const time = msPerChunk && msPerChunk > 0 ? `, roughly ${formatDuration(chunks * msPerChunk)} at the ${Math.round(msPerChunk)} ms a chunk measured so far` : "";
  return `This is a large run: ${count(chunks)} chunks${time}. It keeps going while you browse; Cancel stops it, and what was written stays.`;
}

// ---------------------------------------------------------------------------
// The run_region layer: an annotation layer the box is drawn in
// ---------------------------------------------------------------------------

/** The layer's name in the viewer. */
export const RUN_REGION_LAYER = "run_region";

/** Neuroglancer's name for the tool that draws an axis-aligned box, set as
 * the layer's `tool`: Ctrl+click puts down one corner and Ctrl+click again
 * the other (Neuroglancer's "annotate" action, control+mousedown0, which
 * runs the selected layer's tool). It is one of Neuroglancer's legacy
 * annotation tools (registerLegacyTool), which a layer's `toolBindings`
 * cannot take: bound to Shift+B there, Neuroglancer dropped the binding as
 * it read the state, and Shift+B drew nothing (checked in Chromium against
 * the fork's bundle; as `tool`, two Ctrl+clicks drew the box). */
export const BOX_TOOL = "annotateBoundingBox";

/** The boxes' colour: amber, apart from the good regions' yellow. */
export const RUN_REGION_COLOR = "#f59e0b";

/** The run_region layer, empty: a local annotation layer over `axes` (the
 * viewer's dimension names) at 1 nm, as the good regions' layer is built
 * (viewer/layers.ts goodRegionsLayer), so a box drawn in it is stored in
 * nm, with the bounding-box tool as its tool (Ctrl+click two corners). */
export function runRegionLayer(axes?: readonly string[]): LayerSpec {
  return { ...goodRegionsLayer([], axes), tool: BOX_TOOL, annotationColor: RUN_REGION_COLOR };
}

/** The coordinate space a local annotation layer stores its points in:
 * its source's transform's input dimensions when it names them, else its
 * output dimensions (Neuroglancer's identity transform then), whichever of
 * Neuroglancer's forms the source is in (an object, or a list of them);
 * null when it says neither. */
function annotationDimensions(source: unknown): Record<string, [number, string]> | null {
  const first = Array.isArray(source) ? source[0] : source;
  if (!first || typeof first !== "object") return null;
  const transform = (first as { transform?: { inputDimensions?: unknown; outputDimensions?: unknown } }).transform;
  const dimensions = transform?.inputDimensions ?? transform?.outputDimensions;
  return dimensions && typeof dimensions === "object" ? (dimensions as Record<string, [number, string]>) : null;
}

/** The layer's points are in nm (runRegionLayer) when its source says nothing else. */
const NM_SPACE: Record<string, [number, string]> = { z: [1e-9, "m"], y: [1e-9, "m"], x: [1e-9, "m"] };

/** The id of the box the Run tab draws from its fields (withBox). */
export const TYPED_BOX_ID = "run-region-box";

/** `layer` with `region` (nm) as its one box, in the layer's own coordinate
 * space (latestBox read backwards), so the box in the viewer follows the
 * offset and size typed into the Run tab. The boxes drawn before are
 * replaced, as a run takes the latest box only; anything else in the layer
 * is kept. `layer` as it was when its space cannot hold the box. */
export function withBox(layer: LayerSpec, region: RegionNm): LayerSpec {
  const space = annotationDimensions(layer.source) ?? NM_SPACE;
  const end = region.offset.map((o, a) => o + region.size[a]);
  const pointA = coordsFromNm(region.offset, space), pointB = coordsFromNm(end, space);
  if (!pointA || !pointB) return layer;
  const kept = (Array.isArray(layer.annotations) ? layer.annotations : []).filter(
    (annotation) => (annotation as { type?: unknown } | null)?.type !== "axis_aligned_bounding_box",
  );
  return { ...layer, annotations: [...kept, { type: "axis_aligned_bounding_box", id: TYPED_BOX_ID, pointA, pointB }] };
}

/** Whether two regions are the same box, to a hundredth of a nanometre:
 * Neuroglancer keeps a box's corners as 32-bit floats, so a box written
 * from the fields reads back a little off. */
export function sameRegion(a: RegionNm, b: RegionNm): boolean {
  const close = (u: readonly number[], v: readonly number[]) => u.length === v.length && u.every((x, i) => Math.abs(x - v[i]) < 0.01);
  return close(a.offset, b.offset) && close(a.size, b.size);
}

/** The latest box drawn in an annotation layer's JSON (its `annotations`,
 * oldest first, as Neuroglancer lists a local layer's), as a region in nm:
 * from the smaller of its two corners to the larger along each axis, the
 * corners being wherever the two clicks were. Its coordinates are the
 * layer's coordinate space's, converted by its scales and matched to z, y
 * and x by name, as pointerNm reads the viewer's. null when the layer has no
 * box (none drawn yet, or only points and lines). */
export function latestBox(layer: LayerSpec | null | undefined): RegionNm | null {
  const annotations = layer?.annotations;
  if (!Array.isArray(annotations)) return null;
  const space = annotationDimensions(layer?.source) ?? NM_SPACE;
  for (let i = annotations.length - 1; i >= 0; i--) {
    const box = annotations[i] as { type?: unknown; pointA?: unknown; pointB?: unknown } | null;
    if (!box || box.type !== "axis_aligned_bounding_box") continue;
    const a = pointerNm(box.pointA as ArrayLike<number>, space), b = pointerNm(box.pointB as ArrayLike<number>, space);
    if (!a || !b) continue;
    const offset = a.map((v, k) => Math.min(v, b[k]));
    return { offset, size: a.map((v, k) => Math.max(v, b[k]) - offset[k]) };
  }
  return null;
}
