// Block geometry: what a model reads and writes (the manifest's blocks and
// voxel sizes), which raw level it reads, and the output grid it is served
// on. Pure functions, from Marwan Zouinkhi's web/legacy/pipeline.js (context,
// volumeShape, blockShape) and zarr-reader.js (the level choice), made to
// agree with the Python server (models/geometry.py, io/multiscale.py,
// serving/chunkmirage_server.py) where the two differed. The input box of a
// chunk, its clipping to the volume and the padding past it are chunkmirage's
// engine's (web/chunkmirage/web/src/stages.ts: inputBox, padded), given the
// halo and the ratio from here.
//
// Axis order is (z, y, x) throughout. Lengths named *_shape are in nm, as
// Python's ModelGeometry has them; a shape in voxels says so.
import type { DatasetInfo, ModelEntry } from "../contracts";
import type { SourceInfo } from "../../chunkmirage/web/src/types";

/** What a model reads and writes, in nm (models/geometry.py: ModelGeometry). */
export interface BlockGeometry {
  read_shape: number[];
  write_shape: number[];
  input_voxel_size: number[];
  output_voxel_size: number[];
}

export interface Level { shape: number[]; voxel: number[]; origin: number[] }

const AXES = 3;

/** A manifest's scalar or per-axis size as three numbers: the Hugging Face
 * manifests give one number (a cube, "_in148"), the contracts a list. */
export function asTriple(value: number | number[], what: string): number[] {
  const out = Array.isArray(value) ? value : [value, value, value];
  if (out.length !== AXES || out.some((v) => !Number.isFinite(v))) throw new Error(`${what} must be ${AXES} numbers, got ${JSON.stringify(value)}`);
  return out.map(Number);
}

/** The geometry of a served model: the variant's block sizes in voxels times
 * the manifest's voxel sizes. */
export function geometryOf(entry: ModelEntry): BlockGeometry {
  const ivs = asTriple(entry.manifest.input_voxel_size_nm, `${entry.name}: input_voxel_size_nm`);
  const ovs = asTriple(entry.manifest.output_voxel_size_nm, `${entry.name}: output_voxel_size_nm`);
  const input = asTriple(entry.variant.input_size, `${entry.name}: input_size`);
  const output = asTriple(entry.variant.output_size, `${entry.name}: output_size`);
  return {
    read_shape: input.map((n, a) => n * ivs[a]),
    write_shape: output.map((n, a) => n * ovs[a]),
    input_voxel_size: ivs,
    output_voxel_size: ovs,
  };
}

/** Voxels the model reads. */
export function inputShape(g: BlockGeometry): number[] {
  return g.read_shape.map((s, a) => wholeVoxels(s / g.input_voxel_size[a], "the model's input block"));
}

/** Voxels the model writes: one output chunk (blockShape in the prototype). */
export function outputShape(g: BlockGeometry): number[] {
  return g.write_shape.map((s, a) => wholeVoxels(s / g.output_voxel_size[a], "the model's output block"));
}

/** nm read beyond the written region on each side (inferencer.py:64,
 * ModelGeometry.context): (read_shape - write_shape) / 2. */
export function context(g: BlockGeometry): number[] {
  return g.read_shape.map((r, a) => (r - g.write_shape[a]) / 2);
}

/** The context in voxels of the raw level, which must be whole
 * (chunkmirage_server.py's `_whole` check on the halo): the engine's `halo`
 * for the model's stage, in its input's voxels. */
export function halo(g: BlockGeometry): number[] {
  return context(g).map((c, a) => wholeVoxels(c / g.input_voxel_size[a], "the model's context"));
}

/** Input voxels per output voxel (chunkmirage_ops.py's `ratio`, fused.py's
 * `scale`): 1 for a model writing at the voxel size it reads, 2 for one
 * reading 8 nm and writing 16. The engine grows an output box by it to find
 * the input box on the raw level's grid. */
export function ratio(g: BlockGeometry): number[] {
  return g.output_voxel_size.map((o, a) => o / g.input_voxel_size[a]);
}

/** The part of the box [begin, begin + shape) inside [0, levelShape): `lo`
 * and `hi` in level voxels and `offset`, where that part sits in the box;
 * null when nothing of it is inside (zarr-reader.js's readRoi, BlockInput in
 * chunkmirage_ops.py). The model's stage tells the data from the engine's
 * padding by it. */
export function clipBox(begin: number[], shape: number[], levelShape: number[]): { lo: number[]; hi: number[]; offset: number[] } | null {
  const lo = begin.map((b, a) => Math.min(Math.max(b, 0), levelShape[a]));
  const hi = begin.map((b, a) => Math.max(Math.min(b + shape[a], levelShape[a]), lo[a]));
  if (hi.some((h, a) => h <= lo[a])) return null;
  return { lo, hi, offset: lo.map((l, a) => l - begin[a]) };
}

/** The output volume's shape in voxels (server.py's shape: raw voxels read as
 * if at the model's input voxel size, floor-divided into output voxels). */
export function volumeShape(rawShape: number[], g: BlockGeometry): number[] {
  return rawShape.map((n, a) => Math.floor((n * g.input_voxel_size[a]) / g.output_voxel_size[a]));
}

// ---------------------------------------------------------------------------
// Levels: which raw level to read, and the output level served
// ---------------------------------------------------------------------------

/** Equal voxel sizes, as io/multiscale.py's same_voxel_size: within 1e-6
 * relative (the pyramids' floats are not exact). */
export function sameVoxelSize(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= 1e-6 * Math.max(Math.abs(v), Math.abs(b[i])) + 1e-9);
}

/** `a` is coarser than `b` along some axis (coarser_anywhere). */
function coarserAnywhere(a: number[], b: number[]): boolean {
  return a.some((v, i) => v > b[i] && Math.abs(v - b[i]) > 1e-6 * Math.max(Math.abs(v), Math.abs(b[i])) + 1e-9);
}

/** The index of the raw level a model at `voxel` reads, as the Python server
 * chooses it (io/multiscale.py select_level, mode "floor", how cellmap-flow
 * has always chosen): the level at `voxel` if there is one, else the last
 * level before the first that is coarser than `voxel` along some axis (the
 * finest level that is not too coarse), and the first level when even that
 * one is. The prototype's "closest by voxel size" gives the same level
 * whenever one matches exactly, the usual case. Levels are finest first. */
export function chooseRawLevel(levels: Level[], voxel: number[]): number {
  if (!levels.length) throw new Error("the dataset has no levels to choose from");
  let last = 0;
  for (let i = 0; i < levels.length; i++) {
    if (sameVoxelSize(levels[i].voxel, voxel)) return i;
    if (coarserAnywhere(levels[i].voxel, voxel)) return last;
    last = i;
  }
  return last;
}

/** The one level a model's view is served at, from the raw level it reads,
 * in that level's own units (the engine's SourceInfo gives it; the ratio is
 * unitless, so nothing is converted).
 *
 * The raw level is read as if it were at the model's input voxel size, voxel
 * for voxel (the Python server's "relabel"): the output has one voxel per
 * `ratio` raw voxels, so its shape is floor(raw voxels / ratio). The level is
 * declared at the size its voxels really have, the raw voxel times the
 * ratio: the Python server declares its array at `output_voxel_size` and
 * the viewer rescales it by model_info's `effective_output_voxel_size`
 * (effectiveOutputVoxelSize below), but chunkmirage's engine reads the raw
 * level at its real voxel size and takes the ratio of the two grids from
 * the sizes declared, so the view must say the true one. The viewer's
 * transform from model_info then redeclares the array at the size it
 * already has, which changes nothing. With the raw level at the model's
 * voxel size the two agree anyway.
 *
 * Origins are voxel centres, the OME translation chunkmirage's reader gives
 * and its engine writes (ome.ts, engine.ts omeGroup). The output grid starts
 * at the raw level's corner (chunkmirage_server.py: "output voxel 0 starts
 * at the corner of the level the model reads"), so the served origin is that
 * corner plus half an output voxel. With equal voxel sizes it is the raw
 * origin. */
export function outputLevel(raw: Level, g: BlockGeometry): Level {
  const voxel = raw.voxel.map((v, a) => v * ratio(g)[a]);
  return {
    shape: volumeShape(raw.shape, g),
    voxel,
    origin: raw.origin.map((o, a) => o - raw.voxel[a] / 2 + voxel[a] / 2),
  };
}

/** What an output voxel really measures when the raw level read is not at the
 * model's input voxel size (ModelGeometry.to_model_info): actual * output /
 * input per axis; null when the level is at the model's size. */
export function effectiveOutputVoxelSize(raw: Level, g: BlockGeometry): number[] | null {
  if (sameVoxelSize(raw.voxel, g.input_voxel_size)) return null;
  return raw.voxel.map((v, a) => (v * g.output_voxel_size[a]) / g.input_voxel_size[a]);
}

// ---------------------------------------------------------------------------
// The dataset as the engine opened it, in nm
// ---------------------------------------------------------------------------

/** nm per unit of each length unit a source may declare: OME-Zarr's long
 * names, the short forms chunkmirage's engine maps (its OME_UNIT), and the
 * micro sign. An empty unit is taken as nm, which is what CellMap's own
 * volumes are. */
export const NM_PER_UNIT: Record<string, number> = {
  "": 1, nm: 1, nanometer: 1, nanometre: 1,
  angstrom: 0.1, picometer: 1e-3, picometre: 1e-3,
  um: 1e3, "µm": 1e3, "μm": 1e3, micrometer: 1e3, micrometre: 1e3, micron: 1e3,
  mm: 1e6, millimeter: 1e6, millimetre: 1e6,
  cm: 1e7, centimeter: 1e7, centimetre: 1e7,
  m: 1e9, meter: 1e9, metre: 1e9,
};

/** The dataset's levels in nm, from the engine's SourceInfo for its "raw"
 * view: voxel sizes and origins converted by each axis's unit, and the axes
 * then declared in nanometers. An axis in a unit that is not a length (time,
 * degrees) is left as it is, with its unit. */
export function datasetInfoFrom(url: string, src: SourceInfo): DatasetInfo {
  const factors = src.axes.map((a) => NM_PER_UNIT[a.unit.trim().toLowerCase()] ?? NM_PER_UNIT[a.unit] ?? null);
  return {
    url,
    dtype: src.dtype,
    channels: src.channels,
    axes: src.axes.map((a, i) => ({ name: a.name, unit: factors[i] === null ? a.unit : "nanometer" })),
    levels: src.levels.map((l) => ({
      shape: [...l.shape],
      voxel: l.voxel.map((v, a) => v * (factors[a] ?? 1)),
      origin: l.origin.map((o, a) => o * (factors[a] ?? 1)),
    })),
  };
}

function wholeVoxels(value: number, what: string): number {
  const n = Math.round(value);
  if (Math.abs(value - n) > 1e-6) throw new Error(`${what} (${value} voxels) is not whole voxels`);
  return n;
}
