// The training samples of a whole-network finetune, read once and kept, so
// the epochs cost the GPU alone: for a 2D model, square tiles of the volume
// cut in the plane each stroke of paint was drawn in; for a 3D model, cubes.
// Each comes with the raw around it read and normalized as the model stage
// feeds the model, and its paint as the loss's targets.
//
// The 2D tiles are cut as the server's 2D training cuts its patches ("paint
// in any plane": cellmap_flow/finetune/data/sampler.py paint_depth_axes,
// reader.py slice_axes). Where the server draws random patch centres from the
// paint every epoch, the browser lays tiles over it once: the painted voxels
// of one slice of one plane are covered by a grid of output-sized tiles,
// centred on their bounding box, and each tile that holds paint is a sample.
// An epoch is one pass over the tiles, so every painted voxel counts once an
// epoch (a stroke a slice crosses counts in that slice's tile too, as in a
// patch of the server's).
//
// A tile is handed over as the network takes an x–y slice: its slice axis
// first, then the other two in order (rows z, columns y for a z–y slice, as
// Cellpose's 3D mode cuts it, and the server's slice_axes). The targets are
// made on the tile so turned, so a z–y tile's "y" affinity is along its
// rows, as the network sees them. Side planes are used only where the raw
// level and the model's voxels are cubes; elsewhere every tile is an x–y
// slice and the log says how much paint lay in another plane.
//
// The 3D blocks are the same grid one axis up (planBlocks): output-sized
// cubes over the painted voxels' bounding box, centred on it, and each cube
// that holds paint is a sample, its input the cube grown by half the context
// each way. A 3D model sees the paint as it lies, in whichever planes it was
// drawn, so a block is never turned into another plane, and its affinities
// pair z neighbours as well as y and x ones.
//
// Both are boxes of the volume in (z, y, x), read the same way (boxLabels,
// boxRaw): a tile is a box one voxel thick along its slice axis, whose C
// order is already its rows and columns.
import { emptyLabels } from "../../ai-annotate/fill";
import type { DepthAxis } from "../../ai-annotate/types";
import type { DatasetInfo, Inference, PaintStore, PaintVolume, Step } from "../../contracts";
import { copyBox } from "../../infer/stages";
import { applyNormalizers } from "../../infer/norms";
import { padded } from "../../../chunkmirage/web/src/stages";
import type { PaintedChunk, TargetTransform } from "../types";
import { chunkKey, paintedChunkIndices, readPaintedChunk } from "../volume";

/** What each axis a slice is cut across makes of it (sampler.py PLANE_NAMES). */
export const PLANE_NAMES = ["x–y", "z–x", "z–y"] as const;

/** One tile: the plane it lies in (the axis it is cut across), the slice
 * along that axis, and its output box's corner in the plane, in volume
 * voxels along the plane's two axes in order (planeAxes). */
export interface Tile {
  depth: DepthAxis;
  slice: number;
  corner: [number, number];
}

/** One 3D block: its output cube's corner, in volume voxels (z, y, x). */
export interface Block {
  corner: [number, number, number];
}

/** A tile or a block read: the network's input (side² or side³, float32,
 * normalized, zero past the volume), and the targets and the mask per
 * output voxel and channel (output² or output³ × channels, channels last,
 * as TF.js's NHWC and NDHWC tensors). */
export interface TrainingSample {
  input: Float32Array;
  target: Uint8Array;
  mask: Uint8Array;
  /** Mask entries set, over all channels: what the loss averages over. */
  masked: number;
  /** Kept for augmentation, which draws each step's input and targets from
   * them (run-full.ts): the raw as stored and which of it is inside the
   * level (tileRaw, blockRaw), and the paint (tileLabels, blockLabels). */
  raw?: Float32Array;
  inside?: Uint8Array;
  labels?: Uint8Array | Uint16Array;
}

export interface TileSample extends TrainingSample {
  tile: Tile;
}

export interface BlockSample extends TrainingSample {
  block: Block;
}

/** A box of the volume, [lo, hi) in voxels along (z, y, x). */
interface Box {
  lo: number[];
  hi: number[];
}

/** The two axes of the plane cut across `depth`, in the order a tile hands them over. */
export function planeAxes(depth: DepthAxis): [number, number] {
  return depth === 0 ? [1, 2] : depth === 1 ? [0, 2] : [0, 1];
}

/** The volume's painted chunks, by chunk key. */
export async function readPaint(paint: PaintStore, volume: PaintVolume): Promise<Map<string, PaintedChunk>> {
  const chunks = new Map<string, PaintedChunk>();
  for (const index of await paintedChunkIndices(paint, volume.id)) {
    if (index.length !== 3) continue;
    const chunk = await readPaintedChunk(paint, volume, index);
    if (chunk) chunks.set(chunkKey(index), chunk);
  }
  return chunks;
}

/** Every painted voxel (label above 0) of `chunks`, as volume voxels
 * (z, y, x) in one flat array, three numbers each. */
export function paintedVoxels(chunks: Map<string, PaintedChunk>): Int32Array {
  let n = 0;
  for (const chunk of chunks.values()) for (const label of chunk.labels) if (label > 0) n++;
  const out = new Int32Array(3 * n);
  let at = 0;
  for (const chunk of chunks.values()) {
    const [sz, sy, sx] = chunk.shape, [cz, cy, cx] = chunk.index.map((i, a) => i * chunk.shape[a]);
    for (let z = 0, i = 0; z < sz; z++) for (let y = 0; y < sy; y++) for (let x = 0; x < sx; x++, i++) {
      if (chunk.labels[i] > 0) {
        out[at++] = cz + z;
        out[at++] = cy + y;
        out[at++] = cx + x;
      }
    }
  }
  return out;
}

/** The axis each painted voxel's paint is thinnest across (sampler.py
 * paint_depth_axes): 0 (z, an x–y slice) unless y or x is clearly thinner.
 * A voxel's paint is measured by the run of painted voxels through it along
 * each axis: paint drawn in a z–y slice has runs of 1 along x and long ones
 * along z and y. y or x is taken when its run is at most half z's, so a
 * blob painted in 3D, or a lone voxel, stays an x–y slice, and a line along
 * z, which lies in both side planes, goes to y. `voxels` as paintedVoxels
 * gives them, each once; `extent` bounds them (the volume's shape). */
export function paintDepthAxes(voxels: Int32Array, extent: number[]): Int8Array {
  const n = voxels.length / 3;
  const depth = new Int8Array(n);
  if (!n) return depth;
  const [, ey, ex] = extent;
  const key = (z: number, y: number, x: number) => (z * ey + y) * ex + x;
  const index = new Map<number, number>();
  for (let i = 0; i < n; i++) index.set(key(voxels[3 * i], voxels[3 * i + 1], voxels[3 * i + 2]), i);
  const runs = [new Int32Array(n), new Int32Array(n), new Int32Array(n)];
  const at = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    const run = runs[axis];
    for (let i = 0; i < n; i++) {
      if (run[i]) continue;
      // The run through voxel i along `axis`: walked to both ends, then its length given to each of its voxels.
      const members = [i];
      for (const dir of [-1, 1]) {
        at[0] = voxels[3 * i]; at[1] = voxels[3 * i + 1]; at[2] = voxels[3 * i + 2];
        for (;;) {
          at[axis] += dir;
          if (at[axis] < 0 || at[axis] >= extent[axis]) break;
          const j = index.get(key(at[0], at[1], at[2]));
          if (j === undefined) break;
          members.push(j);
        }
      }
      for (const j of members) run[j] = members.length;
    }
  }
  for (let i = 0; i < n; i++) {
    const side = runs[1][i] <= runs[2][i] ? 1 : 2;  // argmin, y on a tie
    if (2 * runs[side][i] <= runs[0][i]) depth[i] = side;
  }
  return depth;
}

/** Tiles of `output` × `output` over the painted voxels, slice by slice of
 * each plane: those of one slice (their depth axis and coordinate along it)
 * are covered by a grid of tiles centred on their bounding box in the
 * plane, and each tile of the grid that holds one of them is a tile. In the
 * order of the planes, the slices, then the tiles' rows and columns. */
export function planTiles(voxels: Int32Array, depth: Int8Array, output: number): Tile[] {
  const groups = new Map<string, { depth: DepthAxis; slice: number; members: number[] }>();
  for (let i = 0; i < depth.length; i++) {
    const d = depth[i] as DepthAxis, slice = voxels[3 * i + d], k = `${d}:${slice}`;
    let group = groups.get(k);
    if (!group) groups.set(k, (group = { depth: d, slice, members: [] }));
    group.members.push(i);
  }
  const tiles: Tile[] = [];
  const ordered = [...groups.values()].sort((a, b) => a.depth - b.depth || a.slice - b.slice);
  for (const { depth: d, slice, members } of ordered) {
    const [pu, pv] = planeAxes(d);
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const i of members) {
      const u = voxels[3 * i + pu], v = voxels[3 * i + pv];
      u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
    // The grid's origin: the bounding box's corner, moved back by half the
    // slack the whole tiles leave over it.
    const ou = u0 - Math.floor((Math.ceil((u1 - u0 + 1) / output) * output - (u1 - u0 + 1)) / 2);
    const ov = v0 - Math.floor((Math.ceil((v1 - v0 + 1) / output) * output - (v1 - v0 + 1)) / 2);
    const cells = new Set<string>();
    for (const i of members) cells.add(`${Math.floor((voxels[3 * i + pu] - ou) / output)}:${Math.floor((voxels[3 * i + pv] - ov) / output)}`);
    const sorted = [...cells].map((c) => c.split(":").map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    for (const [r, c] of sorted) tiles.push({ depth: d, slice, corner: [ou + r * output, ov + c * output] });
  }
  return tiles;
}

/** Output-sized cubes over the painted voxels, the 3D planTiles: a grid of
 * `output`³ cubes centred on the painted voxels' bounding box, and each cube
 * of it that holds one of them a block, in z, y, x order. `voxels` as
 * paintedVoxels gives them. */
export function planBlocks(voxels: Int32Array, output: number): Block[] {
  const n = voxels.length / 3;
  if (!n) return [];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
    lo[a] = Math.min(lo[a], voxels[3 * i + a]);
    hi[a] = Math.max(hi[a], voxels[3 * i + a]);
  }
  // The grid's origin, as planTiles's: the bounding box's corner, moved back
  // by half the slack the whole cubes leave over it.
  const origin = lo.map((l, a) => {
    const length = hi[a] - l + 1;
    return l - Math.floor((Math.ceil(length / output) * output - length) / 2);
  });
  const cells = origin.map((o, a) => Math.floor((hi[a] - o) / output) + 1);
  const held = new Set<number>();
  for (let i = 0; i < n; i++) {
    const [z, y, x] = [0, 1, 2].map((a) => Math.floor((voxels[3 * i + a] - origin[a]) / output));
    held.add((z * cells[1] + y) * cells[2] + x);
  }
  return [...held].sort((a, b) => a - b).map((k) => {
    const x = k % cells[2], y = Math.floor(k / cells[2]) % cells[1], z = Math.floor(k / (cells[1] * cells[2]));
    return { corner: [origin[0] + z * output, origin[1] + y * output, origin[2] + x * output] };
  });
}

/** `tile`'s output box, grown by `halo` each way in its plane: one voxel thick along its slice axis. */
function tileBox(tile: Tile, output: number, halo = 0): Box {
  const [pu, pv] = planeAxes(tile.depth);
  const lo = [0, 0, 0], hi = [0, 0, 0];
  lo[tile.depth] = tile.slice; hi[tile.depth] = tile.slice + 1;
  lo[pu] = tile.corner[0] - halo; hi[pu] = tile.corner[0] + output + halo;
  lo[pv] = tile.corner[1] - halo; hi[pv] = tile.corner[1] + output + halo;
  return { lo, hi };
}

/** `block`'s output cube, grown by `halo` each way. */
function blockBox(block: Block, output: number, halo = 0): Box {
  return { lo: block.corner.map((c) => c - halo), hi: block.corner.map((c) => c + output + halo) };
}

const sizeOf = ({ lo, hi }: Box) => hi.reduce((n, h, a) => n * (h - lo[a]), 1);

/** The paint of `tile`'s output box (output × output, rows then columns),
 * 0 where nothing is painted and past the volume. */
export function tileLabels(tile: Tile, output: number, chunks: Map<string, PaintedChunk>, volume: Pick<PaintVolume, "chunk_size" | "dataset_shape_voxels" | "dtype">): Uint8Array | Uint16Array {
  return boxLabels(tileBox(tile, output), chunks, volume);
}

/** The paint of `block`'s output cube (output³, C order), 0 where nothing
 * is painted and past the volume. */
export function blockLabels(block: Block, output: number, chunks: Map<string, PaintedChunk>, volume: Pick<PaintVolume, "chunk_size" | "dataset_shape_voxels" | "dtype">): Uint8Array | Uint16Array {
  return boxLabels(blockBox(block, output), chunks, volume);
}

/** The paint in `box`, C order, copied row by row from each painted chunk it overlaps. */
function boxLabels(box: Box, chunks: Map<string, PaintedChunk>, volume: Pick<PaintVolume, "chunk_size" | "dataset_shape_voxels" | "dtype">): Uint8Array | Uint16Array {
  const { lo, hi } = box;
  const labels = emptyLabels(volume, sizeOf(box));
  const shape = hi.map((h, a) => h - lo[a]), size = volume.chunk_size;
  const clo = lo.map((l) => Math.max(l, 0)), chi = hi.map((h, a) => Math.min(h, volume.dataset_shape_voxels[a]));
  if (chi.some((h, a) => h <= clo[a])) return labels;
  const first = clo.map((l, a) => Math.floor(l / size[a])), last = chi.map((h, a) => Math.floor((h - 1) / size[a]));
  for (let iz = first[0]; iz <= last[0]; iz++) for (let iy = first[1]; iy <= last[1]; iy++) for (let ix = first[2]; ix <= last[2]; ix++) {
    const chunk = chunks.get(chunkKey([iz, iy, ix]));
    if (!chunk) continue;
    const corner = [iz, iy, ix].map((i, a) => i * size[a]);
    const from = corner.map((c, a) => Math.max(c, clo[a])), to = corner.map((c, a) => Math.min(c + size[a], chi[a]));
    const [, cy, cx] = chunk.shape;
    for (let z = from[0]; z < to[0]; z++) for (let y = from[1]; y < to[1]; y++) {
      const src = ((z - corner[0]) * cy + (y - corner[1])) * cx + (from[2] - corner[2]);
      labels.set(chunk.labels.subarray(src, src + to[2] - from[2]), ((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2] + (from[2] - lo[2]));
    }
  }
  return labels;
}

/** The raw slice around `tile`'s output box (`side` × `side`, the box grown
 * by half the context each way) on raw level `level`, as stored, and which
 * of its voxels are inside the level (1) or past it (0, where the value is
 * 0). The raw level's voxels are the volume's (a ratio of 1). */
export function tileRaw(tile: Tile, side: number, output: number, infer: Pick<Inference, "readRaw">, dataset: DatasetInfo, level: number): Promise<{ raw: Float32Array; inside: Uint8Array }> {
  return boxRaw(tileBox(tile, output, (side - output) / 2), infer, dataset, level);
}

/** The raw block around `block`'s output cube (`side`³, the cube grown by
 * half the context each way, C order), as tileRaw reads a tile's. */
export function blockRaw(block: Block, side: number, output: number, infer: Pick<Inference, "readRaw">, dataset: DatasetInfo, level: number): Promise<{ raw: Float32Array; inside: Uint8Array }> {
  return boxRaw(blockBox(block, output, (side - output) / 2), infer, dataset, level);
}

/** The raw in `box` on raw level `level`, C order, and which of it is inside the level. */
async function boxRaw(box: Box, infer: Pick<Inference, "readRaw">, dataset: DatasetInfo, level: number): Promise<{ raw: Float32Array; inside: Uint8Array }> {
  const { lo, hi } = box;
  const shape = hi.map((h, a) => h - lo[a]);
  const raw = new Float32Array(sizeOf(box)), inside = new Uint8Array(raw.length);
  const rawShape = dataset.levels[level].shape;
  const readLo = lo.map((o) => Math.max(o, 0)), readHi = hi.map((h, a) => Math.min(h, rawShape[a]));
  const span = readHi.map((h, a) => h - readLo[a]);
  if (span.some((n) => n <= 0)) return { raw, inside };
  const read = padded(await infer.readRaw(level, readLo, readHi), dataset.dtype, 1, readLo, readHi, readLo, readHi, "zero");
  // What was read, copied into the box where it lies in it.
  const at = readLo.map((o, a) => o - lo[a]);
  copyBox(raw, shape, at, Float32Array.from(read.data as ArrayLike<number>), span, [0, 0, 0], span);
  for (let z = 0; z < span[0]; z++) for (let y = 0; y < span[1]; y++) {
    const from = ((z + at[0]) * shape[1] + y + at[1]) * shape[2] + at[2];
    inside.fill(1, from, from + span[2]);
  }
  return { raw, inside };
}

/** `raw`'s voxels `inside` normalized through `norm`, in order, as the model
 * stage normalizes a block, and those past the level zero, as it pads one
 * (infer/stages.ts modelInput): a tile's or a block's alike. */
export function normalizeTile(raw: Float32Array, inside: Uint8Array, norm: Step[]): Float32Array {
  let n = 0;
  for (let i = 0; i < inside.length; i++) n += inside[i];
  const values = new Float32Array(n);
  for (let i = 0, k = 0; i < raw.length; i++) if (inside[i]) values[k++] = raw[i];
  const normalized = applyNormalizers(values, norm), out = new Float32Array(raw.length);
  for (let i = 0, k = 0; i < raw.length; i++) if (inside[i]) out[i] = normalized[k++];
  return out;
}

/** The network's input for `tile`: tileRaw normalized (normalizeTile). */
export async function tileInput(tile: Tile, side: number, output: number, infer: Pick<Inference, "readRaw">, dataset: DatasetInfo, level: number, norm: Step[]): Promise<Float32Array> {
  const { raw, inside } = await tileRaw(tile, side, output, infer, dataset, level);
  return normalizeTile(raw, inside, norm);
}

/** A tile's targets from its labels, through the model's transform
 * (targets.ts) on the tile as a one-slice chunk, channels last. */
export function tileTargets(labels: Uint8Array | Uint16Array, output: number, transform: TargetTransform): { target: Uint8Array; mask: Uint8Array; channels: number; masked: number } {
  return blockTargets(labels, [1, output, output], transform);
}

/** The targets of labels of `shape` (z, y, x; a tile's is one slice), through
 * the model's transform (targets.ts) on them as a chunk, channels last: a
 * cube's affinities pair z neighbours too. */
export function blockTargets(labels: Uint8Array | Uint16Array, shape: readonly [number, number, number], transform: TargetTransform): { target: Uint8Array; mask: Uint8Array; channels: number; masked: number } {
  const t = transform({ index: [0, 0, 0], shape: [...shape], labels });
  const n = shape[0] * shape[1] * shape[2], C = t.channels;
  const target = new Uint8Array(n * C), mask = new Uint8Array(n * C);
  let masked = 0;
  for (let c = 0; c < C; c++) for (let p = 0; p < n; p++) {
    if (t.target[c * n + p] > 0.5) target[p * C + c] = 1;
    if (t.mask[c * n + p] > 0) {
      mask[p * C + c] = 1;
      masked++;
    }
  }
  return { target, mask, channels: C, masked };
}
