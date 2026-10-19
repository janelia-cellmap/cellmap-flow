// The training tiles of a whole-network finetune: square slices of the
// volume cut in the plane each stroke of paint was drawn in, as the server's
// 2D training cuts its patches ("paint in any plane":
// cellmap_flow/finetune/data/sampler.py paint_depth_axes, reader.py
// slice_axes), each with the raw slice around it read and normalized as the
// model stage feeds the model, and its paint as the loss's targets.
//
// Where the server draws random patch centres from the paint every epoch,
// the browser lays tiles over it once: the painted voxels of one slice of
// one plane are covered by a grid of output-sized tiles, centred on their
// bounding box, and each tile that holds paint is a sample. An epoch is one
// pass over the tiles, so every painted voxel counts once an epoch (a
// stroke a slice crosses counts in that slice's tile too, as in a patch of
// the server's); the tiles are read once and kept, so the epochs cost the
// GPU alone.
//
// A tile is handed over as the network takes an x–y slice: its slice axis
// first, then the other two in order (rows z, columns y for a z–y slice, as
// Cellpose's 3D mode cuts it, and the server's slice_axes). The targets are
// made on the tile so turned, so a z–y tile's "y" affinity is along its
// rows, as the network sees them. Side planes are used only where the raw
// level and the model's voxels are cubes; elsewhere every tile is an x–y
// slice and the log says how much paint lay in another plane.
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

/** A tile read: the network's input (side², float32, normalized, zero past
 * the volume), and the targets and the mask per output voxel and channel
 * (output² × channels, channels last, as TF.js's NHWC tensors). */
export interface TileSample {
  tile: Tile;
  input: Float32Array;
  target: Uint8Array;
  mask: Uint8Array;
  /** Mask entries set, over all channels: what the loss averages over. */
  masked: number;
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

/** The volume voxel at (row a, column b) of `tile`'s output box, (z, y, x). */
function voxelOf(tile: Tile, a: number, b: number, out: number[]): number[] {
  const [pu, pv] = planeAxes(tile.depth);
  out[tile.depth] = tile.slice;
  out[pu] = tile.corner[0] + a;
  out[pv] = tile.corner[1] + b;
  return out;
}

/** The paint of `tile`'s output box (output × output, rows then columns),
 * 0 where nothing is painted and past the volume. */
export function tileLabels(tile: Tile, output: number, chunks: Map<string, PaintedChunk>, volume: Pick<PaintVolume, "chunk_size" | "dataset_shape_voxels" | "dtype">): Uint8Array | Uint16Array {
  const labels = emptyLabels(volume, output * output);
  const [cz, cy, cx] = volume.chunk_size, shape = volume.dataset_shape_voxels, v = [0, 0, 0];
  for (let a = 0; a < output; a++) for (let b = 0; b < output; b++) {
    const [z, y, x] = voxelOf(tile, a, b, v);
    if (z < 0 || y < 0 || x < 0 || z >= shape[0] || y >= shape[1] || x >= shape[2]) continue;
    const chunk = chunks.get(`${Math.floor(z / cz)}.${Math.floor(y / cy)}.${Math.floor(x / cx)}`);
    if (chunk) labels[a * output + b] = chunk.labels[((z % cz) * cy + (y % cy)) * cx + (x % cx)];
  }
  return labels;
}

/** The network's input for `tile`: the raw slice around its output box
 * (`side` × `side`, the box grown by half the context each way) on raw
 * level `level`, the voxels inside the level normalized through `norm` and
 * those past it zero, as the model stage pads a block (infer/stages.ts
 * modelInput). The raw level's voxels are the volume's (a ratio of 1). */
export async function tileInput(tile: Tile, side: number, output: number, infer: Pick<Inference, "readRaw">, dataset: DatasetInfo, level: number, norm: Step[]): Promise<Float32Array> {
  const halo = (side - output) / 2;
  const [pu, pv] = planeAxes(tile.depth);
  const lo = [0, 0, 0], hi = [0, 0, 0];
  lo[tile.depth] = tile.slice; hi[tile.depth] = tile.slice + 1;
  lo[pu] = tile.corner[0] - halo; hi[pu] = tile.corner[0] + output + halo;
  lo[pv] = tile.corner[1] - halo; hi[pv] = tile.corner[1] + output + halo;
  const shape = hi.map((h, a) => h - lo[a]);
  const block = new Float32Array(side * side);
  const rawShape = dataset.levels[level].shape;
  const readLo = lo.map((o) => Math.max(o, 0)), readHi = hi.map((h, a) => Math.min(h, rawShape[a]));
  const span = readHi.map((h, a) => h - readLo[a]);
  if (span.some((n) => n <= 0)) return block;
  const read = padded(await infer.readRaw(level, readLo, readHi), dataset.dtype, 1, readLo, readHi, readLo, readHi, "zero");
  const values = applyNormalizers(Float32Array.from(read.data as ArrayLike<number>), norm);
  // The box with its depth axis one voxel thick, in C order, is the tile's
  // rows and columns: copied into the block where it lies in the box.
  copyBox(block, shape, readLo.map((o, a) => o - lo[a]), values, span, [0, 0, 0], span);
  return block;
}

/** A tile's targets from its labels, through the model's transform
 * (targets.ts) on the tile as a one-slice chunk, channels last. */
export function tileTargets(labels: Uint8Array | Uint16Array, output: number, transform: TargetTransform): { target: Uint8Array; mask: Uint8Array; channels: number; masked: number } {
  const t = transform({ index: [0, 0, 0], shape: [1, output, output], labels });
  const n = output * output, C = t.channels;
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
