// The paint of one annotation volume copied into another over the same
// dataset, on another model's grid: Which model to start from's "Use this
// model" (backend/routes/rank.ts), so that the strokes made for one model
// train the model that ranked first, without painting again.
//
// Each voxel of the new volume takes the paint of the old volume's voxels
// whose centres lie inside it: one of them for a grid as fine as the old,
// several for a coarser one. Where it holds none of their centres (a finer
// grid), it takes the old voxel its own centre lies in, so a 128 nm stroke
// becomes 8³ voxels of 16 nm. A new voxel whose old voxels were painted
// with different labels (object against background, or two objects) is
// left unpainted rather than guessed: it lies on a boundary the coarser
// grid cannot draw. Unpainted old voxels inside it do not count against
// it, so a stroke thinner than the new voxel still lands, a voxel wide.
//
// Both grids are placed by their volume's dataset_offset_nm (voxel 0's
// centre, fill.ts volumeCornerNm), as Seed from Prediction places a served
// grid on a volume's. Only the painted voxels' box in each painted chunk is
// visited, so a few strokes on a coarse volume do not walk a fine volume's
// whole extent.
import { chunkBytes, emptyLabels, volumeCornerNm } from "../ai-annotate/fill";
import type { PaintStore, PaintVolume } from "../contracts";
import type { PaintedChunk } from "./types";
import { CHUNK_DIR, chunkKey, paintedChunkIndices, readPaintedChunk } from "./volume";

/** What carryPaint did: the new volume's voxels painted, the old volume's
 * painted voxels they came from, the new voxels left unpainted because
 * their old voxels disagreed, and the new volume's chunks written. */
export interface Carried {
  painted: number;
  source: number;
  conflicts: number;
  chunks: number;
}

type Grid = Pick<PaintVolume, "output_voxel_size" | "dataset_offset_nm" | "dataset_shape_voxels">;

/** Per voxel of `to` along axis `a`, the voxels of `from` it takes, as
 * [first, end) at 2j and 2j + 1 (end ≤ first where none lies in `from`). */
export function axisSources(from: Grid, to: Grid, a: number): Int32Array {
  const fv = from.output_voxel_size[a], tv = to.output_voxel_size[a];
  const fc = volumeCornerNm(from.dataset_offset_nm, from.output_voxel_size)[a];
  const tc = volumeCornerNm(to.dataset_offset_nm, to.output_voxel_size)[a];
  const n = to.dataset_shape_voxels[a], limit = from.dataset_shape_voxels[a];
  const out = new Int32Array(2 * n), eps = 1e-6;
  for (let j = 0; j < n; j++) {
    const lo = tc + j * tv;
    // The old voxels whose centres fc + (i + ½)·fv lie in [lo, lo + tv).
    let first = Math.ceil((lo - fc) / fv - 0.5 - eps), end = Math.ceil((lo + tv - fc) / fv - 0.5 - eps);
    if (end <= first) { first = Math.floor((lo + tv / 2 - fc) / fv); end = first + 1; }
    out[2 * j] = Math.max(first, 0);
    out[2 * j + 1] = Math.min(end, limit);
  }
  return out;
}

/** Copy the paint of `from` into `to`, an empty volume over the same
 * dataset (createVolume'd): see the top of the file. */
export async function carryPaint(store: PaintStore, from: PaintVolume, to: PaintVolume): Promise<Carried> {
  const maps = [0, 1, 2].map((a) => axisSources(from, to, a));
  // The old volume's painted chunks, read once, and per chunk the new
  // voxels whose sources meet the box of its painted voxels.
  const fromChunks = new Map<string, PaintedChunk>();
  const boxes: { lo: number[]; hi: number[] }[] = [];
  let source = 0;
  for (const index of await paintedChunkIndices(store, from.id)) {
    const chunk = await readPaintedChunk(store, from, index);
    const box = chunk && paintedBox(chunk);
    if (!chunk || !box) continue;
    fromChunks.set(index.join(","), chunk);
    chunk.labels.forEach((v) => { if (v) source++; });
    const origin = index.map((i, a) => i * from.chunk_size[a]);
    const toBox = maps.map((m, a) => meeting(m, box.lo[a] + origin[a], box.hi[a] + origin[a]));
    if (toBox.every((r) => r)) boxes.push({ lo: toBox.map((r) => r![0]), hi: toBox.map((r) => r![1]) });
  }
  const [cz, cy, cx] = from.chunk_size;
  const oldLabel = (z: number, y: number, x: number): number => {
    const chunk = fromChunks.get(`${Math.floor(z / cz)},${Math.floor(y / cy)},${Math.floor(x / cx)}`);
    if (!chunk) return 0;
    const [, h, w] = chunk.shape;
    return chunk.labels[((z % cz) * h + (y % cy)) * w + (x % cx)];
  };
  /** The paint of new voxel (z, y, x): 0 for none, null where its old voxels disagree. */
  const newLabel = (z: number, y: number, x: number): number | null => {
    let found = 0;
    for (let iz = maps[0][2 * z]; iz < maps[0][2 * z + 1]; iz++) for (let iy = maps[1][2 * y]; iy < maps[1][2 * y + 1]; iy++) for (let ix = maps[2][2 * x]; ix < maps[2][2 * x + 1]; ix++) {
      const v = oldLabel(iz, iy, ix);
      if (!v) continue;
      if (found && v !== found) return null;
      found = v;
    }
    return found;
  };
  // Each new chunk the boxes reach: its labels, the voxels already visited
  // (two old chunks' boxes meet on a coarser grid) and how many it paints.
  const out = new Map<string, { index: number[]; labels: ReturnType<typeof emptyLabels>; visited: Uint8Array; painted: number }>();
  const [tz, ty, tx] = to.chunk_size, n = tz * ty * tx, cap = to.dtype === "uint16" ? 65535 : 255;
  let painted = 0, conflicts = 0;
  for (const box of boxes) {
    for (let z = box.lo[0]; z < box.hi[0]; z++) for (let y = box.lo[1]; y < box.hi[1]; y++) for (let x = box.lo[2]; x < box.hi[2]; x++) {
      const index = [Math.floor(z / tz), Math.floor(y / ty), Math.floor(x / tx)], key = index.join(".");
      let c = out.get(key);
      if (!c) out.set(key, (c = { index, labels: emptyLabels(to, n), visited: new Uint8Array(n), painted: 0 }));
      const i = ((z % tz) * ty + (y % ty)) * tx + (x % tx);
      if (c.visited[i]) continue;
      c.visited[i] = 1;
      const v = newLabel(z, y, x);
      if (v === null) { conflicts++; continue; }
      if (!v) continue;
      c.labels[i] = Math.min(v, cap);
      c.painted++;
      painted++;
    }
  }
  let chunks = 0;
  for (const c of out.values()) {
    if (!c.painted) continue;
    await store.put(to.id, CHUNK_DIR + chunkKey(c.index), chunkBytes(c.labels));
    chunks++;
  }
  return { painted, source, conflicts, chunks };
}

/** The new voxels [first, end) along one axis whose sources meet old voxels [lo, hi); null when none does. */
function meeting(map: Int32Array, lo: number, hi: number): [number, number] | null {
  let first = -1, end = -1;
  for (let j = 0; j < map.length / 2; j++) {
    if (map[2 * j + 1] <= map[2 * j] || map[2 * j + 1] <= lo || map[2 * j] >= hi) continue;
    if (first < 0) first = j;
    end = j + 1;
  }
  return first < 0 ? null : [first, end];
}

/** The box of `chunk`'s painted voxels, in its own voxels; null when none is painted. */
export function paintedBox(chunk: PaintedChunk): { lo: number[]; hi: number[] } | null {
  const [, h, w] = chunk.shape, lo = [Infinity, Infinity, Infinity], hi = [-1, -1, -1];
  chunk.labels.forEach((label, i) => {
    if (!label) return;
    const at = [Math.floor(i / (h * w)), Math.floor(i / w) % h, i % w];
    for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], at[a]); hi[a] = Math.max(hi[a], at[a] + 1); }
  });
  return hi[0] < 0 ? null : { lo, hi };
}
