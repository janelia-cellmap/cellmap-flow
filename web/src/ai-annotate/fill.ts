// Labelling a whole box of an annotation volume at once, in the paint store:
// the port of cellmap_flow/finetune/session/fill.py's box_voxels, paint_box
// and restore_box (and of finetune/session/volume.py's volume_corner_nm, which
// box_voxels places voxels by) for the browser, where the volume is the paint
// store's zarr v2 chunks (finetune/volume.ts) instead of MinIO's.
//
// What the server does through zarr over MinIO (`arr[box]` read and written)
// is done here chunk by chunk: readBox assembles the box from the chunks it
// touches (a chunk never painted is zeros, zarr's fill value), writeBox puts
// each touched chunk back whole, read-modify-write, as the fork's writable
// zarr source does when the brush paints. paintBox and restoreBox are
// fill.paint_box and fill.restore_box over those two, with the same write
// rule, the same counts and the same "nothing written when nothing changes".
//
// Where voxel 0 is: a volume's `dataset_offset_nm` is voxel 0's CENTRE (the
// OME translation, as volume.py's docstring and volume_corner_nm say, and as
// finetune/volume.ts planVolume sets it from the served level's origin), so
// its corner is half a voxel before it, and a voxel is in a box when its
// centre is (boxVoxels).
import type { PaintStore, PaintVolume } from "../contracts";
import { CHUNK_DIR, chunkKey, readPaintedChunk } from "../finetune/volume";
import { type Labels, UNDO_DEPTH, type UndoEntry } from "./types";

/** The volume fields the box arithmetic reads: its grid. */
export type VolumeGrid = Pick<PaintVolume, "output_voxel_size" | "dataset_offset_nm">;

/** The world position of an annotation volume's voxel-0 lower corner, in nm
 * (volume.volume_corner_nm): `datasetOffsetNm` is voxel 0's centre, so the
 * corner is half a voxel before it on each axis; zeros when the volume has
 * no offset. */
export function volumeCornerNm(datasetOffsetNm: readonly number[] | null | undefined, outputVoxelSize: readonly number[]): number[] {
  return outputVoxelSize.map((v, a) => (datasetOffsetNm ? datasetOffsetNm[a] : 0) - v / 2);
}

/** `{lo, hi}`: the annotation voxels whose centres lie in a box, clipped to
 * the volume (fill.box_voxels). `offsetNm` and `shapeNm` are the box's
 * lower corner and size in world nm (z, y, x); `volumeShape` the volume's
 * extent in voxels. null when no voxel of the volume is in the box. */
export function boxVoxels(volume: VolumeGrid, offsetNm: readonly number[], shapeNm: readonly number[], volumeShape: readonly number[]): { lo: number[]; hi: number[] } | null {
  const voxel = volume.output_voxel_size;
  const corner = volumeCornerNm(volume.dataset_offset_nm, voxel);
  const lo: number[] = [], hi: number[] = [];
  for (let a = 0; a < 3; a++) {
    const start = (offsetNm[a] - corner[a]) / voxel[a];
    const end = start + shapeNm[a] / voxel[a];
    // Voxel i's centre is at i + 0.5 voxels from the corner.
    lo.push(Math.max(Math.ceil(start - 0.5), 0));
    hi.push(Math.min(Math.ceil(end - 0.5), volumeShape[a]));
  }
  if (hi.some((h, a) => h <= lo[a])) return null;
  return { lo, hi };
}

// ---------------------------------------------------------------------------
// Boxes of the store's chunks
// ---------------------------------------------------------------------------

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** A zero array of the volume's label dtype. */
export function emptyLabels(volume: Pick<PaintVolume, "dtype">, n: number): Labels {
  return volume.dtype === "uint16" ? new Uint16Array(n) : new Uint8Array(n);
}

/** A zero array of the same kind as `like`. */
function labelsLike(like: Labels, n: number): Labels {
  return like instanceof Uint16Array ? new Uint16Array(n) : new Uint8Array(n);
}

function checkBox(volume: Pick<PaintVolume, "dataset_shape_voxels">, lo: readonly number[], hi: readonly number[]): void {
  const shape = volume.dataset_shape_voxels;
  if (lo.length !== 3 || hi.length !== 3) throw new Error(`a box is three lo and three hi indices, got ${lo.join(",")}..${hi.join(",")}`);
  for (let a = 0; a < 3; a++) {
    if (!Number.isInteger(lo[a]) || !Number.isInteger(hi[a]) || lo[a] < 0 || hi[a] > shape[a] || hi[a] <= lo[a]) {
      throw new Error(`the box ${lo.join(",")}..${hi.join(",")} is not within the volume's ${shape.join("×")} voxels`);
    }
  }
}

/** The grid indices of the chunks the box [lo, hi) touches, in C order. */
function chunksTouched(volume: Pick<PaintVolume, "chunk_size">, lo: readonly number[], hi: readonly number[]): number[][] {
  const chunk = volume.chunk_size;
  const first = lo.map((l, a) => Math.floor(l / chunk[a])), last = hi.map((h, a) => Math.floor((h - 1) / chunk[a]));
  const indices: number[][] = [];
  for (let z = first[0]; z <= last[0]; z++) for (let y = first[1]; y <= last[1]; y++) for (let x = first[2]; x <= last[2]; x++) indices.push([z, y, x]);
  return indices;
}

/** Copy the overlap of the box [lo, hi) and the chunk at `index` between
 * `box` (laid out over [lo, hi)) and `chunk` (laid out over the chunk), in
 * the direction `into`. Both are C order (z, y, x), so each x run is one
 * `set` of a subarray. */
function copyOverlap(volume: Pick<PaintVolume, "chunk_size">, index: readonly number[], chunk: Labels, box: Labels, lo: readonly number[], hi: readonly number[], into: "box" | "chunk"): void {
  const size = volume.chunk_size;
  const cLo = index.map((i, a) => i * size[a]);
  const oLo = cLo.map((c, a) => Math.max(c, lo[a])), oHi = cLo.map((c, a) => Math.min(c + size[a], hi[a]));
  const boxShape = hi.map((h, a) => h - lo[a]);
  const run = oHi[2] - oLo[2];
  for (let z = oLo[0]; z < oHi[0]; z++) {
    for (let y = oLo[1]; y < oHi[1]; y++) {
      const inChunk = ((z - cLo[0]) * size[1] + (y - cLo[1])) * size[2] + (oLo[2] - cLo[2]);
      const inBox = ((z - lo[0]) * boxShape[1] + (y - lo[1])) * boxShape[2] + (oLo[2] - lo[2]);
      if (into === "box") (box as Uint16Array).set(chunk.subarray(inChunk, inChunk + run) as Uint16Array, inBox);
      else (chunk as Uint16Array).set(box.subarray(inBox, inBox + run) as Uint16Array, inChunk);
    }
  }
}

/** A chunk's labels as the bytes the store holds (the layout labelsOf reads):
 * uint8 as they are, uint16 little-endian. A copy: the store keeps what it is given. */
export function chunkBytes(labels: Labels): Uint8Array {
  if (labels instanceof Uint16Array) {
    const bytes = new Uint8Array(labels.length * 2);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < labels.length; i++) view.setUint16(i * 2, labels[i], true);
    return bytes;
  }
  return new Uint8Array(labels);
}

/** The box [lo, hi) of the volume's labels as the store holds them, C order
 * (z, y, x), in the volume's dtype: the chunks it touches read
 * (readPaintedChunk), a chunk never painted being zeros. */
export async function readBox(store: PaintStore, volume: PaintVolume, lo: readonly number[], hi: readonly number[]): Promise<Labels> {
  checkBox(volume, lo, hi);
  const box = emptyLabels(volume, prod(hi.map((h, a) => h - lo[a])));
  for (const index of chunksTouched(volume, lo, hi)) {
    const painted = await readPaintedChunk(store, volume, index);
    if (painted) copyOverlap(volume, index, painted.labels, box, lo, hi, "box");
  }
  return box;
}

/** Write `labels` (the box [lo, hi), C order, the volume's dtype) into the
 * store: each chunk the box touches is read (zeros when never painted), the
 * box's part of it replaced and the chunk put back whole as
 * `CHUNK_DIR + chunkKey(index)`, as the viewer's brush writes a chunk. */
export async function writeBox(store: PaintStore, volume: PaintVolume, lo: readonly number[], hi: readonly number[], labels: Labels): Promise<void> {
  checkBox(volume, lo, hi);
  const n = prod(hi.map((h, a) => h - lo[a]));
  if (labels.length !== n) throw new Error(`the box ${lo.join(",")}..${hi.join(",")} holds ${n} voxels, the labels ${labels.length}`);
  for (const index of chunksTouched(volume, lo, hi)) {
    const painted = await readPaintedChunk(store, volume, index);
    const chunk = painted ? painted.labels : emptyLabels(volume, prod(volume.chunk_size));
    copyOverlap(volume, index, chunk, labels, lo, hi, "chunk");
    await store.put(volume.id, CHUNK_DIR + chunkKey(index), chunkBytes(chunk));
  }
}

// ---------------------------------------------------------------------------
// paint_box and restore_box
// ---------------------------------------------------------------------------

/** What paintBox did (fill.paint_box's return): how many voxels were written
 * with foreground (2 and up) and with background (1), and how many of those
 * held a label before. The routes answer these as filled_foreground,
 * filled_background and overwritten. */
export interface PaintCounts {
  n_foreground: number;
  n_background: number;
  n_overwritten: number;
}

/** `labels` as the volume's dtype (np.asarray(..., dtype=arr.dtype)): a
 * value past the dtype wraps as numpy's astype wraps it, which never
 * happens for ids from freshIds, which refuses them first. */
function asDtype(labels: Labels, like: Labels): Labels {
  if ((like instanceof Uint16Array) === (labels instanceof Uint16Array)) return labels;
  const out = labelsLike(like, labels.length);
  for (let i = 0; i < labels.length; i++) out[i] = labels[i];
  return out;
}

/** Write `labelsFor(existing)` into the box [lo, hi) (fill.paint_box).
 *
 * `labelsFor` gets the box as the store holds it (readBox) and returns labels
 * of the same length (0 for "leave as it is"). Without `overwrite` only
 * unannotated voxels (0) are written, as for a seed: what the user painted
 * stays. With it every voxel the labels cover is written, painted or not,
 * for a result the user has looked at and prefers to what is there; note
 * that the accept route (backend/routes/ai-annotate.ts, as the Python's
 * ai_annotate_accept) zeroes `labels == 1 & existing != 0` in its own
 * labelsFor before this sees them when overwrite is on, so the model's
 * background never erases paint; that is the route's rule, not this
 * function's, which writes exactly what the labels say. Nothing is written
 * when nothing would change (and nothing is pushed on `undo`). When
 * something is, `options.undo`, when given, gets `{lo, hi, before, after}`
 * pushed (pushUndo: UNDO_DEPTH entries at most) for restoreBox. */
export async function paintBox(
  store: PaintStore,
  volume: PaintVolume,
  lo: readonly number[],
  hi: readonly number[],
  labelsFor: (existing: Labels) => Labels,
  options: { overwrite: boolean; undo?: UndoEntry[] },
): Promise<PaintCounts> {
  const existing = await readBox(store, volume, lo, hi);
  const labels = asDtype(labelsFor(existing), existing);
  if (labels.length !== existing.length) throw new Error(`labelsFor returned ${labels.length} labels for a box of ${existing.length} voxels`);
  const write = new Uint8Array(existing.length);
  let nWrite = 0, nForeground = 0, nOverwritten = 0;
  for (let i = 0; i < existing.length; i++) {
    const w = options.overwrite ? labels[i] > 0 && labels[i] !== existing[i] : existing[i] === 0 && labels[i] > 0;
    if (!w) continue;
    write[i] = 1;
    nWrite++;
    if (labels[i] >= 2) nForeground++;
    if (existing[i] !== 0) nOverwritten++;
  }
  if (nWrite) {
    const before = existing.slice();
    for (let i = 0; i < existing.length; i++) if (write[i]) existing[i] = labels[i];
    await writeBox(store, volume, lo, hi, existing);
    if (options.undo) pushUndo(options.undo, { lo: [...lo], hi: [...hi], before, after: existing.slice() });
  }
  return { n_foreground: nForeground, n_background: nWrite - nForeground, n_overwritten: nOverwritten };
}

/** Put the box back to `entry.before` where it still holds `entry.after`
 * (fill.restore_box): an undo. Only voxels the action changed and nobody has
 * touched since go back, so a stroke painted in the box after the action is
 * kept. Returns how many voxels were restored. */
export async function restoreBox(store: PaintStore, volume: PaintVolume, entry: UndoEntry): Promise<number> {
  const { lo, hi, before, after } = entry;
  const current = await readBox(store, volume, lo, hi);
  if (before.length !== current.length || after.length !== current.length) throw new Error(`the undo entry's ${before.length} voxels are not the box's ${current.length}`);
  let restored = 0;
  for (let i = 0; i < current.length; i++) {
    if (before[i] !== after[i] && current[i] === after[i]) {
      current[i] = before[i];
      restored++;
    }
  }
  if (restored) await writeBox(store, volume, lo, hi, current);
  return restored;
}

/** Push `entry` on an undo stack that keeps the last UNDO_DEPTH entries (the
 * server's deque(maxlen=UNDO_DEPTH) per volume): the oldest goes when it is full. */
export function pushUndo(stack: UndoEntry[], entry: UndoEntry): void {
  stack.push(entry);
  while (stack.length > UNDO_DEPTH) stack.shift();
}
