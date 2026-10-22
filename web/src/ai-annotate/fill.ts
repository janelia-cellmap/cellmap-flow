// A whole box of an annotation volume read from and written to the paint
// store at once, and the undo stack the box writes push onto: the I/O
// around the box decisions, which are the Python both modes run
// (cellmap_flow/portable/fill.py). The Label it tools
// (backend/routes/view-labels.ts, through portable/fill_calls.py) and an
// accepted AI annotation (backend/ai-annotate-job.ts, through
// portable/ai_annotate_calls.py accept) read the box with readBox, hand it
// to that Python in the engine's worker, write the box it answers with
// writeBox and push the box as it was and as it is (pushUndo), as the
// server's finetune/session/fill.py does around the same functions over
// MinIO.
//
// What the server does through zarr over MinIO (`arr[box]` read and written)
// is done here chunk by chunk, the volume being the paint store's zarr v2
// chunks (finetune/volume.ts): readBox assembles the box from the chunks it
// touches (a chunk never painted is zeros, zarr's fill value), writeBox puts
// each touched chunk back whole, read-modify-write, as the fork's writable
// zarr source does when the brush paints.
//
// Where voxel 0 is: a volume's `dataset_offset_nm` is voxel 0's CENTRE (the
// OME translation, as volume.py's docstring and volume_corner_nm say, and as
// finetune/volume.ts planVolume sets it from the served level's origin), so
// its corner is half a voxel before it (volumeCornerNm, which the ranking's
// carry and planVolume place the grid by).
import type { PaintStore, PaintVolume } from "../contracts";
import { CHUNK_DIR, chunkKey, readPaintedChunk } from "../finetune/volume";
import { type Labels, UNDO_DEPTH, type UndoEntry } from "./types";

/** The world position of an annotation volume's voxel-0 lower corner, in nm
 * (volume.volume_corner_nm): `datasetOffsetNm` is voxel 0's centre, so the
 * corner is half a voxel before it on each axis; zeros when the volume has
 * no offset. */
export function volumeCornerNm(datasetOffsetNm: readonly number[] | null | undefined, outputVoxelSize: readonly number[]): number[] {
  return outputVoxelSize.map((v, a) => (datasetOffsetNm ? datasetOffsetNm[a] : 0) - v / 2);
}

// ---------------------------------------------------------------------------
// Boxes of the store's chunks
// ---------------------------------------------------------------------------

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** A zero array of the volume's label dtype. */
export function emptyLabels(volume: Pick<PaintVolume, "dtype">, n: number): Labels {
  return volume.dtype === "uint16" ? new Uint16Array(n) : new Uint8Array(n);
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
function chunkBytes(labels: Labels): Uint8Array {
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
// The undo stack
// ---------------------------------------------------------------------------

/** Push `entry` on an undo stack that keeps the last UNDO_DEPTH entries (the
 * server's deque(maxlen=UNDO_DEPTH) per volume): the oldest goes when it is full. */
export function pushUndo(stack: UndoEntry[], entry: UndoEntry): void {
  stack.push(entry);
  while (stack.length > UNDO_DEPTH) stack.shift();
}
