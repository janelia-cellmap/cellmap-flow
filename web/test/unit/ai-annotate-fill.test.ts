// Labelling a box of an annotation volume in the paint store
// (src/ai-annotate/fill.ts) against cellmap_flow/finetune/session/fill.py:
// box_voxels's arithmetic, the box assembled from and written back to the
// store's chunks, paint_box's write rule and counts (the cases of
// tests/ai_annotate/test_paint_box.py: an 8^3 volume in 4^3 chunks with a
// painted corner), restore_box, and the undo stack's depth.
import { describe, expect, it } from "vitest";

import type { PaintVolume } from "../../src/contracts";
import { boxVoxels, chunkBytes, emptyLabels, paintBox, pushUndo, readBox, restoreBox, volumeCornerNm, writeBox } from "../../src/ai-annotate/fill";
import { type Labels, UNDO_DEPTH, type UndoEntry } from "../../src/ai-annotate/types";
import { CHUNK_DIR, chunkKey, labelsOf } from "../../src/finetune/volume";
import { MemoryPaintStore } from "../../src/state/paint-store";

function volumeOf(dtype: "uint8" | "uint16" = "uint8", shape = [8, 8, 8], chunk = [4, 4, 4]): PaintVolume {
  return {
    id: "vol-1", type: "annotation_volume", model_name: "m", model_id: "m", dataset_path: "x", chunk_size: chunk, input_size: chunk,
    output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8], dataset_offset_nm: [4, 4, 4], dataset_shape_voxels: shape, dtype,
    input_norm: [], postprocess: [], created_at: "2026-10-15T12:00:00Z",
  };
}

const at = (z: number, y: number, x: number, shape = [8, 8, 8]) => (z * shape[1] + y) * shape[2] + x;
const LO = [0, 0, 0], HI = [4, 4, 4], WHOLE = [8, 8, 8];

/** The store of the Python test: the user painted background (1) and an
 * object (5) in the first slice of the first chunk; every other chunk is
 * unpainted (no file). */
async function painted(dtype: "uint8" | "uint16" = "uint8") {
  const store = new MemoryPaintStore();
  const volume = volumeOf(dtype);
  const chunk = emptyLabels(volume, 64);
  chunk[at(0, 0, 0, [4, 4, 4])] = chunk[at(0, 0, 1, [4, 4, 4])] = 1;
  chunk[at(0, 1, 0, [4, 4, 4])] = chunk[at(0, 1, 1, [4, 4, 4])] = 5;
  await store.put(volume.id, CHUNK_DIR + chunkKey([0, 0, 0]), chunkBytes(chunk));
  return { store, volume };
}

/** An object (2) over the box's first two rows of y, background (1) elsewhere. */
function seed(existing: Labels): Labels {
  const labels = existing.slice().fill(1);
  const n = Math.round(Math.cbrt(existing.length));
  for (let z = 0; z < n; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < n; x++) labels[(z * n + y) * n + x] = 2;
  return labels;
}

describe("volumeCornerNm and boxVoxels", () => {
  const grid = { output_voxel_size: [8, 8, 8], dataset_offset_nm: [4, 4, 4] };

  it("the offset is voxel 0's centre; the corner is half a voxel before it", () => {
    expect(volumeCornerNm([4, 4, 4], [8, 8, 8])).toEqual([0, 0, 0]);
    expect(volumeCornerNm([100, 100, 100], [8, 16, 4])).toEqual([96, 92, 98]);
    expect(volumeCornerNm(null, [8, 8, 8])).toEqual([-4, -4, -4]);
  });

  it("takes the voxels whose centres lie in the box, clipped to the volume", () => {
    expect(boxVoxels(grid, [32, 32, 32], [8, 16, 24], [100, 200, 300])).toEqual({ lo: [4, 4, 4], hi: [5, 6, 7] });
    // [30, 42) in z holds the centres 36 (voxel 4) only; [36, 44) in y holds 36 only; [44, 52) in x holds 44 (voxel 5).
    expect(boxVoxels(grid, [30, 36, 44], [12, 8, 8], [100, 200, 300])).toEqual({ lo: [4, 4, 5], hi: [5, 5, 6] });
    expect(boxVoxels(grid, [-20, 1580, 2380], [40, 40, 40], [100, 200, 300])).toEqual({ lo: [0, 197, 297], hi: [2, 200, 300] });
    expect(boxVoxels(grid, [-100, 0, 0], [40, 40, 40], [100, 200, 300])).toBeNull();
    expect(boxVoxels(grid, [0, 0, 0], [3, 8, 8], [100, 200, 300])).toBeNull();  // no centre in [0, 3)
  });
});

describe("readBox and writeBox", () => {
  it("reads unpainted chunks as zeros and painted ones as stored, across chunk boundaries", async () => {
    const { store, volume } = await painted();
    const whole = await readBox(store, volume, LO, WHOLE);
    expect(whole.length).toBe(512);
    expect(whole[at(0, 0, 0)]).toBe(1);
    expect(whole[at(0, 1, 1)]).toBe(5);
    expect(whole[at(0, 2, 0)]).toBe(0);
    expect(whole[at(7, 7, 7)]).toBe(0);
    const part = await readBox(store, volume, [0, 1, 1], [1, 3, 5]);
    expect(Array.from(part)).toEqual([5, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("writes the touched chunks whole, little-endian for uint16, and leaves the others absent", async () => {
    const { store, volume } = await painted("uint16");
    const labels = new Uint16Array(2 * 2 * 2).fill(300);
    await writeBox(store, volume, [3, 3, 3], [5, 5, 5], labels);
    // The box [3, 5) on each axis touches the 8 chunks with indices 0 and 1,
    // one of them the painted one; the 19 others have no file still.
    const files = await store.list(volume.id, CHUNK_DIR);
    const touched = [];
    for (const z of [0, 1]) for (const y of [0, 1]) for (const x of [0, 1]) touched.push(CHUNK_DIR + chunkKey([z, y, x]));
    expect(files).toEqual(touched);
    const bytes = (await store.get(volume.id, `${CHUNK_DIR}1.1.1`))!;
    expect(bytes.length).toBe(128);
    expect([bytes[0], bytes[1]]).toEqual([300 & 0xff, 300 >> 8]);
    const chunk = labelsOf(volume, bytes);
    expect(chunk[0]).toBe(300);
    expect(chunk[1]).toBe(0);
    // The painted chunk keeps its paint beside the new labels.
    const first = labelsOf(volume, (await store.get(volume.id, `${CHUNK_DIR}0.0.0`))!);
    expect(first[at(0, 1, 1, [4, 4, 4])]).toBe(5);
    expect(first[at(3, 3, 3, [4, 4, 4])]).toBe(300);
    expect(Array.from(await readBox(store, volume, [3, 3, 3], [5, 5, 5]))).toEqual(Array.from(labels));
  });

  it("refuses a box outside the volume or labels of another size", async () => {
    const { store, volume } = await painted();
    await expect(readBox(store, volume, [0, 0, 0], [9, 4, 4])).rejects.toThrow(/not within the volume/);
    await expect(readBox(store, volume, [2, 0, 0], [2, 4, 4])).rejects.toThrow(/not within the volume/);
    await expect(writeBox(store, volume, LO, HI, new Uint8Array(10))).rejects.toThrow(/64 voxels, the labels 10/);
  });
});

describe("paintBox (fill.paint_box)", () => {
  it("without overwrite only unannotated voxels are written", async () => {
    const { store, volume } = await painted();
    const before = await readBox(store, volume, LO, WHOLE);
    const undo: UndoEntry[] = [];
    const counts = await paintBox(store, volume, LO, HI, seed, { overwrite: false, undo });
    const after = await readBox(store, volume, LO, WHOLE);
    // 2 x 4 x 4 object voxels and 2 x 4 x 4 background ones, less the 4 painted.
    expect(counts).toEqual({ n_foreground: 32 - 4, n_background: 32, n_overwritten: 0 });
    for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) expect(after[at(0, y, x)]).toBe(before[at(0, y, x)]);  // the painted voxels kept
    for (let z = 0; z < 4; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 4; x++) if (before[at(z, y, x)] === 0) expect(after[at(z, y, x)]).toBe(2);
    for (let z = 0; z < 4; z++) for (let y = 2; y < 4; y++) for (let x = 0; x < 4; x++) expect(after[at(z, y, x)]).toBe(1);
    for (let z = 4; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) expect(after[at(z, y, x)]).toBe(0);  // nothing outside the box
    for (let z = 0; z < 4; z++) for (let y = 4; y < 8; y++) for (let x = 0; x < 8; x++) expect(after[at(z, y, x)]).toBe(0);
    expect(undo.length).toBe(1);
    expect(undo[0].lo).toEqual(LO);
    expect(undo[0].hi).toEqual(HI);
    expect(undo[0].before[at(0, 2, 0, [4, 4, 4])]).toBe(0);
    expect(undo[0].after[at(0, 2, 0, [4, 4, 4])]).toBe(1);
  });

  it("with overwrite painted voxels are written too, and counted", async () => {
    const { store, volume } = await painted();
    const undo: UndoEntry[] = [];
    const counts = await paintBox(store, volume, LO, HI, seed, { overwrite: true, undo });
    const after = await readBox(store, volume, LO, WHOLE);
    for (let z = 0; z < 4; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 4; x++) expect(after[at(z, y, x)]).toBe(2);
    for (let z = 0; z < 4; z++) for (let y = 2; y < 4; y++) for (let x = 0; x < 4; x++) expect(after[at(z, y, x)]).toBe(1);
    // 32 object voxels, 2 of them the user's 5s and 2 the user's 1s; 32 background.
    expect(counts).toEqual({ n_foreground: 32, n_background: 32, n_overwritten: 4 });
    expect(undo.length).toBe(1);
  });

  it("with overwrite a voxel already holding the label is not written, nor one the labels leave 0", async () => {
    const { store, volume } = await painted();
    const labels = (existing: Labels) => {
      const out = seed(existing);
      out[at(0, 1, 0, [4, 4, 4])] = 5;  // the user's 5 stays a 5: no write
      out[at(0, 0, 0, [4, 4, 4])] = 0;  // left alone (an object voxel of the seed, so one foreground write fewer)
      return out;
    };
    const counts = await paintBox(store, volume, LO, HI, labels, { overwrite: true });
    // 32 object voxels less the 5 kept and the one left; 32 background; the
    // other 5 and the user's 1 at (0, 0, 1) overwritten with 2.
    expect(counts).toEqual({ n_foreground: 30, n_background: 32, n_overwritten: 2 });
    const after = await readBox(store, volume, LO, HI);
    expect(after[at(0, 1, 0, [4, 4, 4])]).toBe(5);
    expect(after[at(0, 0, 0, [4, 4, 4])]).toBe(1);
  });

  it("writes nothing, and pushes nothing on undo, when nothing would change", async () => {
    const { store, volume } = await painted();
    const undo: UndoEntry[] = [];
    let counts = await paintBox(store, volume, LO, HI, (existing) => existing.slice(), { overwrite: true, undo });
    expect(counts).toEqual({ n_foreground: 0, n_background: 0, n_overwritten: 0 });
    counts = await paintBox(store, volume, [4, 4, 4], WHOLE, (existing) => emptyLabels(volume, existing.length), { overwrite: false, undo });
    expect(counts).toEqual({ n_foreground: 0, n_background: 0, n_overwritten: 0 });
    expect(undo).toEqual([]);
    expect(await store.list(volume.id, CHUNK_DIR)).toEqual([`${CHUNK_DIR}0.0.0`]);  // no chunk file appeared
  });

  it("takes labels of another dtype as the volume's, and a box across chunks", async () => {
    const { store, volume } = await painted();
    const counts = await paintBox(store, volume, [1, 2, 3], [5, 6, 7], (existing) => new Uint16Array(existing.length).fill(9), { overwrite: false });
    expect(counts).toEqual({ n_foreground: 64, n_background: 0, n_overwritten: 0 });
    const after = await readBox(store, volume, LO, WHOLE);
    expect(after).toBeInstanceOf(Uint8Array);
    expect(after[at(1, 2, 3)]).toBe(9);
    expect(after[at(4, 5, 6)]).toBe(9);
    expect(after[at(0, 2, 3)]).toBe(0);
    expect(after[at(5, 5, 6)]).toBe(0);
    expect((await store.list(volume.id, CHUNK_DIR)).length).toBe(8);
    await expect(paintBox(store, volume, LO, HI, () => new Uint8Array(3), { overwrite: false })).rejects.toThrow(/3 labels for a box of 64/);
  });

  it("the accept route's rule is the route's: the labels given are written as they are", async () => {
    // With overwrite on, the Python's accept route zeroes labels == 1 where
    // existing != 0 before paint_box sees them (so the model's background
    // never erases paint); paintBox itself writes a 1 over a 5 when told to.
    const { store, volume } = await painted();
    const counts = await paintBox(store, volume, LO, HI, (existing) => existing.slice().fill(1), { overwrite: true });
    expect(counts.n_overwritten).toBe(2);  // the two 5s; the two 1s already were 1
    expect((await readBox(store, volume, LO, HI))[at(0, 1, 0, [4, 4, 4])]).toBe(1);
  });
});

describe("restoreBox (fill.restore_box)", () => {
  it("puts back what the action changed where it still holds the action's value", async () => {
    const { store, volume } = await painted();
    const undo: UndoEntry[] = [];
    await paintBox(store, volume, LO, HI, seed, { overwrite: false, undo });
    // Strokes after the action: over a voxel it wrote (kept), over one it
    // left (kept), and outside the box (untouched).
    await writeBox(store, volume, [0, 2, 0], [1, 3, 1], Uint8Array.of(9));
    await writeBox(store, volume, [0, 0, 0], [1, 1, 1], Uint8Array.of(9));
    await writeBox(store, volume, [6, 6, 6], [7, 7, 7], Uint8Array.of(9));
    const restored = await restoreBox(store, volume, undo[0]);
    expect(restored).toBe(60 - 1);  // the 60 written, less the one painted over since
    const after = await readBox(store, volume, LO, WHOLE);
    expect(after[at(0, 2, 0)]).toBe(9);
    expect(after[at(0, 0, 0)]).toBe(9);
    expect(after[at(6, 6, 6)]).toBe(9);
    expect(after[at(0, 1, 0)]).toBe(5);
    expect(after[at(3, 3, 3)]).toBe(0);
    expect(after[at(0, 2, 1)]).toBe(0);
    // Nothing left to restore: nothing written.
    expect(await restoreBox(store, volume, undo[0])).toBe(0);
  });

  it("refuses an entry that does not fit the box", async () => {
    const { store, volume } = await painted();
    await expect(restoreBox(store, volume, { lo: LO, hi: HI, before: new Uint8Array(3), after: new Uint8Array(64) })).rejects.toThrow(/3 voxels/);
  });
});

describe("pushUndo", () => {
  it("keeps the last UNDO_DEPTH entries", () => {
    const stack: UndoEntry[] = [];
    const entry = (n: number): UndoEntry => ({ lo: [n, 0, 0], hi: [n + 1, 1, 1], before: new Uint8Array(1), after: Uint8Array.of(1) });
    for (let n = 0; n < UNDO_DEPTH + 3; n++) pushUndo(stack, entry(n));
    expect(stack.length).toBe(UNDO_DEPTH);
    expect(stack[0].lo[0]).toBe(3);
    expect(stack[UNDO_DEPTH - 1].lo[0]).toBe(UNDO_DEPTH + 2);
  });
});
