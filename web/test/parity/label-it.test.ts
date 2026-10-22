// Parity of the browser's Label it tools with the server's, where the
// browser runs TypeScript of its own: a seed's mutex watershed
// (web/src/label-it/segment.ts mutexObjects) with
// cellmap_flow/dashboard/routes/finetune/view_labels.py's
// (segment_prediction, on mwatershed), and the Label it parts of
// web/src/ai-annotate (labels.ts relabelObjects, fill.ts rewriteForeground)
// with cellmap_flow/finetune/session/fill.py (relabel_objects,
// rewrite_foreground, restore_box). The rest of a seed is the Python both
// modes run (cellmap_flow/portable/label_it.py), tested by pytest.
//
// The fixture (test/fixtures/label_it.json) was written by the Python
// functions themselves (test/fixtures/make_label_it_fixtures.py, whose
// docstring lists the sections): affinities, as the browser's Python hands
// them to the watershed, segmented into objects; label boxes relabelled by
// Split Objects' rule, in memory and in a chunked volume with its undo.
// Objects and ids match exactly. Regenerate the fixture when the Python
// changes; tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import { chunkBytes, readBox, restoreBox, rewriteForeground, writeBox } from "../../src/ai-annotate/fill";
import { relabelObjects } from "../../src/ai-annotate/labels";
import { AIAnnotateError, type Labels, type UndoEntry } from "../../src/ai-annotate/types";
import type { PaintVolume } from "../../src/contracts";
import { CHUNK_DIR, chunkKey } from "../../src/finetune/volume";
import { mutexObjects } from "../../src/label-it/segment";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { type ArrayRecord, expectArrayMatches, loadFixture } from "./helpers";

type Counts = { objects: number; split: number; merged: number };
interface Fixture {
  seed: number;
  mutex_watershed: {
    id: string; output_class: string | null; offsets: number[][]; threshold: number; min_size: number;
    affinities: ArrayRecord; objects: ArrayRecord; count: number;
  }[];
  relabel_objects: { id: string; existing: ArrayRecord; connectivity: number; per_slice: boolean; labels?: ArrayRecord; counts?: Counts; error?: string }[];
  rewrite_foreground: {
    id: string; dtype: "uint8" | "uint16"; shape: number[]; chunks: number[]; initial: ArrayRecord; lo: number[]; hi: number[];
    connectivity: number; per_slice: boolean; changed: number; counts: Counts; after: ArrayRecord;
    undo: { before: ArrayRecord; after: ArrayRecord } | null;
    restore?: { stroke: [number[], number][]; restored: number; final: ArrayRecord };
  }[];
}

const fixture = loadFixture<Fixture>("label_it.json");

/** A fixture array as the labels' typed array (uint8 or uint16, the paint layer's dtypes). */
function labelsOf(record: ArrayRecord): Labels {
  if (record.dtype === "uint16") return Uint16Array.from(record.data);
  if (record.dtype === "uint8") return Uint8Array.from(record.data);
  throw new Error(`${record.dtype} is not a labels dtype`);
}

describe("mutex_watershed", () => {
  for (const c of fixture.mutex_watershed) {
    it(c.id, () => {
      const affinities = { data: Float32Array.from(c.affinities.data), shape: c.affinities.shape };
      const objects = mutexObjects(affinities, c.offsets, c.threshold, c.min_size);
      expectArrayMatches(objects, c.objects, c.id);
      expect(Math.max(0, ...objects)).toBe(c.count);
    });
  }

  it("has objects to find, and cases that drop some by size", () => {
    expect(fixture.mutex_watershed.some((c) => c.count > 2)).toBe(true);
    expect(fixture.mutex_watershed.some((c) => c.min_size > 1)).toBe(true);
  });
});

describe("relabel_objects", () => {
  for (const c of fixture.relabel_objects) {
    it(c.id, () => {
      const existing = labelsOf(c.existing);
      const run = () => relabelObjects(existing, c.existing.shape, { connectivity: c.connectivity, perSlice: c.per_slice });
      if (c.error) {
        expect(run).toThrow(AIAnnotateError);
        expect(run).toThrow(c.error);
        return;
      }
      const { labels, counts } = run();
      expect(labels.constructor).toBe(existing.constructor);
      expectArrayMatches(labels, c.labels!, c.id);
      expect(counts).toEqual(c.counts);
    });
  }
});

describe("rewrite_foreground and restore_box", () => {
  /** A paint volume of the case's shape and chunks in a fresh store, its
   * nonzero chunks written as the brush would have written them. */
  async function storeOf(c: Fixture["rewrite_foreground"][number]): Promise<{ store: MemoryPaintStore; volume: PaintVolume }> {
    const store = new MemoryPaintStore();
    const volume: PaintVolume = {
      id: "vol-1", type: "annotation_volume", model_name: "m", model_id: "m", dataset_path: "x", chunk_size: c.chunks, input_size: c.chunks,
      output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8], dataset_offset_nm: [4, 4, 4], dataset_shape_voxels: c.shape, dtype: c.dtype,
      input_norm: [], postprocess: [], created_at: "2026-10-16T12:00:00Z",
    };
    const initial = labelsOf(c.initial);
    const [nz, ny, nx] = c.shape, [cz, cy, cx] = c.chunks;
    for (let iz = 0; iz < nz / cz; iz++) for (let iy = 0; iy < ny / cy; iy++) for (let ix = 0; ix < nx / cx; ix++) {
      const chunk = c.dtype === "uint16" ? new Uint16Array(cz * cy * cx) : new Uint8Array(cz * cy * cx);
      let any = false;
      for (let z = 0; z < cz; z++) for (let y = 0; y < cy; y++) for (let x = 0; x < cx; x++) {
        const v = initial[((iz * cz + z) * ny + iy * cy + y) * nx + ix * cx + x];
        chunk[(z * cy + y) * cx + x] = v;
        if (v) any = true;
      }
      if (any) await store.put(volume.id, CHUNK_DIR + chunkKey([iz, iy, ix]), chunkBytes(chunk));
    }
    return { store, volume };
  }

  for (const c of fixture.rewrite_foreground) {
    it(c.id, async () => {
      const { store, volume } = await storeOf(c);
      const undo: UndoEntry[] = [];
      const shape = c.hi.map((h, a) => h - c.lo[a]);
      const result = await rewriteForeground(store, volume, c.lo, c.hi, (existing) => relabelObjects(existing, shape, { connectivity: c.connectivity, perSlice: c.per_slice }), { undo });
      expect(result).toEqual({ changed: c.changed, counts: c.counts });
      const whole = [0, 0, 0];
      expectArrayMatches(await readBox(store, volume, whole, c.shape), c.after, `${c.id}: after`);
      if (!c.undo) {
        expect(undo).toEqual([]);
        return;
      }
      expect(undo.length).toBe(1);
      expect([undo[0].lo, undo[0].hi]).toEqual([c.lo, c.hi]);
      expectArrayMatches(undo[0].before, c.undo.before, `${c.id}: undo before`);
      expectArrayMatches(undo[0].after, c.undo.after, `${c.id}: undo after`);
      if (c.restore) {
        for (const [index, value] of c.restore.stroke) {
          const one = c.dtype === "uint16" ? Uint16Array.of(value) : Uint8Array.of(value);
          await writeBox(store, volume, index, index.map((i) => i + 1), one);
        }
        expect(await restoreBox(store, volume, undo[0])).toBe(c.restore.restored);
        expectArrayMatches(await readBox(store, volume, whole, c.shape), c.restore.final, `${c.id}: restored`);
      }
    });
  }
});
