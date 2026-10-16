// Parity of the browser's Label it tools with the server's:
// web/src/label-it (segment.ts, prediction.ts) with
// cellmap_flow/dashboard/routes/finetune/view_labels.py (probability,
// segment_prediction, read_prediction), and the Label it parts of
// web/src/ai-annotate (labels.ts relabelObjects, fill.ts rewriteForeground)
// with cellmap_flow/finetune/session/fill.py (relabel_objects,
// rewrite_foreground, restore_box).
//
// The fixture (test/fixtures/label_it.json) was written by the Python
// functions themselves (test/fixtures/make_label_it_fixtures.py, whose
// docstring lists the sections): predictions of every output class turned
// into probabilities and segmented into objects by components and the mutex
// watershed; label boxes relabelled by Split Objects' rule, in memory and in
// a chunked volume with its undo; and a served array read under annotation
// boxes on other grids. Objects, ids and boxes match exactly, probabilities
// within the parity tolerance (scipy's float32 expit and the port's agree to
// an ulp). Regenerate the fixture when the Python changes;
// tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import { chunkBytes, readBox, restoreBox, rewriteForeground, writeBox } from "../../src/ai-annotate/fill";
import { relabelObjects } from "../../src/ai-annotate/labels";
import { AIAnnotateError, type Labels, type UndoEntry } from "../../src/ai-annotate/types";
import type { PaintVolume } from "../../src/contracts";
import { CHUNK_DIR, chunkKey } from "../../src/finetune/volume";
import { boxFromChunks } from "../../src/infer/output";
import { readPrediction } from "../../src/label-it/prediction";
import { type Block, probability, segmentPrediction } from "../../src/label-it/segment";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { type ArrayRecord, expectArrayMatches, loadFixture } from "./helpers";

type Counts = { objects: number; split: number; merged: number };
interface Fixture {
  seed: number;
  blocks: Record<string, ArrayRecord>;
  probability: { id: string; block: string; output_class: string | null; channels: number[] | null; probability: ArrayRecord }[];
  segment: {
    id: string; method: string; block: string; output_class: string | null; channels: number[] | null; offsets: number[][] | null;
    threshold: number; min_size: number; connectivity: number; per_slice: boolean; objects: ArrayRecord; count: number;
  }[];
  relabel_objects: { id: string; existing: ArrayRecord; connectivity: number; per_slice: boolean; labels?: ArrayRecord; counts?: Counts; error?: string }[];
  rewrite_foreground: {
    id: string; dtype: "uint8" | "uint16"; shape: number[]; chunks: number[]; initial: ArrayRecord; lo: number[]; hi: number[];
    connectivity: number; per_slice: boolean; changed: number; counts: Counts; after: ArrayRecord;
    undo: { before: ArrayRecord; after: ArrayRecord } | null;
    restore?: { stroke: [number[], number][]; restored: number; final: ArrayRecord };
  }[];
  read_prediction: {
    id: string; served: { shape: number[]; voxel: number[]; translation: number[]; chunks: number[] };
    volume: { output_voxel_size: number[]; dataset_offset_nm: number[] }; lo: number[]; hi: number[];
    prediction?: ArrayRecord; new_lo?: number[]; new_hi?: number[]; error?: { message: string; status: number };
  }[];
}

const fixture = loadFixture<Fixture>("label_it.json");

function blockOf(name: string): Block {
  const record = fixture.blocks[name];
  return { data: Float32Array.from(record.data), shape: record.shape };
}

/** A fixture array as the labels' typed array (uint8 or uint16, the paint layer's dtypes). */
function labelsOf(record: ArrayRecord): Labels {
  if (record.dtype === "uint16") return Uint16Array.from(record.data);
  if (record.dtype === "uint8") return Uint8Array.from(record.data);
  throw new Error(`${record.dtype} is not a labels dtype`);
}

describe("probability", () => {
  for (const c of fixture.probability) {
    it(c.id, () => {
      const out = probability(blockOf(c.block), c.output_class, c.channels);
      expect(out.shape).toEqual(c.probability.shape);
      expectArrayMatches(out.data, c.probability, c.id);
    });
  }
});

describe("segment_prediction", () => {
  for (const c of fixture.segment) {
    it(`${c.id} (${c.method})`, () => {
      const objects = segmentPrediction(c.method, blockOf(c.block), c.output_class, c.channels, c.offsets, {
        threshold: c.threshold, minSize: c.min_size, connectivity: c.connectivity, perSlice: c.per_slice,
      });
      expectArrayMatches(objects, c.objects, c.id);
      expect(Math.max(0, ...objects)).toBe(c.count);
    });
  }

  it("covers every method the browser offers, with objects to find", () => {
    expect(new Set(fixture.segment.map((c) => c.method))).toEqual(new Set(["components", "mutex_watershed"]));
    expect(fixture.segment.some((c) => c.method === "mutex_watershed" && c.count > 2)).toBe(true);
    expect(fixture.segment.some((c) => c.per_slice && c.count > 10)).toBe(true);
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

describe("read_prediction", () => {
  /** The served array's value at (c, z, y, x): make_label_it_fixtures' _served_values. */
  const valueAt = (c: number, z: number, y: number, x: number) => c * 1_000_000 + z * 10_000 + y * 100 + x;

  for (const c of fixture.read_prediction) {
    it(c.id, async () => {
      const [C, ...levelShape] = c.served.shape;
      const chunk = c.served.chunks;
      // The model's output as the engine serves it: whole chunks, the part
      // past the array zero (its fill), assembled into the box asked for.
      const infer = {
        readOutput: (_view: string, lo: number[], hi: number[]) =>
          boxFromChunks(lo, hi, chunk, C, async (index) => {
            const values = new Float32Array(C * chunk[0] * chunk[1] * chunk[2]);
            let i = 0;
            for (let ch = 0; ch < C; ch++) for (let z = 0; z < chunk[0]; z++) for (let y = 0; y < chunk[1]; y++) for (let x = 0; x < chunk[2]; x++, i++) {
              const at = [index[0] * chunk[0] + z, index[1] * chunk[1] + y, index[2] * chunk[2] + x];
              if (at.every((v, a) => v < levelShape[a])) values[i] = valueAt(ch, at[0], at[1], at[2]);
            }
            return values;
          }),
      };
      const level = { shape: levelShape, voxel: c.served.voxel, origin: c.served.translation };
      const read = await readPrediction(infer, "view", level, c.volume, c.lo, c.hi);
      if (c.error) {
        expect(read).toBeNull();
        return;
      }
      expect(read).not.toBeNull();
      expect([read!.lo, read!.hi]).toEqual([c.new_lo, c.new_hi]);
      expect(read!.prediction.shape).toEqual(c.prediction!.shape);
      expectArrayMatches(read!.prediction.data, c.prediction!, c.id);
    });
  }
});
