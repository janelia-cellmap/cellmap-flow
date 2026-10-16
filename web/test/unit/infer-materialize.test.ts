// Inference.materialize in src/infer/engine.ts: a run on a region asks
// chunkmirage's Engine.materialize for the served layer's own view, at its
// one level (index 0, single-level serving), as zarr v2 with OME-Zarr 0.4
// metadata, gzip-compressed, empty chunks skipped, the box, the attributes,
// the signal and the progress callback passed through; a box outside the
// level and a model not served are refused before the engine is asked.
// chunkmirage's Engine is replaced by a stand-in with the members the
// wrapper touches, so this runs in node with no service worker or reader.
import { describe, expect, it, vi } from "vitest";

import type { RegionStore } from "../../src/contracts";
import { InferenceEngine, type ModelRunner } from "../../src/infer/engine";
import { memorySession } from "../../src/state";
import { entry } from "./backend-fakes";

const asked = vi.hoisted(() => ({ materialize: [] as { view: string; store: unknown; options: Record<string, unknown> }[], plans: [] as { view: string; options: Record<string, unknown> }[] }));

vi.mock("../../chunkmirage/web/src/engine", () => ({
  Engine: class {
    /** The raw volume as the engine opens it: one 8 nm level, 1000 voxels a side. */
    sources = {
      raw: {
        dtype: "uint8", channels: 1,
        axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
        levels: [{ shape: [1000, 1000, 1000], voxel: [8, 8, 8], origin: [0, 0, 0] }],
      },
    };
    async start() {}
    url(view: string) { return `http://localhost/virtual/page/${view}/`; }
    serveStages() {}
    unserve() {}
    stop() {}
    async materializePlan(view: string, options: Record<string, unknown>) {
      asked.plans.push({ view, options });
      return { peakBytes: 7, split: [0, 2] };
    }
    async materialize(view: string, store: unknown, options: Record<string, unknown>) {
      asked.materialize.push({ view, store, options });
      return { chunks: 1, written: 1, skipped: 0, bytes: 10, aborted: false };
    }
  },
}));

/** A runner that is never run: materialize goes through the engine, which is a stand-in here. */
const runner = {
  run: async () => { throw new Error("no forward pass in these tests"); },
  retain() {}, release: async () => {}, reset: async () => {},
  limits: async () => ({ maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false }),
  device: async () => "fake", provider: null,
} as unknown as ModelRunner;

const CHAIN = { input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }], postprocess: [{ name: "SigmoidPostprocessor" }] };

describe("Inference.materialize", () => {
  it("asks the engine for the layer's own view at its one level, as zarr v2, gzip, empty chunks skipped", async () => {
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const served = await inference.serve(entry("setup_54"), CHAIN, "blob", async () => new ArrayBuffer(8));
    const store: RegionStore = { put: async () => {} };
    const signal = new AbortController().signal;
    const onProgress = () => {};
    const written = await inference.materialize(served.view, store, { lo: [0, 60, 120], hi: [60, 120, 1000], attributes: { cellmap_flow: { model_name: "m" } }, signal, onProgress });
    expect(written).toEqual({ chunks: 1, written: 1, skipped: 0, bytes: 10, aborted: false });
    expect(asked.materialize).toEqual([{
      view: served.view, store,
      options: { level: 0, lo: [0, 60, 120], hi: [60, 120, 1000], extent: "box", format: "zarr2", compression: "gzip", skipEmpty: true, attributes: { cellmap_flow: { model_name: "m" } }, signal, onProgress },
    }]);
  });

  it("writes the chunks uncompressed when the run asks for none (Fiji reads no gzip)", async () => {
    asked.materialize.length = 0;
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const served = await inference.serve(entry("setup_54"), CHAIN, "blob", async () => new ArrayBuffer(8));
    await inference.materialize(served.view, { put: async () => {} }, { lo: [0, 60, 120], hi: [60, 120, 1000], compression: null });
    expect(asked.materialize[0].options.compression).toBeNull();
  });

  it("refuses a box outside the model's level, and a model that is not served, without asking the engine", async () => {
    asked.materialize.length = 0;
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const served = await inference.serve(entry("setup_54"), CHAIN, "blob", async () => new ArrayBuffer(8));
    const store: RegionStore = { put: async () => {} };
    // The 16 nm model reads the one 8 nm level as if it were: 1000 voxels a side.
    await expect(inference.materialize(served.view, store, { lo: [0, 0, 0], hi: [1001, 10, 10] })).rejects.toThrow("is not within the model's 1000×1000×1000 output voxels");
    await expect(inference.materialize(served.view, store, { lo: [5, 0, 0], hi: [5, 10, 10] })).rejects.toThrow("is not within");
    await expect(inference.materialize(served.view, store, { lo: [0.5, 0, 0], hi: [5, 10, 10] })).rejects.toThrow("is not within");
    await expect(inference.materialize("other__CFLOW_ARGS__b__CFLOW_ARGS__", store, { lo: [0, 0, 0], hi: [1, 1, 1] })).rejects.toThrow("other is not served");
    await served.dispose();
    await expect(inference.materialize(served.view, store, { lo: [0, 0, 0], hi: [1, 1, 1] })).rejects.toThrow("is not served");
    expect(asked.materialize).toEqual([]);
  });
  it("asks the engine for the plan of the box a run would write, as the run asks", async () => {
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const served = await inference.serve(entry("setup_54"), CHAIN, "blob", async () => new ArrayBuffer(8));
    expect(await inference.materializePlan(served.view, { lo: [0, 60, 120], hi: [60, 120, 1000] })).toEqual({ peakBytes: 7, split: [0, 2] });
    expect(asked.plans.at(-1)).toEqual({ view: served.view, options: { level: 0, lo: [0, 60, 120], hi: [60, 120, 1000], extent: "box", format: "zarr2" } });
  });
});
