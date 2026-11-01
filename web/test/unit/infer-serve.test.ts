// A model's chains in the browser engine. The viewer keeps what it fetched
// by URL, failures included. A chain whose layer was replaced (an Apply)
// stays served while its model is, as a model server's would, and comes
// back under its URL; a view unserved and served again (its model removed,
// then added back) needs a URL of its own, and the backend must still see
// that it serves through that chain.
import { describe, expect, it, vi } from "vitest";

import { servesAs } from "../../src/backend/serving";
import type { Chain } from "../../src/contracts";
import { InferenceEngine, type ModelRunner } from "../../src/infer/engine";
import { memorySession } from "../../src/state";
import { entry } from "./backend-fakes";

const engineCalls = vi.hoisted(() => ({ served: [] as string[], unserved: [] as string[] }));

vi.mock("../../chunkmirage/web/src/engine", () => ({
  Engine: class {
    sources = {
      raw: {
        dtype: "uint8", channels: 1,
        axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
        levels: [{ shape: [1000, 1000, 1000], voxel: [8, 8, 8], origin: [0, 0, 0] }],
      },
    };
    ready = Promise.resolve();
    async start() {}
    url(view: string) { return `http://localhost/virtual/page/${view}/`; }
    serveStages(view: string) { engineCalls.served.push(view); }
    unserve(view: string) { engineCalls.unserved.push(view); }
    stop() {}
  },
}));

const runner = {
  run: async () => { throw new Error("no forward pass in these tests"); },
  retain() {}, release: async () => {}, reset: async () => {},
  limits: async () => ({ maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false }),
  device: async () => "fake", provider: null,
} as unknown as ModelRunner;
const SIGMOID: Chain = { input_norm: [], postprocess: [{ name: "SigmoidPostprocessor" }] };
const THRESHOLD: Chain = { input_norm: [], postprocess: [{ name: "ThresholdPostprocessor", threshold: "0.5" }] };

describe("a chain applied back", () => {
  it("comes back as it was, under its URL, and goes with its model", async () => {
    engineCalls.served.length = engineCalls.unserved.length = 0;
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const model = entry("setup_54");
    const bytes = async () => new ArrayBuffer(8);
    const sigmoid = await inference.serve(model, SIGMOID, "sigmoid", bytes);
    const threshold = await inference.serve(model, THRESHOLD, "threshold", bytes);
    await sigmoid.park();
    expect(inference.served()).toEqual([threshold]);
    const servedBefore = engineCalls.served.length;
    const back = await inference.serve(model, SIGMOID, "sigmoid", bytes);
    expect(back.url).toBe(sigmoid.url);
    expect(engineCalls.served).toHaveLength(servedBefore);
    expect(engineCalls.unserved).toEqual([]);
    // The model removed: every chain it was served through stops.
    await threshold.park();
    await back.dispose();
    expect(engineCalls.unserved.sort()).toEqual([back.view, threshold.view].sort());
    expect(inference.served()).toEqual([]);
  });
});

describe("a chain served again after it was unserved", () => {
  it("gets a URL of its own, and still counts as served through its chain", async () => {
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const model = entry("setup_54");
    const bytes = async () => new ArrayBuffer(8);
    const first = await inference.serve(model, SIGMOID, "sigmoid", bytes);
    await first.dispose();
    await (await inference.serve(model, THRESHOLD, "threshold", bytes)).dispose();
    const again = await inference.serve(model, SIGMOID, "sigmoid", bytes);
    expect(again.url).not.toBe(first.url);
    expect(servesAs(again, model, "sigmoid")).toBe(true);
    expect(servesAs(again, model, "threshold")).toBe(false);
  });
});
