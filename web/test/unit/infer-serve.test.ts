// A model served again through a chain it was served through before (an
// Apply back to it): the viewer keeps what it fetched by URL, failures
// included, so the second serving needs a URL of its own, and the backend
// must still see that it serves through that chain.
import { describe, expect, it, vi } from "vitest";

import { servesAs } from "../../src/backend/serving";
import type { Chain } from "../../src/contracts";
import { InferenceEngine, type ModelRunner } from "../../src/infer/engine";
import { memorySession } from "../../src/state";
import { entry } from "./backend-fakes";

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
    serveStages() {}
    unserve() {}
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

describe("a chain served again", () => {
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
