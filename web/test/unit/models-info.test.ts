// modelInfo: the /__control__/model_info answer for a browser-served model,
// and the output class the engine's serve puts in it, read off the network
// it runs (the fixtures of test/fixtures/make_onnx_fixture.py, which end in
// a Conv, the fp16 one then a Cast) by the portable rule in node's Pyodide,
// over a chunkmirage Engine stand-in with the members serve touches.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it, vi } from "vitest";

import type { Chain, GpuLimits } from "../../src/contracts";
import { InferenceEngine, type ModelRunner } from "../../src/infer/engine";
import { modelInfo, outputClassOf } from "../../src/shared/model-info";
import { memorySession } from "../../src/state";
import { entry } from "./backend-fakes";
import { entryFixture } from "./models-helpers";
import { portablePython } from "./pyodide";

vi.mock("../../chunkmirage/web/src/engine", async () => {
  const { portablePython } = await import("./pyodide");
  return {
    Engine: class {
      sources = {
        raw: {
          dtype: "uint8", channels: 1,
          axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
          levels: [{ shape: [1000, 1000, 1000], voxel: [16, 16, 16], origin: [0, 0, 0] }],
        },
      };
      ready = Promise.resolve();
      async start() {}
      url(view: string) { return `http://localhost/virtual/page/${view}/`; }
      serveStages() {}
      unserve() {}
      stop() {}
      async call(fn: string, args: Record<string, unknown>, arrays: ArrayBuffer[]) {
        const call = await portablePython();
        if (!call) throw new Error("no Pyodide in this run");
        return call(fn, args, arrays);
      }
    },
  };
});

const python = await portablePython();
const FIXTURES = resolve(import.meta.dirname, "../fixtures");

describe("modelInfo", () => {
  it("gives the geometry of the chosen variant, channels first", () => {
    const info = modelInfo(entryFixture("setup_54"));
    expect(info).toEqual({
      output_channels: 3,
      channels: ["mito_aff_z", "mito_aff_y", "mito_aff_x"],
      write_shape: [704, 704, 704], // 44 voxels at 16 nm
      read_shape: [2112, 2112, 2112], // 132 voxels
      output_voxel_size: [16, 16, 16],
      input_voxel_size: [16, 16, 16],
      effective_output_voxel_size: [16, 16, 16],
      has_channel: true,
      output_axes: ["c", "z", "y", "x"],
      available: true,
      output_class: "unbounded",
      output_class_from: "manifest",
      output_op: null,
      output_min: null,
      output_max: null,
      display_channel: null,
      engine: "browser",
    });
  });

  it("follows the variant: a bigger block on a bigger GPU", () => {
    const big: GpuLimits = { maxBufferSize: 4 * 1024 ** 3, maxStorageBufferBindingSize: null, shaderF16: true };
    const info = modelInfo(entryFixture("setup_53", big));
    expect(info.read_shape).toEqual([2752, 2752, 2752]); // 172 voxels at 16 nm
    expect(info.write_shape).toEqual([1344, 1344, 1344]); // 84 voxels
    expect(info.output_channels).toBe(3);
  });
});

describe("outputClassOf", () => {
  it("maps the manifest's output text to probe.py's classes", () => {
    expect(outputClassOf("raw logits (apply sigmoid / threshold at 0)")).toBe("unbounded");
    expect(outputClassOf("logits")).toBe("unbounded");
    expect(outputClassOf("signed distances")).toBe("unbounded");
    expect(outputClassOf("probabilities (sigmoid applied)")).toBe("unit");
    expect(outputClassOf("softmax over classes")).toBe("unit");
    expect(outputClassOf("tanh, in [-1, 1]")).toBe("signed_unit");
    expect(outputClassOf("")).toBeNull();
    expect(outputClassOf("something else")).toBeNull();
  });
});

describe.skipIf(!python)("a served model's output class", () => {
  const runner = {
    run: async () => { throw new Error("no forward pass in these tests"); },
    retain() {}, release: async () => {}, reset: async () => {},
    limits: async () => ({ maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false }),
    device: async () => "fake", provider: null,
  } as unknown as ModelRunner;
  const chain: Chain = { input_norm: [], postprocess: [] };

  it.each(["head.onnx", "head_fp16.onnx"])("is its network's, read off the bytes it runs, over a manifest that says otherwise: %s", async (file) => {
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const model = entry("setup_54");
    const saysSigmoid = { ...model, manifest: { ...model.manifest, output: "probabilities (sigmoid applied)" } };
    const bytes = readFileSync(resolve(FIXTURES, file));
    let reads = 0;
    const modelBytes = async () => {
      reads++;
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    };
    const served = await inference.serve(saysSigmoid, chain, "blob", modelBytes);
    expect(served.info).toMatchObject({ output_class: "unbounded", output_class_from: "graph", output_op: "Conv" });
    // Read once per file: the next serve, through another chain, reads nothing.
    await inference.serve(saysSigmoid, { ...chain, postprocess: [{ name: "SigmoidPostprocessor" }] }, "blob2", modelBytes);
    expect(reads).toBe(1);
  }, 120_000);  // Pyodide and numpy load the first time, a few seconds
});
