// The ONNX side of finetuning the head, against onnxruntime-web running the
// fixtures of test/fixtures/make_onnx_fixture.py (head.onnx, float32, and
// head_fp16.onnx, float16 between two Casts, both shaped like the real
// exports' tails): headOf finds the 1×1 Conv through the shape-only ops;
// withFeatureOutput's second output lays the features out as the trainer
// assumes, and the model's logits are W · features + b computed here (the
// reconstruction that makes training the head outside the graph sound);
// withHead's new weights change the model's output as predicted and an
// unchanged head round-trips the bytes exactly (the field-level protobuf
// reader and writer reproducing the file); metadata_props written in
// are read back, and onnxruntime still runs the model; and float16 bits,
// whether a runtime hands them back as bits or as a Float16Array, decode to
// their values.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it, vi } from "vitest";

import { asFloat32 } from "../../src/finetune/features";
import { mulberry32 } from "../../src/finetune/head-trainer";
import { float16ArrayToFloat32, float32ArrayToFloat16, headOf, withFeatureOutput, withHead } from "../../src/finetune/onnx-head";
import { parseFields, readMetadata, readModel, withMetadata } from "../../src/finetune/onnx-proto";
import type { Head } from "../../src/finetune/types";

// The page's ORT entry (WebGPU + WASM, which fetches its runtime over HTTP)
// replaced by the package's Node entry, the same WASM runtime read from disk.
vi.mock("onnxruntime-web/webgpu", () => import("onnxruntime-web"));

const FIXTURES = resolve(import.meta.dirname, "../fixtures");
const fixture = (name: string): Uint8Array => new Uint8Array(readFileSync(resolve(FIXTURES, name)));
const HEAD = fixture("head.onnx"), HEAD_FP16 = fixture("head_fp16.onnx");
const IN = [1, 1, 1, 8, 8], SPATIAL = 36;  // the fixture's input and its 6×6 output

/** A deterministic input block, values in [-1, 1). */
function input(seed = 1): Float32Array {
  const random = mulberry32(seed);
  return Float32Array.from({ length: 64 }, () => random() * 2 - 1);
}

/** W · features + b at every voxel, in the logits' layout ([1, C_out, ...]: channel outermost). */
function reconstruct(head: Head, weight: Float32Array, bias: Float32Array, features: Float32Array): Float32Array {
  const out = new Float32Array(head.outChannels * SPATIAL);
  for (let c = 0; c < head.outChannels; c++) for (let v = 0; v < SPATIAL; v++) {
    let z = bias[c];
    for (let i = 0; i < head.inChannels; i++) z += weight[c * head.inChannels + i] * features[i * SPATIAL + v];
    out[c * SPATIAL + v] = z;
  }
  return out;
}

async function runOrt(bytes: Uint8Array | ArrayBuffer, data: Float32Array, outputs?: string[]) {
  const session = await ort.InferenceSession.create(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes), { executionProviders: ["wasm"] });
  try {
    const results = await session.run({ input: new ort.Tensor("float32", data, IN) }, outputs ?? session.outputNames);
    return { names: session.outputNames, results };
  } finally {
    await session.release();
  }
}

describe("metadata_props", () => {
  it("writes entries after every other field, which stay byte for byte, and reads them back", () => {
    expect(readMetadata(HEAD)).toEqual({});
    const out = withMetadata(HEAD, { "cellmap_flow.manifest": '{"model":"x-ft1","channels":["a"]}', note: "é ✓" });
    expect(out.subarray(0, HEAD.length)).toEqual(HEAD);
    expect(readMetadata(out)).toEqual({ "cellmap_flow.manifest": '{"model":"x-ft1","channels":["a"]}', note: "é ✓" });
    // Each entry is one field 14, a StringStringEntryProto: 1 the key, 2 the value.
    const added = readModel(out).fields.filter((f) => f.number === 14);
    expect(added).toHaveLength(2);
    expect(parseFields(added[1].value).map((f) => [f.number, new TextDecoder().decode(f.value)])).toEqual([[1, "note"], [2, "é ✓"]]);
  });

  it("leaves the model one onnxruntime runs, with the same output", async () => {
    const data = input(3);
    const out = withMetadata(HEAD, { "cellmap_flow.manifest": "{}" });
    const before = (await runOrt(HEAD, data)).results.output.data as Float32Array;
    const session = await ort.InferenceSession.create(out, { executionProviders: ["wasm"] });
    try {
      const after = (await session.run({ input: new ort.Tensor("float32", data, IN) })).output.data as Float32Array;
      expect(Array.from(after)).toEqual(Array.from(before));
    } finally {
      await session.release();
    }
  });
});

describe("headOf", () => {
  it("finds the 1×1 Conv behind the Unsqueeze in head.onnx", () => {
    const head = headOf(HEAD);
    expect(head).toMatchObject({ featureTensor: "features", weightName: "head.weight", biasName: "head.bias", inChannels: 4, outChannels: 2, dtype: "float32" });
    expect(head.weight.length).toBe(8);
    expect(head.bias.length).toBe(2);
  });

  it("finds it behind the Cast of head_fp16.onnx and reads the weights as float32", () => {
    const head = headOf(HEAD_FP16), fp32 = headOf(HEAD);
    expect(head).toMatchObject({ featureTensor: "features", weightName: "head.weight", biasName: "head.bias", inChannels: 4, outChannels: 2, dtype: "float16" });
    for (let i = 0; i < 8; i++) expect(head.weight[i]).toBeCloseTo(fp32.weight[i], 2);
    expect(Array.from(head.weight)).toEqual(Array.from(float16ArrayToFloat32(float32ArrayToFloat16(fp32.weight))));
  });
});

describe("the features output and the head's weights in onnxruntime", () => {
  it("adds the feature map as a second output whose layout is the trainer's, and the logits are W · features + b", async () => {
    const head = headOf(HEAD);
    const data = input();
    const { names, results } = await runOrt(withFeatureOutput(HEAD, head), data);
    expect(names).toEqual(["output", "features"]);
    expect(results.features.dims).toEqual([1, 4, 6, 6]);  // channel on axis 1, the other axes the 1×6×6 chunk
    expect(results.output.dims).toEqual([1, 2, 1, 6, 6]);
    const features = results.features.data as Float32Array, logits = results.output.data as Float32Array;
    expect(Math.max(...features)).toBeGreaterThan(0);  // ReLU features that are not all dead
    const expected = reconstruct(head, head.weight, head.bias, features);
    for (let i = 0; i < expected.length; i++) expect(logits[i]).toBeCloseTo(expected[i], 4);
    // The model without the extra output computes the same logits.
    const base = await runOrt(HEAD, data);
    expect(Array.from(base.results.output.data as Float32Array)).toEqual(Array.from(logits));
  }, 30000);

  it("withHead writes new weights the model then computes with; the same weights change nothing", async () => {
    const head = headOf(HEAD);
    const random = mulberry32(7);
    const weight = Float32Array.from({ length: 8 }, () => random() * 2 - 1), bias = Float32Array.from([0.25, -1.5]);
    const changed = withHead(HEAD, head, weight, bias);
    expect(changed.byteLength).toBe(HEAD.byteLength);
    expect(headOf(changed)).toMatchObject({ weight, bias, featureTensor: "features" });
    const data = input(5);
    const { results } = await runOrt(withFeatureOutput(changed, head), data);
    const expected = reconstruct(head, weight, bias, results.features.data as Float32Array);
    const logits = results.output.data as Float32Array;
    for (let i = 0; i < expected.length; i++) expect(logits[i]).toBeCloseTo(expected[i], 4);
    const base = await runOrt(HEAD, data);
    expect(Array.from(base.results.output.data as Float32Array)).not.toEqual(Array.from(logits));
    expect(new Uint8Array(withHead(HEAD, head, head.weight, head.bias))).toEqual(HEAD);
  }, 30000);

  it("withHead writes float16 bits into a float16 model, which then computes with them", async () => {
    const head = headOf(HEAD_FP16);
    const weight = Float32Array.from([1.00048828125, -3.3, 0.1, 0.7, -0.25, 2.5, 0.01, 0]), bias = Float32Array.from([0.3333, -2]);
    const changed = withHead(HEAD_FP16, head, weight, bias);
    expect(changed.byteLength).toBe(HEAD_FP16.byteLength);
    const written = headOf(changed);
    expect(Array.from(written.weight)).toEqual(Array.from(float16ArrayToFloat32(float32ArrayToFloat16(weight))));
    const { results } = await runOrt(withFeatureOutput(changed, head), input(6));
    const expected = reconstruct(head, written.weight, written.bias, float16ArrayToFloat32(results.features.data as Uint16Array));
    const logits = results.output.data as Float32Array;
    for (let i = 0; i < expected.length; i++) expect(Math.abs(logits[i] - expected[i])).toBeLessThan(0.05 * Math.max(1, Math.abs(expected[i])));
    expect(new Uint8Array(withHead(HEAD_FP16, head, head.weight, head.bias))).toEqual(HEAD_FP16);
  }, 30000);
});

describe("float16", () => {
  it("round-trips every finite float16 bit pattern through float32", () => {
    const bits = Uint16Array.from({ length: 65536 }, (_, i) => i).filter((b) => (b & 0x7c00) !== 0x7c00 || (b & 0x3ff) === 0);  // not NaN
    expect(float32ArrayToFloat16(float16ArrayToFloat32(bits))).toEqual(bits);
  });

  it("decodes float16 bits handed back in a Uint16Array, and takes values handed back in a Float16Array as they are, not as bits", () => {
    const values = [0.37, -1.5, 3.75, 0];
    const bits = float32ArrayToFloat16(values);
    expect(Array.from(asFloat32({ type: "float16", data: bits }, "f"))).toEqual(Array.from(float16ArrayToFloat32(bits)));
    // Node 22 has no Float16Array; any typed array of the values stands in
    // for one, since what matters is that it is not a Uint16Array of bits.
    const F16 = (globalThis as { Float16Array?: new (v: number[]) => ArrayLike<number> }).Float16Array ?? Float64Array;
    const out = asFloat32({ type: "float16", data: new F16(values) as unknown as ort.Tensor["data"] }, "f");
    for (let i = 0; i < values.length; i++) expect(out[i]).toBeCloseTo(values[i], 2);
  });
});
