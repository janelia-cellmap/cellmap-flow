// The ONNX side of finetuning the head, against onnxruntime-web running the
// fixtures of test/fixtures/make_onnx_fixture.py (head.onnx, float32, and
// head_fp16.onnx, float16 between two Casts, both shaped like the real
// exports' tails): the field-level protobuf reader reproduces a file byte
// for byte; headOf finds the 1×1 Conv through the shape-only ops and
// refuses other last layers in words that name them; withFeatureOutput's
// second output lays the features out as the trainer assumes, and the
// model's logits are W · features + b computed here (the reconstruction
// that makes training the head outside the graph sound); withHead's new
// weights change the model's output as predicted and an unchanged head
// round-trips the bytes exactly; and the float16 conversions, by hand,
// agree with IEEE 754 on every bit pattern.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it, vi } from "vitest";

import { createFeatureRunner } from "../../src/finetune/features";
import { mulberry32 } from "../../src/finetune/head-trainer";
import { float16ArrayToFloat32, float16ToFloat32, float32ArrayToFloat16, float32ToFloat16, headOf, withFeatureOutput, withHead } from "../../src/finetune/onnx-head";
import { concatBytes, encodeFields, encodeLengthDelimited, encodeVarint, encodeVarintField, parseFields, readGraph, readModel, readVarint, withGraphOutput } from "../../src/finetune/onnx-proto";
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

const utf8 = (s: string) => new TextEncoder().encode(s);
const float32Raw = (values: number[]) => new Uint8Array(Float32Array.from(values).buffer);

// --- A ModelProto from parts, for the shapes the fixtures do not have ---
function attrInts(name: string, ints: number[]): Uint8Array {
  return encodeLengthDelimited(5, concatBytes([encodeLengthDelimited(1, utf8(name)), ...ints.map((i) => encodeVarintField(8, i)), encodeVarintField(20, 7)]));
}
function attrInt(name: string, i: number): Uint8Array {
  return encodeLengthDelimited(5, concatBytes([encodeLengthDelimited(1, utf8(name)), encodeVarintField(3, i), encodeVarintField(20, 2)]));
}
function node(opType: string, inputs: string[], outputs: string[], attributes: Uint8Array[] = [], name = ""): Uint8Array {
  return encodeLengthDelimited(1, concatBytes([
    ...inputs.map((s) => encodeLengthDelimited(1, utf8(s))),
    ...outputs.map((s) => encodeLengthDelimited(2, utf8(s))),
    ...(name ? [encodeLengthDelimited(3, utf8(name))] : []),
    encodeLengthDelimited(4, utf8(opType)),
    ...attributes,
  ]));
}
function tensor(name: string, dims: number[], dataType: number, raw: Uint8Array): Uint8Array {
  return encodeLengthDelimited(5, concatBytes([...dims.map((d) => encodeVarintField(1, d)), encodeVarintField(2, dataType), encodeLengthDelimited(8, utf8(name)), encodeLengthDelimited(9, raw)]));
}
function valueInfo(name: string, elemType: number): Uint8Array {
  return concatBytes([encodeLengthDelimited(1, utf8(name)), encodeLengthDelimited(2, encodeLengthDelimited(1, encodeVarintField(1, elemType)))]);
}
function model(nodes: Uint8Array[], initializers: Uint8Array[], outputs: string[]): Uint8Array {
  const graph = concatBytes([...nodes, ...initializers, encodeLengthDelimited(11, valueInfo("x", 1)), ...outputs.map((o) => encodeLengthDelimited(12, valueInfo(o, 1)))]);
  const opset = concatBytes([encodeLengthDelimited(1, utf8("")), encodeVarintField(2, 18)]);
  return concatBytes([encodeVarintField(1, 8), encodeLengthDelimited(7, graph), encodeLengthDelimited(8, opset)]);
}
/** A 4 → 2 head on "x" with the given Conv attributes and weight dims. */
function convModel(options: { kernel?: number[]; dims?: number[]; group?: number; bias?: boolean; after?: "Sigmoid" | "Relu" | null; dataType?: number } = {}): Uint8Array {
  const kernel = options.kernel ?? [1, 1], dims = options.dims ?? [2, 4, ...kernel], bias = options.bias ?? true, dataType = options.dataType ?? 1;
  const n = dims.reduce((a, b) => a * b, 1);
  const weightRaw = dataType === 11 ? new Uint8Array(Float64Array.from({ length: n }, (_, i) => i / 10).buffer) : float32Raw(Array.from({ length: n }, (_, i) => i / 10));
  const attributes = [attrInts("kernel_shape", kernel), attrInt("group", options.group ?? 1)];
  const convOut = options.after ? "pre" : "output";
  const nodes = [node("Conv", bias ? ["x", "w", "b"] : ["x", "w"], [convOut], attributes, "head")];
  if (options.after) nodes.push(node(options.after, ["pre"], ["output"]));
  const initializers = [tensor("w", dims, dataType, weightRaw), ...(bias ? [tensor("b", [2], dataType, dataType === 11 ? new Uint8Array(Float64Array.from([0.5, -0.5]).buffer) : float32Raw([0.5, -0.5]))] : [])];
  return model(nodes, initializers, ["output"]);
}

describe("protobuf fields", () => {
  it("round-trips varints of every size, including int64 below zero", () => {
    for (const v of [0, 1, 127, 128, 300, 2 ** 31, 2 ** 49 - 1, 2 ** 49, Number.MAX_SAFE_INTEGER]) {
      const bytes = encodeVarint(v);
      expect(readVarint(bytes, 0)).toEqual({ value: v, next: bytes.length });
    }
    const minusOne = encodeVarint(-1);
    expect(minusOne.length).toBe(10);
    expect(readVarint(minusOne, 0).value).toBe(2n ** 64n - 1n);
    expect(encodeVarint(2n ** 60n)).toEqual(encodeVarint(Number(2n ** 60n)));
    expect(() => readVarint(new Uint8Array([0x80]), 0)).toThrow(/runs past the end/);
  });

  it.each([["head.onnx", HEAD], ["head_fp16.onnx", HEAD_FP16]])("parses %s and encodes it back byte for byte, at every level", (_name, bytes) => {
    const fields = parseFields(bytes, 0);
    expect(encodeFields(fields)).toEqual(bytes);
    expect(fields.map((f) => f.number)).toContain(7);
    const graph = fields.find((f) => f.number === 7)!;
    expect(bytes.subarray(graph.start, graph.end)).toEqual(graph.value);
    const graphFields = parseFields(graph.value, graph.start);
    expect(encodeFields(graphFields)).toEqual(graph.value);
    for (const f of graphFields) expect(bytes.subarray(f.start, f.end)).toEqual(f.value);  // offsets are absolute
    const nodes = graphFields.filter((f) => f.number === 1);
    for (const n of nodes) expect(encodeFields(parseFields(n.value, n.start))).toEqual(n.value);
  });

  it("reads the graph: nodes with attributes, initializers with where their bytes are, the I/O names", () => {
    const { graph } = readModel(HEAD);
    const g = readGraph(graph);
    expect(g.inputs).toEqual(["input"]);
    expect(g.outputs).toEqual(["output"]);
    expect(g.nodes.map((n) => n.opType)).toEqual(["Constant", "Gather", "Conv", "Relu", "Conv", "Constant", "Unsqueeze"]);
    const head = g.nodes[4];
    expect(head).toMatchObject({ name: "head", inputs: ["features", "head.weight", "head.bias"], outputs: ["logits"] });
    expect(head.attributes.find((a) => a.name === "kernel_shape")).toMatchObject({ ints: [1, 1], type: 7 });
    expect(head.attributes.find((a) => a.name === "group")).toMatchObject({ int: 1, type: 2 });
    expect(g.nodes[1].attributes.find((a) => a.name === "axis")?.int).toBe(2);
    expect(g.initializers.map((t) => [t.name, t.dims, t.dataType])).toEqual([["body.weight", [4, 1, 3, 3], 1], ["body.bias", [4], 1], ["head.weight", [2, 4, 1, 1], 1], ["head.bias", [2], 1]]);
    const w = g.initializers[2];
    expect(w.rawData!.end - w.rawData!.start).toBe(8 * 4);
    expect(w.typedData).toBe(false);
    expect(w.external).toBe(false);
    // The raw bytes at those offsets are the weights: the first is finite and not 0.
    const first = new DataView(HEAD.buffer, HEAD.byteOffset + w.rawData!.start, 4).getFloat32(0, true);
    expect(Number.isFinite(first) && first !== 0).toBe(true);
  });

  it("adds a graph output and leaves every other byte as it was", () => {
    const out = withGraphOutput(HEAD, "features", 1);
    const before = readModel(HEAD), after = readModel(out);
    expect(readGraph(after.graph).outputs).toEqual(["output", "features"]);
    // The model's other fields, byte for byte, and the graph's old bytes as its prefix.
    const others = (m: typeof before, bytes: Uint8Array) => m.fields.filter((f) => f.number !== 7).map((f) => Array.from(bytes.subarray(f.tagStart, f.end)));
    expect(others(after, out)).toEqual(others(before, HEAD));
    expect(after.graph.value.subarray(0, before.graph.value.length)).toEqual(before.graph.value);
    expect(out.length).toBe(HEAD.length + (after.graph.value.length - before.graph.value.length) + (encodeVarint(after.graph.value.length).length - encodeVarint(before.graph.value.length).length));
  });
});

describe("headOf", () => {
  it("finds the 1×1 Conv behind the Unsqueeze in head.onnx", () => {
    const head = headOf(HEAD);
    expect(head).toMatchObject({ featureTensor: "features", weightName: "head.weight", biasName: "head.bias", inChannels: 4, outChannels: 2, dtype: "float32" });
    expect(head.weight.length).toBe(8);
    expect(head.bias.length).toBe(2);
    expect(headOf(HEAD.buffer.slice(HEAD.byteOffset, HEAD.byteOffset + HEAD.byteLength) as ArrayBuffer)).toEqual(head);  // an ArrayBuffer does as well
  });

  it("finds it behind the Cast of head_fp16.onnx and reads the weights as float32", () => {
    const head = headOf(HEAD_FP16), fp32 = headOf(HEAD);
    expect(head).toMatchObject({ featureTensor: "features", weightName: "head.weight", biasName: "head.bias", inChannels: 4, outChannels: 2, dtype: "float16" });
    for (let i = 0; i < 8; i++) expect(head.weight[i]).toBeCloseTo(fp32.weight[i], 2);
    expect(Array.from(head.weight)).toEqual(Array.from(float16ArrayToFloat32(float32ArrayToFloat16(fp32.weight))));
  });

  it("walks through a Sigmoid: the head's output is then the logits before it", () => {
    const head = headOf(convModel({ after: "Sigmoid" }));
    expect(head).toMatchObject({ featureTensor: "x", weightName: "w", biasName: "b", inChannels: 4, outChannels: 2 });
    expect(Array.from(head.bias)).toEqual([0.5, -0.5]);
    expect(head.weight[7]).toBeCloseTo(0.7, 6);
  });

  it("gives a head without bias zeros and no bias name", () => {
    const head = headOf(convModel({ bias: false }));
    expect(head.biasName).toBeNull();
    expect(Array.from(head.bias)).toEqual([0, 0]);
  });

  it("refuses other last layers, naming what it found", () => {
    expect(() => headOf(convModel({ kernel: [3, 3] }))).toThrow("the model's last layer is a 3×3 convolution; the browser trains a 1×1 head only");
    expect(() => headOf(convModel({ group: 2, dims: [2, 2, 1, 1] }))).toThrow(/grouped convolution \(group 2\)/);
    expect(() => headOf(convModel({ after: "Relu" }))).toThrow(/comes from a Relu, not from a convolution/);
    expect(() => headOf(convModel({ dataType: 11 }))).toThrow(/is float64, not float32 or float16/);
    expect(() => headOf(model([node("Identity", ["x"], ["output"])], [], ["output"]))).toThrow(/no node of the graph produces "x"/);
    expect(() => headOf(model([node("Conv", ["x", "missing", "b"], ["output"], [attrInts("kernel_shape", [1, 1])])], [], ["output"]))).toThrow(/weight missing is not an initializer/);
    expect(() => headOf(new Uint8Array([0x08, 0x08]))).toThrow(/no graph/);
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

  it("does the same for the float16 model: float16 features, converted, reconstruct the float32 logits", async () => {
    const head = headOf(HEAD_FP16);
    const data = input(2);
    const { results } = await runOrt(withFeatureOutput(HEAD_FP16, head), data);
    expect(results.features.type).toBe("float16");
    expect(results.output.type).toBe("float32");
    const features = float16ArrayToFloat32(results.features.data as Uint16Array), logits = results.output.data as Float32Array;
    const expected = reconstruct(head, head.weight, head.bias, features);
    for (let i = 0; i < expected.length; i++) expect(Math.abs(logits[i] - expected[i])).toBeLessThan(0.02 * Math.max(1, Math.abs(expected[i])));
  }, 30000);

  it("createFeatureRunner returns both as float32 with their dims, one pass at a time, and releases the session", async () => {
    const head = headOf(HEAD_FP16);
    const lines: string[] = [];
    const runner = await createFeatureRunner({ bytes: HEAD_FP16, head, log: (l) => lines.push(l) });
    expect(runner.provider).toBe("wasm");
    expect(lines.join("\n")).toMatch(/features session ready on WASM/);
    const [a, b] = await Promise.all([runner.run(input(3), IN), runner.run(input(4), IN)]);
    expect(a.featuresDims).toEqual([1, 4, 6, 6]);
    expect(a.logitsDims).toEqual([1, 2, 1, 6, 6]);
    expect(a.features).toBeInstanceOf(Float32Array);
    expect(a.logits).toBeInstanceOf(Float32Array);
    expect(Array.from(a.logits)).not.toEqual(Array.from(b.logits));
    const expected = reconstruct(head, head.weight, head.bias, a.features);
    for (let i = 0; i < expected.length; i++) expect(Math.abs(a.logits[i] - expected[i])).toBeLessThan(0.02 * Math.max(1, Math.abs(expected[i])));
    await runner.dispose();
    await expect(createFeatureRunner({ bytes: HEAD, head: headOf(HEAD), inputName: "nope" })).rejects.toThrow(/no input nope \(it has input\)/);
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
    expect(() => withHead(HEAD, head, new Float32Array(7), bias)).toThrow(/7 weights for a 2 × 4 head/);
  }, 30000);

  it("withHead writes float16 bits into a float16 model, rounded to nearest even", async () => {
    const head = headOf(HEAD_FP16);
    // The extremes, read back: ties to even, saturation, the smallest subnormal.
    const extremes = Float32Array.from([1.00048828125, 1.00146484375, -3.3, 0.1, 65519, 1e-8, 2 ** -24, 0]);
    expect(Array.from(headOf(withHead(HEAD_FP16, head, extremes, head.bias)).weight)).toEqual([1, 1.001953125, float16ToFloat32(float32ToFloat16(-3.3)), float16ToFloat32(float32ToFloat16(0.1)), 65504, 0, 2 ** -24, 0]);
    // Weights the float16 arithmetic can hold, run.
    const weight = Float32Array.from([1.00048828125, -3.3, 0.1, 0.7, -0.25, 2.5, 0.01, 0]), bias = Float32Array.from([0.3333, -2]);
    const changed = withHead(HEAD_FP16, head, weight, bias);
    expect(changed.byteLength).toBe(HEAD_FP16.byteLength);
    const written = headOf(changed);
    expect(Array.from(written.weight)).toEqual(Array.from(float16ArrayToFloat32(float32ArrayToFloat16(weight))));
    expect(Array.from(written.bias)).toEqual([float16ToFloat32(float32ToFloat16(0.3333)), -2]);
    const { results } = await runOrt(withFeatureOutput(changed, head), input(6));
    const expected = reconstruct(head, written.weight, written.bias, float16ArrayToFloat32(results.features.data as Uint16Array));
    const logits = results.output.data as Float32Array;
    for (let i = 0; i < expected.length; i++) expect(Math.abs(logits[i] - expected[i])).toBeLessThan(0.05 * Math.max(1, Math.abs(expected[i])));
    expect(new Uint8Array(withHead(HEAD_FP16, head, head.weight, head.bias))).toEqual(HEAD_FP16);
  }, 30000);

  it("withHead skips the bias of a head that has none", () => {
    const bytes = convModel({ bias: false }), head = headOf(bytes);
    const out = withHead(bytes, head, new Float32Array(8).fill(2), Float32Array.from([9, 9]));
    expect(headOf(out)).toMatchObject({ biasName: null, bias: new Float32Array([0, 0]) });
    expect(Array.from(headOf(out).weight)).toEqual(new Array(8).fill(2));
  });
});

describe("float16 by hand", () => {
  it("converts known values both ways", () => {
    const known: [number, number][] = [[0, 0x0000], [1, 0x3c00], [-2, 0xc000], [0.5, 0x3800], [65504, 0x7bff], [2 ** -14, 0x0400], [2 ** -24, 0x0001], [0.0999755859375, 0x2e66], [Infinity, 0x7c00], [-Infinity, 0xfc00]];
    for (const [value, bits] of known) {
      expect(float32ToFloat16(value)).toBe(bits);
      expect(float16ToFloat32(bits)).toBe(value);
    }
    expect(Object.is(float16ToFloat32(0x8000), -0)).toBe(true);
    expect(float32ToFloat16(-0)).toBe(0x8000);
    expect(Number.isNaN(float16ToFloat32(0x7e00))).toBe(true);
    expect(float32ToFloat16(NaN) & 0x7c00).toBe(0x7c00);
    expect(float32ToFloat16(NaN) & 0x3ff).not.toBe(0);
  });

  it("rounds to nearest, ties to even, and saturates to infinity or zero", () => {
    expect(float32ToFloat16(1 + 2 ** -11)).toBe(0x3c00);  // halfway between 1 and the next half: even
    expect(float32ToFloat16(1 + 3 * 2 ** -11)).toBe(0x3c02);  // halfway: even again, upwards this time
    expect(float32ToFloat16(1 + 2 ** -11 + 2 ** -20)).toBe(0x3c01);  // just above halfway: up
    expect(float32ToFloat16(65519)).toBe(0x7bff);  // below the midpoint to 65536: the largest finite
    expect(float32ToFloat16(65520)).toBe(0x7c00);  // the midpoint: even is infinity
    expect(float32ToFloat16(70000)).toBe(0x7c00);
    expect(float32ToFloat16(2 ** -25)).toBe(0);  // the midpoint to the smallest subnormal: even is zero
    expect(float32ToFloat16(3 * 2 ** -25)).toBe(0x0002);  // the midpoint between the two smallest: even
    expect(float32ToFloat16(2 ** -25 + 2 ** -30)).toBe(0x0001);
    expect(float32ToFloat16(1e-9)).toBe(0);
    expect(float32ToFloat16(2 ** -14 - 2 ** -25)).toBe(0x0400);  // rounds up into the smallest normal
  });

  it("round-trips every finite float16 bit pattern through float32", () => {
    const bits = Uint16Array.from({ length: 65536 }, (_, i) => i).filter((b) => (b & 0x7c00) !== 0x7c00 || (b & 0x3ff) === 0);  // not NaN
    expect(float32ArrayToFloat16(float16ArrayToFloat32(bits))).toEqual(bits);
  });
});
