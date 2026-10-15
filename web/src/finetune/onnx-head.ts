// A model's head in its ONNX bytes: the last 1×1 convolution, found by
// walking back from the graph's output; the model with the convolution's
// input, the feature map, as a second graph output, so one run yields
// features and logits; and the model with new weights in the convolution's
// initializers. Why the head trains and nothing else is in types.ts.
//
// The exports in cellmap/onnx-web-models (checked 2026-10-15) all end the
// same way: `final_conv.*.weight` [C_out, 32, 1, 1] and a bias, a Conv on a
// 32-channel feature map, then shape-only ops (Unsqueeze, Transpose, and in
// the fp16 exports a Cast back to float32) to `output`. None has a sigmoid,
// but one that did would be walked through too: the head's output is then
// the logits before it, which is what the trainer's BCE-with-logits wants.
// The weights are read as float32 whatever the file stores (fp16 exports
// store float16), and written back in the file's own type: Node 22 and the
// browsers have no Float16Array, so the conversion is done by hand.
import type { Head } from "./types";
import { type Graph, type Initializer, type Node, readGraph, readModel, withGraphOutput } from "./onnx-proto";

/** Ops that move values around without changing them: walked through from the output to the head. */
export const SHAPE_ONLY_OPS: ReadonlySet<string> = new Set(["Unsqueeze", "Squeeze", "Transpose", "Reshape", "Identity", "Cast", "Flatten"]);

const FLOAT32 = 1, FLOAT16 = 10;
const DATA_TYPE_NAMES: Record<number, string> = {
  1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64", 8: "string", 9: "bool",
  10: "float16", 11: "float64", 12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128", 16: "bfloat16",
};

const asBytes = (bytes: ArrayBuffer | Uint8Array): Uint8Array => (bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));

/** The model's head: from the graph's first output back through the
 * shape-only ops (and a Sigmoid, if any) to a Conv, which must be a 1×1 (or
 * 1×1×1), ungrouped, unstrided, unpadded convolution with its weight (and
 * bias, if it has one) in float32 or float16 initializers. */
export function headOf(bytes: ArrayBuffer | Uint8Array): Head {
  const graph = readGraph(readModel(asBytes(bytes)).graph);
  const conv = headConv(graph);
  const [featureTensor, weightName, biasName = ""] = conv.inputs;
  if (!featureTensor || !weightName) throw new Error(`the head convolution "${conv.name}" names no input or weight`);
  const group = attribute(conv, "group")?.int ?? 1;
  if (group !== 1) throw new Error(`the model's last layer is a grouped convolution (group ${group}); the browser trains a 1×1 head only`);
  const weight = initializer(graph, weightName, "weight");
  if (weight.dims.length < 2) throw new Error(`the head's weight ${weightName} has dims [${weight.dims.join(", ")}], not [out, in, 1, ...]`);
  const kernel = attribute(conv, "kernel_shape")?.ints ?? weight.dims.slice(2);
  if (kernel.some((k) => k !== 1)) throw new Error(`the model's last layer is a ${kernel.join("×")} convolution; the browser trains a 1×1 head only`);
  if (weight.dims.slice(2).some((k) => k !== 1)) throw new Error(`the head's weight ${weightName} has dims [${weight.dims.join(", ")}], not a 1×1 kernel's`);
  for (const [name, unit] of [["strides", 1], ["dilations", 1], ["pads", 0]] as const) {
    const values = attribute(conv, name)?.ints;
    if (values?.some((v) => v !== unit)) throw new Error(`the model's last layer has ${name} [${values.join(", ")}]; the browser trains a plain 1×1 head only`);
  }
  const [outChannels, inChannels] = weight.dims;
  const dtype = dtypeOf(weight, weightName);
  const weightValues = values(asBytes(bytes), weight, outChannels * inChannels, weightName);
  let biasValues: Float32Array = new Float32Array(outChannels);
  if (biasName) {
    const bias = initializer(graph, biasName, "bias");
    if (dtypeOf(bias, biasName) !== dtype) throw new Error(`the head's bias ${biasName} is ${dtypeOf(bias, biasName)} where its weight is ${dtype}`);
    if (bias.dims.length !== 1 || bias.dims[0] !== outChannels) throw new Error(`the head's bias ${biasName} has dims [${bias.dims.join(", ")}] for ${outChannels} output channels`);
    biasValues = values(asBytes(bytes), bias, outChannels, biasName);
  }
  return { featureTensor, weightName, biasName: biasName || null, inChannels, outChannels, weight: weightValues, bias: biasValues, dtype };
}

/** The model with the head's feature map as a second graph output, in the
 * head's own element type. Unchanged when it already is one. */
export function withFeatureOutput(bytes: ArrayBuffer | Uint8Array, head: Head): Uint8Array {
  const model = asBytes(bytes);
  if (readGraph(readModel(model).graph).outputs.includes(head.featureTensor)) return model;
  return withGraphOutput(model, head.featureTensor, head.dtype === "float16" ? FLOAT16 : FLOAT32);
}

/** A copy of the model with `weight` ([outChannels × inChannels], as
 * Head.weight) and `bias` written over the head's initializers, in the
 * file's own type (float16 rounded to nearest, ties to even); the bias is
 * dropped when the head has none. The bytes are otherwise identical. */
export function withHead(bytes: ArrayBuffer | Uint8Array, head: Head, weight: Float32Array, bias: Float32Array): ArrayBuffer {
  if (weight.length !== head.outChannels * head.inChannels) throw new Error(`${weight.length} weights for a ${head.outChannels} × ${head.inChannels} head`);
  if (bias.length !== head.outChannels) throw new Error(`${bias.length} biases for ${head.outChannels} output channels`);
  const copy = new Uint8Array(asBytes(bytes));  // a fresh buffer of exactly the model's bytes
  const graph = readGraph(readModel(copy).graph);
  write(copy, initializer(graph, head.weightName, "weight"), weight, head);
  if (head.biasName !== null) write(copy, initializer(graph, head.biasName, "bias"), bias, head);
  return copy.buffer;
}

/** The Conv the graph's first output comes from, through the shape-only ops. */
function headConv(graph: Graph): Node {
  if (!graph.outputs.length) throw new Error("the model's graph has no output");
  const producers = new Map<string, Node>();
  for (const node of graph.nodes) for (const out of node.outputs) if (out) producers.set(out, node);
  let name = graph.outputs[0];
  for (let steps = 0; steps <= graph.nodes.length; steps++) {
    const node = producers.get(name);
    if (!node) throw new Error(`no node of the graph produces "${name}" (the output is an input or an initializer passed through)`);
    if (node.opType === "Conv") return node;
    if (SHAPE_ONLY_OPS.has(node.opType) || node.opType === "Sigmoid") {
      name = node.inputs[0];
      continue;
    }
    throw new Error(`the model's output comes from a ${node.opType}${node.name ? ` ("${node.name}")` : ""}, not from a convolution; the browser trains a 1×1 convolution head only`);
  }
  throw new Error("the graph's output never reaches a convolution (a cycle of shape-only ops)");
}

function attribute(node: Node, name: string) {
  return node.attributes.find((a) => a.name === name) ?? null;
}

function initializer(graph: Graph, name: string, what: string): Initializer {
  const init = graph.initializers.find((t) => t.name === name);
  if (!init) throw new Error(`the head's ${what} ${name} is not an initializer of the graph (a constant node's, or computed?)`);
  if (init.external) throw new Error(`the head's ${what} ${name} is stored outside the model file (external data)`);
  if (!init.rawData) throw new Error(`the head's ${what} ${name} is stored ${init.typedData ? "in a typed field, not raw_data" : "nowhere (no raw_data)"}`);
  return init;
}

function dtypeOf(init: Initializer, name: string): Head["dtype"] {
  if (init.dataType === FLOAT32) return "float32";
  if (init.dataType === FLOAT16) return "float16";
  throw new Error(`the head's ${name} is ${DATA_TYPE_NAMES[init.dataType] ?? `data type ${init.dataType}`}, not float32 or float16`);
}

/** `count` values of the initializer's raw_data (little-endian, as ONNX stores it) as float32. */
function values(model: Uint8Array, init: Initializer, count: number, name: string): Float32Array {
  const { start, end } = init.rawData!;
  const size = init.dataType === FLOAT16 ? 2 : 4;
  if (end - start !== count * size) throw new Error(`the head's ${name} holds ${end - start} bytes, not the ${count * size} its dims [${init.dims.join(", ")}] take`);
  const view = new DataView(model.buffer, model.byteOffset + start, end - start);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = size === 2 ? float16ToFloat32(view.getUint16(2 * i, true)) : view.getFloat32(4 * i, true);
  return out;
}

function write(model: Uint8Array, init: Initializer, data: Float32Array, head: Head): void {
  const { start, end } = init.rawData!;
  const size = head.dtype === "float16" ? 2 : 4;
  if (end - start !== data.length * size) throw new Error(`${init.name} holds ${end - start} bytes; ${data.length} ${head.dtype} values take ${data.length * size}`);
  const view = new DataView(model.buffer, model.byteOffset + start, end - start);
  for (let i = 0; i < data.length; i++) {
    if (size === 2) view.setUint16(2 * i, float32ToFloat16(data[i]), true);
    else view.setFloat32(4 * i, data[i], true);
  }
}

// ---------------------------------------------------------------------------
// IEEE 754 binary16 by hand

/** The float32 value of float16 bits. */
export function float16ToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >> 10) & 0x1f, fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;  // zero or subnormal
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);

/** The float16 bits nearest `value` (ties to even, as hardware converts);
 * past 65504 in magnitude it is infinity, below 2^-25 it is zero. */
export function float32ToFloat16(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000, exponent = (x >>> 23) & 0xff, mantissa = x & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 | (mantissa >>> 13) : 0);  // infinity, or a quiet NaN
  const e = exponent - 127 + 15;  // the exponent rebiased for half
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    // A subnormal half, or zero: the significand with its hidden bit, shifted
    // down to the 2^-24 unit, rounded to nearest even. A carry out of the ten
    // fraction bits makes the smallest normal, which is right.
    if (e < -10) return sign;
    const m = mantissa | 0x800000, shift = 14 - e;
    let half = m >>> shift;
    const rest = m & ((1 << shift) - 1), halfway = 1 << (shift - 1);
    if (rest > halfway || (rest === halfway && half & 1)) half++;
    return sign | half;
  }
  let half = sign | (e << 10) | (mantissa >>> 13);
  const rest = mantissa & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++;  // a carry rolls into the exponent, and past 0x7bff makes infinity, both right
  return half;
}

export function float16ArrayToFloat32(bits: ArrayLike<number>): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) out[i] = float16ToFloat32(bits[i]);
  return out;
}

export function float32ArrayToFloat16(values: ArrayLike<number>): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = float32ToFloat16(values[i]);
  return out;
}
