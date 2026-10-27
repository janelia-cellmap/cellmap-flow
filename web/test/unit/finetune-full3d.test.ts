// Whole-network finetuning of a 3D model (src/finetune/full): net3d.ts's
// lean training step held to Marwan's plain one (cellmap-tfjs-train.ts
// buildNet on an NDHWC graph) on networks small enough for TF.js's CPU
// backend, and the weights' way from the served .onnx into PyTorch's layout
// and back (readTorchParams, toOnnxWeights).
import { describe, expect, it } from "vitest";

import type { InitializerOffsets, TorchParam, TorchWeight, WeightMapEntry } from "../../src/finetune/full/cellmap-tfjs-train";
import { buildNet, maskedBce, readTorchParams, toOnnxWeights, trainStep, writeOnnxWeights } from "../../src/finetune/full/cellmap-tfjs-train";
import { opInputs, parseTrainGraph, type TrainGraph3d } from "../../src/finetune/full/graph";
import { buildNet3d } from "../../src/finetune/full/net3d";
import { loadTf } from "../../src/finetune/full/tfjs";


const conv = (x: string, y: string, w: string, stride = [1, 1, 1], b: string | undefined = `${w}.b`) => ({ op: "conv", x, y, w, ...(b ? { b } : {}), stride });
const bn = (x: string, y: string, p: string) => ({ op: "bn", x, y, scale: `${p}.g`, bias: `${p}.beta`, mean: `${p}.mean`, var: `${p}.var`, eps: 1e-5 });
const crops = (x: string, y: string, by: number) => ["d", "h", "w"].map((axis, i) => ({ op: "crop", x: i ? `${y}.${i - 1}` : x, y: i < 2 ? `${y}.${i}` : y, axis, start: by, end_margin: by }));
const convParams = (w: string, o: number, i: number, k: number, bias = true) => ({ [w]: { shape: [o, i, k, k, k], trainable: true }, ...(bias ? { [`${w}.b`]: { shape: [o], trainable: true } } : {}) });
const bnParams = (p: string, c: number) => ({
  [`${p}.g`]: { shape: [c], trainable: true }, [`${p}.beta`]: { shape: [c], trainable: true },
  [`${p}.mean`]: { shape: [c], trainable: false }, [`${p}.var`]: { shape: [c], trainable: false },
});

// A UNet as setup_53's, in miniature: 14³ in, 4³ out. Conv → BatchNorm →
// ReLU twice, a 2× max pool, a BatchNorm and a ReLU on their own, conv →
// BN → ReLU, a transposed conv of kernel and stride 2, the skip cropped by
// 2 on d, h and w and concatenated, conv → BN → ReLU, and a 1×1×1 head
// without a bias.
const UNET = parseTrainGraph({
  format: "cellmap-train-graph/1", layout: "NDHWC", input: "input", output: "logits", context: 10,
  params: {
    ...convParams("c1", 4, 1, 3), ...bnParams("n1", 4), ...convParams("c2", 4, 4, 3), ...bnParams("n2", 4), ...bnParams("n0", 4),
    ...convParams("c3", 6, 4, 3), ...bnParams("n3", 6), "up.w": { shape: [6, 4, 2, 2, 2], trainable: true }, "up.b": { shape: [4], trainable: true },
    ...convParams("c4", 4, 8, 3), ...bnParams("n4", 4), ...convParams("head", 2, 4, 1, false),
  },
  ops: [
    conv("input", "z1", "c1"), bn("z1", "b1", "n1"), { op: "relu", x: "b1", y: "a1" },
    conv("a1", "z2", "c2"), bn("z2", "b2", "n2"), { op: "relu", x: "b2", y: "a2" },
    { op: "maxpool", x: "a2", y: "p", k: [2, 2, 2], stride: [2, 2, 2] }, bn("p", "pn", "n0"), { op: "relu", x: "pn", y: "pr" },
    conv("pr", "z3", "c3"), bn("z3", "b3", "n3"), { op: "relu", x: "b3", y: "a3" },
    { op: "conv_t", x: "a3", y: "u", w: "up.w", b: "up.b", stride: [2, 2, 2] },
    ...crops("a2", "skip", 2), { op: "concat", xs: ["skip", "u"], y: "cat" },
    conv("cat", "z4", "c4"), bn("z4", "b4", "n4"), { op: "relu", x: "b4", y: "a4" },
    conv("a4", "logits", "head", [1, 1, 1], ""),
  ],
}) as TrainGraph3d;

// A ResUNet as setup_54's: 12³ in, 4³ out. A stem of two convs with
// InstanceNorm, the first with a ReLU, the second added to a 1×1×1 conv of
// the input cropped by 2, then a ReLU; a strided 2×2×2 conv down; conv →
// InstanceNorm → ReLU; nearest upsampling by 2 and a 1×1×1 conv; the stem
// cropped by 2 and concatenated; a 1×1×1 head.
const RESUNET = parseTrainGraph({
  format: "cellmap-train-graph/1", layout: "NDHWC", input: "input", output: "logits", context: 8,
  params: {
    ...convParams("skip", 4, 1, 1), ...convParams("s1", 4, 1, 3), ...convParams("s2", 4, 4, 3), ...convParams("down", 6, 4, 2), ...convParams("e1", 6, 6, 3),
    ...convParams("reduce", 4, 6, 1), ...convParams("head", 2, 8, 1),
    ...Object.fromEntries(["i1", "i2", "i3"].flatMap((p, k) => [[`${p}.g`, { shape: [k < 2 ? 4 : 6], trainable: true }], [`${p}.beta`, { shape: [k < 2 ? 4 : 6], trainable: true }]])),
  },
  ops: [
    conv("input", "k", "skip"),
    conv("input", "y1", "s1"), { op: "inorm", x: "y1", y: "n1", scale: "i1.g", bias: "i1.beta", eps: 1e-5 }, { op: "relu", x: "n1", y: "r1" },
    conv("r1", "y2", "s2"), { op: "inorm", x: "y2", y: "n2", scale: "i2.g", bias: "i2.beta", eps: 1e-5 },
    ...crops("k", "kc", 2), { op: "add", x: "n2", x2: "kc", y: "sum" }, { op: "relu", x: "sum", y: "stem" },
    conv("stem", "dn", "down", [2, 2, 2]),
    conv("dn", "y3", "e1"), { op: "inorm", x: "y3", y: "n3", scale: "i3.g", bias: "i3.beta", eps: 1e-5 }, { op: "relu", x: "n3", y: "r3" },
    { op: "upsample", x: "r3", y: "up", factor: 2 }, conv("up", "red", "reduce"),
    ...crops("stem", "sc", 2), { op: "concat", xs: ["red", "sc"], y: "cat" },
    conv("cat", "logits", "head"),
  ],
}) as TrainGraph3d;

/** Values for `graph`'s params, the same each run: convolutions scaled by
 * their fan-in, norms' scales near 1, running variances near 1. */
function weightsFor(graph: TrainGraph3d): Record<string, Float32Array> {
  return Object.fromEntries(Object.entries(graph.params).map(([name, { shape }], p) => {
    const n = shape.reduce((a, b) => a * b, 1), fanIn = shape.slice(1).reduce((a, b) => a * b, 1);
    const wave = (i: number) => Math.sin(i * 12.9898 + p * 78.233 + 0.5) * 0.5;
    const value = name.endsWith(".g") || name.endsWith(".var") ? (i: number) => 1 + 0.4 * wave(i)
      : shape.length === 1 ? (i: number) => 0.2 * wave(i)
      : (i: number) => (2 * wave(i)) / Math.sqrt(fanIn);
    return [name, Float32Array.from({ length: n }, (_, i) => value(i))];
  }));
}

const block = (n: number) => Float32Array.from({ length: n ** 3 }, (_, i) => Math.sin(i / 3) * Math.cos(i / 7) * 0.9);
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

describe("net3d's lean training step", () => {
  // Two Adam steps of each network both ways, on TF.js's CPU backend; the
  // UNet again with every param trained, its BatchNorms' statistics too,
  // which then do not fuse with their convs.
  it("trains as Marwan's plain network does: the same losses, the same weights after", { timeout: 120_000 }, async () => {
    const { tf } = await loadTf();
    const everything = new Set(Object.keys(UNET.params));
    for (const [graph, side, trainable] of [[UNET, 14, null], [UNET, 14, everything], [RESUNET, 12, null]] as const) {
      const weights = weightsFor(graph), out = side - graph.context, channels = 2, lr = 1e-2;
      const x = tf.tensor5d(block(side), [1, side, side, side, 1]);
      const size = out ** 3 * channels;
      const target = tf.tensor5d(Float32Array.from({ length: size }, (_, i) => (i % 3 === 0 ? 1 : 0)), [1, out, out, out, channels]);
      const mask = tf.tensor5d(Float32Array.from({ length: size }, (_, i) => (i % 5 === 0 ? 0 : 1)), [1, out, out, out, channels]);

      const plain = buildNet(tf, graph, weights, { trainable }), opt = tf.train.adam(lr);
      const plainLosses = [0, 1].map(() => {
        const loss = trainStep(tf, plain, opt, x, target, mask);
        const v = loss.dataSync()[0];
        loss.dispose();
        return v;
      });

      const lean = buildNet3d(tf, graph, weights, { trainable, learningRate: lr });
      const teacher = buildNet3d(tf, graph, weights, { learningRate: 0 });
      // The teacher runs forward only, as the plain network did before its steps.
      const before = buildNet(tf, graph, weights);
      const z0 = teacher.forward(x), z1 = tf.tidy(() => before.forward(x)), z2 = await teacher.forwardAsync(x);
      expect(maxAbs(z0.dataSync(), z1.dataSync())).toBeLessThan(1e-5);
      expect(Array.from(z2.dataSync())).toEqual(Array.from(z0.dataSync()));
      tf.dispose([z0, z1, z2]);
      await expect(teacher.step(x, () => { throw new Error("not called"); })).rejects.toThrow(/forward only/);
      before.dispose();

      const leanLosses: number[] = [];
      for (let i = 0; i < 2; i++) {
        const terms = await lean.step(x, (z) => {
          const supervised = maskedBce(tf, z, target, mask);
          return { total: tf.clone(supervised), supervised, distillation: null, otherChannels: null };
        });
        expect(terms.supervised).toBeCloseTo(terms.total, 6);
        leanLosses.push(terms.total);
      }
      for (let i = 0; i < 2; i++) expect(Math.abs(leanLosses[i] - plainLosses[i])).toBeLessThan(1e-5 * plainLosses[i]);
      expect(leanLosses[1]).toBeLessThan(leanLosses[0]);

      const a = lean.exportWeights(), b = plain.exportWeights();
      expect(Object.keys(a).sort()).toEqual(Object.keys(b).sort());
      // A conv's bias that only shifts what an InstanceNorm then centres
      // (straight after the conv, or through another conv) has no gradient:
      // only rounding, which Adam scales to a step of any size, differently
      // each way. Its value changes nothing the network computes.
      const intoNorm = (t: string): boolean => {
        const readers = graph.ops.filter((o) => opInputs(o).includes(t));
        return readers.length > 0 && readers.every((o) => o.op === "inorm" || (o.op === "conv" && intoNorm(o.y)));
      };
      const cancelled = new Set(graph.ops.flatMap((op) => (op.op === "conv" && op.b && intoNorm(op.y) ? [op.b] : [])));
      for (const [name, { data, shape }] of Object.entries(a)) {
        expect(shape).toEqual(graph.params[name].shape);
        if (cancelled.has(name)) continue;
        // Each step moves a weight by up to lr: the two agree to a thousandth of that.
        expect(maxAbs(data, b[name].data), name).toBeLessThan(lr * 1e-3);
        expect(maxAbs(data, weights[name]), name).toBeGreaterThan(lr / 2);
      }
      [x, target, mask].forEach((t) => t.dispose());
      opt.dispose(); plain.dispose(); lean.dispose(); teacher.dispose();
    }
  });
});

/** PyTorch's layouts by hand: the ONNX initializers of a Conv3d split per
 * depth tap ([O,I,kd,kh,kw] → kd of [O,I,kh,kw]), of a ConvTranspose3d of
 * kernel 2 as a 1×1 conv to 8× the channels ([I,O,2,2,2] → [O·8, I, 1, 1],
 * row o·8 + 4a + 2b + e) and its bias repeated 8×, as Marwan's exports
 * write them. */
function exportedOnnx(conv: Float32Array, convBias: Float32Array, up: Float32Array, upBias: Float32Array) {
  const [O, I] = [2, 3], init: Record<string, { shape: number[]; data: Float32Array }> = {};
  for (let k = 0; k < 3; k++) {
    const data = new Float32Array(O * I * 9);
    for (let o = 0; o < O; o++) for (let i = 0; i < I; i++) for (let hw = 0; hw < 9; hw++) data[(o * I + i) * 9 + hw] = conv[((o * I + i) * 3 + k) * 9 + hw];
    init[`c.convs.${k}.weight`] = { shape: [O, I, 3, 3], data };
  }
  init["c.convs.0.bias"] = { shape: [O], data: convBias };
  const [UI, UO] = [3, 2], data = new Float32Array(UO * 8 * UI);
  for (let i = 0; i < UI; i++) for (let o = 0; o < UO; o++) for (let t = 0; t < 8; t++) data[(o * 8 + t) * UI + i] = up[(i * UO + o) * 8 + t];
  init["up.conv.convs.0.weight"] = { shape: [UO * 8, UI, 1, 1], data };
  init["up.conv.convs.0.bias"] = { shape: [UO * 8], data: Float32Array.from({ length: UO * 8 }, (_, j) => upBias[Math.floor(j / 8)]) };
  // Laid out in one buffer with gaps, as a protobuf has between its tensors.
  let at = 16;
  const offsets: InitializerOffsets = {};
  for (const [name, t] of Object.entries(init)) {
    offsets[name] = { offset: at, nbytes: t.data.length * 4, dtype: "float32", shape: t.shape };
    at += t.data.length * 4 + 12;
  }
  const bytes = new Uint8Array(at);
  bytes.fill(0xa5);
  for (const [name, t] of Object.entries(init)) new Uint8Array(bytes.buffer, offsets[name].offset, offsets[name].nbytes).set(new Uint8Array(t.data.buffer));
  return { bytes, offsets };
}

describe("readTorchParams and toOnnxWeights", () => {
  it("read PyTorch's layout out of the served initializers, and write exactly them back", () => {
    const wave = (n: number, seed: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + seed));
    const torch = { "c.weight": wave(2 * 3 * 27, 1), "c.bias": wave(2, 2), "up.weight": wave(3 * 2 * 8, 3), "up.bias": wave(2, 4) };
    const { bytes, offsets } = exportedOnnx(torch["c.weight"], torch["c.bias"], torch["up.weight"], torch["up.bias"]);
    // The recipes as in setup_53's torch_params.json and weight_map.json.
    const torchParams: Record<string, TorchParam> = {
      "c.weight": { shape: [2, 3, 3, 3, 3], from_onnx: [{ stack: ["c.convs.0.weight", "c.convs.1.weight", "c.convs.2.weight"], axis: 2 }] },
      "c.bias": { shape: [2], from_onnx: [{ from: "c.convs.0.bias" }] },
      "up.weight": { shape: [3, 2, 2, 2, 2], from_onnx: [{ from: "up.conv.convs.0.weight" }, { reshape: [2, 2, 2, 2, 3] }, { permute: [4, 0, 1, 2, 3] }] },
      "up.bias": { shape: [2], from_onnx: [{ from: "up.conv.convs.0.bias" }, { reshape: [2, 8] }, { index: [1, 0] }] },
    };
    const weightMap: Record<string, WeightMapEntry> = {
      ...Object.fromEntries([0, 1, 2].map((k) => [`c.convs.${k}.weight`, { torch: "c.weight", to_onnx: [{ index: [2, k] as [number, number] }] }])),
      "c.convs.0.bias": { torch: "c.bias", to_onnx: [] },
      "up.conv.convs.0.weight": { torch: "up.weight", to_onnx: [{ permute: [1, 2, 3, 4, 0] }, { reshape: [16, 3, 1, 1] }] },
      "up.conv.convs.0.bias": { torch: "up.bias", to_onnx: [{ repeat_interleave: [0, 8] }] },
    };
    const read = readTorchParams(bytes, offsets, torchParams);
    for (const [name, values] of Object.entries(torch)) expect(Array.from(read[name]), name).toEqual(Array.from(values));
    const params: Record<string, TorchWeight> = Object.fromEntries(Object.entries(read).map(([n, data]) => [n, { data, shape: torchParams[n].shape }]));
    const back = toOnnxWeights(weightMap, params, offsets);
    expect(Object.keys(back).sort()).toEqual(Object.keys(offsets).sort());
    expect(Array.from(writeOnnxWeights(bytes, offsets, back))).toEqual(Array.from(bytes));
    // An initializer whose param did not train is left alone.
    const { "up.bias": _, ...trained } = params;
    expect(Object.keys(toOnnxWeights(weightMap, trained, offsets))).not.toContain("up.conv.convs.0.bias");
    // A recipe that does not fit its tensor is refused, not misread.
    expect(() => readTorchParams(bytes, offsets, { "c.bias": { shape: [2], from_onnx: [{ from: "c.convs.0.bias" }, { index: [1, 0] }] } })).toThrow(/index 0 of axis 1/);
  });
});
