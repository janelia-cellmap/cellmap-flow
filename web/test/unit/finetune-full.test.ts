// Whole-network finetuning (src/finetune/full) against onnxruntime-web:
// Marwan's TF.js runtime on a network small enough for TF.js's CPU backend,
// head.onnx (a 3×3 body and a 1×1 head, shaped as the exports end): a few
// steps lower the masked loss, and the weights trained, written back at
// their offsets, are what onnxruntime then reads and runs. A 3D model's
// training block is read with its paint where the network sees the two in
// register, and the TF.js loss is the head's (head-trainer.ts) for every
// loss the form offers.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import type { DatasetInfo } from "../../src/contracts";
import { transformBlock } from "../../src/finetune/augment";
import type { Tf } from "../../src/finetune/full/cellmap-tfjs-train";
import { createAdam } from "../../src/finetune/full/adam";
import { buildNet, maskedBce, readOnnxWeights, trainStep, writeOnnxWeights } from "../../src/finetune/full/cellmap-tfjs-train";
import { parseTrainGraph } from "../../src/finetune/full/graph";
import { fullTrainStep, lossTerms, OTHER_CHANNELS_WEIGHT, supervisedLoss } from "../../src/finetune/full/loss";
import { onnxInitializerOffsets } from "../../src/finetune/full/run-full";
import { loadTf } from "../../src/finetune/full/tfjs";
import { blockLabels, blockRaw, normalizeTile, paintedVoxels, planBlocks } from "../../src/finetune/full/tiles";
import { batchLoss, mulberry32 } from "../../src/finetune/head-trainer";
import type { PaintedChunk } from "../../src/finetune/types";

const FIXTURES = resolve(import.meta.dirname, "../fixtures");

// head.onnx as a training graph: 8×8 in, a valid 3×3 convolution, 6×6 out.
const TINY = parseTrainGraph({
  format: "cellmap-train-graph/1", layout: "NHWC", input: "input", output: "logits", context: 2,
  params: {
    "body.weight": { shape: [4, 1, 3, 3], trainable: true }, "body.bias": { shape: [4], trainable: true },
    "head.weight": { shape: [2, 4, 1, 1], trainable: true }, "head.bias": { shape: [2], trainable: true },
  },
  ops: [
    { op: "conv", x: "input", y: "body", w: "body.weight", b: "body.bias", stride: [1, 1] },
    { op: "relu", x: "body", y: "act" },
    { op: "conv", x: "act", y: "logits", w: "head.weight", b: "head.bias", stride: [1, 1] },
  ],
});

/** ORT's output for `bytes` on an n × n slice, channels last as TF.js has it. */
async function ortForward(bytes: Uint8Array, x: Float32Array, n: number): Promise<Float32Array> {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  try {
    const out = (await session.run({ input: new ort.Tensor("float32", x, [1, 1, 1, n, n]) })).output;
    const [, c, , h, w] = out.dims, data = out.data as Float32Array, nhwc = new Float32Array(data.length);
    for (let k = 0; k < c; k++) for (let p = 0; p < h * w; p++) nhwc[p * c + k] = data[k * h * w + p];
    return nhwc;
  } finally {
    await session.release();
  }
}

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const slice = (n: number) => new Float32Array(n * n).map((_, i) => Math.sin(Math.floor(i / n) / 3) * Math.cos((i % n) / 2) * 0.9);

/** head.onnx (or its fp16 twin) with a few Adam steps toward a fixed target on part of its output. */
function trained(tf: Tf, bytes: Uint8Array, steps: number) {
  const offsets = onnxInitializerOffsets(bytes);
  const net = buildNet(tf, TINY, readOnnxWeights(bytes, offsets, Object.keys(TINY.params)));
  const x = tf.tensor4d(slice(8), [1, 8, 8, 1]);
  const target = tf.tensor4d(Float32Array.from({ length: 72 }, (_, i) => (i % 3 === 0 ? 1 : 0)), [1, 6, 6, 2]);
  const mask = tf.tensor4d(Float32Array.from({ length: 72 }, (_, i) => (i % 5 === 0 ? 0 : 1)), [1, 6, 6, 2]);
  const opt = tf.train.adam(0.05);
  const losses: number[] = [];
  for (let i = 0; i < steps; i++) {
    const loss = trainStep(tf, net, opt, x, target, mask);
    losses.push(loss.dataSync()[0]);
    loss.dispose();
  }
  const after = tf.tidy(() => maskedBce(tf, net.forward(x), target, mask).dataSync()[0]);
  const forward = tf.tidy(() => Float32Array.from(net.forward(x).dataSync() as Float32Array));
  const weights = net.exportWeights();
  [x, target, mask].forEach((t) => t.dispose());
  opt.dispose();
  net.dispose();
  return { offsets, losses, after, forward, weights };
}

describe("a 3D model's training blocks", () => {
  it("hold the raw around the paint's cube in register with it, through flips of z and turns", async () => {
    // A 20³ volume in 4³ chunks, read for a network of 6³ in and 2³ out
    // (context 4: the input is the output cube grown by 2 each way). The
    // paint: a 2³ object at (8..9)³, its corner voxel another object (3);
    // the raw: 0 but for a mark on that corner voxel.
    const n = 20, side = 6, output = 2, halo = 2;
    const volume = { chunk_size: [4, 4, 4], dataset_shape_voxels: [n, n, n], dtype: "uint16" as const };
    const chunk: PaintedChunk = { index: [2, 2, 2], shape: [4, 4, 4], labels: new Uint16Array(64) };
    for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) chunk.labels[(z * 4 + y) * 4 + x] = z + y + x ? 2 : 3;
    const chunks = new Map([["2.2.2", chunk]]);
    const rawVolume = new Uint8Array(n ** 3);
    rawVolume[(8 * n + 8) * n + 8] = 1;
    const dataset = { url: "", dtype: "uint8", channels: 1, axes: [], levels: [{ shape: [n, n, n], voxel: [16, 16, 16], origin: [0, 0, 0] }] } as unknown as DatasetInfo;
    const infer = {
      async readRaw(_level: number, lo: number[], hi: number[]) {
        const out = new Uint8Array((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
        let i = 0;
        for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[i++] = rawVolume[(z * n + y) * n + x];
        return out.buffer;
      },
    };

    const blocks = planBlocks(paintedVoxels(chunks), output);
    expect(blocks).toEqual([{ corner: [8, 8, 8] }]);
    const labels = blockLabels(blocks[0], output, chunks, volume);
    const { raw, inside } = await blockRaw(blocks[0], side, output, infer, dataset, 0, 0);
    expect(inside.every((v) => v === 1)).toBe(true);
    // The mark on the output's corner voxel is at the input's corner + context/2.
    expect(raw.indexOf(1)).toBe((halo * side + halo) * side + halo);
    expect(labels.indexOf(3)).toBe(0);
    // And stays there through any draw: the same flips and turn of each about its own centre.
    const at = (i: number, s: number) => [Math.floor(i / (s * s)), Math.floor(i / s) % s, i % s];
    for (const draw of [{ flips: [true, false, false], turns: 0 }, { flips: [true, true, false], turns: 1 }, { flips: [false, true, true], turns: 3 }] as const) {
      const d = { flips: [...draw.flips] as [boolean, boolean, boolean], turns: draw.turns, scale: 1 };
      const r = at(transformBlock(raw, [side, side, side], d).indexOf(1), side);
      const l = at(transformBlock(labels, [output, output, output], d).indexOf(3), output);
      expect(r.map((v) => v - halo)).toEqual(l);
    }
  });

  it("repeat the volume's normalized edge past it, as the model stage serves a block", async () => {
    // A 4³ volume and the 6³ block around the output cube at its corner: the
    // block starts 2 voxels before the volume on every axis.
    const n = 4, side = 6, output = 2;
    const value = (z: number, y: number, x: number) => 100 * z + 10 * y + x + 1;
    const dataset = { url: "", dtype: "uint16", channels: 1, axes: [], levels: [{ shape: [n, n, n], voxel: [16, 16, 16], origin: [0, 0, 0] }] } as unknown as DatasetInfo;
    const infer = {
      async readRaw(_level: number, lo: number[], hi: number[]) {
        const out = new Uint16Array((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
        let i = 0;
        for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[i++] = value(z, y, x);
        return out.buffer;
      },
    };
    const { raw, inside } = await blockRaw({ corner: [0, 0, 0] }, side, output, infer, dataset, 0, 0);
    const input = normalizeTile(raw, inside, [{ name: "LambdaNormalizer", expression: "x/1000" }], [side, side, side]);
    const near = (i: number) => Math.min(Math.max(i - 2, 0), n - 1);
    for (const [z, y, x] of [[0, 0, 0], [1, 3, 5], [2, 2, 2], [5, 0, 4]]) {
      expect(input[(z * side + y) * side + x]).toBeCloseTo(value(near(z), near(y), near(x)) / 1000, 6);
    }
  });
});

describe("the whole network's loss", () => {
  it("is the head's supervised term for every loss, balanced or not, smoothed or not", { timeout: 60_000 }, async () => {
    const { tf } = await loadTf();
    const random = mulberry32(3), n = 64;
    const z = Float64Array.from({ length: n }, () => 6 * random() - 3);
    const y = Float32Array.from({ length: n }, () => (random() < 0.3 ? 1 : 0));
    const m = Float32Array.from({ length: n }, () => (random() < 0.7 ? 1 : 0));
    for (const lossType of ["bce", "mse", "margin"] as const) for (const balanceClasses of [false, true]) for (const labelSmoothing of [0, 0.1]) {
      const o = { lossType, margin: 0.3, balanceClasses, labelSmoothing, distillationLambda: 0, distillationScope: "unlabeled" as const };
      const expected = batchLoss(z, y, m, o, new Float64Array(n));
      const got = tf.tidy(() => supervisedLoss(tf, tf.tensor1d(Float32Array.from(z)), tf.tensor1d(y), tf.tensor1d(m), o).dataSync()[0]);
      expect(got).toBeCloseTo(expected, 5);
    }
  });
});

describe("the whole network's loss for one channel of several", () => {
  it("trains the channel picked on its paint and holds every other channel to the starting network on every voxel", { timeout: 60_000 }, async () => {
    const { tf } = await loadTf();
    // Three voxels of two channels, channels last: the starting network's
    // logits, and the student's with channel 0 moved by d0 and channel 1 by
    // d1; the paint (one channel, the picked one's) on the first two voxels.
    const teacher = [0.5, -1, 0.2, 1.5, -0.3, 2];
    const moved = (d0: number, d1: number) => teacher.map((t, i) => t + (i % 2 ? d1 : d0));
    const y = [1, 0, 1], m = [1, 1, 0];
    const o = { lossType: "bce" as const, margin: 0.3, balanceClasses: false, labelSmoothing: 0, distillationLambda: 0.5, distillationScope: "unlabeled" as const, selectChannel: 1 };
    const terms = (z: number[]) => tf.tidy(() => {
      const t = lossTerms(tf, tf.tensor2d(z, [3, 2]), tf.tensor2d(y, [3, 1]), tf.tensor2d(m, [3, 1]), tf.tensor2d(teacher, [3, 2]), o);
      return { total: t.total.dataSync()[0], supervised: t.supervised.dataSync()[0], distillation: t.distillation!.dataSync()[0], others: t.otherChannels!.dataSync()[0] };
    });
    // Channel 0 moved by 0.4 on every voxel: held at 0.4², painted or not;
    // channel 1's supervised term is the head's on channel 1 alone.
    const away = terms(moved(0.4, 0));
    expect(away.others).toBeCloseTo(0.16, 6);
    expect(away.distillation).toBeCloseTo(0, 6);
    const z1 = Float64Array.from([teacher[1], teacher[3], teacher[5]]);
    expect(away.supervised).toBeCloseTo(batchLoss(z1, Float32Array.from(y), Float32Array.from(m), o, new Float64Array(3)), 5);
    expect(away.total).toBeCloseTo(away.supervised + OTHER_CHANNELS_WEIGHT * away.others, 5);
    // Channel 1 moved by 0.3: nothing held, and its own distillation on its
    // one unpainted voxel, as the form's scope says.
    const trained = terms(moved(0, 0.3));
    expect(trained.others).toBeCloseTo(0, 6);
    expect(trained.distillation).toBeCloseTo(0.09, 6);
    expect(trained.total).toBeCloseTo(trained.supervised + 0.5 * trained.distillation, 5);
  });
});

describe("Marwan's TF.js runtime on head.onnx", () => {
  // TF.js's CPU backend: seconds a test on a quiet machine, near a minute on a busy one.
  it("lowers the masked loss in a few steps", { timeout: 120_000 }, async () => {
    const { tf } = await loadTf();
    const { losses, after } = trained(tf, new Uint8Array(readFileSync(resolve(FIXTURES, "head.onnx"))), 20);
    expect(after).toBeLessThan(losses[0] * 0.8);
    expect(losses.at(-1)!).toBeLessThan(losses[0]);
  });

  it("holds the network to its starting self: the distillation term is 0 before the first step and grows as it trains", { timeout: 120_000 }, async () => {
    const { tf } = await loadTf();
    const bytes = new Uint8Array(readFileSync(resolve(FIXTURES, "head.onnx")));
    const offsets = onnxInitializerOffsets(bytes), names = Object.keys(TINY.params);
    const net = buildNet(tf, TINY, readOnnxWeights(bytes, offsets, names));
    const teacher = buildNet(tf, TINY, readOnnxWeights(bytes, offsets, names), { trainable: new Set() });
    const x = tf.tensor4d(slice(8), [1, 8, 8, 1]);
    const target = tf.tensor4d(Float32Array.from({ length: 72 }, (_, i) => (i % 3 === 0 ? 1 : 0)), [1, 6, 6, 2]);
    const mask = tf.tensor4d(Float32Array.from({ length: 72 }, (_, i) => (i % 5 === 0 ? 0 : 1)), [1, 6, 6, 2]);
    const z0 = tf.tidy(() => teacher.forward(x));
    const bce = tf.tidy(() => maskedBce(tf, z0, target, mask).dataSync()[0]);
    const opt = createAdam(tf);
    const o = { lossType: "bce" as const, margin: 0.3, labelSmoothing: 0, balanceClasses: false, distillationLambda: 1, distillationScope: "unlabeled" as const };
    const terms = [0, 1, 2].map(() => {
      const step = fullTrainStep(tf, net, opt, x, target, mask, z0, o, 0.05);
      const out = { supervised: step.supervised.dataSync()[0], distillation: step.distillation!.dataSync()[0], total: step.total.dataSync()[0] };
      [step.total, step.supervised, step.distillation!].forEach((s) => s.dispose());
      return out;
    });
    expect(terms[0].distillation).toBeCloseTo(0, 6);
    expect(terms[0].supervised).toBeCloseTo(bce, 5);  // without the head's options, Marwan's masked BCE
    expect(terms[2].distillation).toBeGreaterThan(0);
    expect(terms[2].total).toBeCloseTo(terms[2].supervised + terms[2].distillation, 5);
    [x, target, mask, z0].forEach((t) => t.dispose());
    opt.dispose(); net.dispose(); teacher.dispose();
  });

  // TF.js's optimizer keeps the rate it was made with, so the whole network
  // steps an Adam of its own (adam.ts), each step at the schedule's rate:
  // given one rate throughout, it must step as TF.js's does.
  it("steps as TF.js's Adam does at the rate it is given", { timeout: 120_000 }, async () => {
    const { tf } = await loadTf();
    const bytes = new Uint8Array(readFileSync(resolve(FIXTURES, "head.onnx")));
    const offsets = onnxInitializerOffsets(bytes), names = Object.keys(TINY.params);
    const ours = buildNet(tf, TINY, readOnnxWeights(bytes, offsets, names)), theirs = buildNet(tf, TINY, readOnnxWeights(bytes, offsets, names));
    const x = tf.tensor4d(slice(8), [1, 8, 8, 1]);
    const target = tf.tensor4d(Float32Array.from({ length: 72 }, (_, i) => (i % 3 === 0 ? 1 : 0)), [1, 6, 6, 2]);
    const mask = tf.tensor4d(Float32Array.from({ length: 72 }, (_, i) => (i % 5 === 0 ? 0 : 1)), [1, 6, 6, 2]);
    // Marwan's masked BCE, which trainStep minimizes.
    const o = { lossType: "bce" as const, margin: 0.3, labelSmoothing: 0, balanceClasses: false, distillationLambda: 0, distillationScope: "unlabeled" as const };
    const adam = createAdam(tf), tfAdam = tf.train.adam(0.05);
    for (let i = 0; i < 3; i++) {
      const step = fullTrainStep(tf, ours, adam, x, target, mask, null, o, 0.05);
      const loss = trainStep(tf, theirs, tfAdam, x, target, mask);
      expect(step.total.dataSync()[0]).toBeCloseTo(loss.dataSync()[0], 6);
      tf.dispose([step.total, step.supervised, loss]);
    }
    const a = ours.exportWeights(), b = theirs.exportWeights();
    for (const name of names) expect(maxAbs(a[name], b[name])).toBeLessThan(1e-6);
    [x, target, mask].forEach((t) => t.dispose());
    adam.dispose(); tfAdam.dispose(); ours.dispose(); theirs.dispose();
  });

  it("writes the trained weights back where onnxruntime reads them, in fp32 and fp16", { timeout: 120_000 }, async () => {
    const { tf } = await loadTf();
    for (const [file, tolerance] of [["head.onnx", 1e-5], ["head_fp16.onnx", 5e-3]] as const) {
      const bytes = new Uint8Array(readFileSync(resolve(FIXTURES, file)));
      const { offsets, forward, weights } = trained(tf, bytes, 10);
      const written = writeOnnxWeights(bytes, offsets, weights);
      expect(written.byteLength).toBe(bytes.byteLength);
      // Read back from the written file, at offsets found in it afresh.
      const back = readOnnxWeights(written, onnxInitializerOffsets(written), Object.keys(weights));
      for (const name of Object.keys(weights)) expect(maxAbs(back[name], weights[name])).toBeLessThanOrEqual(tolerance * 2);
      // And run: onnxruntime on the written model computes what TF.js does with the trained net.
      expect(maxAbs(await ortForward(written, slice(8), 8), forward)).toBeLessThan(tolerance * 20);
      expect(maxAbs(await ortForward(bytes, slice(8), 8), forward)).toBeGreaterThan(1e-3);
    }
  });
});
