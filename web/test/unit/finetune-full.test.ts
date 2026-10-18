// Whole-network finetuning (src/finetune/full) against onnxruntime-web: the
// tile sides the exported training graphs take, and Marwan's TF.js runtime
// on a network small enough for TF.js's CPU backend, head.onnx (a 3×3 body
// and a 1×1 head, shaped as the exports end): a few steps lower the masked
// loss, and the weights trained, written back at their offsets, are what
// onnxruntime then reads and runs. A 3D model's training block is read with
// its paint where the network sees the two in register. The real
// setup_53_2d is opt-in, as models-live is: it downloads 7.75 MB from
// Hugging Face, and TF.js's CPU backend takes about 30 s for one forward
// pass at 300².
//
//   CELLMAP_FLOW_LIVE=1 npx vitest run test/unit/finetune-full
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import type { DatasetInfo } from "../../src/contracts";
import { transformBlock } from "../../src/finetune/augment";
import type { Tf } from "../../src/finetune/full/cellmap-tfjs-train";
import { buildNet, maskedBce, readOnnxWeights, trainStep, writeOnnxWeights } from "../../src/finetune/full/cellmap-tfjs-train";
import { parseTrainGraph, tileSides } from "../../src/finetune/full/graph";
import { fullTrainStep } from "../../src/finetune/full/loss";
import { checkOffsets, onnxInitializerOffsets } from "../../src/finetune/full/run-full";
import { loadTf } from "../../src/finetune/full/tfjs";
import { blockLabels, blockRaw, blockTargets, paintedVoxels, planBlocks } from "../../src/finetune/full/tiles";
import { affinityOffsets, affinityTargets } from "../../src/finetune/targets";
import type { PaintedChunk } from "../../src/finetune/types";
import { readFixture } from "./models-helpers";

const FIXTURES = resolve(import.meta.dirname, "../fixtures");
const live = process.env.CELLMAP_FLOW_LIVE === "1";

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

describe("the exported training graphs", () => {
  it("take every tile side of 92 + 8k pixels, setup_53_2d's UNet and setup_54_2d's ResUNet alike", () => {
    const sides = Array.from({ length: 64 }, (_, k) => 92 + 8 * k);
    for (const folder of ["setup_53_2d", "setup_54_2d"]) {
      expect(tileSides(parseTrainGraph(readFixture(`${folder}.train_graph.json`)), sides.at(-1)!)).toEqual(sides);
    }
  });
});

describe("a 3D model's training blocks", () => {
  it("hold the raw around the paint's cube in register with it, through flips of z and turns, and pair z neighbours", async () => {
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
    const { raw, inside } = await blockRaw(blocks[0], side, output, infer, dataset, 0);
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
    // The z affinity pairs each voxel of the lower slice with the one above:
    // 3 over 2 is a boundary, 2 over 2 the same object.
    const t = blockTargets(labels, [output, output, output], affinityTargets(affinityOffsets(["mito_aff_z", "mito_aff_y", "mito_aff_x"]), 3));
    const zPairs = [0, 1, 2, 3].map((p) => [t.mask[p * 3], t.target[p * 3]]);
    expect(zPairs).toEqual([[1, 0], [1, 1], [1, 1], [1, 1]]);
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
    const opt = tf.train.adam(0.05);
    const o = { lossType: "bce" as const, margin: 0.3, labelSmoothing: 0, balanceClasses: false, distillationLambda: 1, distillationScope: "unlabeled" as const };
    const terms = [0, 1, 2].map(() => {
      const step = fullTrainStep(tf, net, opt, x, target, mask, z0, o);
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

describe.skipIf(!live)("setup_53_2d from the Hub", () => {
  it("runs in TF.js as in onnxruntime", { timeout: 300_000 }, async () => {
    const base = "https://huggingface.co/cellmap/onnx-web-models/resolve/main/setup_53_2d";
    const file = "setup_53_mito_affinities_16nm_2d_in300_fp32.onnx";
    const get = async (path: string) => {
      const response = await fetch(`${base}/${path}`);
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
      return response;
    };
    const graph = parseTrainGraph(await (await get("finetune/train_graph.json")).json());
    const offsets = (await (await get("finetune/initializers.json")).json())[file];
    const bytes = new Uint8Array(await (await get(file)).arrayBuffer());
    checkOffsets(bytes, offsets, Object.keys(graph.params), file);
    const { tf } = await loadTf();
    const net = buildNet(tf, graph, readOnnxWeights(bytes, offsets, Object.keys(graph.params)));
    const x = slice(300);
    const tfOut = tf.tidy(() => Float32Array.from(net.forward(tf.tensor4d(x, [1, 300, 300, 1])).dataSync() as Float32Array));
    net.dispose();
    const ortOut = await ortForward(bytes, x, 300);
    expect(tfOut.length).toBe(212 * 212 * 3);
    expect(maxAbs(tfOut, ortOut)).toBeLessThan(1e-3);
  });
});
