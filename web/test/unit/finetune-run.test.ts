// One finetune end to end over the real ONNX step: createRunFinetune with
// samples and targets made here (two painted chunks of the head.onnx
// fixture's 1×6×6 output) and the real feature runner, head finder, trainer
// and weight writer. The run logs its plan and each chunk's painted count,
// trains the epochs asked with the dashboard's epoch lines, and the
// Finetuned's bytes run in onnxruntime and compute W · features + b with
// the trained head; a volume with no paint is refused in words that say to
// paint; the run starts from initialHead when given; a cancel ends it
// early; the runner is released however the run ends.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it, vi } from "vitest";

import type { Chain, DatasetInfo, ModelEntry, PaintVolume } from "../../src/contracts";
import { createFeatureRunner, type FeatureRunner } from "../../src/finetune/features";
import { headOf } from "../../src/finetune/onnx-head";
import { createRunFinetune, runFinetune, type RunDeps } from "../../src/finetune/run";
import { DEFAULT_OPTIONS, type Finetuned, type FinetuneRun, type PaintedChunk, type Sample, type Targets } from "../../src/finetune/types";
import { CHUNK_DIR, chunkKey } from "../../src/finetune/volume";
import { MemoryPaintStore } from "../../src/state/paint-store";

// The page's ORT entry replaced by the package's Node entry (see finetune-onnx.test.ts).
vi.mock("onnxruntime-web/webgpu", () => import("onnxruntime-web"));

const HEAD = new Uint8Array(readFileSync(resolve(import.meta.dirname, "../fixtures/head.onnx")));
const BASE = HEAD.buffer.slice(HEAD.byteOffset, HEAD.byteOffset + HEAD.byteLength) as ArrayBuffer;
const CHUNK = [1, 6, 6], IN_DIMS = [1, 1, 1, 8, 8];

const ENTRY: ModelEntry = {
  id: "test/head", name: "head_fixture", repo: "test", folder: "head",
  manifest: {
    model: "head", task: "fixture", architecture: "tiny", checkpoint: "none", params: 60, channels: ["a", "b"],
    input_voxel_size_nm: [8, 8, 8], output_voxel_size_nm: [8, 8, 8], output: "raw logits", input_norm: [],
    io: { input: "input", output: "output" },
    variants: [{ variant: "fp32", input_size: [1, 8, 8], output_size: [1, 6, 6], file: "head.onnx", file_mb: 0.001 }],
  },
  variant: { variant: "fp32", input_size: [1, 8, 8], output_size: [1, 6, 6], file: "head.onnx", file_mb: 0.001 },
  onnxUrl: "test:head",
};
const VOLUME: PaintVolume = {
  id: "vol-test", type: "annotation_volume", model_name: ENTRY.name, model_id: ENTRY.id, dataset_path: "mem://raw",
  chunk_size: CHUNK, input_size: [1, 8, 8], output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8], dataset_offset_nm: [0, 0, 0],
  dataset_shape_voxels: [1, 12, 12], dtype: "uint8", input_norm: [], postprocess: [], created_at: "2026-10-15T00:00:00Z",
};
const CHAIN: Chain = { input_norm: [], postprocess: [] };
const DATASET: DatasetInfo = { url: "mem://raw", dtype: "uint8", channels: 1, axes: [], levels: [{ shape: [1, 12, 12], voxel: [8, 8, 8], origin: [0, 0, 0] }] };

/** A chunk whose labels come from `label(v)`: 0 unpainted, 1 background, 2 object. */
function chunk(index: number[], label: (v: number) => number): PaintedChunk {
  return { index, shape: CHUNK, labels: Uint8Array.from({ length: 36 }, (_, v) => label(v)) };
}

/** The input block of chunk `index`: a deterministic pattern that differs per chunk. */
function block(index: number[]): Float32Array {
  return Float32Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.37 + index[2] * 1.7) + 0.3 * Math.cos(i * 1.3));
}

/** The binary target broadcast to both channels: object where the label is 2 or more, masked where painted. */
function targets(c: PaintedChunk): Targets {
  const n = 36, target = new Float32Array(2 * n), mask = new Float32Array(2 * n);
  for (let v = 0; v < n; v++) for (let ch = 0; ch < 2; ch++) {
    target[ch * n + v] = c.labels[v] >= 2 ? 1 : 0;
    mask[ch * n + v] = c.labels[v] > 0 ? 1 : 0;
  }
  return { channels: 2, target, mask };
}

/** The samples and the paint store listing them, for `chunks`. */
async function painted(chunks: PaintedChunk[]) {
  const paint = new MemoryPaintStore();
  for (const c of chunks) await paint.put(VOLUME.id, CHUNK_DIR + chunkKey(c.index), new Uint8Array(c.labels.buffer, c.labels.byteOffset, c.labels.byteLength));
  const samples: Sample[] = chunks.map((c) => ({ chunk: c, input: block(c.index), inputDims: IN_DIMS }));
  // eslint-disable-next-line require-yield
  async function* collectSamples(): AsyncGenerator<Sample> { for (const s of samples) yield s; }
  return { paint, collectSamples };
}

function reporter(cancelWhen: (lines: string[]) => boolean = () => false) {
  const lines: string[] = [];
  return { lines, log: (l: string) => lines.push(l), cancelled: () => cancelWhen(lines) };
}

/** The deps with the fake samples and targets, and the real runner, watched. */
function deps(collectSamples: () => AsyncGenerator<Sample>, disposed: { count: number }): RunDeps {
  return {
    collectSamples,
    targetTransformFor: () => targets,
    createFeatureRunner: async (options) => {
      const runner: FeatureRunner = await createFeatureRunner(options);
      return { ...runner, dispose: async () => { disposed.count++; await runner.dispose(); } };
    },
  };
}

function run(over: Partial<FinetuneRun>, paint: MemoryPaintStore, report: ReturnType<typeof reporter>): FinetuneRun {
  return { entry: ENTRY, baseBytes: BASE, volume: VOLUME, chain: CHAIN, dataset: DATASET, infer: { readRaw: async () => new ArrayBuffer(0) }, paint, options: { ...DEFAULT_OPTIONS, epochs: 8, learningRate: 0.05, batchPixels: 16 }, report, ...over };
}

async function logitsOf(bytes: ArrayBuffer, input: Float32Array): Promise<Float32Array> {
  const session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] });
  try {
    return (await session.run({ input: new ort.Tensor("float32", input, IN_DIMS) })).output.data as Float32Array;
  } finally {
    await session.release();
  }
}

// Two chunks: the first painted on its left half (object) and right half
// (background), 36 voxels; the second painted on a few voxels, 5 of them.
const CHUNKS = [chunk([0, 0, 0], (v) => (v % 6 < 3 ? 2 : 1)), chunk([0, 0, 1], (v) => ([3, 9, 15].includes(v) ? 2 : [20, 33].includes(v) ? 1 : 0))];

describe("runFinetune", () => {
  it("trains the head on the painted voxels of every chunk and writes it into a model that runs", async () => {
    const { paint, collectSamples } = await painted(CHUNKS);
    const disposed = { count: 0 }, report = reporter();
    const finetuned: Finetuned = await createRunFinetune(deps(collectSamples, disposed))(run({}, paint, report));
    const log = report.lines.join("\n");
    expect(log).toMatch(/Finetuning head_fixture's head on volume vol-test: 2 painted chunks of 1×6×6 voxels/);
    expect(log).toMatch(/The head: a 1×1 convolution, 4 → 2 channels \(head\.weight, head\.bias, float32\); 8 epochs of 16 voxels per step/);
    expect(log).toMatch(/chunk 1\/2 \[0, 0, 0\]: 36 painted voxels/);
    expect(log).toMatch(/chunk 2\/2 \[0, 0, 1\]: 5 painted voxels/);
    expect(log).toMatch(/41 painted voxels in 2 chunks; training the head/);
    expect(report.lines.filter((l) => /^Epoch \d+\/8 - Loss: \d\.\d{6}$/.test(l))).toHaveLength(8);
    expect(finetuned).toMatchObject({ epochs: 8, pixels: 41, chunks: 2 });
    expect(finetuned.losses).toHaveLength(8);
    expect(finetuned.finalLoss).toBe(finetuned.losses[7]);
    expect(finetuned.losses[7]).toBeLessThan(finetuned.losses[0]);
    expect(finetuned.head).toEqual(headOf(BASE));
    expect(finetuned.weight).not.toEqual(finetuned.head.weight);
    expect(disposed.count).toBe(1);
    // The bytes are the base's with the trained head in them, and the model computes with it.
    expect(finetuned.bytes.byteLength).toBe(BASE.byteLength);
    expect(headOf(finetuned.bytes)).toMatchObject({ weight: finetuned.weight, bias: finetuned.bias });
    const input = block([0, 0, 0]);
    const trained = await logitsOf(finetuned.bytes, input), base = await logitsOf(BASE, input);
    expect(Array.from(trained)).not.toEqual(Array.from(base));
    const features = await createFeatureRunner({ bytes: BASE, head: finetuned.head });
    try {
      const f = await features.run(input, IN_DIMS);
      for (let c = 0; c < 2; c++) for (let v = 0; v < 36; v++) {
        let z = finetuned.bias[c];
        for (let i = 0; i < 4; i++) z += finetuned.weight[c * 4 + i] * f.features[i * 36 + v];
        expect(trained[c * 36 + v]).toBeCloseTo(z, 4);
      }
    } finally {
      await features.dispose();
    }
  }, 60000);

  it("refuses a volume with nothing painted, saying to paint first", async () => {
    const { paint, collectSamples } = await painted([chunk([0, 0, 0], () => 0)]);
    const disposed = { count: 0 };
    await expect(createRunFinetune(deps(collectSamples, disposed))(run({}, paint, reporter()))).rejects.toThrow(/no painted voxels: paint background \(1\) and objects \(2 and up\) on the annotation layer first/);
    expect(disposed.count).toBe(1);
  }, 60000);

  it("starts from initialHead when given", async () => {
    const { paint, collectSamples } = await painted(CHUNKS);
    const initialHead = { weight: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), bias: Float32Array.from([-1, 1]) };
    const finetuned = await createRunFinetune(deps(collectSamples, { count: 0 }))(run({ initialHead, options: { ...DEFAULT_OPTIONS, epochs: 1, learningRate: 1e-9, batchPixels: 64 } }, paint, reporter()));
    expect(Array.from(finetuned.weight).map((w) => Math.round(w))).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(finetuned.bias).map((b) => Math.round(b))).toEqual([-1, 1]);
    expect(Array.from(headOf(finetuned.bytes).weight).map((w) => Math.round(w))).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  }, 60000);

  it("stops early when cancelled, keeping what was trained", async () => {
    const { paint, collectSamples } = await painted(CHUNKS);
    const disposed = { count: 0 };
    const report = reporter((lines) => lines.filter((l) => l.startsWith("Epoch")).length >= 2);
    const finetuned = await createRunFinetune(deps(collectSamples, disposed))(run({}, paint, report));
    expect(finetuned.epochs).toBe(2);
    expect(finetuned.losses).toHaveLength(2);
    expect(finetuned.finalLoss).toBe(finetuned.losses[1]);
    expect(report.lines.at(-1)).toMatch(/Stopped after 2 of 8 epochs; loss \d\.\d{6}/);
    expect(disposed.count).toBe(1);
    // Cancelled before any sample: no training, the runner still released.
    const atOnce = reporter(() => true);
    await expect(createRunFinetune(deps(collectSamples, disposed))(run({}, paint, atOnce))).rejects.toThrow(/cancelled before training/);
    expect(disposed.count).toBe(2);
  }, 60000);

  it("exports the run with the real samples and targets", () => {
    expect(typeof runFinetune).toBe("function");
  });
});
