// One finetune end to end over the real ONNX step: createRunFinetune with
// samples and targets made here (two painted chunks of the head.onnx
// fixture's 1×6×6 output) and the real feature runner, head finder, trainer
// and weight writer. The run trains the epochs asked with the dashboard's
// epoch lines, and the Finetuned's bytes, the trained head in them, run in
// onnxruntime; a volume with no paint is refused; a cancel ends it early;
// the runner is released however the run ends. Distillation, on by default,
// holds the painted chunks' unpainted voxels; with good regions it holds
// their unpainted voxels instead, running the model on a region's block
// nothing was painted in, and skips what is not a region in the volume. A
// run of one channel trains that channel's row of the head and leaves the
// other rows as it started them, bit for bit.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it, vi } from "vitest";

import type { Chain, DatasetInfo, ModelEntry, PaintVolume } from "../../src/contracts";
import { createFeatureRunner, type FeatureRunner } from "../../src/finetune/features";
import { headOf } from "../../src/finetune/onnx-head";
import { createRunFinetune, type RunDeps } from "../../src/finetune/run";
import { DEFAULT_OPTIONS, type FinetuneRun, type HeadFinetuned, type PaintedChunk, type Sample, type Targets } from "../../src/finetune/types";
import { CHUNK_DIR, chunkKey } from "../../src/finetune/volume";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { portablePython, type PythonCall } from "./pyodide";

// The page's ORT entry replaced by the package's Node entry (see finetune-onnx.test.ts).
vi.mock("onnxruntime-web/webgpu", () => import("onnxruntime-web"));

// The good regions' anchors are the portable Python's, here in node's
// Pyodide (pyodide.ts), loaded once for the file; without it (no network for
// numpy's wheel) the run with good regions skips.
const python = await portablePython();
const callPython: PythonCall = python ?? (() => Promise.reject(new Error("Pyodide is not available in this test run")));

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
async function targets(c: PaintedChunk): Promise<Targets> {
  const n = 36, target = new Float32Array(2 * n), mask = new Float32Array(2 * n);
  for (let v = 0; v < n; v++) for (let ch = 0; ch < 2; ch++) {
    target[ch * n + v] = c.labels[v] >= 2 ? 1 : 0;
    mask[ch * n + v] = c.labels[v] > 0 ? 1 : 0;
  }
  return { channels: 2, target, mask };
}

/** The binary target of one channel, as target-call.ts makes it for a channel picked. */
async function oneChannel(c: PaintedChunk): Promise<Targets> {
  const both = await targets(c), n = 36;
  return { channels: 1, target: both.target.slice(0, n), mask: both.mask.slice(0, n) };
}

/** A float32 array's bits, to compare weights exactly (not by value: -0 is 0 by value). */
const bitsOf = (a: Float32Array) => Array.from(new Uint32Array(a.buffer, a.byteOffset, a.length));

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
  return { entry: ENTRY, baseBytes: BASE, volume: VOLUME, chain: CHAIN, dataset: DATASET, infer: { readRaw: async () => new ArrayBuffer(0), python: callPython }, paint, options: { ...DEFAULT_OPTIONS, epochs: 8, learningRate: 0.05, batchPixels: 16 }, report, ...over };
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
    const finetuned: HeadFinetuned = await createRunFinetune(deps(collectSamples, disposed))(run({}, paint, report));
    expect(report.lines.filter((l) => /^Epoch \d+\/8 - Loss: \d\.\d{6} - Supervised: \d\.\d{6} - Distillation: \d\.\d{6}$/.test(l))).toHaveLength(8);
    // The default distillation holds the painted chunks' unpainted voxels: the second chunk's 31.
    expect(finetuned).toMatchObject({ epochs: 8, pixels: 41, chunks: 2, distilled: 31 });
    expect(finetuned.losses).toHaveLength(8);
    expect(finetuned.supervisedLosses).toHaveLength(8);
    expect(finetuned.finalLoss).toBe(finetuned.losses[7]);
    expect(finetuned.losses[7]).toBeLessThan(finetuned.losses[0]);
    expect(finetuned.head).toEqual(headOf(BASE));
    expect(finetuned.weight).not.toEqual(finetuned.head.weight);
    expect(disposed.count).toBe(1);
    // The bytes are the base's with the trained head in them, and the model computes with it.
    expect(finetuned.bytes.byteLength).toBe(BASE.byteLength);
    expect(headOf(finetuned.bytes)).toMatchObject({ weight: finetuned.weight, bias: finetuned.bias });
    const input = block([0, 0, 0]);
    // (That it computes W · features + b with them is finetune-onnx.test.ts's withHead test.)
    const trained = await logitsOf(finetuned.bytes, input), base = await logitsOf(BASE, input);
    expect(Array.from(trained)).not.toEqual(Array.from(base));
  }, 60000);

  it("refuses a volume whose chunks hold no paint (all erased)", async () => {
    const { paint, collectSamples } = await painted([chunk([0, 0, 0], () => 0)]);
    const disposed = { count: 0 };
    await expect(createRunFinetune(deps(collectSamples, disposed))(run({}, paint, reporter()))).rejects.toThrow(/no painted voxels/);
    expect(disposed.count).toBe(1);
  }, 60000);

  it("stops early when cancelled, keeping what was trained", async () => {
    const { paint, collectSamples } = await painted(CHUNKS);
    const disposed = { count: 0 };
    const report = reporter((lines) => lines.filter((l) => l.startsWith("Epoch")).length >= 2);
    const finetuned = await createRunFinetune(deps(collectSamples, disposed))(run({}, paint, report));
    expect(finetuned.epochs).toBe(2);
    expect(finetuned.losses).toHaveLength(2);
    expect(finetuned.finalLoss).toBe(finetuned.losses[1]);
    expect(disposed.count).toBe(1);
    // Cancelled before any sample: no training, the runner still released.
    const atOnce = reporter(() => true);
    await expect(createRunFinetune(deps(collectSamples, disposed))(run({}, paint, atOnce))).rejects.toThrow(/cancelled/);
    expect(disposed.count).toBe(2);
  }, 60000);
});

/** collectSamples over `chunks` that takes the run's chunk indices as the
 * real one does: a chunk not among the painted is a sample of zeros. */
function byIndex(chunks: PaintedChunk[]) {
  return async function* (_run?: unknown, indices?: number[][]): AsyncGenerator<Sample> {
    for (const index of indices ?? chunks.map((c) => c.index)) {
      const painted = chunks.find((c) => chunkKey(c.index) === chunkKey(index)) ?? chunk(index, () => 0);
      yield { chunk: painted, input: block(index), inputDims: IN_DIMS };
    }
  };
}

// The volume's grid: voxel 0 centred at 0 nm (its corner at -4), 8 nm voxels,
// 1×6×6 blocks. A good region centred on block (0, 1, 1), which nothing is
// painted in, one on block (0, 0, 1), painted on 5 voxels, and one far
// outside the volume.
const ON_UNPAINTED = { id: "aaaa0001", label: "good-1", offset_nm: [-4, 44, 44], shape_nm: [8, 48, 48] };
const ON_PAINTED = { id: "aaaa0002", label: "good-2", offset_nm: [-4, -4, 44], shape_nm: [8, 48, 48] };
const OUTSIDE = { id: "aaaa0003", label: "good-3", offset_nm: [-4, 476, 476], shape_nm: [8, 48, 48] };

describe("runFinetune's distillation", () => {
  it.skipIf(!python)("with good regions, holds their unpainted voxels instead of the form's scope, running the model on a region's unpainted block", async () => {
    const { paint } = await painted(CHUNKS);
    const report = reporter();
    const goodRegions = [ON_UNPAINTED, ON_PAINTED, OUTSIDE, { not: "a region" }];
    const options = { ...DEFAULT_OPTIONS, epochs: 3, learningRate: 0.05, batchPixels: 16, distillationScope: "all" as const };
    const finetuned = await createRunFinetune(deps(byIndex(CHUNKS), { count: 0 }))(run({ options, goodRegions }, paint, report));
    // The painted chunks first, then the region's block nothing was painted
    // in; the region outside the volume and the one that is not a region are
    // set aside. Held: the painted region's 31 unpainted voxels and the
    // unpainted block's 36, whatever the form's scope (all) says.
    expect(report.lines.filter((l) => l.startsWith("chunk "))).toHaveLength(3);
    expect(finetuned).toMatchObject({ pixels: 41, chunks: 2, distilled: 67 });
    expect(report.lines.filter((l) => / - Supervised: .* - Distillation: /.test(l))).toHaveLength(3);
  }, 60000);
});

describe("runFinetune of one channel", () => {
  it("trains that channel's row of the head alone, the other row written back as the run started it, bit for bit", async () => {
    const base = headOf(BASE), cin = base.inChannels;
    const asked: (number | null)[] = [];
    const oneChannelDeps = (collect: () => AsyncGenerator<Sample>): RunDeps => ({
      ...deps(collect, { count: 0 }),
      targetTransformFor: (_entry, _infer, channel = null) => {
        asked.push(channel);
        return oneChannel;
      },
    });
    const { paint, collectSamples } = await painted(CHUNKS);
    const options = { ...DEFAULT_OPTIONS, epochs: 8, learningRate: 0.05, batchPixels: 16, selectChannel: 1 };
    const finetuned = await createRunFinetune(oneChannelDeps(collectSamples))(run({ options }, paint, reporter()));
    expect(asked).toEqual([1]);
    const written = headOf(finetuned.bytes);
    expect(bitsOf(written.weight.subarray(0, cin))).toEqual(bitsOf(base.weight.subarray(0, cin)));
    expect(bitsOf(written.bias.subarray(0, 1))).toEqual(bitsOf(base.bias.subarray(0, 1)));
    expect(Array.from(written.weight.subarray(cin))).not.toEqual(Array.from(base.weight.subarray(cin)));
    expect(written.bias[1]).not.toBe(base.bias[1]);
    expect(finetuned.losses[7]).toBeLessThan(finetuned.losses[0]);
  }, 60000);
});
