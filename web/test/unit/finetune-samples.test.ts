// collectSamples (src/finetune/samples.ts) on a hand-made annotation volume
// in a MemoryPaintStore and a fake Inference whose readRaw records the boxes
// asked and answers a ramp: a sample is the block the model stage would feed
// the model for the same chunk (test/unit/infer-stages.test.ts checks that
// block's edges, clipped and repeated where the level ends), so the checks
// here are the box read, the dims, the normalization through the volume's
// input_norm (not the run's chain), and which chunks are samples.
import { describe, expect, it } from "vitest";

import type { Chain, DatasetInfo, ModelEntry, ModelManifest, PaintVolume, Step } from "../../src/contracts";
import { collectSamples } from "../../src/finetune/samples";
import type { Sample } from "../../src/finetune/types";
import { CHUNK_DIR, readVolume } from "../../src/finetune/volume";
import { entryFor } from "../../src/models/manifest";
import { MemoryPaintStore } from "../../src/state/paint-store";

/** A model reading `input` voxels and writing `output`, at 8 nm. */
function model(input: number[], output: number[]): ModelEntry {
  const manifest: ModelManifest = {
    model: "setup_t", task: "mito", architecture: "", checkpoint: "", params: 0, channels: ["mito"],
    input_voxel_size_nm: [8, 8, 8], output_voxel_size_nm: [8, 8, 8],
    output: "logits", input_norm: [], io: { input: "input", output: "output" },
    variants: [{ variant: "fp32", input_size: input, output_size: output, file: "setup_t_fp32.onnx", file_mb: 1 }],
  };
  return entryFor("setup_t", manifest, manifest.variants[0]);
}

/** A dataset of one level of `shape` voxels at 8 nm. */
function dataset(shape: number[], dtype = "uint16"): DatasetInfo {
  return {
    url: "https://example.org/raw.zarr", dtype, channels: 1,
    axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
    levels: [{ shape, voxel: [8, 8, 8], origin: [0, 0, 0] }],
  };
}

/** The raw volume of the tests: voxel (z, y, x) holds 100 z + 10 y + x + 1, never 0. */
const rawValue = (z: number, y: number, x: number) => 100 * z + 10 * y + x + 1;

/** An Inference's readRaw over a level of `shape` whose voxel (z, y, x) is
 * `value(z, y, x)` in `dtype`, recording every box asked and refusing one
 * outside the level, as the engine's reader would. */
function fakeInfer(shape: number[], value: (z: number, y: number, x: number) => number = rawValue, dtype: "uint8" | "uint16" = "uint16") {
  const reads: { level: number; lo: number[]; hi: number[] }[] = [];
  const Typed = dtype === "uint8" ? Uint8Array : Uint16Array;
  return {
    reads,
    async readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
      reads.push({ level, lo: [...lo], hi: [...hi] });
      for (let a = 0; a < 3; a++) {
        if (lo[a] < 0 || hi[a] > shape[a] || hi[a] <= lo[a]) throw new Error(`a box ${lo.join(",")}..${hi.join(",")} is not within the level ${shape.join("×")}`);
      }
      const out = new Typed((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
      let k = 0;
      for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[k++] = value(z, y, x);
      return out.buffer;
    },
  };
}

const NO_STEPS: Step[] = [];
const CHAIN: Chain = { input_norm: [], postprocess: [] };

/** A volume for `entry` over `data` in a fresh store, its record written as
 * the volume's `.zattrs` (JSON) and read back, with `.zarray` beside the
 * chunks and the given chunks' bytes put as the brush would put them
 * (uint8 labels). */
async function volumeIn(entry: ModelEntry, data: DatasetInfo, chunks: Record<string, number[]>, inputNorm: Step[] = NO_STEPS) {
  const store = new MemoryPaintStore();
  const id = "vol-0123abcd-20261015-120000";
  const level = data.levels[0];
  const record: Omit<PaintVolume, "id"> = {
    type: "annotation_volume", model_name: entry.name, model_id: entry.id, dataset_path: data.url,
    chunk_size: entry.variant.output_size, input_size: entry.variant.input_size,
    output_voxel_size: entry.manifest.output_voxel_size_nm, input_voxel_size: entry.manifest.input_voxel_size_nm,
    dataset_offset_nm: [0, 0, 0],
    dataset_shape_voxels: level.shape.map((n, a) => Math.ceil(n / entry.variant.output_size[a]) * entry.variant.output_size[a]),
    dtype: "uint8", input_norm: inputNorm, postprocess: [], created_at: "2026-10-15T12:00:00Z",
  };
  const text = new TextEncoder();
  await store.put(id, ".zattrs", text.encode(JSON.stringify(record)));
  await store.put(id, `${CHUNK_DIR}.zarray`, text.encode(JSON.stringify({ zarr_format: 2 })));
  for (const [key, labels] of Object.entries(chunks)) await store.put(id, CHUNK_DIR + key, Uint8Array.from(labels));
  const volume = await readVolume(store, id);
  if (!volume) throw new Error("the volume's .zattrs did not read back");
  return { store, volume };
}

async function samplesOf(entry: ModelEntry, data: DatasetInfo, chunks: Record<string, number[]>, options: { inputNorm?: Step[]; value?: (z: number, y: number, x: number) => number; chain?: Chain } = {}) {
  const { store, volume } = await volumeIn(entry, data, chunks, options.inputNorm);
  const infer = fakeInfer(data.levels[0].shape, options.value, data.dtype === "uint8" ? "uint8" : "uint16");
  const samples: Sample[] = [];
  for await (const sample of collectSamples({ volume, entry, chain: options.chain ?? CHAIN, dataset: data, infer, paint: store })) samples.push(sample);
  return { samples, reads: infer.reads };
}

/** The block's value at (z, y, x) for a block of `shape`. */
const at = (sample: Sample, shape: number[], z: number, y: number, x: number) => sample.input[(z * shape[1] + y) * shape[2] + x];

/** 6 voxels in, 2 out, at 8 nm: a context of 2 voxels (as the stage tests' geometry). */
const G3 = model([6, 6, 6], [2, 2, 2]);
const RAW3 = dataset([7, 8, 8]);
const EIGHT = [1, 2, 3, 4, 5, 6, 7, 8];

describe("collectSamples on a 3D model", () => {
  it("reads the chunk's box grown by the halo and feeds the model the block as read", async () => {
    const { samples, reads } = await samplesOf(G3, RAW3, { "1.1.1": EIGHT });
    expect(reads).toEqual([{ level: 0, lo: [0, 0, 0], hi: [6, 6, 6] }]);
    expect(samples.length).toBe(1);
    const [s] = samples;
    expect(s.inputDims).toEqual([1, 1, 6, 6, 6]);
    expect(s.input).toBeInstanceOf(Float32Array);
    expect(s.input.length).toBe(216);
    // No normalizers: the raw values, as float32.
    expect(at(s, [6, 6, 6], 0, 0, 0)).toBe(rawValue(0, 0, 0));
    expect(at(s, [6, 6, 6], 5, 5, 5)).toBe(rawValue(5, 5, 5));
    expect(at(s, [6, 6, 6], 2, 3, 4)).toBe(rawValue(2, 3, 4));
    // The chunk as painted.
    expect(s.chunk.index).toEqual([1, 1, 1]);
    expect(s.chunk.shape).toEqual([2, 2, 2]);
    expect(s.chunk.labels).toBeInstanceOf(Uint8Array);
    expect(Array.from(s.chunk.labels)).toEqual(EIGHT);
  });

  it("normalizes the data through the volume's input_norm and repeats it past the level", async () => {
    // The manifests' chain: MinMax 0..255 then 2x-1, so 255 is 1.0, and the
    // voxels past the level repeat that.
    const steps: Step[] = [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }];
    const raw = dataset([7, 8, 8], "uint8");
    // The run's chain says otherwise: the volume's normalizers are the ones
    // the paint was drawn over, and the trainer follows them.
    const chain: Chain = { input_norm: [{ name: "ZScoreNormalizer", mean: 255, std: 1 }], postprocess: [] };
    const { samples } = await samplesOf(G3, raw, { "0.0.0": EIGHT, "1.1.1": EIGHT }, { inputNorm: steps, value: () => 255, chain });
    const [origin, inside] = samples;
    expect(origin.input.every((v) => v === 1)).toBe(true);
    expect(inside.input.every((v) => v === 1)).toBe(true);
  });

  it("yields the chunks asked for, in that order, one never painted as zeros (a good region's block), and none past the level", async () => {
    const { store, volume } = await volumeIn(G3, RAW3, { "1.1.1": EIGHT });
    const infer = fakeInfer(RAW3.levels[0].shape);
    const samples: Sample[] = [];
    for await (const s of collectSamples({ volume, entry: G3, chain: CHAIN, dataset: RAW3, infer, paint: store }, [[1, 1, 1], [0, 2, 3], [9, 0, 0]])) samples.push(s);
    expect(samples.map((s) => s.chunk.index)).toEqual([[1, 1, 1], [0, 2, 3]]);
    expect(Array.from(samples[0].chunk.labels)).toEqual(EIGHT);
    expect(samples[1].chunk).toEqual({ index: [0, 2, 3], shape: [2, 2, 2], labels: new Uint8Array(8) });
    expect(infer.reads.length).toBe(2);
  });
});
