// collectSamples (src/finetune/samples.ts) on a hand-made annotation volume
// in a MemoryPaintStore and a fake Inference whose readRaw records the boxes
// asked and answers a ramp: a sample must be the block the model stage would
// feed the model for the same chunk (test/unit/infer-stages.test.ts builds
// that block the same way from chunkmirage's inputBox and padded), so the
// checks are the box read (whole, or clipped where the level ends), the
// zeros where the block leaves the level, the normalization through the
// volume's input_norm, the dims, and which chunks of the store are samples.
import { describe, expect, it } from "vitest";

import type { Chain, DatasetInfo, ModelEntry, ModelManifest, PaintVolume, Step } from "../../src/contracts";
import { collectSamples } from "../../src/finetune/samples";
import type { Sample } from "../../src/finetune/types";
import { CHUNK_DIR, readVolume } from "../../src/finetune/volume";
import { entryFor } from "../../src/models/manifest";
import { MemoryPaintStore } from "../../src/state/paint-store";

/** A model reading `input` voxels at `voxel` nm and writing `output` at `outputVoxel` nm. */
function model(input: number[], output: number[], voxel = 8, outputVoxel = voxel): ModelEntry {
  const manifest: ModelManifest = {
    model: "setup_t", task: "mito", architecture: "", checkpoint: "", params: 0, channels: ["mito"],
    input_voxel_size_nm: [voxel, voxel, voxel], output_voxel_size_nm: [outputVoxel, outputVoxel, outputVoxel],
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
 * (uint8 labels; a Uint16Array's as little-endian uint16). */
async function volumeIn(entry: ModelEntry, data: DatasetInfo, chunks: Record<string, number[] | Uint16Array>, inputNorm: Step[] = NO_STEPS) {
  const store = new MemoryPaintStore();
  const id = "vol-0123abcd-20261015-120000";
  const uint16 = Object.values(chunks).some((c) => c instanceof Uint16Array);
  const level = data.levels[0];
  const record: Omit<PaintVolume, "id"> = {
    type: "annotation_volume", model_name: entry.name, model_id: entry.id, dataset_path: data.url,
    chunk_size: entry.variant.output_size, input_size: entry.variant.input_size,
    output_voxel_size: entry.manifest.output_voxel_size_nm, input_voxel_size: entry.manifest.input_voxel_size_nm,
    dataset_offset_nm: [0, 0, 0],
    dataset_shape_voxels: level.shape.map((n, a) => Math.ceil(n / entry.variant.output_size[a]) * entry.variant.output_size[a]),
    dtype: uint16 ? "uint16" : "uint8", input_norm: inputNorm, postprocess: [], created_at: "2026-10-15T12:00:00Z",
  };
  const text = new TextEncoder();
  await store.put(id, ".zattrs", text.encode(JSON.stringify(record)));
  await store.put(id, `${CHUNK_DIR}.zarray`, text.encode(JSON.stringify({ zarr_format: 2 })));
  for (const [key, labels] of Object.entries(chunks)) {
    const bytes = labels instanceof Uint16Array ? new Uint8Array(labels.buffer, labels.byteOffset, labels.byteLength) : Uint8Array.from(labels);
    await store.put(id, CHUNK_DIR + key, bytes);
  }
  const volume = await readVolume(store, id);
  if (!volume) throw new Error("the volume's .zattrs did not read back");
  return { store, volume };
}

async function samplesOf(entry: ModelEntry, data: DatasetInfo, chunks: Record<string, number[] | Uint16Array>, options: { inputNorm?: Step[]; value?: (z: number, y: number, x: number) => number; chain?: Chain } = {}) {
  const { store, volume } = await volumeIn(entry, data, chunks, options.inputNorm);
  const infer = fakeInfer(data.levels[0].shape, options.value, data.dtype === "uint8" ? "uint8" : "uint16");
  const samples: Sample[] = [];
  for await (const sample of collectSamples({ volume, entry, chain: options.chain ?? CHAIN, dataset: data, infer, paint: store })) samples.push(sample);
  return { samples, reads: infer.reads, volume };
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

  it("clips the box where the level ends and leaves the block zero past it", async () => {
    // Chunk [3, 1, 1] covers z 6..7 on a level of 7: the engine clips it to
    // one voxel deep, and its input box, z 4..9, is read to 7. The block the
    // model sees is still 6 deep: the data at the front, zeros behind.
    const { samples, reads } = await samplesOf(G3, RAW3, { "3.1.1": EIGHT });
    expect(reads).toEqual([{ level: 0, lo: [4, 0, 0], hi: [7, 6, 6] }]);
    const [s] = samples;
    expect(s.inputDims).toEqual([1, 1, 6, 6, 6]);
    expect(at(s, [6, 6, 6], 0, 0, 0)).toBe(rawValue(4, 0, 0));
    expect(at(s, [6, 6, 6], 2, 5, 5)).toBe(rawValue(6, 5, 5));
    expect(at(s, [6, 6, 6], 3, 0, 0)).toBe(0);
    expect(at(s, [6, 6, 6], 5, 5, 5)).toBe(0);
  });

  it("is zero before the level for a chunk at the origin", async () => {
    const { samples, reads } = await samplesOf(G3, RAW3, { "0.0.0": EIGHT });
    expect(reads).toEqual([{ level: 0, lo: [0, 0, 0], hi: [4, 4, 4] }]);
    const [s] = samples;
    expect(at(s, [6, 6, 6], 0, 0, 0)).toBe(0);
    expect(at(s, [6, 6, 6], 1, 1, 1)).toBe(0);
    expect(at(s, [6, 6, 6], 1, 2, 2)).toBe(0);
    expect(at(s, [6, 6, 6], 2, 2, 2)).toBe(rawValue(0, 0, 0));
    expect(at(s, [6, 6, 6], 5, 5, 5)).toBe(rawValue(3, 3, 3));
  });

  it("normalizes the data through the volume's input_norm and leaves the padding at zero", async () => {
    // The manifests' chain: MinMax 0..255 then 2x-1, so 255 is 1.0; the
    // zeros past the level stay 0, not -1, as the server pads after the chain.
    const steps: Step[] = [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }];
    const raw = dataset([7, 8, 8], "uint8");
    // The run's chain says otherwise: the volume's normalizers are the ones
    // the paint was drawn over, and the trainer follows them.
    const chain: Chain = { input_norm: [{ name: "ZScoreNormalizer", mean: 255, std: 1 }], postprocess: [] };
    const { samples } = await samplesOf(G3, raw, { "0.0.0": EIGHT, "1.1.1": EIGHT }, { inputNorm: steps, value: () => 255, chain });
    const [origin, inside] = samples;
    expect(at(origin, [6, 6, 6], 0, 0, 0)).toBe(0);
    expect(at(origin, [6, 6, 6], 2, 2, 2)).toBe(1);
    expect(at(origin, [6, 6, 6], 5, 5, 5)).toBe(1);
    expect(inside.input.every((v) => v === 1)).toBe(true);
  });

  it("yields the painted chunks in key order, skipping nothing painted, .zarray and a chunk past the level", async () => {
    const { samples, reads } = await samplesOf(G3, RAW3, { "1.1.1": EIGHT, "0.0.0": EIGHT, "9.0.0": EIGHT });
    expect(samples.map((s) => s.chunk.index)).toEqual([[0, 0, 0], [1, 1, 1]]);
    expect(reads.length).toBe(2);  // nothing read for the chunk past the level
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

  it("carries uint16 labels", async () => {
    const ids = Uint16Array.from([0, 1, 2, 300, 40000, 1, 2, 2]);
    const { samples, volume } = await samplesOf(G3, RAW3, { "1.1.1": ids });
    expect(volume.dtype).toBe("uint16");
    expect(samples[0].chunk.labels).toBeInstanceOf(Uint16Array);
    expect(Array.from(samples[0].chunk.labels)).toEqual(Array.from(ids));
  });

  it("refuses a volume whose chunks are not the model's output block", async () => {
    const { store, volume } = await volumeIn(G3, RAW3, { "0.0.0": EIGHT });
    const other = model([8, 8, 8], [4, 4, 4]);
    const run = collectSamples({ volume, entry: other, chain: CHAIN, dataset: RAW3, infer: fakeInfer(RAW3.levels[0].shape), paint: store });
    await expect(run.next()).rejects.toThrow(/2×2×2 voxels, but .* writes 4×4×4/);
  });
});

describe("collectSamples on a 2D model", () => {
  /** One slice in, one out: a halo of 2 along y and x, none along z. */
  const G2 = model([1, 6, 6], [1, 2, 2]);
  const FOUR = [1, 2, 2, 1];

  it("reads one slice, grown by the halo in the plane", async () => {
    const { samples, reads } = await samplesOf(G2, dataset([3, 8, 8]), { "2.1.1": FOUR });
    expect(reads).toEqual([{ level: 0, lo: [2, 0, 0], hi: [3, 6, 6] }]);
    const [s] = samples;
    expect(s.inputDims).toEqual([1, 1, 1, 6, 6]);
    expect(s.input.length).toBe(36);
    expect(at(s, [1, 6, 6], 0, 0, 0)).toBe(rawValue(2, 0, 0));
    expect(at(s, [1, 6, 6], 0, 5, 5)).toBe(rawValue(2, 5, 5));
    expect(s.chunk.shape).toEqual([1, 2, 2]);
    expect(Array.from(s.chunk.labels)).toEqual(FOUR);
  });

  it("clips a slice's box at the far edge of the plane and zero-pads", async () => {
    // A 7-voxel plane: chunk [0, 3, 3] covers y, x 6..7, its box 4..10 is
    // read to 7, and the block has data in its first three rows and columns.
    const { samples, reads } = await samplesOf(G2, dataset([3, 7, 7]), { "0.3.3": FOUR });
    expect(reads).toEqual([{ level: 0, lo: [0, 4, 4], hi: [1, 7, 7] }]);
    const [s] = samples;
    expect(at(s, [1, 6, 6], 0, 0, 0)).toBe(rawValue(0, 4, 4));
    expect(at(s, [1, 6, 6], 0, 2, 2)).toBe(rawValue(0, 6, 6));
    expect(at(s, [1, 6, 6], 0, 3, 0)).toBe(0);
    expect(at(s, [1, 6, 6], 0, 0, 3)).toBe(0);
    expect(at(s, [1, 6, 6], 0, 5, 5)).toBe(0);
  });
});

describe("collectSamples on a model writing coarser voxels than it reads", () => {
  it("scales the chunk's box onto the raw grid before growing it by the halo", async () => {
    // 8 voxels in at 8 nm, 2 out at 16 nm: a 64 nm block writing 32 nm, a
    // context of 16 nm = 2 raw voxels, two raw voxels to each output voxel.
    const G = model([8, 8, 8], [2, 2, 2], 8, 16);
    const raw = dataset([12, 12, 12]);
    const { samples, reads, volume } = await samplesOf(G, raw, { "1.0.0": EIGHT });
    expect(volume.output_voxel_size).toEqual([16, 16, 16]);
    // Output chunk [1, 0, 0] is z 2..4 on the 16 nm grid, raw z 4..8, grown to 2..10; y and x 0..4 grown to -2..6.
    expect(reads).toEqual([{ level: 0, lo: [2, 0, 0], hi: [10, 6, 6] }]);
    const [s] = samples;
    expect(s.inputDims).toEqual([1, 1, 8, 8, 8]);
    expect(at(s, [8, 8, 8], 0, 0, 0)).toBe(0);
    expect(at(s, [8, 8, 8], 0, 2, 2)).toBe(rawValue(2, 0, 0));
    expect(at(s, [8, 8, 8], 7, 7, 7)).toBe(rawValue(9, 5, 5));
  });
});
