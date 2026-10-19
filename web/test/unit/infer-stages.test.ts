// The two stages of a served model's view, fed what chunkmirage's engine
// feeds them (its own inputBox and padded, from web/chunkmirage/web/src/
// stages.ts, over a fake raw volume) and checked as it checks them (its
// fitted): the model stage's input (normalized where there is data, zero
// past the volume, whole for a chunk the volume ends in) and output
// (channel first, cropped to the chunk), and the postprocess stage's dtype,
// channels and chunk context.
import { describe, expect, it } from "vitest";

import { fitted, inputBox, padded, type Block, type StageAt } from "../../chunkmirage/web/src/stages";
import type { Chain, ModelEntry, Step } from "../../src/contracts";
import { type BlockGeometry, halo as haloOf, ratio } from "../../src/infer/geometry";
import { buildPostprocessors, type NdArray, type Postprocessor } from "../../src/infer/post";
import { copyBox, leading, modelStage, modelStageKey, type ModelOutput, outputBlock, postprocessStage, postprocessStageKey } from "../../src/infer/stages";

/** 6 voxels in, 2 out, at 8 nm: a context of 2 voxels. */
const G: BlockGeometry = { read_shape: [48, 48, 48], write_shape: [16, 16, 16], input_voxel_size: [8, 8, 8], output_voxel_size: [8, 8, 8] };
const RAW = [5, 4, 6];  // the raw level's voxels: three chunks along x, the last chunk along z partial
const CHUNK = [2, 2, 2];
const VOXEL = [8, 8, 8];

/** The raw volume: voxel (z, y, x) holds 100 z + 10 y + x + 1, never 0. */
function rawValue(z: number, y: number, x: number) { return 100 * z + 10 * y + x + 1; }

/** The engine's view of chunk `index`: its box, clipped to the level as
 * Engine.stagedChunk clips it. */
function at(index: number[]): StageAt {
  const lo = index.map((i, a) => i * CHUNK[a]), hi = lo.map((o, a) => Math.min(o + CHUNK[a], RAW[a]));
  return { view: "m", level: 0, index, lo, hi, voxel: VOXEL };
}

/** The model stage's input for `at`, as Engine.input makes it: the box grown
 * by the halo on the raw grid, what lies inside the volume (of `raw` voxels)
 * read (uint16), the rest zero. */
function engineInput(a: StageAt, raw = RAW): Block {
  const { lo, hi } = inputBox(a.lo, a.hi, ratio(G), haloOf(G));
  const readLo = lo.map((o) => Math.max(o, 0)), readHi = hi.map((h, i) => Math.min(h, raw[i]));
  let data: ArrayBuffer | null = null;
  if (readHi.every((h, i) => h > readLo[i])) {
    const shape = readHi.map((h, i) => h - readLo[i]);
    const out = new Uint16Array(shape[0] * shape[1] * shape[2]);
    let k = 0;
    for (let z = readLo[0]; z < readHi[0]; z++) for (let y = readLo[1]; y < readHi[1]; y++) for (let x = readLo[2]; x < readHi[2]; x++) out[k++] = rawValue(z, y, x);
    data = out.buffer;
  }
  return padded(data, "uint16", 1, readLo, readHi, lo, hi, "zero");
}

/** A fake model: C channels, each the centre 2³ of the input times (c + 1);
 * it keeps the last input it was fed. */
function model(channels: number) {
  const fed: { data: Float32Array; dims: number[] }[] = [];
  const run = async (data: Float32Array, dims: number[]): Promise<ModelOutput> => {
    fed.push({ data, dims });
    const out = new Float32Array(channels * 8);
    let i = 0;
    for (let c = 0; c < channels; c++) for (let z = 2; z < 4; z++) for (let y = 2; y < 4; y++) for (let x = 2; x < 4; x++) out[i++] = data[(z * 6 + y) * 6 + x] * (c + 1);
    return { data: out, dims: [1, channels, 2, 2, 2], type: "float32", ms: 1 };
  };
  return { run, fed };
}

function stage(normalizers: Step[], channels = 1, hasChannel = true, run = model(channels).run) {
  return modelStage({ key: "k", geometry: G, rawShape: RAW, inputVoxelSize: VOXEL, normalizers, channels, hasChannel, run });
}

/** `stage` on the engine's input for chunk `index`, fitted as the engine fits
 * a first stage on its own grid (the result must be the box's shape). */
async function produce(s: ReturnType<typeof stage>, index: number[]): Promise<Block> {
  const a = at(index), input = engineInput(a);
  const box = a.hi.map((h, i) => h - a.lo[i]);
  return fitted(await s.produce(input, a), s.dtype, [...(s.added ?? []), ...box], input.shape, null);
}

const NONE: Step[] = [];

describe("the model stage", () => {
  it("declares the model's context as its halo and the raw level as what it reads", () => {
    const s = stage(NONE, 3);
    expect(s.halo).toEqual([2, 2, 2]);
    expect(s.inputVoxelSize).toEqual(VOXEL);
    expect(s.dtype).toBe("float32");
    expect(s.added).toEqual([3]);
  });

  it("feeds the model the chunk grown by the halo and serves its output channel first", async () => {
    const m = model(2);
    const out = await produce(stage(NONE, 2, true, m.run), [1, 0, 1]);
    expect(m.fed[0].dims).toEqual([1, 1, 6, 6, 6]);
    expect(out.shape).toEqual([2, 2, 2, 2]);
    const data = out.data as Float32Array;
    // output voxel (2, 0, 2) = raw voxel (2, 0, 2), channel 1 doubled
    expect(data[0]).toBe(rawValue(2, 0, 2));
    expect(data[8]).toBe(2 * rawValue(2, 0, 2));
    expect(data[7]).toBe(rawValue(3, 1, 3));
  });

  it("normalizes the data and repeats its normalized edge past the volume, as the Python server pads after the chain", async () => {
    const m = model(1);
    await produce(stage([{ name: "ZScoreNormalizer", mean: "1", std: "1" }], 1, true, m.run), [0, 0, 0]);  // raw - 1
    const fed = m.fed[0].data, v = (z: number, y: number, x: number) => fed[(z * 6 + y) * 6 + x];
    // Before the volume: the nearest voxel inside it, normalized (not its raw value, not 0).
    expect(v(1, 2, 3)).toBe(rawValue(0, 0, 1) - 1);
    expect(v(0, 4, 5)).toBe(rawValue(0, 2, 3) - 1);
    expect(v(2, 2, 2)).toBe(rawValue(0, 0, 0) - 1);  // the volume starts at block voxel 2
    expect(v(5, 5, 5)).toBe(rawValue(3, 3, 3) - 1);
  });

  it("normalizes the whole block in place when all of it is data", async () => {
    const m = model(1);
    const s = modelStage({ key: "k", geometry: G, rawShape: [20, 20, 20], inputVoxelSize: VOXEL, normalizers: [{ name: "LambdaNormalizer", expression: "x/1000" }], channels: 1, hasChannel: true, run: m.run });
    const a = { ...at([1, 1, 1]), hi: [4, 4, 4] };
    const input = engineInput(a, [20, 20, 20]);
    await s.produce(input, a);
    expect(m.fed[0].data[0]).toBeCloseTo(rawValue(0, 0, 0) / 1000, 6);
    expect(m.fed[0].data[215]).toBeCloseTo(rawValue(5, 5, 5) / 1000, 6);
  });

  it("runs the model on the whole block for a chunk the volume ends in and crops the output to the chunk", async () => {
    // chunk [2, 1, 0] covers z 4..5 (the level ends at 5), so the engine's
    // block is 5 deep, not the model's 6
    const m = model(1);
    const a = at([2, 1, 0]);
    expect(a.hi).toEqual([5, 4, 2]);
    expect(engineInput(a).shape).toEqual([5, 6, 6]);
    const out = await produce(stage(NONE, 1, true, m.run), [2, 1, 0]);
    expect(m.fed[0].dims).toEqual([1, 1, 6, 6, 6]);
    const fed = m.fed[0].data, v = (z: number, y: number, x: number) => fed[(z * 6 + y) * 6 + x];
    expect(v(2, 2, 2)).toBe(rawValue(4, 2, 0));  // the block starts at raw (2, 0, -2)
    expect(v(3, 2, 2)).toBe(rawValue(4, 2, 0));  // raw z 5: past the volume, its last voxel repeated
    expect(v(5, 2, 2)).toBe(rawValue(4, 2, 0));  // raw z 7: past what the engine read, the same
    expect(out.shape).toEqual([1, 1, 2, 2]);
    expect(Array.from(out.data as Float32Array)).toEqual([rawValue(4, 2, 0), rawValue(4, 2, 1), rawValue(4, 3, 0), rawValue(4, 3, 1)]);
  });

  it("is zeros for a chunk wholly outside the volume", async () => {
    const m = model(1);
    const s = modelStage({ key: "k", geometry: G, rawShape: [1, 1, 1], inputVoxelSize: VOXEL, normalizers: [{ name: "ZScoreNormalizer", mean: "1", std: "1" }], channels: 1, hasChannel: true, run: m.run });
    const a: StageAt = { view: "m", level: 0, index: [5, 5, 5], lo: [10, 10, 10], hi: [12, 12, 12], voxel: VOXEL };
    const { lo, hi } = inputBox(a.lo, a.hi, [1, 1, 1], [2, 2, 2]);
    await s.produce(padded(null, "uint16", 1, lo, lo, lo, hi, "zero"), a);
    expect(m.fed[0].data.every((x) => x === 0)).toBe(true);
  });

  it("squeezes the channel for a view without a channel axis, and refuses two", async () => {
    const out = await produce(stage(NONE, 1, false), [0, 0, 0]);
    expect(out.shape).toEqual([2, 2, 2]);
    await expect(produce(stage(NONE, 2, false), [0, 0, 0])).rejects.toThrow(/cannot serve 2 channels/);
  });

  it("refuses an output that is not what the manifest says", async () => {
    const wrongSize = async (): Promise<ModelOutput> => ({ data: new Float32Array(27), dims: [1, 1, 3, 3, 3], type: "float32", ms: 0 });
    await expect(produce(stage(NONE, 1, true, wrongSize), [0, 0, 0])).rejects.toThrow(/wrote 3×3×3 voxels, not the 2×2×2/);
    await expect(produce(stage(NONE, 3, true, model(2).run), [0, 0, 0])).rejects.toThrow(/wrote 2 channel\(s\), not the 3/);
    const half = async (): Promise<ModelOutput> => ({ data: new Uint16Array(8), dims: [1, 1, 2, 2, 2], type: "float16", ms: 0 });
    await expect(produce(stage(NONE, 1, true, half), [0, 0, 0])).rejects.toThrow(/float16/);
  });

  it("refuses a raw volume with channels", async () => {
    const a = at([0, 0, 0]), input = engineInput(a);
    await expect(stage(NONE).produce({ data: new Uint16Array(2 * 216), shape: [2, ...input.shape] }, a)).rejects.toThrow(/2 channels/);
  });

  it("refuses a block that is not the engine's for the box", async () => {
    const a = at([0, 0, 0]);
    await expect(stage(NONE).produce({ data: new Uint16Array(125), shape: [5, 5, 5] }, a)).rejects.toThrow(/block of 5×5×5 for an input box of 6×6×6/);
  });
});

describe("outputBlock", () => {
  it("drops the batch axis of [1, C, z, y, x] and adds a channel to [1, z, y, x]", () => {
    expect(outputBlock({ data: new Float32Array(16), dims: [1, 2, 2, 2, 2], type: "float32", ms: 0 }, 2, [2, 2, 2]).shape).toEqual([2, 2, 2, 2]);
    expect(outputBlock({ data: new Float32Array(8), dims: [1, 2, 2, 2], type: "float32", ms: 0 }, 1, [2, 2, 2]).shape).toEqual([1, 2, 2, 2]);
    expect(() => outputBlock({ data: new Float32Array(8), dims: [2, 2, 2], type: "float32", ms: 0 }, 1, [2, 2, 2])).toThrow(/rank 3/);
    expect(() => outputBlock({ data: new Float32Array(16), dims: [2, 1, 2, 2, 2], type: "float32", ms: 0 }, 1, [2, 2, 2])).toThrow(/batch of 2/);
  });
});

describe("leading and copyBox", () => {
  it("takes the leading voxels of every channel", () => {
    const nd: NdArray = { data: Float32Array.from({ length: 16 }, (_, i) => i), shape: [2, 2, 2, 2] };
    const out = leading(nd, [2, 1, 2, 2]);
    expect(out.shape).toEqual([2, 1, 2, 2]);
    expect(Array.from(out.data as Float32Array)).toEqual([0, 1, 2, 3, 8, 9, 10, 11]);
    expect(() => leading(nd, [2, 3, 2, 2])).toThrow(/no leading/);
  });

  it("copies a box between blocks at offsets, and refuses one that does not fit", () => {
    const dst = new Float32Array(27);
    copyBox(dst, [3, 3, 3], [1, 1, 1], new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2], [0, 0, 0], [2, 2, 2]);
    const v = (z: number, y: number, x: number) => dst[(z * 3 + y) * 3 + x];
    expect(v(0, 0, 0)).toBe(0);
    expect(v(1, 1, 1)).toBe(1);
    expect(v(1, 1, 2)).toBe(2);
    expect(v(2, 2, 2)).toBe(8);
    expect(Array.from(dst).filter((x) => x !== 0).length).toBe(8);
    expect(() => copyBox(new Float32Array(8), [2, 2, 2], [1, 0, 0], new Float32Array(8), [2, 2, 2], [0, 0, 0], [2, 2, 2])).toThrow(/does not fit/);
  });
});

describe("the postprocess stage", () => {
  const block = (channels: number): Block => ({ data: Float32Array.from({ length: channels * 8 }, (_, i) => i * 10), shape: [channels, 2, 2, 2] });
  function post(steps: Step[], modelChannels = 1, hasChannel = true, postprocessors: Postprocessor[] = buildPostprocessors(steps)) {
    return postprocessStage({ key: "p", geometry: G, steps, postprocessors, modelChannels, hasChannel });
  }

  it("declares the chain's dtype and channels", () => {
    expect(post([{ name: "ThresholdPostprocessor", threshold: "0.5" }]).dtype).toBe("uint8");
    expect(post([{ name: "ChannelSelection", channels: "0,2" }], 3).added).toEqual([2]);
    expect(post([], 3).dtype).toBe("float32");
    expect(post([{ name: "MortonSegmentationRelabeling" }]).dtype).toBe("uint64");
  });

  it("returns a segmentation's uint64 ids as the BigUint64Array the engine fits", async () => {
    const s = post([{ name: "ThresholdPostprocessor", threshold: "35" }, { name: "MortonSegmentationRelabeling" }]);
    const input = block(1);
    const out = fitted(await s.produce(input, at([1, 0, 0])), s.dtype, [1, 2, 2, 2], input.shape, null);
    expect(out.data).toBeInstanceOf(BigUint64Array);
    // the foreground is one component, its id the chunk's Morton code times the block's voxels plus 1
    const ids = new Set(Array.from(out.data as BigUint64Array).filter((v) => v !== 0n));
    expect(ids.size).toBe(1);
    expect([...ids][0]).toBeGreaterThan(0n);
  });

  it("runs the steps and returns the block in the declared dtype, as the engine checks", async () => {
    const s = post([{ name: "ThresholdPostprocessor", threshold: "35" }]);
    const input = block(1);
    const out = fitted(await s.produce(input, at([0, 0, 0])), s.dtype, [1, 2, 2, 2], input.shape, null);
    expect(out.data).toBeInstanceOf(Uint8Array);
    expect(Array.from(out.data as ArrayLike<number>)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  });

  it("selects channels and casts into the chain's dtype", async () => {
    const s = post([{ name: "DefaultPostprocessor", clip_min: "0", clip_max: "1000", bias: "0", multiplier: "1" }, { name: "ChannelSelection", channels: "2" }], 3);
    const out = (await s.produce(block(3), at([0, 0, 0]))) as Block;
    expect(out.shape).toEqual([1, 2, 2, 2]);
    expect(Array.from(out.data as ArrayLike<number>)).toEqual([160, 170, 180, 190, 200, 210, 220, 230]);
  });

  it("gives the steps the chunk's index and the voxels of one whole block", async () => {
    const seen: { index: number[]; voxels: number }[] = [];
    const spy: Postprocessor = (nd, ctx) => { seen.push({ index: ctx.chunkIndex, voxels: ctx.chunkNumVoxels }); return nd; };
    const s = post([], 1, true, [spy]);
    await s.produce(block(1), at([2, 1, 0]));  // a clipped chunk: the block's voxels all the same
    expect(seen).toEqual([{ index: [2, 1, 0], voxels: 8 }]);
  });

  it("squeezes the channel for a view without a channel axis, lifting a plain block for the steps", async () => {
    const s = post([{ name: "SigmoidPostprocessor" }], 1, false);
    const out = (await s.produce({ data: new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]), shape: [2, 2, 2] }, at([0, 0, 0]))) as Block;
    expect(out.shape).toEqual([2, 2, 2]);
    expect(Array.from(out.data as ArrayLike<number>)).toEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    await expect(post([], 2, false).produce(block(2), at([0, 0, 0]))).rejects.toThrow(/cannot serve 2 channels/);
  });
});

describe("the stage keys", () => {
  const entry = { id: "cellmap/onnx-web-models/setup_54", variant: { file: "setup_54_in148_fp32.onnx" } } as ModelEntry;
  const chain: Chain = { input_norm: [{ name: "ZScoreNormalizer", std: "1", mean: "0" }], postprocess: [{ name: "ThresholdPostprocessor", threshold: "0.5" }] };

  it("name the model file and the normalizers, canonically", () => {
    const key = modelStageKey(entry, chain);
    expect(key).toBe('cellmap/onnx-web-models/setup_54|setup_54_in148_fp32.onnx|norm:[{"mean":"0","name":"ZScoreNormalizer","std":"1"}]|model');
    expect(modelStageKey(entry, { ...chain, input_norm: [{ name: "ZScoreNormalizer", mean: "0", std: "1" }] })).toBe(key);  // key order does not matter
    expect(modelStageKey(entry, { ...chain, postprocess: [] })).toBe(key);  // nor the postprocessing
    expect(modelStageKey(entry, { ...chain, input_norm: [] })).not.toBe(key);
  });

  it("and the postprocess key adds the steps", () => {
    const key = modelStageKey(entry, chain);
    expect(postprocessStageKey(key, chain)).toBe(`${key}|post:[{"name":"ThresholdPostprocessor","threshold":"0.5"}]`);
    expect(postprocessStageKey(key, { ...chain, postprocess: [] })).toBe(`${key}|post:[]`);
  });
});
