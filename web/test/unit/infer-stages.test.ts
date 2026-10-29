// The two stages of a served model's view, fed what chunkmirage's engine
// feeds them (its own inputBox and padded, from web/chunkmirage/web/src/
// stages.ts, over a fake raw volume) and checked as it checks them (its
// fitted): the model stage's input (normalized where there is data, its
// normalized edge repeated past the volume, whole for a chunk the volume
// ends in) and output (channel first, cropped to the chunk, refused when it
// is not what the manifest says), the postprocess stage's uint64 ids and
// the chunk context its steps get, and the keys a stage's output is kept
// under for revisits.
import { describe, expect, it } from "vitest";

import { fitted, inputBox, padded, type Block, type StageAt } from "../../chunkmirage/web/src/stages";
import type { Chain, ModelEntry, Step } from "../../src/contracts";
import { type BlockGeometry, halo as haloOf, ratio } from "../../src/infer/geometry";
import { buildPostprocessors, type Postprocessor } from "../../src/infer/post";
import { modelStage, modelStageKey, type ModelOutput, postprocessStage, postprocessStageKey } from "../../src/infer/stages";

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
 * read (uint16), the rest zero; of `channels` read together, as the reader
 * reads a precomputed volume's, channel c 1000 above the one before. */
function engineInput(a: StageAt, raw = RAW, channels = 1): Block {
  const { lo, hi } = inputBox(a.lo, a.hi, ratio(G), haloOf(G));
  const readLo = lo.map((o) => Math.max(o, 0)), readHi = hi.map((h, i) => Math.min(h, raw[i]));
  let data: ArrayBuffer | null = null;
  if (readHi.every((h, i) => h > readLo[i])) {
    const shape = readHi.map((h, i) => h - readLo[i]);
    const out = new Uint16Array(channels * shape[0] * shape[1] * shape[2]);
    let k = 0;
    for (let c = 0; c < channels; c++) {
      for (let z = readLo[0]; z < readHi[0]; z++) for (let y = readLo[1]; y < readHi[1]; y++) for (let x = readLo[2]; x < readHi[2]; x++) out[k++] = rawValue(z, y, x) + 1000 * c;
    }
    data = out.buffer;
  }
  return padded(data, "uint16", channels, readLo, readHi, lo, hi, "zero");
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

  it("feeds the model the raw channel the chain's ChannelSelector picks, of a volume read with its channels together", async () => {
    const m = model(1);
    const a = at([1, 0, 1]);
    await stage([{ name: "ChannelSelector", channel: "1" }], 1, true, m.run).produce(engineInput(a, RAW, 2), a);
    const fed = m.fed[0].data;
    expect(fed.length).toBe(216);
    expect(fed[(2 * 6 + 2) * 6 + 2]).toBe(rawValue(2, 0, 2) + 1000);  // the block starts at raw (0, -2, 0)
  });

  it("refuses an output that is not what the manifest says", async () => {
    const wrongSize = async (): Promise<ModelOutput> => ({ data: new Float32Array(27), dims: [1, 1, 3, 3, 3], type: "float32", ms: 0 });
    await expect(produce(stage(NONE, 1, true, wrongSize), [0, 0, 0])).rejects.toThrow(/3×3×3/);
    await expect(produce(stage(NONE, 3, true, model(2).run), [0, 0, 0])).rejects.toThrow(/channel/);
    const half = async (): Promise<ModelOutput> => ({ data: new Uint16Array(8), dims: [1, 1, 2, 2, 2], type: "float16", ms: 0 });
    await expect(produce(stage(NONE, 1, true, half), [0, 0, 0])).rejects.toThrow(/float16/);
  });
});

describe("the postprocess stage", () => {
  it("returns a segmentation's uint64 ids as the BigUint64Array the engine fits", async () => {
    const steps: Step[] = [{ name: "ThresholdPostprocessor", threshold: "35" }, { name: "MortonSegmentationRelabeling" }];
    const s = postprocessStage({ key: "p", geometry: G, steps, postprocessors: buildPostprocessors(steps), modelChannels: 1, hasChannel: true });
    const input: Block = { data: Float32Array.from({ length: 8 }, (_, i) => i * 10), shape: [1, 2, 2, 2] };
    const out = fitted(await s.produce(input, at([1, 0, 0])), s.dtype, [1, 2, 2, 2], input.shape, null);
    expect(out.data).toBeInstanceOf(BigUint64Array);
    // the foreground is one component, its id the chunk's Morton code times the block's voxels plus 1
    const ids = new Set(Array.from(out.data as BigUint64Array).filter((v) => v !== 0n));
    expect(ids.size).toBe(1);
    expect([...ids][0]).toBeGreaterThan(0n);
  });

  it("gives the steps the chunk's index and the voxels of one whole block", async () => {
    const seen: { index: number[]; voxels: number }[] = [];
    const spy: Postprocessor = (nd, ctx) => { seen.push({ index: ctx.chunkIndex, voxels: ctx.chunkNumVoxels }); return nd; };
    const s = postprocessStage({ key: "p", geometry: G, steps: [], postprocessors: [spy], modelChannels: 1, hasChannel: true });
    await s.produce({ data: new Float32Array(8), shape: [1, 2, 2, 2] }, at([2, 1, 0]));  // a clipped chunk: the block's voxels all the same
    expect(seen).toEqual([{ index: [2, 1, 0], voxels: 8 }]);
  });
});

describe("the stage keys", () => {
  const entry = { id: "cellmap/onnx-web-models/setup_54", variant: { file: "setup_54_in148_fp32.onnx" } } as ModelEntry;
  const chain: Chain = { input_norm: [{ name: "ZScoreNormalizer", std: "1", mean: "0" }], postprocess: [{ name: "ThresholdPostprocessor", threshold: "0.5" }] };

  it("name the model file and the normalizers, canonically", () => {
    const key = modelStageKey(entry, chain);
    expect(modelStageKey(entry, { ...chain, input_norm: [{ name: "ZScoreNormalizer", mean: "0", std: "1" }] })).toBe(key);  // key order does not matter
    expect(modelStageKey(entry, { ...chain, postprocess: [] })).toBe(key);  // nor the postprocessing
    expect(modelStageKey(entry, { ...chain, input_norm: [] })).not.toBe(key);
    expect(modelStageKey({ ...entry, variant: { file: "setup_54_in116_fp32.onnx" } } as ModelEntry, chain)).not.toBe(key);
  });

  it("and the postprocess key adds the steps", () => {
    const key = modelStageKey(entry, chain);
    expect(postprocessStageKey(key, chain)).toBe(postprocessStageKey(key, { ...chain, input_norm: [] }));
    expect(postprocessStageKey(key, { ...chain, postprocess: [] })).not.toBe(postprocessStageKey(key, chain));
    expect(postprocessStageKey(key, chain)).not.toBe(key);
  });
});
