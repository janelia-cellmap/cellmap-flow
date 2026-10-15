// The whole chunk pipeline with a fake raw volume and a fake model: the input
// block the model sees (clipped, padded, normalized), and the chunk's bytes
// (channel first, in the chain's dtype) for the output it returns.
import { describe, expect, it } from "vitest";

import type { Chain } from "../../src/contracts";
import { type BlockGeometry, inputBox } from "../../src/infer/geometry";
import { type ModelOutput, type ProduceContext, outputToNdArray, produceChunk, readInputBlock } from "../../src/infer/produce";
import { buildPostprocessors } from "../../src/infer/post";

/** 6 voxels in, 2 out, at 8 nm: a context of 2 voxels. */
const G: BlockGeometry = { read_shape: [48, 48, 48], write_shape: [16, 16, 16], input_voxel_size: [8, 8, 8], output_voxel_size: [8, 8, 8] };
const RAW = [5, 4, 6];  // the raw level's voxels: three chunks along x, the last one partial

/** The raw volume: voxel (z, y, x) holds 100 z + 10 y + x + 1, never 0. */
function rawValue(z: number, y: number, x: number) { return 100 * z + 10 * y + x + 1; }

async function read(lo: number[], hi: number[]): Promise<ArrayBuffer> {
  const shape = hi.map((h, a) => h - lo[a]);
  const out = new Uint16Array(shape[0] * shape[1] * shape[2]);
  let i = 0;
  for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) out[i++] = rawValue(z, y, x);
  return out.buffer;
}

/** A fake model: C channels, each the centre 2³ of the input times (c + 1). */
function model(channels: number): (input: Float32Array, dims: number[]) => Promise<ModelOutput> {
  return async (input, dims) => {
    expect(dims).toEqual([1, 1, 6, 6, 6]);
    const out = new Float32Array(channels * 8);
    let i = 0;
    for (let c = 0; c < channels; c++) for (let z = 2; z < 4; z++) for (let y = 2; y < 4; y++) for (let x = 2; x < 4; x++) out[i++] = input[(z * 6 + y) * 6 + x] * (c + 1);
    return { data: out, dims: [1, channels, 2, 2, 2], type: "float32" };
  };
}

function context(chain: Chain, channels = 1, hasChannel = true): ProduceContext {
  return { geometry: G, rawShape: RAW, rawDtype: "uint16", chain, postprocessors: buildPostprocessors(chain.postprocess), hasChannel, read, run: model(channels) };
}

const NONE: Chain = { input_norm: [], postprocess: [] };

describe("readInputBlock", () => {
  it("reads the output chunk grown by the context, padding what lies outside the data", async () => {
    const block = await readInputBlock(context(NONE), [0, 0, 0]);
    expect(block.length).toBe(216);
    const { begin } = inputBox(G, [0, 0, 0]);
    expect(begin).toEqual([-2, -2, -2]);
    const at = (z: number, y: number, x: number) => block[(z * 6 + y) * 6 + x];
    expect(at(0, 0, 0)).toBe(0);  // before the data
    expect(at(1, 1, 1)).toBe(0);
    expect(at(2, 2, 2)).toBe(rawValue(0, 0, 0));  // the data starts at block voxel 2
    expect(at(5, 5, 5)).toBe(rawValue(3, 3, 3));
  });

  it("pads past the data's end", async () => {
    // chunk [0, 0, 2] covers x 4..6; its input block x 2..8 while the data ends at 6
    const block = await readInputBlock(context(NONE), [0, 0, 2]);
    const at = (z: number, y: number, x: number) => block[(z * 6 + y) * 6 + x];
    expect(at(2, 2, 0)).toBe(rawValue(0, 0, 2));
    expect(at(2, 2, 3)).toBe(rawValue(0, 0, 5));
    expect(at(2, 2, 4)).toBe(0);
    expect(at(2, 2, 5)).toBe(0);
  });

  it("is zeros for a block wholly outside", async () => {
    const block = await readInputBlock({ ...context(NONE), rawShape: [1, 1, 1] }, [5, 5, 5]);
    expect(Array.from(block).every((v) => v === 0)).toBe(true);
  });

  it("returns the data itself when the whole block is inside", async () => {
    const block = await readInputBlock({ ...context(NONE), rawShape: [20, 20, 20] }, [1, 1, 1]);
    expect(block[0]).toBe(rawValue(0, 0, 0));
    expect(block[215]).toBe(rawValue(5, 5, 5));
  });
});

describe("outputToNdArray", () => {
  it("drops the batch axis of [1, C, z, y, x] and adds a channel to [1, z, y, x]", () => {
    expect(outputToNdArray({ data: new Float32Array(16), dims: [1, 2, 2, 2, 2], type: "float32" }).shape).toEqual([2, 2, 2, 2]);
    expect(outputToNdArray({ data: new Float32Array(8), dims: [1, 2, 2, 2], type: "float32" }).shape).toEqual([1, 2, 2, 2]);
    expect(() => outputToNdArray({ data: new Float32Array(8), dims: [2, 2, 2], type: "float32" })).toThrow(/rank 3/);
    expect(() => outputToNdArray({ data: new Uint16Array(8), dims: [1, 1, 2, 2, 2], type: "float16" })).toThrow(/float16/);
  });
});

describe("produceChunk", () => {
  it("serves the model's output, channel first, as float32 with an empty chain", async () => {
    const buf = await produceChunk(context(NONE, 2), [1, 0, 1]);
    const out = new Float32Array(buf);
    expect(out.length).toBe(2 * 8);
    // output voxel (2, 0, 2) = raw voxel (2, 0, 2), channel 1 doubled
    expect(out[0]).toBe(rawValue(2, 0, 2));
    expect(out[8]).toBe(2 * rawValue(2, 0, 2));
    expect(out[7]).toBe(rawValue(3, 1, 3));
  });

  it("normalizes the input before the model and postprocesses after", async () => {
    const chain: Chain = {
      input_norm: [{ name: "ZScoreNormalizer", mean: "1", std: "1" }],  // raw - 1
      postprocess: [{ name: "ThresholdPostprocessor", threshold: "50" }],
    };
    const buf = await produceChunk(context(chain), [0, 0, 0]);
    const out = new Uint8Array(buf);
    expect(out.length).toBe(8);
    // output voxels are raw (0..1, 0..1, 0..1): raw - 1 = 100 z + 10 y + x, > 50 only for z = 1
    expect(Array.from(out)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  });

  it("selects channels and writes the chain's dtype", async () => {
    const chain: Chain = {
      input_norm: [{ name: "LambdaNormalizer", expression: "x/1000" }],
      postprocess: [{ name: "DefaultPostprocessor", clip_min: "0", clip_max: "1", bias: "0", multiplier: "100" }, { name: "ChannelSelection", channels: "2" }],
    };
    const buf = await produceChunk(context(chain, 3), [0, 0, 0]);
    const out = new Uint8Array(buf);
    expect(out.length).toBe(8);
    // channel 2 is raw * 3 / 1000, * 100 into uint8: raw (1, 1, 1) = 112 → 33.6 → 33
    expect(out[7]).toBe(33);
    expect(out[0]).toBe(0);  // raw 1 → 0.3
  });

  it("squeezes the channel for a view without a channel axis", async () => {
    const buf = await produceChunk(context(NONE, 1, false), [0, 0, 0]);
    expect(new Float32Array(buf).length).toBe(8);
    await expect(produceChunk(context(NONE, 2, false), [0, 0, 0])).rejects.toThrow(/cannot serve 2 channels/);
  });

  it("refuses an output of another size than the manifest says", async () => {
    const ctx = { ...context(NONE), run: async (): Promise<ModelOutput> => ({ data: new Float32Array(27), dims: [1, 1, 3, 3, 3], type: "float32" }) };
    await expect(produceChunk(ctx, [0, 0, 0])).rejects.toThrow(/wrote 3×3×3 voxels, not the 2×2×2/);
  });
});
