// The pieces of Seed from Prediction that the parity suite does not reach
// (test/parity/label-it.test.ts checks the numerics against the Python):
// which methods and channels a model offers, the refusals, the float32
// rounding at the threshold, and the served level a prediction is read on.
import { describe, expect, it } from "vitest";

import { planVolume } from "../../src/finetune/volume";
import { readPrediction, servedIndices, servedLevel } from "../../src/label-it/prediction";
import { classifyOutputRange, expit32, foregroundMask, methodsFor, outputTypeOf, probability, seedPlan, segmentPrediction } from "../../src/label-it/segment";
import { DATASET, entry } from "./backend-fakes";

const AFFINITY = entry("setup_54");
const NUC = entry("setup_56", "nuc", ["nuc"], 128);

describe("methods and plans", () => {
  it("offers the mutex watershed first for an affinity model, components for every model", () => {
    expect(outputTypeOf(AFFINITY)).toBe("affinities");
    expect(methodsFor(outputTypeOf(AFFINITY))).toEqual(["mutex_watershed", "components"]);
    expect(outputTypeOf(NUC)).toBe("binary");
    expect(methodsFor(outputTypeOf(NUC))).toEqual(["components"]);
  });

  it("reads an affinity model's offset channels, at the offsets their names say, and every channel of another", () => {
    expect(seedPlan(AFFINITY, null)).toEqual({ outputType: "affinities", channels: [0, 1, 2], offsets: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] });
    expect(seedPlan(NUC, null)).toEqual({ outputType: "binary", channels: null, offsets: null });
    // A channel without an offset (an LSD, say) is left out of the affinities.
    const withLsd = entry("setup_x", "mito_affinities", ["mito_aff_z", "mito_lsd_0", "mito_aff_x"]);
    expect(seedPlan(withLsd, null)).toEqual({ outputType: "affinities", channels: [0, 2], offsets: [[1, 0, 0], [0, 0, 1]] });
  });

  it("thresholds the one channel a request selects, and refuses a negative one", () => {
    expect(seedPlan(AFFINITY, 1).channels).toEqual([1]);
    expect(() => seedPlan(AFFINITY, -1)).toThrow("select_channel must be 0 or more, got -1");
  });

  it("refuses a method the browser does not run, and a mutex watershed without offsets", () => {
    const block = { data: new Float32Array(8), shape: [1, 2, 2, 2] };
    const options = { threshold: 0.5, minSize: 0, connectivity: 1, perSlice: false };
    expect(() => segmentPrediction("distance_watershed", block, "unbounded", null, null, options)).toThrow(/method must be one of components, mutex_watershed, distance_watershed, instances/);
    expect(() => segmentPrediction("mutex_watershed", block, "unbounded", null, null, options)).toThrow(/offsets are unknown/);
  });
});

describe("the probability", () => {
  it("is scipy's float32 expit of a logit, 0.5 at 0, saturating at 0 and 1", () => {
    expect(expit32(0)).toBe(0.5);
    expect(expit32(Math.fround(1e-9))).toBe(0.5);  // exp(-x) rounds to 1 in float32
    expect(expit32(100)).toBe(1);
    expect(expit32(-100)).toBe(0);
    expect(expit32(2)).toBe(Math.fround(1 / Math.fround(1 + Math.fround(Math.exp(-2)))));
  });

  it("classifies an output by its range as serving/probe.py does", () => {
    expect(classifyOutputRange(0, 1)).toBe("unit");
    expect(classifyOutputRange(-0.00005, 1.00005)).toBe("unit");
    expect(classifyOutputRange(-1, 1)).toBe("signed_unit");
    expect(classifyOutputRange(-1.01, 0)).toBe("unbounded");
  });

  it("names a channel the model does not have", () => {
    expect(() => probability({ data: new Float32Array(2), shape: [2, 1, 1, 1] }, "unit", [5])).toThrow("The model has no channel 5");
  });

  it("compares with the threshold as a float32, strictly, as numpy does", () => {
    const t = Math.fround(0.3);
    const block = { data: Float32Array.of(t, Math.fround(0.30000004), 0.29), shape: [1, 1, 1, 3] };
    // 0.3 as a float32 is not over 0.3 as a float32; the next float32 up is.
    expect(Array.from(foregroundMask(block, "unit", null, 0.3))).toEqual([0, 1, 0]);
  });

  it("averages the channels in float32, one after another", () => {
    const block = { data: Float32Array.of(0.1, 0.2, 0.7), shape: [3, 1, 1, 1] };
    const sum = Math.fround(Math.fround(Math.fround(0.1) + Math.fround(0.2)) + Math.fround(0.7));
    expect(Array.from(foregroundMask(block, "unit", null, Math.fround(sum / 3) - 1e-6))).toEqual([1]);
  });
});

describe("the served level", () => {
  it("is the grid planVolume lays the model's volume on", () => {
    const level = servedLevel(DATASET, AFFINITY);
    const volume = planVolume(DATASET, AFFINITY, { input_norm: [], postprocess: [] });
    expect(level.voxel).toEqual(volume.output_voxel_size);
    expect(level.origin).toEqual(volume.dataset_offset_nm);
    expect(volume.dataset_shape_voxels.every((n, a) => n >= level.shape[a])).toBe(true);
  });

  it("maps a box on the same grid onto itself, and refuses one the level does not reach", () => {
    const level = { shape: [10, 10, 10], voxel: [8, 8, 8], origin: [0, 0, 0] };
    const volume = { output_voxel_size: [8, 8, 8], dataset_offset_nm: [0, 0, 0] };
    expect(servedIndices(volume, [2, 3, 4], [5, 6, 7], level)).toEqual({ lo: [2, 3, 4], hi: [5, 6, 7], indices: [[2, 3, 4], [3, 4, 5], [4, 5, 6]] });
    expect(servedIndices(volume, [8, 0, 0], [12, 2, 2], level)?.hi).toEqual([10, 2, 2]);
    expect(servedIndices(volume, [10, 0, 0], [12, 2, 2], level)).toBeNull();
  });

  it("reads the served box once and returns null outside", async () => {
    const asked: number[][][] = [];
    const infer = {
      readOutput: async (_view: string, lo: number[], hi: number[]) => {
        asked.push([lo, hi]);
        const shape = [1, ...hi.map((h, a) => h - lo[a])];
        return { data: new Float32Array(shape.reduce((a, b) => a * b, 1)), shape };
      },
    };
    const level = { shape: [10, 10, 10], voxel: [16, 16, 16], origin: [4, 4, 4] };
    const volume = { output_voxel_size: [8, 8, 8], dataset_offset_nm: [0, 0, 0] };
    const read = await readPrediction(infer, "v", level, volume, [0, 0, 0], [4, 6, 8]);
    expect(read?.prediction.shape).toEqual([1, 4, 6, 8]);
    expect(asked).toEqual([[[0, 0, 0], [2, 3, 4]]]);
    expect(await readPrediction(infer, "v", level, volume, [40, 0, 0], [44, 2, 2])).toBeNull();
    expect(asked.length).toBe(1);
  });
});
