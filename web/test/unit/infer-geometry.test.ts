// Block geometry: the raw level a model reads and the output level it is
// served at, on the Hugging Face models' sizes (setup_54: 148³ voxels in,
// 60³ out, 16 nm): the level choice, the output grid's shape and origin (a
// voxel centre), the voxel size declared for a level read at another, and
// the engine's levels in nm whatever unit each axis is in.
import { describe, expect, it } from "vitest";

import {
  type BlockGeometry, chooseRawLevel, datasetInfoFrom, effectiveOutputVoxelSize, outputLevel, sameVoxelSize, volumeShape,
} from "../../src/infer/geometry";

/** setup_54's 148³ variant: 148 voxels in, 60 out, at 16 nm. */
const G: BlockGeometry = {
  read_shape: [2368, 2368, 2368],
  write_shape: [960, 960, 960],
  input_voxel_size: [16, 16, 16],
  output_voxel_size: [16, 16, 16],
};

/** A model whose output is coarser than its input (anisotropic too). */
const COARSE: BlockGeometry = {
  read_shape: [40 * 8, 40 * 8, 40 * 8],
  write_shape: [8 * 16, 8 * 16, 8 * 16],
  input_voxel_size: [8, 8, 8],
  output_voxel_size: [16, 16, 16],
};

describe("levels", () => {
  const levels = [
    { shape: [1000, 2000, 3000], voxel: [8, 8, 8], origin: [4, 4, 4] },
    { shape: [500, 1000, 1500], voxel: [16, 16, 16], origin: [8, 8, 8] },
    { shape: [250, 500, 750], voxel: [32, 32, 32], origin: [16, 16, 16] },
  ];

  it("chooses the level at the model's voxel size", () => {
    expect(chooseRawLevel(levels, [16, 16, 16])).toBe(1);
    expect(chooseRawLevel(levels, [8, 8, 8])).toBe(0);
    expect(chooseRawLevel(levels, [32, 32, 32])).toBe(2);
    expect(chooseRawLevel(levels, [16.0000001, 16, 16])).toBe(1);
  });

  it("else the finest level that is not coarser on any axis", () => {
    expect(chooseRawLevel(levels, [20, 20, 20])).toBe(1);
    expect(chooseRawLevel(levels, [128, 128, 128])).toBe(2);
    expect(chooseRawLevel(levels, [32, 32, 20])).toBe(1);
  });

  it("and the first level when even that one is too coarse", () => {
    expect(chooseRawLevel(levels, [4, 4, 4])).toBe(0);
    expect(() => chooseRawLevel([], [8, 8, 8])).toThrow(/no levels/);
  });

  it("compares voxel sizes within a relative tolerance", () => {
    expect(sameVoxelSize([8, 8, 8], [8.000001, 8, 8])).toBe(true);
    expect(sameVoxelSize([8, 8, 8], [8.1, 8, 8])).toBe(false);
  });

  it("serves one output level at the output voxel size, from the raw level's corner", () => {
    const out = outputLevel(levels[1], G);
    expect(out.voxel).toEqual([16, 16, 16]);
    expect(out.shape).toEqual([500, 1000, 1500]);
    expect(out.origin).toEqual([8, 8, 8]);  // equal voxel sizes: the raw origin
    expect(volumeShape([500, 1000, 1500], G)).toEqual([500, 1000, 1500]);
  });

  it("floors the shape and centres the origin when the output is coarser", () => {
    const out = outputLevel({ shape: [101, 100, 100], voxel: [8, 8, 8], origin: [4, 4, 4] }, COARSE);
    expect(out.shape).toEqual([50, 50, 50]);
    expect(out.voxel).toEqual([16, 16, 16]);
    expect(out.origin).toEqual([8, 8, 8]);  // raw corner 0 + half an output voxel
    expect(effectiveOutputVoxelSize(levels[0], COARSE)).toBeNull();
  });

  it("declares a level read at another voxel size at the size its output really has, which model_info says too", () => {
    // a 16 nm model on a pyramid with 12 nm: the 12 nm level is read as if
    // 16, voxel for voxel, so the output lies at 12 nm and is declared there
    const raw = { shape: [100, 100, 100], voxel: [12, 12, 12], origin: [6, 6, 6] };
    const out = outputLevel(raw, G);
    expect(out.shape).toEqual([100, 100, 100]);
    expect(out.voxel).toEqual([12, 12, 12]);
    expect(out.origin).toEqual([6, 6, 6]);  // the corner (0) plus half a real voxel
    expect(effectiveOutputVoxelSize(raw, G)).toEqual([12, 12, 12]);
    expect(effectiveOutputVoxelSize(raw, COARSE)).toEqual([24, 24, 24]);
    expect(outputLevel(raw, COARSE)).toEqual({ shape: [50, 50, 50], voxel: [24, 24, 24], origin: [12, 12, 12] });
  });

  it("works in the source's own units, since the ratio has none", () => {
    // the same level in micrometers, as a source in µm declares it
    const raw = { shape: [100, 100, 100], voxel: [0.012, 0.012, 0.012], origin: [0.006, 0.006, 0.006] };
    const out = outputLevel(raw, COARSE);
    expect(out.voxel.map((v) => +v.toFixed(9))).toEqual([0.024, 0.024, 0.024]);
    expect(out.origin.map((v) => +v.toFixed(9))).toEqual([0.012, 0.012, 0.012]);
  });
});

describe("datasetInfoFrom", () => {
  it("converts the engine's levels to nm by each axis's unit", () => {
    const info = datasetInfoFrom("https://x/raw.zarr", {
      dtype: "uint8", channels: 1,
      axes: [{ name: "z", unit: "micrometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "" }],
      levels: [{ shape: [10, 20, 30], voxel: [0.008, 8, 8], origin: [0.004, 4, 4] }],
    }, { count: 1, labels: null });
    expect(info.url).toBe("https://x/raw.zarr");
    expect(info.dtype).toBe("uint8");
    expect(info.axes).toEqual([{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }]);
    expect(info.levels[0].voxel.map((v) => +v.toFixed(9))).toEqual([8, 8, 8]);
    expect(info.levels[0].origin.map((v) => +v.toFixed(9))).toEqual([4, 4, 4]);
    expect(info.levels[0].shape).toEqual([10, 20, 30]);
  });
});
