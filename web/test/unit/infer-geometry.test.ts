// Block geometry: the context and the ratio the engine grows a chunk by,
// clipping, the level choice and the output grid, on the Hugging Face
// models' sizes (setup_54: 148³ voxels in, 60³ out, 16 nm).
import { describe, expect, it } from "vitest";

import type { ModelEntry } from "../../src/contracts";
import {
  type BlockGeometry, asTriple, chooseRawLevel, clipBox, context, datasetInfoFrom, effectiveOutputVoxelSize, geometryOf, halo,
  inputShape, outputLevel, outputShape, ratio, sameVoxelSize, volumeShape,
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

describe("shapes and context", () => {
  it("gives the blocks in voxels", () => {
    expect(inputShape(G)).toEqual([148, 148, 148]);
    expect(outputShape(G)).toEqual([60, 60, 60]);
    expect(inputShape(COARSE)).toEqual([40, 40, 40]);
    expect(outputShape(COARSE)).toEqual([8, 8, 8]);
  });

  it("is (read - write) / 2 in nm, and whole input voxels", () => {
    expect(context(G)).toEqual([704, 704, 704]);
    expect(halo(G)).toEqual([44, 44, 44]);
    expect(context(COARSE)).toEqual([96, 96, 96]);
    expect(halo(COARSE)).toEqual([12, 12, 12]);
  });

  it("refuses a context that is not whole voxels", () => {
    expect(() => halo({ ...G, read_shape: [2372, 2368, 2368] })).toThrow(/context .* not whole voxels/);
  });

  it("builds the geometry from a model entry, scalar or per-axis sizes", () => {
    const entry = {
      name: "setup_54",
      manifest: { input_voxel_size_nm: [16, 16, 16], output_voxel_size_nm: [16, 16, 16] },
      variant: { input_size: [148, 148, 148], output_size: [60, 60, 60] },
    } as unknown as ModelEntry;
    expect(geometryOf(entry)).toEqual(G);
    expect(asTriple(148, "n")).toEqual([148, 148, 148]);
    expect(() => asTriple([1, 2], "input_size")).toThrow(/input_size must be 3 numbers/);
  });
});

describe("the ratio the engine grows a chunk by", () => {
  it("is input voxels per output voxel", () => {
    expect(ratio(G)).toEqual([1, 1, 1]);
    expect(ratio(COARSE)).toEqual([2, 2, 2]);
  });

  it("makes an output block grown by the halo the model's input block", () => {
    // what stages.ts's inputBox does with the halo and the ratio: the output
    // block, scaled onto the raw grid and padded by the halo on each side
    for (const g of [G, COARSE]) {
      const grown = outputShape(g).map((n, a) => n * ratio(g)[a] + 2 * halo(g)[a]);
      expect(grown).toEqual(inputShape(g));
    }
  });
});

describe("clipBox", () => {
  const shape = [148, 148, 148], level = [500, 300, 200];

  it("keeps a box inside the level as it is", () => {
    expect(clipBox([16, 16, 16], shape, level)).toEqual({ lo: [16, 16, 16], hi: [164, 164, 164], offset: [0, 0, 0] });
  });

  it("clips at the start and records where the data lands in the box", () => {
    expect(clipBox([-44, -44, -44], shape, level)).toEqual({ lo: [0, 0, 0], hi: [104, 104, 104], offset: [44, 44, 44] });
  });

  it("clips at the end", () => {
    expect(clipBox([400, 200, 100], shape, level)).toEqual({ lo: [400, 200, 100], hi: [500, 300, 200], offset: [0, 0, 0] });
  });

  it("is null for a box wholly outside", () => {
    expect(clipBox([600, 0, 0], shape, level)).toBeNull();
    expect(clipBox([-200, 0, 0], shape, level)).toBeNull();
  });
});

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
    });
    expect(info.url).toBe("https://x/raw.zarr");
    expect(info.dtype).toBe("uint8");
    expect(info.axes).toEqual([{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }]);
    expect(info.levels[0].voxel.map((v) => +v.toFixed(9))).toEqual([8, 8, 8]);
    expect(info.levels[0].origin.map((v) => +v.toFixed(9))).toEqual([4, 4, 4]);
    expect(info.levels[0].shape).toEqual([10, 20, 30]);
  });

  it("leaves an axis in a unit that is not a length as it is", () => {
    const info = datasetInfoFrom("u", {
      dtype: "float32", channels: 1,
      axes: [{ name: "t", unit: "second" }, { name: "y", unit: "m" }, { name: "x", unit: "um" }],
      levels: [{ shape: [1, 1, 1], voxel: [60, 1e-9, 1], origin: [0, 0, 0] }],
    });
    expect(info.axes[0]).toEqual({ name: "t", unit: "second" });
    expect(info.levels[0].voxel[0]).toBe(60);
    expect(info.levels[0].voxel[1]).toBeCloseTo(1, 9);
    expect(info.levels[0].voxel[2]).toBe(1000);
  });
});
