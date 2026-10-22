// An annotation volume (src/finetune/volume.ts) against
// cellmap_flow/finetune/session/volume.py: where planVolume lays a volume
// for a model over a dataset (numbers computed by hand from the geometry
// formulas, with the arithmetic beside them), the labels' dtype (common.py's
// annotation_dtype_for), and the round trip through a paint store, the
// painted chunks read back as the brush wrote them.
import { describe, expect, it } from "vitest";

import type { Chain, DatasetInfo, ModelVariant, PaintVolume } from "../../src/contracts";
import { CHUNK_DIR, createVolume, paintedChunkIndices, planVolume, readPaintedChunk, readVolume } from "../../src/finetune/volume";
import { entryFor } from "../../src/models";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { DATASET, entry, manifest, NAME_54 } from "./backend-fakes";

const chain: Chain = {
  input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }],
  postprocess: [{ name: "SigmoidPostprocessor" }],
};
const NOW = new Date(2026, 9, 15, 13, 4, 5);  // 2026-10-15 13:04:05 local

describe("planVolume (plan_volume)", () => {
  it("lays setup_54's volume on the served prediction's grid, padded to whole chunks, with uint16 labels for its affinities", () => {
    // setup_54's fp32 variant: 148³ voxels in, 60³ out, at 16 nm; the dataset
    // one level, 1000³ voxels at 8 nm, origin 0. chooseRawLevel finds no
    // level at 16 nm and no level coarser than 16 nm, so level 0 (8 nm) is
    // read as if it were 16 nm. The ratio (output / input voxel size) is 1:
    //   output voxel  = 8 × 1 = 8 nm
    //   level shape   = floor(1000 × 16 / 16) = 1000 voxels
    //   origin        = 0 - 8/2 + 8/2 = 0 (equal voxel sizes: the raw origin)
    //   padded shape  = ceil(1000 / 60) × 60 = 17 × 60 = 1020 voxels
    const volume = planVolume(DATASET, entry("setup_54"), chain, { id: "vol-abc12345-20261015-130405", now: NOW });
    expect(volume).toMatchObject({
      chunk_size: [60, 60, 60],
      input_size: [148, 148, 148],
      output_voxel_size: [8, 8, 8],
      input_voxel_size: [8, 8, 8],
      claimed_input_voxel_size: [16, 16, 16],
      claimed_output_voxel_size: [16, 16, 16],
      dataset_offset_nm: [0, 0, 0],
      dataset_shape_voxels: [1020, 1020, 1020],
      dtype: "uint16",
      model_name: NAME_54,
      dataset_path: DATASET.url,
      input_norm: chain.input_norm,
      created_at: NOW.toISOString(),
    });
    // Copies: the chain the dashboard goes on editing is not the volume's.
    expect(volume.input_norm).not.toBe(chain.input_norm);
  });

  it("starts an anisotropic level's volume at its corner plus half an output voxel, a voxel centre as the OME translation is", () => {
    // A model reading 8 nm and writing 16 nm (40³ voxels in, 8³ out), over a
    // dataset whose one level is 16 × 8 × 8 nm, 101 × 1000 × 1000 voxels,
    // origin (8, 4, 4) (voxel centres: the level's corner is 0). No level is
    // at 8 nm and level 0 is already coarser along z, so level 0 is read as
    // if it were 8 nm isotropic. The ratio is 2 on every axis:
    //   output voxel  = (16, 8, 8) × 2 = (32, 16, 16) nm
    //   level shape   = floor((101, 1000, 1000) × 8 / 16) = (50, 500, 500)
    //   origin        = corner + half an output voxel = (8 - 8 + 16, 4 - 4 + 8, 4 - 4 + 8) = (16, 8, 8)
    //   padded shape  = ceil((50, 500, 500) / 8) × 8 = (7 × 8, 63 × 8, 63 × 8) = (56, 504, 504)
    const variant: ModelVariant = { variant: "fp32", input_size: [40, 40, 40], output_size: [8, 8, 8], file: "coarse_fp32.onnx", file_mb: 1 };
    const coarse = entryFor("coarse", { ...manifest("coarse", ["nuc"], 8), output_voxel_size_nm: [16, 16, 16], variants: [variant] }, variant);
    const dataset: DatasetInfo = { ...DATASET, levels: [{ shape: [101, 1000, 1000], voxel: [16, 8, 8], origin: [8, 4, 4] }] };
    const volume = planVolume(dataset, coarse, { input_norm: [], postprocess: [] }, { id: "vol-anis", now: NOW });
    expect(volume).toMatchObject({
      input_voxel_size: [16, 8, 8],
      output_voxel_size: [32, 16, 16],
      claimed_input_voxel_size: [8, 8, 8],
      claimed_output_voxel_size: [16, 16, 16],
      dataset_offset_nm: [16, 8, 8],
      chunk_size: [8, 8, 8],
      dataset_shape_voxels: [56, 504, 504],
      dtype: "uint8",  // a plain binary model
    });
  });

  it("refuses a dataset with more than one channel, as serve() does", () => {
    const rgb: DatasetInfo = { ...DATASET, channels: 3, axes: [{ name: "c", unit: "" }, ...DATASET.axes] };
    expect(() => planVolume(rgb, entry("setup_54"), chain)).toThrow(/channels/);
  });
});

describe("createVolume and readVolume on a paint store", () => {
  it("round-trips the volume through its files, the labels placed at voxel 0's centre", async () => {
    const store = new MemoryPaintStore();
    const volume = planVolume(DATASET, entry("setup_54"), chain, { id: "vol-rt", now: NOW });
    await createVolume(store, volume);
    expect(await store.list("vol-rt")).toEqual([".zattrs", ".zgroup", "annotation/.zattrs", "annotation/.zgroup", "annotation/s0/.zarray"]);
    expect(await readVolume(store, "vol-rt")).toEqual(volume);
    expect(await store.volumes()).toEqual(["vol-rt"]);
    // Where Neuroglancer draws the paint: the output voxel size, translated to the volume's offset.
    const labels = JSON.parse(new TextDecoder().decode(await store.get("vol-rt", "annotation/.zattrs")));
    expect(labels.multiscales[0].datasets[0].coordinateTransformations).toEqual([
      { type: "scale", scale: volume.output_voxel_size },
      { type: "translation", translation: volume.dataset_offset_nm },
    ]);
    // Not an annotation volume's .zattrs: none.
    await store.put("other", ".zattrs", new TextEncoder().encode(JSON.stringify({ multiscales: [] })));
    expect(await readVolume(store, "other")).toBeNull();
  });

  it("lists the chunks painted and decodes a uint16 chunk little-endian, in C order", async () => {
    const store = new MemoryPaintStore();
    const volume: PaintVolume = { ...planVolume(DATASET, entry("setup_54"), chain, { id: "vol-chunks", now: NOW }), chunk_size: [1, 2, 2], dtype: "uint16" };
    await createVolume(store, volume);
    // 1, 2, 256, 65535 as little-endian uint16
    await store.put(volume.id, `${CHUNK_DIR}3.0.0`, new Uint8Array([1, 0, 2, 0, 0, 1, 255, 255]));
    await store.put(volume.id, `${CHUNK_DIR}0.1.2`, new Uint8Array(8));
    expect(await paintedChunkIndices(store, volume.id)).toEqual([[0, 1, 2], [3, 0, 0]]);  // the .zarray beside them ignored
    const chunk = await readPaintedChunk(store, volume, [3, 0, 0]);
    expect(chunk).toMatchObject({ index: [3, 0, 0], shape: [1, 2, 2] });
    expect(chunk!.labels).toBeInstanceOf(Uint16Array);
    expect(Array.from(chunk!.labels)).toEqual([1, 2, 256, 65535]);
    expect(await readPaintedChunk(store, volume, [0, 0, 1])).toBeNull();
  });
});
