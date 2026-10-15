// An annotation volume (src/finetune/volume.ts) against
// cellmap_flow/finetune/session/volume.py: where planVolume lays a volume
// for a model over a dataset (numbers computed by hand from the geometry
// formulas, with the arithmetic beside them), the labels' dtype (common.py's
// annotation_dtype_for), the zarr v2 files create_volume_zarr writes, and
// the round trip through a paint store.
import { describe, expect, it } from "vitest";

import type { Chain, DatasetInfo, PaintVolume } from "../../src/contracts";
import {
  annotationDtypeFor, CHUNK_DIR, chunkKey, createVolume, hasAffinityChannels, labelsOf, newVolumeId, paintedChunkIndices, parseChunkKey, planVolume,
  readPaintedChunk, readVolume, volumeFiles,
} from "../../src/finetune/volume";
import { entryFor } from "../../src/models";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { DATASET, entry, manifest, NAME_54 } from "./backend-fakes";

const chain: Chain = {
  input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }],
  postprocess: [{ name: "SigmoidPostprocessor" }],
};
const threshold = { name: "ThresholdPostprocessor", threshold: "0.5" };
const NOW = new Date(2026, 9, 15, 13, 4, 5);  // 2026-10-15 13:04:05 local

describe("planVolume (plan_volume) on the fakes' dataset and setup_54", () => {
  // setup_54's fp32 variant: 148³ voxels in, 60³ out, at 16 nm; the dataset
  // one level, 1000³ voxels at 8 nm, origin 0. chooseRawLevel finds no
  // level at 16 nm and no level coarser than 16 nm, so level 0 (8 nm) is
  // read as if it were 16 nm. The ratio (output / input voxel size) is 1:
  //   output voxel  = 8 × 1 = 8 nm
  //   level shape   = floor(1000 × 16 / 16) = 1000 voxels
  //   origin        = 0 - 8/2 + 8/2 = 0 (equal voxel sizes: the raw origin)
  //   padded shape  = ceil(1000 / 60) × 60 = 17 × 60 = 1020 voxels
  const volume = planVolume(DATASET, entry("setup_54"), chain, { id: "vol-abc12345-20261015-130405", now: NOW });

  it("lays the volume on the served prediction's grid, padded to whole chunks", () => {
    expect(volume.chunk_size).toEqual([60, 60, 60]);
    expect(volume.input_size).toEqual([148, 148, 148]);
    expect(volume.output_voxel_size).toEqual([8, 8, 8]);
    expect(volume.input_voxel_size).toEqual([8, 8, 8]);
    expect(volume.dataset_offset_nm).toEqual([0, 0, 0]);
    expect(volume.dataset_shape_voxels).toEqual([1020, 1020, 1020]);
  });

  it("keeps the model's own voxel sizes as the claimed ones", () => {
    expect(volume.claimed_input_voxel_size).toEqual([16, 16, 16]);
    expect(volume.claimed_output_voxel_size).toEqual([16, 16, 16]);
  });

  it("records the model, the dataset, the chains and the time", () => {
    expect(volume).toMatchObject({
      id: "vol-abc12345-20261015-130405",
      type: "annotation_volume",
      model_name: NAME_54,
      model_id: "cellmap/onnx-web-models/setup_54",
      dataset_path: DATASET.url,
      input_norm: chain.input_norm,
      postprocess: chain.postprocess,
      created_at: NOW.toISOString(),
    });
    expect(volume.resample).toBeUndefined();
    // Copies: the chain the dashboard goes on editing is not the volume's.
    expect(volume.input_norm).not.toBe(chain.input_norm);
    expect(volume.input_norm[0]).not.toBe(chain.input_norm[0]);
  });

  it("gives an affinity model uint16 labels, for the objects' ids", () => {
    expect(volume.dtype).toBe("uint16");
  });

  it("names a new volume as new_volume_id does when no id is given", () => {
    const planned = planVolume(DATASET, entry("setup_54"), chain, { now: NOW });
    expect(planned.id).toMatch(/^vol-[0-9a-f]{8}-20261015-130405$/);
    expect(planVolume(DATASET, entry("setup_54"), chain).id).toMatch(/^vol-[0-9a-f]{8}-\d{8}-\d{6}$/);
  });
});

describe("planVolume on an anisotropic level read by a model whose output is coarser than its input", () => {
  // A model reading 8 nm and writing 16 nm (40³ voxels in, 8³ out), over a
  // dataset whose one level is 16 × 8 × 8 nm, 101 × 1000 × 1000 voxels,
  // origin (8, 4, 4) (voxel centres: the level's corner is 0). No level is
  // at 8 nm and level 0 is already coarser along z, so level 0 is read as
  // if it were 8 nm isotropic. The ratio is 2 on every axis:
  //   output voxel  = (16, 8, 8) × 2 = (32, 16, 16) nm
  //   level shape   = floor((101, 1000, 1000) × 8 / 16) = (50, 500, 500)
  //   origin        = corner + half an output voxel = (8 - 8 + 16, 4 - 4 + 8, 4 - 4 + 8) = (16, 8, 8)
  //   padded shape  = ceil((50, 500, 500) / 8) × 8 = (7 × 8, 63 × 8, 63 × 8) = (56, 504, 504)
  const coarse = entryFor("coarse", {
    ...manifest("coarse", ["nuc"], 8),
    output_voxel_size_nm: [16, 16, 16],
    variants: [{ variant: "fp32", input_size: [40, 40, 40], output_size: [8, 8, 8], file: "coarse_fp32.onnx", file_mb: 1 }],
  }, {
    variant: "fp32", input_size: [40, 40, 40], output_size: [8, 8, 8], file: "coarse_fp32.onnx", file_mb: 1,
  });
  const dataset: DatasetInfo = { ...DATASET, levels: [{ shape: [101, 1000, 1000], voxel: [16, 8, 8], origin: [8, 4, 4] }] };
  const volume = planVolume(dataset, coarse, { input_norm: [], postprocess: [] }, { id: "vol-anis", now: NOW });

  it("takes the raw level's voxel size as the input's and the ratio of the two grids as the output's", () => {
    expect(volume.input_voxel_size).toEqual([16, 8, 8]);
    expect(volume.output_voxel_size).toEqual([32, 16, 16]);
    expect(volume.claimed_input_voxel_size).toEqual([8, 8, 8]);
    expect(volume.claimed_output_voxel_size).toEqual([16, 16, 16]);
  });

  it("starts at the raw level's corner plus half an output voxel, a voxel centre as the OME translation is", () => {
    expect(volume.dataset_offset_nm).toEqual([16, 8, 8]);
  });

  it("floors the extent into output voxels and pads it to whole chunks", () => {
    expect(volume.chunk_size).toEqual([8, 8, 8]);
    expect(volume.input_size).toEqual([40, 40, 40]);
    expect(volume.dataset_shape_voxels).toEqual([56, 504, 504]);
  });

  it("is uint8 for a plain binary model", () => {
    expect(volume.dtype).toBe("uint8");
  });
});

describe("planVolume refuses what serve() refuses", () => {
  it("a dataset with more than one channel", () => {
    const rgb: DatasetInfo = { ...DATASET, channels: 3, axes: [{ name: "c", unit: "" }, ...DATASET.axes] };
    expect(() => planVolume(rgb, entry("setup_54"), chain)).toThrow(/3 channels/);
  });
});

describe("annotationDtypeFor (annotation_dtype_for)", () => {
  const nuc = entry("setup_56", "nuc", ["nuc"], 128);
  const affinities = entry("setup_54");
  const none: Chain = { input_norm: [], postprocess: [] };

  it("is uint16 for a model whose channels are affinities, by the server's _aff test", () => {
    expect(hasAffinityChannels(affinities)).toBe(true);
    expect(hasAffinityChannels(nuc)).toBe(false);
    expect(hasAffinityChannels(entry("x", "t", ["x_aff"]))).toBe(true);
    expect(hasAffinityChannels(entry("x", "t", ["affinity"]))).toBe(false);
    expect(annotationDtypeFor(affinities, none)).toBe("uint16");
  });

  it("is uint16 when the chain serves labels, which are painted over object by object", () => {
    expect(annotationDtypeFor(nuc, { input_norm: [], postprocess: [{ name: "SigmoidPostprocessor" }, threshold] })).toBe("uint16");
    // The last step that says wins: a DefaultPostprocessor after the threshold serves an image again.
    expect(annotationDtypeFor(nuc, { input_norm: [], postprocess: [threshold, { name: "DefaultPostprocessor" }] })).toBe("uint8");
  });

  it("is uint8 otherwise", () => {
    expect(annotationDtypeFor(nuc, none)).toBe("uint8");
    expect(annotationDtypeFor(nuc, chain)).toBe("uint8");
  });
});

describe("volumeFiles (create_volume_zarr)", () => {
  const volume = planVolume(DATASET, entry("setup_54"), chain, { id: "vol-abc12345-20261015-130405", now: NOW });
  const files = volumeFiles(volume);
  const parsed = (path: string) => JSON.parse(files[path]) as Record<string, unknown>;

  it("is the five metadata files of the server's layout and nothing else", () => {
    expect(Object.keys(files).sort()).toEqual([".zattrs", ".zgroup", "annotation/.zattrs", "annotation/.zgroup", "annotation/s0/.zarray"]);
    expect(parsed(".zgroup")).toEqual({ zarr_format: 2 });
    expect(parsed("annotation/.zgroup")).toEqual({ zarr_format: 2 });
  });

  it("puts every field of the volume in the root .zattrs, the server's first and in its order", () => {
    const attrs = parsed(".zattrs");
    expect(attrs).toEqual(volume);
    expect(Object.keys(attrs)).toEqual([
      "type", "model_name", "dataset_path", "chunk_size", "output_voxel_size", "input_size", "input_voxel_size", "dataset_offset_nm",
      "dataset_shape_voxels", "created_at", "claimed_output_voxel_size", "claimed_input_voxel_size", "input_norm", "postprocess",
      "id", "model_id", "dtype",
    ]);
  });

  it("writes resample only when it is on, as the server does", () => {
    expect("resample" in parsed(".zattrs")).toBe(false);
    const attrs = JSON.parse(volumeFiles({ ...volume, resample: true })[".zattrs"]) as Record<string, unknown>;
    expect(attrs.resample).toBe(true);
    expect(Object.keys(attrs).indexOf("resample")).toBe(Object.keys(attrs).indexOf("created_at") + 1);
  });

  it("describes the labels as OME multiscales 0.4 at the output voxel size, translated to voxel 0's centre", () => {
    expect(parsed("annotation/.zattrs")).toEqual({
      multiscales: [{
        version: "0.4",
        name: "annotation",
        axes: [
          { name: "z", type: "space", unit: "nanometer" },
          { name: "y", type: "space", unit: "nanometer" },
          { name: "x", type: "space", unit: "nanometer" },
        ],
        datasets: [{
          path: "s0",
          coordinateTransformations: [
            { type: "scale", scale: [8, 8, 8] },
            { type: "translation", translation: [0, 0, 0] },
          ],
        }],
      }],
    });
  });

  it("declares s0 padded to whole chunks, one chunk per output block, uncompressed, dots between the indices", () => {
    expect(parsed(`${CHUNK_DIR}.zarray`)).toEqual({
      zarr_format: 2,
      shape: [1020, 1020, 1020],
      chunks: [60, 60, 60],
      dtype: "<u2",
      compressor: null,
      fill_value: 0,
      order: "C",
      filters: null,
      dimension_separator: ".",
    });
    expect((JSON.parse(volumeFiles({ ...volume, dtype: "uint8" })["annotation/s0/.zarray"]) as { dtype: string }).dtype).toBe("|u1");
  });
});

describe("createVolume and readVolume on a paint store", () => {
  it("round-trips the volume through its files", async () => {
    const store = new MemoryPaintStore();
    const volume = planVolume(DATASET, entry("setup_54"), chain, { id: "vol-rt", now: NOW });
    await createVolume(store, volume);
    expect(await store.list("vol-rt")).toEqual([".zattrs", ".zgroup", "annotation/.zattrs", "annotation/.zgroup", "annotation/s0/.zarray"]);
    expect(await readVolume(store, "vol-rt")).toEqual(volume);
    expect(await store.volumes()).toEqual(["vol-rt"]);
  });

  it("is null for a volume that is not there, or whose .zattrs is not an annotation volume's", async () => {
    const store = new MemoryPaintStore();
    expect(await readVolume(store, "vol-none")).toBeNull();
    await store.put("other", ".zattrs", new TextEncoder().encode(JSON.stringify({ multiscales: [] })));
    expect(await readVolume(store, "other")).toBeNull();
    await store.put("broken", ".zattrs", new TextEncoder().encode("{not json"));
    expect(await readVolume(store, "broken")).toBeNull();
  });
});

describe("the painted chunks", () => {
  const volume: PaintVolume = {
    ...planVolume(DATASET, entry("setup_54"), chain, { id: "vol-chunks", now: NOW }),
    chunk_size: [1, 2, 2],
    dtype: "uint16",
  };

  it("keys a chunk by its grid index with dots, and reads a key back", () => {
    expect(chunkKey([3, 0, 12])).toBe("3.0.12");
    expect(parseChunkKey("3.0.12")).toEqual([3, 0, 12]);
    expect(parseChunkKey(".zarray")).toBeNull();
    expect(parseChunkKey("3.0.x")).toBeNull();
    expect(parseChunkKey("")).toBeNull();
  });

  it("lists the chunks painted, the .zarray beside them ignored", async () => {
    const store = new MemoryPaintStore();
    await createVolume(store, volume);
    await store.put(volume.id, `${CHUNK_DIR}3.0.0`, new Uint8Array(8));
    await store.put(volume.id, `${CHUNK_DIR}0.1.2`, new Uint8Array(8));
    expect(await paintedChunkIndices(store, volume.id)).toEqual([[0, 1, 2], [3, 0, 0]]);
    expect(await paintedChunkIndices(store, "vol-none")).toEqual([]);
  });

  it("decodes a uint16 chunk little-endian, in C order", async () => {
    const store = new MemoryPaintStore();
    // 1, 2, 256, 65535 as little-endian uint16
    await store.put(volume.id, `${CHUNK_DIR}0.0.0`, new Uint8Array([1, 0, 2, 0, 0, 1, 255, 255]));
    const chunk = await readPaintedChunk(store, volume, [0, 0, 0]);
    expect(chunk).not.toBeNull();
    expect(chunk!.index).toEqual([0, 0, 0]);
    expect(chunk!.shape).toEqual([1, 2, 2]);
    expect(chunk!.labels).toBeInstanceOf(Uint16Array);
    expect(Array.from(chunk!.labels)).toEqual([1, 2, 256, 65535]);
    expect(await readPaintedChunk(store, volume, [0, 0, 1])).toBeNull();
  });

  it("reads a uint8 chunk as a copy of the bytes, and refuses a chunk of the wrong size", () => {
    const bytes = new Uint8Array([0, 1, 2, 3]);
    const labels = labelsOf({ ...volume, dtype: "uint8" }, bytes);
    expect(labels).toBeInstanceOf(Uint8Array);
    expect(Array.from(labels)).toEqual([0, 1, 2, 3]);
    expect(labels.buffer).not.toBe(bytes.buffer);
    expect(() => labelsOf({ ...volume, dtype: "uint8" }, new Uint8Array(3))).toThrow(/3 bytes, not the 4 of 1×2×2 uint8 voxels/);
    expect(() => labelsOf(volume, new Uint8Array(4))).toThrow(/4 bytes, not the 8 of 1×2×2 uint16 voxels/);
  });
});

describe("newVolumeId (new_volume_id)", () => {
  it("is vol-<8 hex>-<YYYYmmdd-HHMMSS>, local time", () => {
    expect(newVolumeId(NOW, "0badcafe")).toBe("vol-0badcafe-20261015-130405");
    expect(newVolumeId(new Date(2026, 0, 2, 3, 4, 5))).toMatch(/^vol-[0-9a-f]{8}-20260102-030405$/);
  });
});
