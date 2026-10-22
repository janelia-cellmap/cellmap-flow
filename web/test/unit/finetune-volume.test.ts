// An annotation volume (src/finetune/volume.ts): its plan answered by the
// portable Python (cellmap_flow/portable/volume.py, whose rules
// tests/portable/test_volume.py tests; here in node's Pyodide,
// test/unit/pyodide.ts, so without it that test skips) on the grid the
// engine serves the model's prediction on, and the round trip through a
// paint store, the painted chunks read back as the brush wrote them.
import { describe, expect, it, vi } from "vitest";

import type { Chain, DatasetInfo, ModelVariant, PaintVolume } from "../../src/contracts";
import { CHUNK_DIR, createVolume, paintedChunkIndices, planVolume, readPaintedChunk, readVolume } from "../../src/finetune/volume";
import { chooseRawLevel, geometryOf, outputLevel } from "../../src/infer/geometry";
import { entryFor } from "../../src/models";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { DATASET, entry, manifest, NAME_54 } from "./backend-fakes";
import { portablePython } from "./pyodide";

// Pyodide and numpy load once for the file, a few seconds; each call after is quick.
vi.setConfig({ testTimeout: 120_000 });
const python = await portablePython();

const chain: Chain = {
  input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }],
  postprocess: [{ name: "SigmoidPostprocessor" }],
};
const NOW = new Date(2026, 9, 15, 13, 4, 5);  // 2026-10-15 13:04:05 local

/** A volume as planVolume makes one, for the paint store's round trip. */
const VOLUME: PaintVolume = {
  id: "vol-rt", type: "annotation_volume", model_name: NAME_54, model_id: "test/setup_54", dataset_path: DATASET.url,
  chunk_size: [60, 60, 60], input_size: [148, 148, 148], output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8],
  claimed_output_voxel_size: [16, 16, 16], claimed_input_voxel_size: [16, 16, 16], dataset_offset_nm: [0, 0, 0],
  dataset_shape_voxels: [1020, 1020, 1020], dtype: "uint16", input_norm: chain.input_norm, postprocess: chain.postprocess,
  created_at: NOW.toISOString(),
};

describe("planVolume (plan_volume)", () => {
  it.skipIf(!python)("is the Python's plan, on the grid the model is served on, with uint16 labels for an affinity model", async () => {
    // A model reading 8 nm and writing 16 over an anisotropic level of 16 ×
    // 8 × 8 nm: the served level is the level's voxel times the ratio, from
    // its corner, and the volume lies on it, one chunk per output block,
    // over the level's extent padded to whole chunks.
    const variant: ModelVariant = { variant: "fp32", input_size: [40, 40, 40], output_size: [8, 8, 8], file: "coarse_fp32.onnx", file_mb: 1 };
    const coarse = entryFor("coarse", { ...manifest("coarse", ["nuc"], 8), output_voxel_size_nm: [16, 16, 16], variants: [variant] }, variant);
    const dataset: DatasetInfo = { ...DATASET, levels: [{ shape: [101, 1000, 1000], voxel: [16, 8, 8], origin: [8, 4, 4] }] };
    const volume = await planVolume({ python: python! }, dataset, coarse, chain, { id: "vol-anis", now: NOW });
    const g = geometryOf(coarse), served = outputLevel(dataset.levels[chooseRawLevel(dataset.levels, g.input_voxel_size)], g);
    expect(volume.output_voxel_size).toEqual(served.voxel);
    expect(volume.dataset_offset_nm).toEqual(served.origin);
    // Whole chunks over all of the served level (the far partial voxel too: the Python's to say).
    expect(volume.dataset_shape_voxels.every((n, a) => n % 8 === 0 && n >= served.shape[a])).toBe(true);
    expect(volume).toMatchObject({ chunk_size: [8, 8, 8], input_size: [40, 40, 40], dtype: "uint8", id: "vol-anis", created_at: NOW.toISOString() });
    // Copies: the chain the dashboard goes on editing is not the volume's.
    expect(volume.input_norm).toEqual(chain.input_norm);
    expect(volume.input_norm).not.toBe(chain.input_norm);

    expect((await planVolume({ python: python! }, DATASET, entry("setup_54"), chain)).dtype).toBe("uint16");
  });

  it("refuses a dataset with more than one channel, as serve() does, before asking the Python", async () => {
    const rgb: DatasetInfo = { ...DATASET, channels: 3, axes: [{ name: "c", unit: "" }, ...DATASET.axes] };
    const noPython = { python: () => Promise.reject(new Error("not asked")) };
    await expect(planVolume(noPython, rgb, entry("setup_54"), chain)).rejects.toThrow(/channels/);
  });
});

describe("createVolume and readVolume on a paint store", () => {
  it("round-trips the volume through its files, the labels placed at voxel 0's centre", async () => {
    const store = new MemoryPaintStore();
    await createVolume(store, VOLUME);
    expect(await store.list("vol-rt")).toEqual([".zattrs", ".zgroup", "annotation/.zattrs", "annotation/.zgroup", "annotation/s0/.zarray"]);
    expect(await readVolume(store, "vol-rt")).toEqual(VOLUME);
    expect(await store.volumes()).toEqual(["vol-rt"]);
    // Where Neuroglancer draws the paint: the output voxel size, translated to the volume's offset.
    const labels = JSON.parse(new TextDecoder().decode(await store.get("vol-rt", "annotation/.zattrs")));
    expect(labels.multiscales[0].datasets[0].coordinateTransformations).toEqual([
      { type: "scale", scale: VOLUME.output_voxel_size },
      { type: "translation", translation: VOLUME.dataset_offset_nm },
    ]);
    // Not an annotation volume's .zattrs: none.
    await store.put("other", ".zattrs", new TextEncoder().encode(JSON.stringify({ multiscales: [] })));
    expect(await readVolume(store, "other")).toBeNull();
  });

  it("lists the chunks painted and decodes a uint16 chunk little-endian, in C order", async () => {
    const store = new MemoryPaintStore();
    const volume: PaintVolume = { ...VOLUME, id: "vol-chunks", chunk_size: [1, 2, 2], dtype: "uint16" };
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
