// Run on a region (src/run-region): a box in nm placed on each served
// model's level (rounded out to the voxels it touches, then counted in the
// model's output blocks, the chunks materialize writes), the box read back
// from the run_region layer's JSON as Neuroglancer keeps it, the run over a
// fake engine (FakeInference.materialize: the view, the box and the
// attributes it is asked for, its progress, a cancel, a failure), the
// provenance, the output folder's overwrite check over a fake directory
// handle, the zip of a run kept in memory, and the layer a saved zarr opens
// as. The DOM (panel.ts) needs a browser and is not tested here.
import { describe, expect, it } from "vitest";

import { serveModels } from "../../src/backend/serving";
import type { Chain, DatasetInfo, RegionStore, RegionWrite, RegionWritten, ServedModel } from "../../src/contracts";
import type { Level } from "../../src/infer/geometry";
import {
  existingOutputs, OUTPUT_FOLDER_KEY, type OutputFolder, overwriteQuestion, prepareFolder, savedLayer, savedLayerName, writable,
  ZIP_LIMIT_BYTES, zarrName, zipFileName, zipRun,
} from "../../src/run-region/output";
import {
  BOX_TOOL, chainOf, chunkGrid, describePlan, formatBytes, formatDuration, isPlan, LARGE_RUN_CHUNKS, largeRunWarning,
  latestBox, levelBox, type ModelPlan, planModel, regionError, roundedOut, RUN_REGION_LAYER, runRegionLayer, wholeVolumeNm,
} from "../../src/run-region/region";
import { describeProgress, provenance, PROVENANCE_KEY, type RunProgress, runRegion, type RunTarget } from "../../src/run-region/run";
import { DATASET, type FakeContext, fakeContext, NAME_54, NAME_56 } from "./backend-fakes";

/** The chain a fresh browser session serves through (render.py's DEFAULT_CHAIN). */
const SIGMOID: Chain = {
  input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }],
  postprocess: [{ name: "SigmoidPostprocessor" }],
};

/** The fakes' dataset as a pyramid, 8 nm to 128 nm, each level's corner
 * the finest's (-4 nm), so voxel 0's centre, the OME translation, is half
 * the level's voxel past it. */
const PYRAMID: DatasetInfo = {
  ...DATASET,
  levels: [8, 16, 32, 64, 128].map((v) => ({ shape: [8000, 8000, 8000].map((n) => Math.floor(n / v)), voxel: [v, v, v], origin: [v / 2 - 4, v / 2 - 4, v / 2 - 4] })),
};

/** The 16 nm model's level on the pyramid: its own 16 nm level, voxel 0's
 * centre at 4 nm, its corner at -4 nm (infer/geometry.ts outputLevel). */
const LEVEL_16: Level = { shape: [500, 500, 500], voxel: [16, 16, 16], origin: [4, 4, 4] };

/** Both catalog models (16 nm affinities, 128 nm nuclei) served through `chain`, in the catalog's order. */
async function served(chain: Chain = SIGMOID): Promise<{ ctx: FakeContext; models: ServedModel[] }> {
  const ctx = fakeContext({ dataset: PYRAMID });
  const models = await serveModels(ctx, ctx.catalog.entries, chain);
  return { ctx, models };
}

function plansFor(models: ServedModel[], region: { offset: number[]; size: number[] }): ModelPlan[] {
  return models.map((m) => {
    const p = planModel(m, PYRAMID, region);
    if (!isPlan(p)) throw new Error(p.error);
    return p;
  });
}

/** A store per model, in memory, and the target over them. */
function memoryTarget(): RunTarget & { stores: Map<string, Map<string, Uint8Array>> } {
  const stores = new Map<string, Map<string, Uint8Array>>();
  return {
    stores,
    where: (name) => `out/${zarrName(name)}`,
    store: (name): RegionStore => {
      const files = new Map<string, Uint8Array>();
      stores.set(name, files);
      return { put: async (path, bytes) => void files.set(path, bytes) };
    },
  };
}

/** A clock that moves 100 ms each time it is read. */
function steppingClock(): () => number {
  let t = 0;
  return () => (t += 100);
}

describe("a region on a model's level", () => {
  it("takes every voxel the box touches, and none whose edge it only meets", () => {
    // Exactly on voxel edges: [-4, 156) nm is voxels 0..9.
    expect(levelBox({ offset: [-4, -4, -4], size: [160, 160, 160] }, LEVEL_16)).toEqual({ lo: [0, 0, 0], hi: [10, 10, 10] });
    // A quarter voxel past them on each side: one more voxel at the far end.
    expect(levelBox({ offset: [0, 0, 0], size: [160, 160, 160] }, LEVEL_16)).toEqual({ lo: [0, 0, 0], hi: [11, 11, 11] });
    // Per axis: z from 100 nm (voxel 6, which spans 92..108), y and x one voxel each.
    expect(levelBox({ offset: [100, 28, 29], size: [100, 16, 1] }, LEVEL_16)).toEqual({ lo: [6, 2, 2], hi: [13, 3, 3] });
  });

  it("gives a flat box the one voxel it lies in", () => {
    // Inside voxel 6 along z, and on the boundary between voxels 0 and 1.
    expect(levelBox({ offset: [100, -4, -4], size: [0, 32, 32] }, LEVEL_16)).toEqual({ lo: [6, 0, 0], hi: [7, 2, 2] });
    expect(levelBox({ offset: [12, -4, -4], size: [0, 32, 32] }, LEVEL_16)).toEqual({ lo: [1, 0, 0], hi: [2, 2, 2] });
  });

  it("clips the box to the level, and has nothing for a box outside it", () => {
    expect(levelBox({ offset: [-1000, -1000, 7000], size: [2000, 2000, 5000] }, LEVEL_16)).toEqual({ lo: [0, 0, 437], hi: [63, 63, 500] });
    expect(levelBox({ offset: [8000, 0, 0], size: [100, 100, 100] }, LEVEL_16)).toBeNull();
    expect(levelBox({ offset: [-500, 0, 0], size: [400, 100, 100] }, LEVEL_16)).toBeNull();
  });

  it("is not moved a voxel by the float noise of nm into voxels", () => {
    // 3 × 4.2 is 12.600000000000001, which over 4.2 is 3.0000000000000004:
    // its ceiling would take a fourth voxel the box does not reach.
    const level: Level = { shape: [100, 100, 100], voxel: [4.2, 4.2, 4.2], origin: [2.1, 2.1, 2.1] };
    expect(levelBox({ offset: [0, 0, 0], size: [3 * 4.2, 3 * 4.2, 3 * 4.2] }, level)).toEqual({ lo: [0, 0, 0], hi: [3, 3, 3] });
  });

  it("makes the whole volume the finest level's extent, which is all of each model's level", async () => {
    expect(wholeVolumeNm(PYRAMID)).toEqual({ offset: [-4, -4, -4], size: [8000, 8000, 8000] });
    const { models } = await served();
    const [p54, p56] = plansFor(models, wholeVolumeNm(PYRAMID));
    expect([p54.lo, p54.hi, p54.chunks]).toEqual([[0, 0, 0], [500, 500, 500], 729]);
    expect([p56.lo, p56.hi, p56.level.voxel, p56.chunks]).toEqual([[0, 0, 0], [62, 62, 62], [128, 128, 128], 8]);
    // On a volume with no 128 nm level the 128 nm model reads the 8 nm one
    // as if it were, and its output is 8 nm voxels, 1000 of them a side.
    const flat = planModel(models[1], DATASET, wholeVolumeNm(DATASET));
    expect(isPlan(flat) && [flat.hi, flat.level.voxel, flat.chunks]).toEqual([[1000, 1000, 1000], [8, 8, 8], 17 ** 3]);
  });

  it("counts the chunks the box reaches, rounded out as materialize writes them", () => {
    expect(chunkGrid([0, 0, 0], [60, 60, 60], [60, 60, 60])).toEqual({ grid: [1, 1, 1], count: 1 });
    expect(chunkGrid([59, 0, 0], [61, 60, 1], [60, 60, 60])).toEqual({ grid: [2, 1, 1], count: 2 });
    expect(chunkGrid([0, 0, 0], [500, 500, 500], [60, 60, 60])).toEqual({ grid: [9, 9, 9], count: 729 });
  });

  it("checks the region's numbers, a flat box among those allowed", () => {
    expect(regionError({ offset: [0, 0, 0], size: [0, 10, 10] })).toBeNull();
    expect(regionError({ offset: [0, Number.NaN, 0], size: [1, 1, 1] })).toMatch(/offset must be three numbers/);
    expect(regionError({ offset: [0, 0, 0], size: [1, 1] })).toMatch(/size must be three numbers/);
    expect(regionError({ offset: [0, 0, 0], size: [1, -1, 1] })).toMatch(/must not be negative/);
  });

  it("rounds a drawn box out to whole nm for the fields", () => {
    expect(roundedOut({ offset: [10.4, -3.2, 0], size: [5.2, 1, 2.5] })).toEqual({ offset: [10, -4, 0], size: [6, 2, 3] });
  });
});

describe("a model's plan", () => {
  it("is the served level, the box in its voxels, its chunks and what they hold", async () => {
    const { models } = await served();
    const p = planModel(models[0], PYRAMID, { offset: [0, 0, 0], size: [960, 960, 1000] });
    if (!isPlan(p)) throw new Error(p.error);
    expect(p.name).toBe(NAME_54);
    expect(p.level).toEqual(LEVEL_16);
    expect(p.chunk).toEqual([60, 60, 60]);
    expect([p.lo, p.hi]).toEqual([[0, 0, 0], [61, 61, 63]]);
    expect([p.grid, p.chunks]).toEqual([[2, 2, 2], 8]);
    // Three affinities in float32, whole chunks before compression.
    expect([p.channels, p.dtype, p.bytes]).toEqual([3, "float32", 8 * 60 ** 3 * 3 * 4]);
    expect(p.chain).toEqual(SIGMOID);
    expect(describePlan(p)).toBe(`${NAME_54}: 61×61×63 voxels at 16×16×16 nm, 2×2×2 = 8 chunks of 60×60×60 (at most 21 MB before compression)`);
  });

  it("follows the chain the view was served through: its channels and its dtype", async () => {
    const chain: Chain = { input_norm: SIGMOID.input_norm, postprocess: [{ name: "ChannelSelection", channels: "0" }, { name: "ThresholdPostprocessor", threshold: 0.5 }] };
    const { models } = await served(chain);
    expect(chainOf(models[0])).toEqual(chain);
    const p = planModel(models[0], PYRAMID, { offset: [0, 0, 0], size: [10, 10, 10] });
    if (!isPlan(p)) throw new Error(p.error);
    expect([p.channels, p.dtype, p.bytes]).toEqual([1, "uint8", 60 ** 3]);
  });

  it("is refused, by name, for a region outside what the model predicts", async () => {
    const { models } = await served();
    const p = planModel(models[0], PYRAMID, { offset: [9000, 0, 0], size: [10, 10, 10] });
    expect(isPlan(p)).toBe(false);
    expect(!isPlan(p) && p.error).toBe(`${NAME_54}: the region is outside what it predicts (500×500×500 voxels at 16×16×16 nm)`);
  });

  it("warns of a large run, with the time it takes at the measured pace", () => {
    expect(largeRunWarning(LARGE_RUN_CHUNKS, 200)).toBeNull();
    expect(largeRunWarning(4913, null)).toBe("This is a large run: 4,913 chunks. It keeps going while you browse; Cancel stops it, and what was written stays.");
    expect(largeRunWarning(4913, 250)).toMatch(/^This is a large run: 4,913 chunks, roughly 20 min at the 250 ms a chunk measured so far\./);
  });

  it("says sizes and durations as a person reads them", () => {
    expect([formatBytes(512), formatBytes(2_500), formatBytes(21_000_000), formatBytes(3.2e9)]).toEqual(["512 B", "2.5 kB", "21 MB", "3.2 GB"]);
    expect([formatDuration(4_400), formatDuration(600_000), formatDuration(9_000_000)]).toEqual(["4 s", "10 min", "2 h 30 min"]);
  });
});

describe("the run_region layer", () => {
  it("is an empty local annotation layer in nm with the bounding-box tool as its tool (Ctrl+click)", () => {
    const layer = runRegionLayer(["z", "y", "x"]);
    expect(layer.type).toBe("annotation");
    expect(layer.annotations).toEqual([]);
    // A legacy tool: as `tool`, not `toolBindings`, which Neuroglancer drops for it.
    expect(layer.tool).toBe("annotateBoundingBox");
    expect(layer.toolBindings).toBeUndefined();
    expect(BOX_TOOL).toBe("annotateBoundingBox");
    expect(layer.source).toEqual([{ url: "local://annotations", transform: { outputDimensions: { z: [1e-9, "m"], y: [1e-9, "m"], x: [1e-9, "m"] } } }]);
    expect(RUN_REGION_LAYER).toBe("run_region");
  });

  it("gives the latest box drawn, its corners in either order", () => {
    const layer = {
      ...runRegionLayer(),
      annotations: [
        { type: "axis_aligned_bounding_box", id: "a", pointA: [0, 0, 0], pointB: [10, 10, 10] },
        { type: "axis_aligned_bounding_box", id: "b", pointA: [500, 120.5, 64], pointB: [100, 20, 960] },
        { type: "point", id: "c", point: [1, 2, 3] },
      ],
    };
    expect(latestBox(layer)).toEqual({ offset: [100, 20, 64], size: [400, 100.5, 896] });
  });

  it("reads the points in the layer's own coordinate space, matched to z, y, x by name", () => {
    // A layer at 8 nm whose axes are listed x, y, z.
    const space = { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"] };
    const layer = { type: "annotation", source: { url: "local://annotations", transform: { outputDimensions: space } }, annotations: [{ type: "axis_aligned_bounding_box", pointA: [1, 2, 3], pointB: [2, 4, 6] }] };
    expect(latestBox(layer)).toEqual({ offset: [24, 16, 8], size: [24, 16, 8] });
    // The source's own (input) dimensions win over the space it is shown in.
    const shown = { ...layer, source: { url: "local://annotations", transform: { inputDimensions: { z: [1e-6, "m"], y: [1e-6, "m"], x: [1e-6, "m"] }, outputDimensions: space } } };
    expect(latestBox(shown)).toEqual({ offset: [1000, 2000, 3000], size: [1000, 2000, 3000] });
  });

  it("has no box without one drawn, or without the layer", () => {
    expect(latestBox(runRegionLayer())).toBeNull();
    expect(latestBox({ type: "annotation", annotations: [{ type: "line", pointA: [0, 0, 0], pointB: [1, 1, 1] }] })).toBeNull();
    expect(latestBox(null)).toBeNull();
  });
});

describe("a run", () => {
  const REGION = { offset: [0, 0, 0], size: [960, 960, 960] };

  it("writes each model's own view over its box into its store, with its attributes, one model after the other", async () => {
    const { ctx, models } = await served();
    const plans = plansFor(models, REGION);
    const target = memoryTarget();
    const progress: RunProgress[] = [];
    const logged: string[] = [];
    const outcome = await runRegion({
      infer: ctx.infer, plans, target, signal: new AbortController().signal,
      attributes: (p) => ({ model: p.name }), onProgress: (p) => progress.push(p), log: (l) => logged.push(l), clock: steppingClock(),
    });
    expect(outcome.aborted).toBe(false);
    expect(outcome.error).toBeNull();
    // Each model over its own grid: 16 nm voxels for the first, 128 nm
    // for the second, whose one chunk covers the region.
    expect(ctx.infer.materialized.map((m) => [m.view, m.options.lo, m.options.hi])).toEqual([
      [models[0].view, [0, 0, 0], [61, 61, 61]],
      [models[1].view, [0, 0, 0], [8, 8, 8]],
    ]);
    expect(ctx.infer.materialized.map((m) => m.options.attributes)).toEqual([{ model: NAME_54 }, { model: NAME_56 }]);
    expect(outcome.results.map((r) => [r.plan.name, r.where, r.written.written, r.written.chunks])).toEqual([
      [NAME_54, `out/${NAME_54}.zarr`, 8, 8],
      [NAME_56, `out/${NAME_56}.zarr`, 1, 1],
    ]);
    expect([...target.stores.get(NAME_54)!.keys()]).toContain("0/1.1.1");
    // The run's count goes on across the models, to their sum; the time
    // left is the pace so far over the chunks to go.
    const last = progress[progress.length - 1];
    expect([last.name, last.model, last.models, last.runDone, last.runTotal, last.remainingMs]).toEqual([NAME_56, 2, 2, 9, 9, 0]);
    const fifth = progress[4];
    expect([fifth.name, fifth.done, fifth.runDone]).toEqual([NAME_54, 5, 5]);
    expect(fifth.remainingMs).toBeCloseTo((fifth.elapsedMs / 5) * 4);
    expect(describeProgress(fifth)).toMatch(new RegExp(`^${NAME_54} \\(1 of 2\\): 5 of 8 chunks, 5 written, \\d+ B in all; \\d+ s so far, about \\d+ s left$`));
    expect(logged).toContain(`${NAME_54}: done, 8 of 8 chunks written (${formatBytes(target.stores.get(NAME_54)!.get(".zattrs")!.length + 8)}) to out/${NAME_54}.zarr`);
    expect(logged[logged.length - 1]).toMatch(/^run done in \d+ s: 2 model\(s\)$/);
  });

  it("stops at a cancel between chunks: the model's chunks written so far stay, the next model is not started", async () => {
    const { ctx, models } = await served();
    const controller = new AbortController();
    let chunks = 0;
    ctx.infer.beforeChunk = () => {
      if (++chunks === 3) controller.abort();
    };
    const logged: string[] = [];
    const outcome = await runRegion({ infer: ctx.infer, plans: plansFor(models, REGION), target: memoryTarget(), signal: controller.signal, attributes: () => ({}), log: (l) => logged.push(l) });
    expect(outcome.aborted).toBe(true);
    expect(outcome.error).toBeNull();
    expect(ctx.infer.materialized.map((m) => m.view)).toEqual([models[0].view]);
    expect(outcome.results.map((r) => [r.plan.name, r.written.written, r.written.aborted])).toEqual([[NAME_54, 2, true]]);
    expect(logged).toContain(`${NAME_54}: cancelled, 2 of 8 chunks written (${formatBytes(outcome.results[0].written.bytes)}) to out/${NAME_54}.zarr`);
    expect(ctx.infer.materialized[0].options.signal).toBe(controller.signal);
  });

  it("takes an abort the engine throws for a cancel, with what the last progress said was written", async () => {
    const { models } = await served();
    const controller = new AbortController();
    const infer = {
      materialize: async (_view: string, _store: RegionStore, options: RegionWrite): Promise<RegionWritten> => {
        options.onProgress?.({ done: 5, total: 8, written: 4, bytes: 400 });
        controller.abort();
        throw new DOMException("The operation was aborted.", "AbortError");
      },
    };
    const outcome = await runRegion({ infer, plans: plansFor(models, REGION), target: memoryTarget(), signal: controller.signal, attributes: () => ({}), log: () => {} });
    expect(outcome.aborted).toBe(true);
    expect(outcome.results.map((r) => r.written)).toEqual([{ chunks: 8, written: 4, skipped: 1, bytes: 400, aborted: true }]);
  });

  it("stops at a model that fails, saying which and why", async () => {
    const { ctx, models } = await served();
    ctx.infer.beforeChunk = (view, index) => {
      if (index.join() === "0,1,0") throw new Error(`chunk 0/0/1/0 of ${view.split("__")[0]}: out of GPU memory`);
    };
    const logged: string[] = [];
    const outcome = await runRegion({ infer: ctx.infer, plans: plansFor(models, REGION), target: memoryTarget(), signal: new AbortController().signal, attributes: () => ({}), log: (l) => logged.push(l) });
    expect(outcome.aborted).toBe(false);
    expect(outcome.error).toBe(`${NAME_54}: chunk 0/0/1/0 of ${NAME_54}: out of GPU memory`);
    expect(ctx.infer.materialized).toHaveLength(1);
    expect(outcome.results[0].written.written).toBe(2);
    expect(logged[logged.length - 1]).toMatch(new RegExp(`^${NAME_54}: failed after 2 of 8 chunks written .* to out/${NAME_54}\\.zarr: chunk`));
  });

  it("starts nothing once cancelled", async () => {
    const { ctx, models } = await served();
    const controller = new AbortController();
    controller.abort();
    const outcome = await runRegion({ infer: ctx.infer, plans: plansFor(models, REGION), target: memoryTarget(), signal: controller.signal, attributes: () => ({}), log: () => {} });
    expect([outcome.aborted, outcome.results.length, ctx.infer.materialized.length]).toEqual([true, 0, 0]);
  });

  it("puts the provenance in each zarr's attributes: the model, its chain, the dataset, the region and the box, when and where", async () => {
    const { models } = await served();
    const [plan] = plansFor(models, REGION);
    const now = new Date(Date.UTC(2026, 9, 16, 12, 30, 0));
    const attrs = provenance(plan, DATASET, REGION, now, "https://janelia-cellmap.github.io/cellmap-flow/app/");
    const entry = models[0].entry;
    expect(attrs).toEqual({
      [PROVENANCE_KEY]: {
        model_name: NAME_54, model_id: entry.id, repo: entry.repo, folder: "setup_54", variant: "fp32", model_file: "setup_54_fp32.onnx",
        input_norm: SIGMOID.input_norm, postprocess: SIGMOID.postprocess,
        dataset_path: DATASET.url, region_offset_nm: [0, 0, 0], region_shape_nm: [960, 960, 960],
        box_voxels: { lo: [0, 0, 0], hi: [61, 61, 61] }, voxel_size_nm: [16, 16, 16],
        created_at: "2026-10-16T12:30:00.000Z", app_url: "https://janelia-cellmap.github.io/cellmap-flow/app/", engine: "browser",
      },
    });
    expect(PROVENANCE_KEY).toBe("cellmap_flow");
  });
});

/** A picked folder: its entries by name, a directory or a file, and its permission. */
function fakeFolder(entries: Record<string, "dir" | "file"> = {}, permission: PermissionState = "granted") {
  const state = { entries: new Map(Object.entries(entries)), permission, asked: 0, failWith: null as string | null };
  const named = (name: string) => Object.assign(new Error(name), { name });
  const folder: OutputFolder = {
    name: "Predictions",
    queryPermission: async () => state.permission,
    requestPermission: async () => {
      state.asked++;
      return state.permission === "prompt" ? (state.permission = "granted") : state.permission;
    },
    getDirectoryHandle: async (name) => {
      if (state.failWith) throw named(state.failWith);
      const kind = state.entries.get(name);
      if (kind === "dir") return {};
      throw named(kind === "file" ? "TypeMismatchError" : "NotFoundError");
    },
    removeEntry: async (name) => void state.entries.delete(name),
  };
  return { folder, state };
}

describe("the output folder", () => {
  it("finds the models' zarrs there already, a file in the way among them", async () => {
    const { folder, state } = fakeFolder({ [zarrName(NAME_54)]: "dir", [zarrName(NAME_56)]: "file", "other.zarr": "dir" });
    expect(await existingOutputs(folder, [NAME_54, NAME_56, "mito"])).toEqual([`${NAME_54}.zarr`, `${NAME_56}.zarr`]);
    state.failWith = "NotAllowedError";
    await expect(existingOutputs(folder, [NAME_54])).rejects.toThrow("NotAllowedError");
  });

  it("asks for write permission only when it is not granted", async () => {
    const granted = fakeFolder();
    expect(await writable(granted.folder)).toBe(true);
    expect(granted.state.asked).toBe(0);
    const prompt = fakeFolder({}, "prompt");
    expect(await writable(prompt.folder)).toBe(true);
    expect(prompt.state.asked).toBe(1);
    expect(await writable(fakeFolder({}, "denied").folder)).toBe(false);
  });

  it("asks before writing over a model's zarr, and removes it whole only on a yes", async () => {
    const { folder, state } = fakeFolder({ [zarrName(NAME_54)]: "dir", "other.zarr": "dir" });
    const questions: string[] = [];
    const no = await prepareFolder(folder, [NAME_54, NAME_56], (q) => (questions.push(q), false));
    expect(no).toEqual({ ready: false, refused: true, reason: "Nothing was run; Predictions is as it was." });
    expect(questions).toEqual([overwriteQuestion("Predictions", [`${NAME_54}.zarr`])]);
    expect(questions[0]).toBe(`This is in the folder already:\n  Predictions/${NAME_54}.zarr\n\nReplace it? It is deleted first, so the old chunks do not mix with the new ones.`);
    expect([...state.entries.keys()]).toEqual([`${NAME_54}.zarr`, "other.zarr"]);
    const yes = await prepareFolder(folder, [NAME_54, NAME_56], () => true);
    expect(yes).toEqual({ ready: true, removed: [`${NAME_54}.zarr`] });
    expect([...state.entries.keys()]).toEqual(["other.zarr"]);
  });

  it("asks nothing when nothing is in the way, and writes nothing without permission", async () => {
    const empty = fakeFolder();
    expect(await prepareFolder(empty.folder, [NAME_54], () => { throw new Error("asked"); })).toEqual({ ready: true, removed: [] });
    const denied = fakeFolder({ [zarrName(NAME_54)]: "dir" }, "denied");
    const answer = await prepareFolder(denied.folder, [NAME_54], () => true);
    expect(answer.ready).toBe(false);
    expect(!answer.ready && [answer.refused, answer.reason]).toEqual([true, "Writing to Predictions was not allowed, so nothing was run. Click Run to be asked again, or choose another folder."]);
    expect(denied.state.entries.size).toBe(1);
  });

  it("keeps its handle under a key no picked volume's key can be", () => {
    // local-folder's keys are six random characters, a hyphen, the name.
    expect(OUTPUT_FOLDER_KEY).toBe("run-region:output");
    expect(OUTPUT_FOLDER_KEY[6]).not.toBe("-");
  });
});

/** The file names in a store-only zip, from its local headers in order (APPNOTE 4.3.7). */
function zipNames(zip: Uint8Array): string[] {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const names: string[] = [];
  for (let at = 0; view.getUint32(at, true) === 0x04034b50;) {
    const size = view.getUint32(at + 18, true), nameLength = view.getUint16(at + 26, true), extra = view.getUint16(at + 28, true);
    names.push(new TextDecoder().decode(zip.subarray(at + 30, at + 30 + nameLength)));
    at += 30 + nameLength + extra + size;
  }
  return names;
}

describe("a run kept in memory", () => {
  it("zips each model's files under its .zarr folder", () => {
    const bytes = (n: number) => new Uint8Array(n).fill(7);
    const stores = new Map([
      ["mito", { files: new Map([["0/0/0/0/0", bytes(3)], [".zattrs", bytes(2)], [".zgroup", bytes(1)]]) }],
      ["nuc", { files: new Map([[".zgroup", bytes(1)]]) }],
    ]);
    expect(zipNames(zipRun(stores, new Date(2026, 9, 16)))).toEqual(["mito.zarr/.zattrs", "mito.zarr/.zgroup", "mito.zarr/0/0/0/0/0", "nuc.zarr/.zgroup"]);
  });

  it("is named for when it was run, and capped at 256 MB of predictions", () => {
    expect(zipFileName(new Date(2026, 9, 16, 9, 5, 3))).toBe("predictions-20261016-090503.zip");
    expect(ZIP_LIMIT_BYTES).toBe(256e6);
  });
});

describe("a saved zarr in the viewer", () => {
  it("is the model's own layer on the zarr, read through the local/ route", async () => {
    const { models } = await served();
    const layer = savedLayer(models[0], SIGMOID, "http://localhost/app/local/abc123-Predictions/", "#00ff00");
    expect(layer.name).toBe(savedLayerName(NAME_54));
    expect(layer.name).toBe(`${NAME_54} (saved)`);
    expect(layer.type).toBe("image");
    expect(layer.source).toBe(`zarr2://http://localhost/app/local/abc123-Predictions/${NAME_54}.zarr/`);
    expect(layer.shader).toContain('color(default="#00ff00")');
  });

  it("is a segmentation layer for a chain that ends in labels, and keeps the live layer's shader", async () => {
    const labels: Chain = { input_norm: SIGMOID.input_norm, postprocess: [{ name: "ThresholdPostprocessor", threshold: 0.5 }] };
    const { models } = await served(labels);
    expect(savedLayer(models[0], labels, "http://localhost/app/local/k/", "#ff0000").type).toBe("segmentation");
    const kept = savedLayer(models[1], SIGMOID, "http://localhost/app/local/k", "#ff0000", { shader: "void main() { emitGrayscale(1.0); }" });
    expect([kept.shader, kept.source]).toEqual(["void main() { emitGrayscale(1.0); }", `zarr2://http://localhost/app/local/k/${NAME_56}.zarr/`]);
  });
});
