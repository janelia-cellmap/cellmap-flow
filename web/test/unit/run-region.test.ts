// Run on a region (src/run-region): a model's plan for a box in nm (the
// served level, the box in its voxels, its chunks and what they hold, or a
// refusal for a box outside what the model predicts), the run_region layer
// and the box read back from it as Neuroglancer keeps it, the run over a
// fake engine (FakeInference.materialize: the view, the box and the
// attributes it is asked for, its progress, a cancel, a failure), the
// output folder's overwrite question over a fake directory handle, the zip
// of a run kept in memory, the layer a saved zarr opens as, and what
// Inference.materialize asks chunkmirage's Engine for (a stand-in with the
// members the wrapper touches, so this runs in node with no service worker
// or reader). The DOM (panel.ts) needs a browser and is not tested here.
import { describe, expect, it, vi } from "vitest";

import { serveModels } from "../../src/backend/serving";
import type { Chain, DatasetInfo, RegionStore, ServedModel } from "../../src/contracts";
import { InferenceEngine, type ModelRunner } from "../../src/infer/engine";
import type { Level } from "../../src/infer/geometry";
import { type OutputFolder, overwriteQuestion, prepareFolder, savedLayer, savedLayerName, zarrName, zipRun } from "../../src/run-region/output";
import { isPlan, latestBox, type ModelPlan, planModel, runRegionLayer, sameRegion } from "../../src/run-region/region";
import { type RunProgress, runRegion, type RunTarget } from "../../src/run-region/run";
import { memorySession } from "../../src/state";
import { DATASET, entry, type FakeContext, fakeContext, NAME_54, NAME_56 } from "./backend-fakes";

const asked = vi.hoisted(() => ({ materialize: [] as { view: string; store: unknown; options: Record<string, unknown> }[] }));

vi.mock("../../chunkmirage/web/src/engine", () => ({
  Engine: class {
    /** The raw volume as the engine opens it: one 8 nm level, 1000 voxels a side. */
    sources = {
      raw: {
        dtype: "uint8", channels: 1,
        axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
        levels: [{ shape: [1000, 1000, 1000], voxel: [8, 8, 8], origin: [0, 0, 0] }],
      },
    };
    ready = Promise.resolve();
    async start() {}
    url(view: string) { return `http://localhost/virtual/page/${view}/`; }
    serveStages() {}
    unserve() {}
    stop() {}
    async materialize(view: string, store: unknown, options: Record<string, unknown>) {
      asked.materialize.push({ view, store, options });
      return { chunks: 1, written: 1, skipped: 0, bytes: 10, aborted: false };
    }
  },
}));

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

describe("a model's plan", () => {
  it("is the served level, the box in its voxels, its chunks and what they hold", async () => {
    const { models } = await served();
    const p = planModel(models[0], PYRAMID, { offset: [0, 0, 0], size: [960, 960, 1000] });
    if (!isPlan(p)) throw new Error(p.error);
    expect(p.name).toBe(NAME_54);
    expect(p.level).toEqual(LEVEL_16);
    expect(p.chunk).toEqual([60, 60, 60]);
    expect([p.lo, p.hi]).toEqual([[0, 0, 0], [61, 61, 63]]);
    expect([p.grid, p.chunks, p.savedChunks]).toEqual([[2, 2, 2], 8, 8]);
    // Three affinities in float32, whole chunks before compression.
    expect([p.channels, p.dtype, p.bytes]).toEqual([3, "float32", 8 * 60 ** 3 * 3 * 4]);
    expect(p.chain).toEqual(SIGMOID);
  });

  it("is refused, by name, for a region outside what the model predicts", async () => {
    const { models } = await served();
    const p = planModel(models[0], PYRAMID, { offset: [9000, 0, 0], size: [10, 10, 10] });
    expect(isPlan(p)).toBe(false);
    expect(!isPlan(p) && p.error).toContain(NAME_54);
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
    expect(layer.source).toEqual([{ url: "local://annotations", transform: { outputDimensions: { z: [1e-9, "m"], y: [1e-9, "m"], x: [1e-9, "m"] } } }]);
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

  it("takes a box read back a little off as the same region (Neuroglancer's 32-bit corners)", () => {
    const region = { offset: [100, 2000, 3000.4], size: [64, 1600, 1600] };
    const read = { offset: [100, 2000, Math.fround(3000.4)], size: [64, 1600, Math.fround(4600.4) - Math.fround(3000.4)] };
    expect(read.offset[2]).not.toBe(region.offset[2]);
    expect(sameRegion(read, region)).toBe(true);
    expect(sameRegion({ offset: [100, 2000, 3001], size: [64, 1600, 1600] }, region)).toBe(false);
  });
});

describe("a run", () => {
  const REGION = { offset: [0, 0, 0], size: [960, 960, 960] };

  it("writes each model's own view over its box into its store, with its attributes, one model after the other", async () => {
    const { ctx, models } = await served();
    const plans = plansFor(models, REGION);
    const target = memoryTarget();
    const progress: RunProgress[] = [];
    const outcome = await runRegion({
      infer: ctx.infer, plans, target, signal: new AbortController().signal,
      attributes: (p) => ({ model: p.name }), onProgress: (p) => progress.push(p), log: () => {}, clock: steppingClock(),
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
  });

  it("stops at a cancel between chunks: the model's chunks written so far stay, the next model is not started", async () => {
    const { ctx, models } = await served();
    const controller = new AbortController();
    let chunks = 0;
    ctx.infer.beforeChunk = () => {
      if (++chunks === 3) controller.abort();
    };
    const outcome = await runRegion({ infer: ctx.infer, plans: plansFor(models, REGION), target: memoryTarget(), signal: controller.signal, attributes: () => ({}), log: () => {} });
    expect(outcome.aborted).toBe(true);
    expect(outcome.error).toBeNull();
    expect(ctx.infer.materialized.map((m) => m.view)).toEqual([models[0].view]);
    expect(outcome.results.map((r) => [r.plan.name, r.written.written, r.written.aborted])).toEqual([[NAME_54, 2, true]]);
    expect(ctx.infer.materialized[0].options.signal).toBe(controller.signal);
  });

  it("stops at a model that fails, saying which and why", async () => {
    const { ctx, models } = await served();
    ctx.infer.beforeChunk = (view, index) => {
      if (index.join() === "0,1,0") throw new Error(`chunk 0/0/1/0 of ${view.split("__")[0]}: out of GPU memory`);
    };
    const outcome = await runRegion({ infer: ctx.infer, plans: plansFor(models, REGION), target: memoryTarget(), signal: new AbortController().signal, attributes: () => ({}), log: () => {} });
    expect(outcome.aborted).toBe(false);
    expect(outcome.error).toBe(`${NAME_54}: chunk 0/0/1/0 of ${NAME_54}: out of GPU memory`);
    expect(ctx.infer.materialized).toHaveLength(1);
    expect(outcome.results[0].written.written).toBe(2);
  });
});

/** A picked folder: its entries by name, a directory or a file, and its permission. */
function fakeFolder(entries: Record<string, "dir" | "file"> = {}, permission: PermissionState = "granted") {
  const state = { entries: new Map(Object.entries(entries)), permission };
  const named = (name: string) => Object.assign(new Error(name), { name });
  const folder: OutputFolder = {
    name: "Predictions",
    queryPermission: async () => state.permission,
    requestPermission: async () => (state.permission === "prompt" ? (state.permission = "granted") : state.permission),
    getDirectoryHandle: async (name) => {
      const kind = state.entries.get(name);
      if (kind === "dir") return {};
      throw named(kind === "file" ? "TypeMismatchError" : "NotFoundError");
    },
    removeEntry: async (name) => void state.entries.delete(name),
  };
  return { folder, state };
}

describe("the output folder", () => {
  it("asks before writing over a model's zarr, and removes it whole only on a yes", async () => {
    const { folder, state } = fakeFolder({ [zarrName(NAME_54)]: "dir", "other.zarr": "dir" });
    const questions: string[] = [];
    const no = await prepareFolder(folder, [NAME_54, NAME_56], (q) => (questions.push(q), false));
    expect(no).toMatchObject({ ready: false, refused: true });
    expect(questions).toEqual([overwriteQuestion("Predictions", [`${NAME_54}.zarr`])]);
    expect([...state.entries.keys()]).toEqual([`${NAME_54}.zarr`, "other.zarr"]);
    const yes = await prepareFolder(folder, [NAME_54, NAME_56], () => true);
    expect(yes).toEqual({ ready: true, removed: [`${NAME_54}.zarr`] });
    expect([...state.entries.keys()]).toEqual(["other.zarr"]);
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
});

describe("a saved zarr in the viewer", () => {
  it("is the model's own layer on the zarr, read through the local/ route", async () => {
    const { models } = await served();
    const layer = savedLayer(models[0], SIGMOID, "http://localhost/app/local/abc123-Predictions/", "#00ff00");
    expect(layer.name).toBe(savedLayerName(NAME_54));
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

describe("Inference.materialize", () => {
  /** A runner that is never run: materialize goes through the engine, which is a stand-in here. */
  const runner = {
    run: async () => { throw new Error("no forward pass in these tests"); },
    retain() {}, release: async () => {}, reset: async () => {},
    limits: async () => ({ maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false }),
    device: async () => "fake", provider: null,
  } as unknown as ModelRunner;

  it("asks the engine for the layer's own view at its one level, as zarr v2, gzip, empty chunks skipped", async () => {
    const inference = new InferenceEngine({ session: memorySession(), log: () => {}, runner });
    await inference.openDataset("https://example.org/data.zarr");
    const served = await inference.serve(entry("setup_54"), SIGMOID, "blob", async () => new ArrayBuffer(8));
    const store: RegionStore = { put: async () => {} };
    const signal = new AbortController().signal;
    const onProgress = () => {};
    const written = await inference.materialize(served.view, store, { lo: [0, 60, 120], hi: [60, 120, 1000], attributes: { cellmap_flow: { model_name: "m" } }, signal, onProgress });
    expect(written).toEqual({ chunks: 1, written: 1, skipped: 0, bytes: 10, aborted: false });
    expect(asked.materialize).toEqual([{
      view: served.view, store,
      options: { level: 0, lo: [0, 60, 120], hi: [60, 120, 1000], extent: "box", format: "zarr2", compression: "gzip", skipEmpty: true, attributes: { cellmap_flow: { model_name: "m" } }, signal, onProgress },
    }]);
  });
});
