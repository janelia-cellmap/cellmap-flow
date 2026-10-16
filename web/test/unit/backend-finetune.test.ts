// The Finetune tab's routes against fake context members, a fake volume
// plan and a fake run: the JSON each answers, checked against what the tab's
// JavaScript reads (crops.js, sessions.js, models.js, job-monitor.js,
// job-card.js, log-stream.js), and what each does to the paint store, the
// session, the viewer and the catalog.
import { describe, expect, it } from "vitest";

import { createBackend } from "../../src/backend";
import { type FinetuneDeps, finetuneRoutes, readTraining, RESTART_ACTION, RESTART_EVENT, RESTART_KEY, sameDataset } from "../../src/backend/routes/finetune";
import { Router } from "../../src/backend/router";
import type { ApiRequest, ApiResponse, Chain, ModelEntry, PaintVolume } from "../../src/contracts";
import { MANIFEST_METADATA_KEY } from "../../src/contracts";
import { readMetadata } from "../../src/finetune/onnx-proto";
import { saveGoodRegions } from "../../src/finetune/good-regions";
import type { Finetuned, FinetuneRun, RunFinetune } from "../../src/finetune/types";
import { CHUNK_DIR, chunkKey } from "../../src/finetune/volume";
import { defaultChain } from "../../src/state";
import { paintLayerName } from "../../src/viewer/layers";
import { DATASET, entry, type FakeContext, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

const encoder = new TextEncoder();
const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/** A volume as planVolume would plan one for setup_54 over DATASET: 2×4×4 chunks over 4×8×8 voxels. */
function volume(id: string, over: Partial<PaintVolume> = {}): PaintVolume {
  return {
    id,
    type: "annotation_volume",
    model_name: NAME_54,
    model_id: "cellmap/onnx-web-models/setup_54",
    dataset_path: DATASET.url,
    chunk_size: [2, 4, 4],
    input_size: [10, 12, 12],
    output_voxel_size: [8, 8, 8],
    input_voxel_size: [8, 8, 8],
    claimed_output_voxel_size: [16, 16, 16],
    claimed_input_voxel_size: [16, 16, 16],
    dataset_offset_nm: [0, 0, 0],
    dataset_shape_voxels: [4, 8, 8],
    dtype: "uint8",
    input_norm: defaultChain().input_norm,
    postprocess: defaultChain().postprocess,
    created_at: "2026-10-15T12:00:00.000Z",
    ...over,
  };
}

/** A hand-off between a test and the fake run: the run stops at each epoch
 * until the test lets it go on, and the test can wait for it to stop. */
class Baton {
  private stops: (() => void)[] = [];
  private waiting: (() => void)[] = [];

  /** The run: wait here until the test says go. */
  stop(): Promise<void> {
    return new Promise((resume) => {
      this.stops.push(resume);
      for (const wake of this.waiting.splice(0)) wake();
    });
  }

  /** The test: resolves once the run is stopped (at once if it already is). */
  async stopped(): Promise<void> {
    if (this.stops.length) return;
    await new Promise<void>((wake) => this.waiting.push(wake));
  }

  /** The test: let the run take its next step. */
  go(): void {
    this.stops.shift()?.();
  }
}

/** Let a gated run go on until it is over. */
async function letRun(baton: Baton, job: { settled(): Promise<void> }): Promise<void> {
  for (;;) {
    const stopped = await Promise.race([baton.stopped().then(() => true), job.settled().then(() => false)]);
    if (!stopped) return;
    baton.go();
  }
}

/** A run that logs one summary line per epoch (loss 0.5/epoch), stops where
 * the reporter says, and makes a head one more than the one it started from
 * (so a restart's initialHead shows in its result). */
function fakeRun(baton?: Baton): { run: RunFinetune; calls: FinetuneRun[]; results: Finetuned[] } {
  const calls: FinetuneRun[] = [];
  const results: Finetuned[] = [];
  const run: RunFinetune = async (r) => {
    calls.push(r);
    const n = r.options.epochs;
    const from = r.initialHead ?? { weight: new Float32Array([1, 2, 3]), bias: new Float32Array([0]) };
    const weight = from.weight.map((w) => w + 1);
    const bias = from.bias.map((b) => b + 1);
    const losses: number[] = [];
    r.report.log(`Reading the painted chunks of ${r.volume.id}`);
    for (let i = 1; i <= n; i++) {
      if (baton) await baton.stop();
      if (r.report.cancelled()) break;
      losses.push(0.5 / i);
      r.report.log(`Epoch ${i}/${n} - Loss: ${(0.5 / i).toFixed(6)}`);
    }
    // A ModelProto as far as its fields go (ir_version 1, then a graph field
    // holding two bytes, the weights' and the epochs' counts), so that the
    // download can write its manifest into it.
    const bytes = new ArrayBuffer(6);
    new Uint8Array(bytes).set([0x08, 0x01, 0x3a, 0x02, weight.length, losses.length]);
    const result: Finetuned = {
      bytes,
      head: { featureTensor: "features", weightName: "final_conv.weight", biasName: "final_conv.bias", inChannels: 32, outChannels: 3, weight, bias, dtype: "float32" },
      weight,
      bias,
      epochs: losses.length,
      losses,
      finalLoss: losses.length ? losses[losses.length - 1] : null,
      supervisedLosses: losses,
      pixels: 40,
      chunks: 1,
      distilled: 0,
    };
    results.push(result);
    return result;
  };
  return { run, calls, results };
}

/** The finetune routes on their own router, with a plan that makes the
 * volume above (numbered) and a clock one second apart per call. */
function harness(ctx: FakeContext, deps: Partial<FinetuneDeps> = {}) {
  const router = new Router();
  const plans: { entry: ModelEntry; chain: Chain; dataset: string }[] = [];
  let second = 0;
  const plan: FinetuneDeps["plan"] = (dataset, e, chain, options = {}) => {
    plans.push({ entry: e, chain, dataset: dataset.url });
    const id = `vol-${String(plans.length).padStart(8, "0")}-20261015-120000`;
    return volume(id, { model_name: e.name, model_id: e.id, dataset_path: dataset.url, created_at: (options.now ?? new Date()).toISOString() });
  };
  const jobs = finetuneRoutes(router, { plan, now: () => new Date(2026, 9, 15, 12, 0, second++), ...deps });
  return { handle: (req: ApiRequest) => router.handle(req, ctx), jobs, plans };
}

/** Serve `entries` in the fake engine and the session, as Submit would. */
async function serve(ctx: FakeContext, ...entries: ModelEntry[]): Promise<void> {
  for (const e of entries) await ctx.infer.serve(e, ctx.session.state.chain, "blob", async () => new ArrayBuffer(8));
  await ctx.session.update({ models: [...ctx.session.state.models, ...entries] });
}

async function paint(ctx: FakeContext, id: string, index: number[]): Promise<void> {
  await ctx.paint.put(id, CHUNK_DIR + chunkKey(index), new Uint8Array(32).fill(1));
}

/** The form's submit body (job-monitor.js), with the browser's loss. */
const FORM = {
  lora_r: 8,
  num_epochs: 3,
  batch_size: 8,
  learning_rate: 0.01,
  loss_type: "bce",
  distillation_lambda: 0.01,
  distillation_scope: "unlabeled",
  balance_classes: true,
  augment: false,
  label_smoothing: 0,
};
const submitBody = (extra: Record<string, unknown> = {}) => ({ model_name: NAME_54, corrections_path: "", ...FORM, auto_serve: true, queue: "", ...extra });

/** Two models served, a volume created for setup_54 and (unless `paint: false`) one chunk painted; `deps` over the harness's. */
async function ready(options: { baton?: Baton; paint?: boolean; deps?: Partial<FinetuneDeps> } = {}) {
  const ctx = fakeContext();
  await serve(ctx, entry("setup_54"), entry("setup_56", "nuc", ["nuc"], 128));
  const { run, calls, results } = fakeRun(options.baton);
  const h = harness(ctx, { run, ...options.deps });
  const created = body(await h.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_54 })));
  const id = created.volume_id as string;
  if (options.paint !== false) await paint(ctx, id, [0, 0, 0]);
  return { ctx, h, id, created, calls, results };
}

const status = (h: ReturnType<typeof harness>, jobId: string) => h.handle(request("GET", `/api/finetune/job/${jobId}/status`)).then(body);

/** An event stream's events one at a time; `soon` gives "nothing" when none comes within `ms`. */
function openStream(res: ApiResponse) {
  if (!("stream" in res)) throw new Error("not a stream");
  expect(res.type).toBe("text/event-stream");
  const reader = res.stream.getReader();
  const decoder = new TextDecoder();
  let pending: Promise<string | null> | null = null;
  const next = () =>
    (pending ??= reader.read().then(({ value, done }) => {
      pending = null;
      return done ? null : decoder.decode(value);
    }));
  return {
    next,
    soon: (ms = 20) => Promise.race([next(), new Promise<"nothing">((resolve) => setTimeout(() => resolve("nothing"), ms))]),
    cancel: () => reader.cancel(),
  };
}

describe("GET /api/finetune/models", () => {
  it("lists the served models with the geometry the picker shows, the head as the way to finetune, and the one model as selected", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    expect(body(await h.handle(request("GET", "/api/finetune/models")))).toEqual({ models: [], selected_model: null });
    await serve(ctx, entry("setup_54"));
    expect(body(await h.handle(request("GET", "/api/finetune/models")))).toEqual({
      models: [{ name: NAME_54, write_shape: [60, 60, 60], output_voxel_size: [16, 16, 16], output_channels: 3, finetune_modes: ["head"] }],
      selected_model: NAME_54,
    });
    await serve(ctx, entry("setup_56", "nuc", ["nuc"], 128));
    const two = body(await h.handle(request("GET", "/api/finetune/models")));
    expect(two.models.map((m: any) => m.name)).toEqual([NAME_54, NAME_56]);
    expect(two.selected_model).toBeNull();
  });
});

describe("POST /api/finetune/create-volume", () => {
  it("plans a volume for the served model over the open dataset, writes its files, makes it the session's, and answers its URLs and geometry", async () => {
    const { ctx, h, id, created } = await ready({ paint: false });
    expect(h.plans).toEqual([{ entry: ctx.infer.served()[0].entry, chain: ctx.session.state.chain, dataset: DATASET.url }]);
    expect(id).toBe("vol-00000001-20261015-120000");
    expect(created).toEqual({
      success: true,
      volume_id: id,
      zarr_path: `paint/${id}`,
      minio_url: `s3+http://localhost/paint/${id}`,
      neuroglancer_url: `s3+http://localhost/paint/${id}/annotation`,
      metadata: { dataset_shape_voxels: [4, 8, 8], chunk_size: [2, 4, 4], output_voxel_size: [8, 8, 8], claimed_output_voxel_size: [16, 16, 16], dataset_offset_nm: [0, 0, 0] },
    });
    expect(await ctx.paint.list(id)).toEqual([".zattrs", ".zgroup", "annotation/.zattrs", "annotation/.zgroup", "annotation/s0/.zarray"]);
    const attrs = JSON.parse(new TextDecoder().decode(await ctx.paint.get(id, ".zattrs")));
    expect(attrs).toMatchObject({ type: "annotation_volume", model_name: NAME_54, dataset_path: DATASET.url, chunk_size: [2, 4, 4] });
    expect(ctx.session.state.paintVolume).toBe(id);
    expect(ctx.logged.join("\n")).toMatch(new RegExp(`New annotation volume ${id} for ${NAME_54}`));
  });

  it("refuses as the Python route does: no body, no models loaded, a model not served, no dataset", async () => {
    const empty = harness(fakeContext());
    expect((await empty.handle(request("POST", "/api/finetune/create-volume", null))).status).toBe(400);
    expect(body(await empty.handle(request("POST", "/api/finetune/create-volume", {}))).error).toBe("model_name is required");
    expect(body(await empty.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_54 })))).toEqual({ success: false, error: "No models loaded" });
    const ctx = fakeContext();
    await serve(ctx, entry("setup_56", "nuc", ["nuc"], 128));
    const missing = await harness(ctx).handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_54 }));
    expect(missing.status).toBe(404);
    expect(body(missing)).toEqual({ success: false, error: `Model ${NAME_54} not found` });
    const closed = fakeContext({ dataset: null });
    await serve(closed, entry("setup_54"));
    const noData = await harness(closed).handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_54 }));
    expect(noData.status).toBe(400);
    expect(body(noData).error).toMatch(/No dataset is open/);
    expect(await closed.paint.volumes()).toEqual([]);
  });
});

describe("POST /api/finetune/add-to-viewer", () => {
  it("puts the volume's paint layer in the viewer under annotation_<id>, on the URL the tab sends back, and selects it on the Draw tab", async () => {
    const { ctx, h, id, created } = await ready({ paint: false });
    const res = body(await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: id, minio_url: created.neuroglancer_url })));
    expect(res).toEqual({ success: true, message: "Layer added to viewer", layer_name: `annotation_${id}` });
    expect(paintLayerName(id)).toBe(`annotation_${id}`);
    const layer = ctx.viewer.layers[`annotation_${id}`];
    expect(layer).toMatchObject({
      type: "segmentation",
      source: { url: `s3+http://localhost/paint/${id}/annotation`, subsources: { default: { writingEnabled: true }, bounds: {} } },
      toolBindings: { A: "vox-brush", F: "vox-flood-fill" },
      tab: "Draw",
    });
    expect(ctx.viewer.selected).toEqual([{ name: `annotation_${id}`, tab: "Draw" }]);
  });

  it("takes a plain http URL into the fork's s3+ scheme, and the volume's own URL when none is sent", async () => {
    const { ctx, h, id } = await ready({ paint: false });
    await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: id, minio_url: `http://localhost/paint/${id}/annotation` }));
    expect((ctx.viewer.layers[`annotation_${id}`].source as any).url).toBe(`s3+http://localhost/paint/${id}/annotation`);
    await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: id }));
    expect((ctx.viewer.layers[`annotation_${id}`].source as any).url).toBe(`s3+http://localhost/paint/${id}/annotation`);
  });

  it("answers 404 for a volume the browser has not, and 400 for an id with a slash", async () => {
    const { h } = await ready({ paint: false });
    const res = await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: "vol-nope", minio_url: "x" }));
    expect(res.status).toBe(404);
    expect(body(res).success).toBe(false);
    expect((await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: "a/b" }))).status).toBe(400);
  });
});

describe("the existing volumes", () => {
  it("lists every volume in the store as a session, newest first, with its painted chunks and dataset, skipping what is not a volume", async () => {
    const { ctx, h, id } = await ready();
    await paint(ctx, id, [1, 1, 1]);
    const second = body(await h.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_56 }))).volume_id;
    await ctx.paint.put("junk", ".zattrs", encoder.encode('{"type": "something_else"}'));
    const res = body(await h.handle(request("POST", "/api/finetune/list-existing-sessions", { output_path: "/ignored" })));
    expect(res.success).toBe(true);
    expect(res.sessions).toEqual([
      expect.objectContaining({ session_id: second, session_path: second, volumes: [{ volume_id: second, path: `paint/${second}` }], chunk_count: 0, dataset_path: DATASET.url, model_name: NAME_56 }),
      expect.objectContaining({ session_id: id, session_path: id, volumes: [{ volume_id: id, path: `paint/${id}` }], chunk_count: 2, dataset_path: DATASET.url, model_name: NAME_54 }),
    ]);
    expect(res.sessions[0].created_at > res.sessions[1].created_at).toBe(true);
  });

  it("resumes a volume as the session's, in place, with its painted chunks counted and its record as metadata", async () => {
    const { ctx, h, id } = await ready();
    await ctx.session.update({ paintVolume: null });
    const res = body(await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: id, output_path: "/ignored", load_id: "abc" })));
    expect(res).toMatchObject({
      success: true,
      volume_id: id,
      new_session_path: id,
      zarr_path: `paint/${id}`,
      minio_url: `s3+http://localhost/paint/${id}`,
      neuroglancer_url: `s3+http://localhost/paint/${id}/annotation`,
      copied_count: 0,
      copied_minio: false,
      painted_chunk_count: 1,
      skipped_chunk_extracts: 0,
    });
    expect(res.metadata).toMatchObject({ id, type: "annotation_volume", model_name: NAME_54 });
    expect(ctx.session.state.paintVolume).toBe(id);
    expect(body(await h.handle(request("GET", "/api/finetune/load-existing-volume-progress?load_id=abc")))).toEqual({ success: true, progress: { phase: "done", done: true } });
  });

  it("refuses, with a 409, a volume painted on another dataset than the open one", async () => {
    const { ctx, h, id } = await ready();
    await ctx.session.update({ datasetUrl: "https://example.org/other.zarr", paintVolume: null });
    const res = await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: id }));
    expect(res.status).toBe(409);
    expect(body(res).error).toBe(`That session was painted on ${DATASET.url}, and this dashboard has https://example.org/other.zarr open. Open ${DATASET.url} to resume it.`);
    expect(ctx.session.state.paintVolume).toBeNull();
    // the same dataset under another spelling is the same dataset
    await ctx.session.update({ datasetUrl: `${DATASET.url}/s0` });
    expect(body(await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: id }))).success).toBe(true);
  });

  it("answers 404 for a volume the browser has not, and 400 without an id", async () => {
    const { h } = await ready();
    expect((await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: "vol-nope" }))).status).toBe(404);
    expect((await h.handle(request("POST", "/api/finetune/load-existing-volume", {}))).status).toBe(400);
  });

  it("tells the tab which volume the session paints into, with its paint counted, or none", async () => {
    const { ctx, h, id } = await ready();
    const res = body(await h.handle(request("GET", "/api/finetune/session-volume")));
    expect(res).toMatchObject({ success: true, volume_id: id, painted_chunk_count: 1 });
    expect(res.volume).toMatchObject({ id, type: "annotation_volume", model_name: NAME_54 });
    await ctx.session.update({ paintVolume: null });
    expect(body(await h.handle(request("GET", "/api/finetune/session-volume")))).toEqual({ success: true, volume_id: null, painted_chunk_count: 0 });
    await ctx.session.update({ paintVolume: "vol-gone" });
    expect(body(await h.handle(request("GET", "/api/finetune/session-volume"))).volume_id).toBeNull();
  });

  it("knows one dataset under its spellings", () => {
    expect(sameDataset("https://x/em.zarr/", "https://x/em.zarr")).toBe(true);
    expect(sameDataset("https://x/em.zarr/s2", "https://x/em.zarr")).toBe(true);
    expect(sameDataset("https://x/em.zarr", "https://x/other.zarr")).toBe(false);
  });

  it("counts the session volume's painted chunks for sync-annotations and refresh-annotated-regions", async () => {
    const { ctx, h, id } = await ready();
    await paint(ctx, id, [0, 0, 1]);
    expect(body(await h.handle(request("POST", "/api/finetune/sync-annotations", { force: true })))).toEqual({ success: true, message: "Synced 2 annotations", synced_count: 2 });
    expect(body(await h.handle(request("POST", "/api/finetune/refresh-annotated-regions", {})))).toEqual({ success: true, count: 2 });
    await ctx.session.update({ paintVolume: null });
    expect(body(await h.handle(request("POST", "/api/finetune/sync-annotations", {}))).message).toBe("Synced 0 annotations");
  });
});

describe("POST /api/finetune/submit", () => {
  it("refuses, with needs_volume, when the session has no annotation volume", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"));
    const res = await harness(ctx).handle(request("POST", "/api/finetune/submit", submitBody()));
    expect(res.status).toBe(400);
    expect(body(res)).toMatchObject({ success: false, needs_volume: true });
    expect(body(res).error).toMatch(/No annotation volume/);
  });

  it("refuses a volume with nothing painted, a loss that needs a server, a model not served, and a bad epoch count", async () => {
    const { h, ctx, id } = await ready({ paint: false });
    const unpainted = await h.handle(request("POST", "/api/finetune/submit", submitBody()));
    expect(unpainted.status).toBe(400);
    expect(body(unpainted).error).toMatch(/Nothing is painted/);
    expect("needs_volume" in body(unpainted)).toBe(false);
    await paint(ctx, id, [0, 0, 0]);
    const dice = await h.handle(request("POST", "/api/finetune/submit", submitBody({ loss_type: "dice" })));
    expect(dice.status).toBe(400);
    expect(body(dice).error).toMatch(/dice loss needs a server: Dice is an overlap over each training patch/);
    expect(body(dice).error).toMatch(/BCE \(bce\), MSE \(mse\) or margin \(margin\)/);
    expect(body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ loss_type: "interval" })))).error).toMatch(/interval loss needs a server: the browser makes only the binary and affinity targets/);
    const missing = await h.handle(request("POST", "/api/finetune/submit", submitBody({ model_name: "nowhere" })));
    expect(missing.status).toBe(404);
    expect(body(missing).error).toBe("Model nowhere not found");
    expect(body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ num_epochs: 0 })))).error).toMatch(/num_epochs/);
    expect(h.jobs.list()).toEqual([]);
  });
});

describe("a job's life", () => {
  it("answers the submit as training.py does, with the output type by the channels", async () => {
    const { h, id } = await ready();
    const res = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    expect(res).toMatchObject({ success: true, lsf_job_id: null, output_dir: "this browser (IndexedDB)", output_type: "affinities", message: "Finetuning job submitted successfully" });
    expect(res.job_id).toMatch(/^ft-1-\d{8}-\d{6}$/);
    expect(res.note).toMatch(/head/);
    await h.jobs.get(res.job_id)!.settled();
    // a binary model, on the same volume
    const nuc = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ model_name: NAME_56 }))));
    expect(nuc.output_type).toBe("binary");
    expect(nuc.job_id).toMatch(/^ft-2-/);
    await h.jobs.get(nuc.job_id)!.settled();
    expect(body(await h.handle(request("GET", "/api/finetune/jobs"))).jobs.map((j: any) => j.job_id)).toEqual([res.job_id, nuc.job_id]);
    expect(h.jobs.get(res.job_id)!.params.corrections_path).toBe(id);
  });

  it("runs the model on the volume with the form's options, and reports as job-card.js reads it once served", async () => {
    const { ctx, h, id, calls, results } = await ready();
    const base = ctx.infer.served().find((s) => s.entry.name === NAME_54)!.entry;
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await h.jobs.get(job_id)!.settled();
    expect(calls).toHaveLength(1);
    const run = calls[0];
    expect(run.entry).toBe(base); // the served entry, whose variant the layer shows
    expect(run.volume.id).toBe(id);
    expect(run.chain).toEqual(ctx.session.state.chain);
    expect(run.dataset).toEqual(DATASET);
    expect(run.paint).toBe(ctx.paint);
    expect(run.infer).toBe(ctx.infer);
    expect(run.options).toEqual({
      epochs: 3, learningRate: 0.01, lossType: "bce", margin: 0.3, balanceClasses: true, labelSmoothing: 0, batchPixels: 4096,
      distillationLambda: 0.01, distillationScope: "unlabeled", seed: 0,
    });
    expect(run.goodRegions).toBeUndefined();
    expect(run.initialHead).toBeUndefined();
    expect(run.baseBytes.byteLength).toBe(8); // the fake catalog's base model

    const s = await status(h, job_id);
    expect(s).toMatchObject({
      success: true,
      job_id,
      status: "WAITING_FOR_RESTART",
      model_name: NAME_54,
      current_epoch: 3,
      total_epochs: 3,
      loss_epoch: 3,
      inference_server_ready: true,
      finetuned_model_name: `${NAME_54}-ft1`,
      corrections_path: id,
      lsf_job_id: null,
      log_file: null,
      model_yaml_path: null,
      inference_server_url: null,
      output_dir: "this browser (IndexedDB)",
      progress_percent: 100,
    });
    expect(s.loss).toBeCloseTo(0.5 / 3, 6);
    expect("latest_loss" in s).toBe(false);
    expect(s.params).toEqual({
      model_name: NAME_54,
      corrections_path: id,
      num_epochs: 3,
      learning_rate: 0.01,
      loss_type: "bce",
      balance_classes: true,
      label_smoothing: 0,
      batch_pixels: 4096,
      distillation_lambda: 0.01,
      distillation_scope: "unlabeled",
      finetune_mode: "head",
      auto_serve: true,
      output_type: "affinities",
    });
    expect(Date.parse(s.created_at)).not.toBeNaN();

    // the log, whole lines with the byte offset after them
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`)));
    expect(logs.success).toBe(true);
    expect(logs.logs).toContain("Epoch 1/3 - Loss: 0.500000\n");
    expect(logs.logs).toContain("Epoch 3/3 - Loss: 0.166667\n");
    expect(logs.logs).toMatch(new RegExp(`Finetuned model ${NAME_54}-ft1 kept in this browser`));
    expect(logs.logs).toMatch(new RegExp(`Serving ${NAME_54}-ft1 beside ${NAME_54}`));
    expect(logs.offset).toBe(encoder.encode(logs.logs).byteLength);
    expect(results).toHaveLength(1);
  });

  it("keeps the result as a finetuned model and, with Auto-serve, serves it beside the base in the session", async () => {
    const { ctx, h, id, results } = await ready();
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await h.jobs.get(job_id)!.settled();
    const name = `${NAME_54}-ft1`;
    expect(ctx.catalog.finetunedRecords).toHaveLength(1);
    const record = ctx.catalog.finetunedRecords[0];
    expect(record).toMatchObject({
      id: `finetuned/${name}`,
      name,
      base: { id: "cellmap/onnx-web-models/setup_54", name: NAME_54, file: "setup_54_fp32.onnx" },
      volume: id,
      epochs: 3,
      entry: { id: `finetuned/${name}`, name, repo: "finetuned" },
    });
    expect(record.final_loss).toBeCloseTo(0.5 / 3, 6);
    expect(record.params.output_type).toBe("affinities");
    expect(ctx.catalog.finetunedBytes.get(record.entry.onnxUrl)).toBe(results[0].bytes);
    // served right after the base, which stays, the other model untouched
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, name, NAME_56]);
    expect(ctx.infer.served().map((s) => s.entry.name).sort()).toEqual([NAME_54, name, NAME_56].sort());
    expect(ctx.viewer.removed).not.toContain(NAME_54);
    expect(Object.keys(ctx.viewer.layers)).toContain(name);
    expect(ctx.logged.join("\n")).toMatch(new RegExp(`Finetuned model ${name} kept`));
  });

  it("completes without serving when Auto-serve is off", async () => {
    const { ctx, h } = await ready();
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ auto_serve: false }))));
    await h.jobs.get(job_id)!.settled();
    const s = await status(h, job_id);
    expect(s).toMatchObject({ status: "COMPLETED", inference_server_ready: false, finetuned_model_name: `${NAME_54}-ft1`, current_epoch: 3 });
    expect(s.params.auto_serve).toBe(false);
    expect(ctx.catalog.finetunedRecords.map((r) => r.name)).toEqual([`${NAME_54}-ft1`]);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, NAME_56]);
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_54, NAME_56]);
  });

  it("fails, saying why in the log, when the run throws", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"));
    const failing = harness(ctx, {
      run: async () => {
        throw new Error("no WebGPU device");
      },
    });
    const vol = body(await failing.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_54 }))).volume_id;
    await paint(ctx, vol, [0, 0, 0]);
    const failed = body(await failing.handle(request("POST", "/api/finetune/submit", submitBody()))).job_id;
    await failing.jobs.get(failed)!.settled();
    expect((await status(failing, failed)).status).toBe("FAILED");
    expect(body(await failing.handle(request("GET", `/api/finetune/job/${failed}/logs`))).logs).toContain("Training failed: no WebGPU device\n");
    const stream = openStream(await failing.handle(request("GET", `/api/finetune/job/${failed}/logs/stream?offset=0`)));
    const events: string[] = [];
    for (let e = await stream.next(); e !== null; e = await stream.next()) events.push(e);
    expect(events[events.length - 1]).toBe("event: done\ndata: FAILED\n\n");
    expect(ctx.catalog.finetunedRecords).toEqual([]);
  });

  it("answers 404 for a job it does not have", async () => {
    const { h } = await ready();
    for (const path of ["/api/finetune/job/ft-9/status", "/api/finetune/job/ft-9/logs"]) {
      const res = await h.handle(request("GET", path));
      expect(res.status).toBe(404);
      expect(body(res)).toEqual({ success: false, error: "Job not found" });
    }
    const stream = openStream(await h.handle(request("GET", "/api/finetune/job/ft-9/logs/stream")));
    expect(await stream.next()).toBe("data: Job ft-9 not found\n\n");
    expect(await stream.next()).toBe("event: done\ndata: NOT_FOUND\n\n");
    expect(await stream.next()).toBeNull();
  });
});

describe("GET /api/finetune/job/:id/logs/stream", () => {
  it("sends the log so far with byte-offset ids, then each line as it is logged, and done when the job ends", async () => {
    const baton = new Baton();
    const { h, id } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ auto_serve: false }))));
    await baton.stopped();
    const stream = openStream(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs/stream`)));
    const first = `Job ${job_id}: ${NAME_54} on the paint of ${id}, 3 epoch(s) in this browser`;
    const firstEnd = encoder.encode(first).byteLength + 1;
    expect(await stream.next()).toBe(`id: ${firstEnd}\ndata: ${first}\n\n`);
    const second = `Reading the painted chunks of ${id}`;
    const secondEnd = firstEnd + encoder.encode(second).byteLength + 1;
    expect(await stream.next()).toBe(`id: ${secondEnd}\ndata: ${second}\n\n`);
    expect(await stream.soon()).toBe("nothing"); // waiting on the run
    baton.go();
    const epoch = await stream.next();
    expect(epoch).toMatch(/^id: \d+\ndata: Epoch 1\/3 - Loss: 0\.500000\n\n$/);
    expect((await status(h, job_id))).toMatchObject({ status: "RUNNING", current_epoch: 1, loss: 0.5, loss_epoch: 1 });
    await baton.stopped();
    baton.go();
    expect(await stream.next()).toMatch(/Epoch 2\/3/);
    await baton.stopped();
    baton.go();
    expect(await stream.next()).toMatch(/Epoch 3\/3/);
    await h.jobs.get(job_id)!.settled();
    expect(await stream.next()).toMatch(/kept in this browser/);
    expect(await stream.next()).toBe("event: done\ndata: COMPLETED\n\n");
    expect(await stream.next()).toBeNull();
  });

  it("starts after the offset a reconnect or GET …/logs gives, as Last-Event-ID or ?offset=", async () => {
    const { h, id } = await ready();
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ auto_serve: false }))));
    await h.jobs.get(job_id)!.settled();
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`)));
    const first = `Job ${job_id}: ${NAME_54} on the paint of ${id}, 3 epoch(s) in this browser`;
    const firstEnd = encoder.encode(first).byteLength + 1;
    const fromSecond = openStream(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs/stream`, null, { "last-event-id": String(firstEnd) })));
    expect(await fromSecond.next()).toMatch(/^id: \d+\ndata: Reading the painted chunks/);
    const fromEnd = openStream(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs/stream?offset=${logs.offset}`)));
    expect(await fromEnd.next()).toBe("event: done\ndata: COMPLETED\n\n");
  });

  it("stays open while the job waits for a restart, carries the restarted run's lines, and ends on cancel", async () => {
    const { h } = await ready();
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await h.jobs.get(job_id)!.settled();
    const stream = openStream(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs/stream`)));
    const seen: string[] = [];
    const drain = async () => {
      for (let event = await stream.soon(); event !== "nothing" && event !== null; event = await stream.soon()) seen.push(event);
    };
    await drain();
    expect(seen.some((e) => e.includes("event: done"))).toBe(false);
    expect(seen[seen.length - 1]).toMatch(/WAITING_FOR_RESTART/);
    expect(body(await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, { ...FORM, num_epochs: 1 }))).success).toBe(true);
    expect(await stream.next()).toMatch(/^id: \d+\ndata: RESTARTING_TRAINING/);
    await h.jobs.get(job_id)!.settled();
    await drain();
    expect(seen.some((e) => e.includes("event: done"))).toBe(false);
    expect(body(await h.handle(request("POST", `/api/finetune/job/${job_id}/cancel`))).success).toBe(true);
    expect(await stream.next()).toMatch(/Cancelled by the user/);
    expect(await stream.next()).toBe("event: done\ndata: CANCELLED\n\n");
    expect(await stream.next()).toBeNull();
  });
});

describe("cancel, stop early, restart", () => {
  it("cancels a running job at once; the run stops at its next step and nothing of it is kept", async () => {
    const baton = new Baton();
    const { ctx, h, results } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await baton.stopped();
    const stream = openStream(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs/stream`)));
    expect(body(await h.handle(request("POST", `/api/finetune/job/${job_id}/cancel`)))).toEqual({ success: true, message: `Job ${job_id} cancelled` });
    expect((await status(h, job_id)).status).toBe("CANCELLED");
    baton.go();
    await h.jobs.get(job_id)!.settled();
    expect(results).toHaveLength(1); // the fake run returned early, as a run does when cancelled
    expect(ctx.catalog.finetunedRecords).toEqual([]);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, NAME_56]);
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`))).logs as string;
    expect(logs).toContain("Cancelled by the user: the run stops at its next step, and nothing of it is kept.\n");
    expect(logs).not.toContain("Epoch 1/3");
    let last = "";
    for (let e = await stream.next(); e !== null; e = await stream.next()) last = e;
    expect(last).toBe("event: done\ndata: CANCELLED\n\n");
    const again = await h.handle(request("POST", `/api/finetune/job/${job_id}/cancel`));
    expect(again.status).toBe(400);
    expect(body(again).error).toMatch(/already finished with status CANCELLED/);
    expect((await h.handle(request("POST", "/api/finetune/job/ft-9/cancel"))).status).toBe(404);
  });

  it("stops early after the epoch now running, keeping and serving the head trained by then", async () => {
    const baton = new Baton();
    const { ctx, h } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ num_epochs: 5 }))));
    await baton.stopped();
    baton.go(); // epoch 1
    await baton.stopped();
    const res = body(await h.handle(request("POST", `/api/finetune/job/${job_id}/stop-early`)));
    expect(res.success).toBe(true);
    expect(res.message).toMatch(/exit after epoch 2/);
    baton.go(); // epoch 2 trains: the stop takes effect after it
    await baton.stopped();
    baton.go();
    await h.jobs.get(job_id)!.settled();
    const s = await status(h, job_id);
    expect(s).toMatchObject({ status: "WAITING_FOR_RESTART", current_epoch: 2, total_epochs: 5, inference_server_ready: true, finetuned_model_name: `${NAME_54}-ft1` });
    expect(ctx.catalog.finetunedRecords[0].epochs).toBe(2);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, `${NAME_54}-ft1`, NAME_56]);
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`))).logs as string;
    expect(logs).toContain("Stop requested: training ends after epoch 2");
    expect(logs).toContain("Stopped early after epoch 2");
    expect(logs).not.toContain("Epoch 3/5");
    // not training any more: nothing to stop
    const waiting = await h.handle(request("POST", `/api/finetune/job/${job_id}/stop-early`));
    expect(waiting.status).toBe(400);
    expect(body(waiting).error).toMatch(/not training/);
    expect((await h.handle(request("POST", "/api/finetune/job/ft-9/stop-early"))).status).toBe(404);
  });

  it("restarts from the trained head with the form's parameters, under the same job id, replacing the served model", async () => {
    const baton = new Baton();
    const { ctx, h, calls, results } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    const job = h.jobs.get(job_id)!;
    await letRun(baton, job);
    expect(job.status).toBe("WAITING_FOR_RESTART");
    const res = body(await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, { ...FORM, num_epochs: 2, learning_rate: 0.001 })));
    expect(res).toMatchObject({ success: true, job_id, annotations_synced: 0 });
    expect(res.message).toMatch(/Restart request sent/);
    // the card's reset: the marker in the log, the progress from zero
    await baton.stopped();
    expect(job.toDict()).toMatchObject({ status: "RUNNING", current_epoch: 0, total_epochs: 2, latest_loss: null, latest_loss_epoch: null, inference_server_ready: false, finetuned_model_name: `${NAME_54}-ft1` });
    expect(job.lines.some((line) => line.startsWith("RESTARTING_TRAINING"))).toBe(true);
    await letRun(baton, job);
    expect(calls).toHaveLength(2);
    expect(calls[1].initialHead).toEqual({ weight: results[0].weight, bias: results[0].bias });
    expect(calls[1].options).toMatchObject({ epochs: 2, learningRate: 0.001 });
    expect(calls[1].entry.name).toBe(NAME_54); // the base again, with the trained head
    const s = await status(h, job_id);
    expect(s).toMatchObject({ status: "WAITING_FOR_RESTART", current_epoch: 2, total_epochs: 2, loss_epoch: 2, finetuned_model_name: `${NAME_54}-ft2`, inference_server_ready: true });
    expect(s.params).toMatchObject({ model_name: NAME_54, num_epochs: 2, learning_rate: 0.001, output_type: "affinities" });
    // ft2 took ft1's place beside the base, which stays served
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, `${NAME_54}-ft2`, NAME_56]);
    expect(ctx.infer.served().map((x) => x.entry.name).sort()).toEqual([NAME_54, `${NAME_54}-ft2`, NAME_56].sort());
    // ft1 is forgotten, record and bytes: the job keeps its latest model only.
    expect(ctx.catalog.finetunedRecords.map((r) => r.name)).toEqual([`${NAME_54}-ft2`]);
    expect([...ctx.catalog.finetunedBytes.keys()].some((url) => url.includes(`${NAME_54}-ft1`))).toBe(false);
    expect(job.lines.some((line) => line.includes(`${NAME_54}-ft2 replaces ${NAME_54}-ft1, which is removed from this browser.`))).toBe(true);
    // ft1's layer was handed to ft2 (its place and the user's settings on
    // it), not removed and a new one appended.
    expect(ctx.viewer.replacings.at(-1)).toEqual({ [`${NAME_54}-ft2`]: `${NAME_54}-ft1` });
    expect(ctx.viewer.removed).not.toContain(`${NAME_54}-ft1`);
    expect(Object.keys(ctx.viewer.layers)).not.toContain(`${NAME_54}-ft1`);
    expect(results[1].weight).toEqual(results[0].weight.map((w) => w + 1));
  });

  it("keeps an earlier job's model when a new job is submitted: only a restart replaces its own job's model", async () => {
    const baton = new Baton();
    const { ctx, h } = await ready({ baton });
    const first = body(await h.handle(request("POST", "/api/finetune/submit", submitBody()))).job_id;
    await letRun(baton, h.jobs.get(first)!);
    expect((await h.handle(request("POST", `/api/finetune/job/${first}/cancel`))).status).toBe(200);
    const second = body(await h.handle(request("POST", "/api/finetune/submit", submitBody()))).job_id;
    const job = h.jobs.get(second)!;
    await letRun(baton, job);
    expect(job.status).toBe("WAITING_FOR_RESTART");
    expect(ctx.catalog.finetunedRecords.map((r) => r.name).sort()).toEqual([`${NAME_54}-ft1`, `${NAME_54}-ft2`]);
    expect(job.lines.some((line) => line.includes("removed from this browser"))).toBe(false);
  });

  it("refuses a restart of a job not waiting for one, in the Python route's words, and a loss that needs a server", async () => {
    const baton = new Baton();
    const { h } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ auto_serve: false }))));
    await baton.stopped();
    const running = await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, FORM));
    expect(running.status).toBe(409);
    expect(body(running).error).toBe(`Job ${job_id} is in state RUNNING - can only restart a job that is waiting for a restart (its training iteration has finished or diverged)`);
    await letRun(baton, h.jobs.get(job_id)!);
    const completed = await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, FORM));
    expect(completed.status).toBe(409);
    expect(body(completed).error).toMatch(/in state COMPLETED/);
    expect((await h.handle(request("POST", "/api/finetune/job/ft-9/restart", FORM))).status).toBe(404);
    const combined = await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, { ...FORM, loss_type: "combined" }));
    expect(combined.status).toBe(400);
    expect(body(combined).error).toMatch(/combined loss needs a server/);
  });
});

describe("Shift+R over the viewer", () => {
  it("is bound when the tab asks for the jobs, before there is one, once per router, as restart_key.py binds it", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    for (let i = 0; i < 2; i++) expect(body(await h.handle(request("GET", "/api/finetune/jobs")))).toEqual({ success: true, jobs: [] });
    expect(ctx.viewer.bindings.map((b) => [b.key, b.action])).toEqual([["shift+keyr", "finetune-restart"]]);
    expect([RESTART_KEY, RESTART_ACTION]).toEqual(["shift+keyr", "finetune-restart"]);
  });

  it("tells the page, whose job-monitor.js restarts the job; the job itself is left to it, and its status carries no count", async () => {
    const page = new EventTarget();
    const heard: string[] = [];
    page.addEventListener(RESTART_EVENT, (event) => heard.push(event.type));
    const baton = new Baton();
    const { ctx, h } = await ready({ baton, deps: { page: () => page } });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    const job = h.jobs.get(job_id)!;
    await letRun(baton, job);
    expect(job.status).toBe("WAITING_FOR_RESTART");
    await h.handle(request("GET", "/api/finetune/jobs"));

    ctx.viewer.bindings[0].handler();

    expect(heard).toEqual(["cellmap-flow:restart-finetune"]);
    // The form's values are the page's: it sends the restart, as its Restart does.
    expect(job.status).toBe("WAITING_FOR_RESTART");
    // The server's key counts presses in the status (restart_requests) for
    // its page to poll; this one has told the page already.
    expect(await status(h, job_id)).not.toHaveProperty("restart_requests");
  });

  it("does nothing, and does not throw, with no document to tell (as here, in node)", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    await h.handle(request("GET", "/api/finetune/jobs"));
    expect(() => ctx.viewer.bindings[0].handler()).not.toThrow();
  });
});

describe("the head's training controls", () => {
  it("reads the loss, its margin, the batch of voxels and the distillation into the run's options and the job's params", () => {
    const read = readTraining({ ...FORM, loss_type: "margin", margin: 0.2, label_smoothing: 0.1, batch_pixels: 1024, distillation_lambda: 0.5, distillation_scope: "all" });
    if ("error" in read) throw new Error(read.error);
    // The margin loss takes no smoothing, as the server sets it.
    expect(read.options).toMatchObject({ lossType: "margin", margin: 0.2, labelSmoothing: 0, batchPixels: 1024, distillationLambda: 0.5, distillationScope: "all" });
    expect(read.params).toEqual({
      num_epochs: 3, learning_rate: 0.01, loss_type: "margin", margin: 0.2, balance_classes: true, label_smoothing: 0,
      batch_pixels: 1024, distillation_lambda: 0.5, distillation_scope: "all", finetune_mode: "head",
    });
    // The margin goes with the margin loss alone; the server's batch of patches is not the browser's.
    const mse = readTraining({ ...FORM, loss_type: "mse", margin: 0.2, label_smoothing: 0.1 });
    if ("error" in mse) throw new Error(mse.error);
    expect(mse.options).toMatchObject({ lossType: "mse", labelSmoothing: 0.1, batchPixels: 4096 });
    expect("margin" in mse.params).toBe(false);
    // Nothing sent: the defaults, distillation on at the server form's 0.01.
    const bare = readTraining({});
    if ("error" in bare) throw new Error(bare.error);
    expect(bare.options).toMatchObject({ lossType: "bce", distillationLambda: 0.01, distillationScope: "unlabeled", batchPixels: 4096 });
  });

  it("refuses values the trainer would refuse, saying which field", () => {
    const error = (extra: Record<string, unknown>) => (readTraining({ ...FORM, ...extra }) as { error: string }).error;
    expect(error({ batch_pixels: 0 })).toMatch(/^batch_pixels: /);
    expect(error({ batch_pixels: 2.5 })).toMatch(/^batch_pixels: /);
    expect(error({ batch_pixels: 1 << 21 })).toMatch(/^batch_pixels: .* 1048576/);
    expect(error({ distillation_lambda: -1 })).toMatch(/^distillation_lambda: /);
    expect(error({ distillation_lambda: "lots" })).toMatch(/^distillation_lambda: /);
    expect(error({ distillation_scope: "anchor" })).toMatch(/^distillation_scope: expected unlabeled or all/);
    expect(error({ loss_type: "margin", margin: 0.7 })).toMatch(/^margin: /);
    expect(error({ loss_type: "dice" })).toMatch(/dice loss needs a server/);
  });

  it("hands the run the volume's good regions as they are when it starts, and a restart the form's new values", async () => {
    const baton = new Baton();
    const { ctx, h, id, calls } = await ready({ baton });
    const regions = [{ id: "a1b2c3d4", label: "good-1", offset_nm: [0, 0, 0], shape_nm: [16, 32, 32] }];
    await saveGoodRegions(ctx.paint, id, regions);
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ loss_type: "mse", batch_pixels: 2048 }))));
    const job = h.jobs.get(job_id)!;
    await letRun(baton, job);
    expect(calls[0].goodRegions).toEqual(regions);
    expect(calls[0].options).toMatchObject({ lossType: "mse", batchPixels: 2048 });
    // A region marked since, and new values in the form: the restart takes both.
    const more = [...regions, { id: "e5f6a7b8", label: "good-2", offset_nm: [16, 0, 0], shape_nm: [16, 32, 32] }];
    await saveGoodRegions(ctx.paint, id, more);
    const restart = { ...FORM, loss_type: "margin", margin: 0.25, batch_pixels: 512, distillation_lambda: 1, distillation_scope: "all" };
    expect(body(await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, restart))).success).toBe(true);
    await letRun(baton, job);
    expect(calls[1].goodRegions).toEqual(more);
    expect(calls[1].options).toMatchObject({ lossType: "margin", margin: 0.25, batchPixels: 512, distillationLambda: 1, distillationScope: "all" });
    expect(job.params).toMatchObject({ loss_type: "margin", margin: 0.25, batch_pixels: 512, distillation_lambda: 1, distillation_scope: "all" });
  });
});

describe("downloads and deletion", () => {
  /** The names and sizes in a ZIP's central directory. */
  function zipEntries(buffer: ArrayBuffer): { name: string; size: number }[] {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    const end = bytes.length - 22;
    expect(view.getUint32(end, true)).toBe(0x06054b50);
    const count = view.getUint16(end + 10, true);
    let at = view.getUint32(end + 16, true);
    const entries: { name: string; size: number }[] = [];
    for (let i = 0; i < count; i++) {
      expect(view.getUint32(at, true)).toBe(0x02014b50);
      const nameLength = view.getUint16(at + 28, true);
      entries.push({ name: new TextDecoder().decode(bytes.subarray(at + 46, at + 46 + nameLength)), size: view.getUint32(at + 24, true) });
      at += 46 + nameLength + view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
    }
    return entries;
  }

  it("downloads the volume as a zip of its zarr, every file under <id>.zarr/", async () => {
    const { ctx, h, id } = await ready();
    const res = await h.handle(request("GET", `/api/finetune/volume/${id}/download`));
    expect(res.status).toBe(200);
    if (!("body" in res)) throw new Error("not a body answer");
    expect(res.type).toBe("application/zip");
    expect(res.headers).toEqual({ "content-disposition": `attachment; filename="${id}.zarr.zip"` });
    const buffer = res.body as ArrayBuffer;
    expect(buffer).toBeInstanceOf(ArrayBuffer);
    const attrs = (await ctx.paint.get(id, ".zattrs"))!;
    expect(zipEntries(buffer)).toEqual([
      { name: `${id}.zarr/.zattrs`, size: attrs.byteLength },
      { name: `${id}.zarr/.zgroup`, size: expect.any(Number) },
      { name: `${id}.zarr/annotation/.zattrs`, size: expect.any(Number) },
      { name: `${id}.zarr/annotation/.zgroup`, size: expect.any(Number) },
      { name: `${id}.zarr/annotation/s0/.zarray`, size: expect.any(Number) },
      { name: `${id}.zarr/annotation/s0/0.0.0`, size: 32 },
    ]);
    expect((await h.handle(request("GET", "/api/finetune/volume/vol-nope/download"))).status).toBe(404);
  });

  it("downloads a finetuned model's bytes as .onnx, by its name, its manifest written into them", async () => {
    const { ctx, h, results } = await ready();
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await h.jobs.get(job_id)!.settled();
    const res = await h.handle(request("GET", `/api/finetune/model/${NAME_54}-ft1/download`));
    expect(res.status).toBe(200);
    if (!("body" in res)) throw new Error("not a body answer");
    expect(res.type).toBe("application/octet-stream");
    expect(res.headers).toEqual({ "content-disposition": `attachment; filename="${NAME_54}-ft1.onnx"` });
    // The bytes the run wrote, then the manifest the server's onnx type
    // reads: the record's, whose one variant is this file.
    const downloaded = new Uint8Array(res.body as ArrayBuffer), kept = new Uint8Array(results[0].bytes);
    expect(downloaded.subarray(0, kept.length)).toEqual(kept);
    const [record] = await ctx.catalog.finetuned();
    const manifest = JSON.parse(readMetadata(downloaded)[MANIFEST_METADATA_KEY]);
    expect(manifest).toEqual(record.entry.manifest);
    expect(manifest).toMatchObject({ model: `${NAME_54}-ft1`, channels: ["mito_aff_z", "mito_aff_y", "mito_aff_x"], variants: [{ file: `${NAME_54}-ft1.onnx` }] });
    // What the catalog keeps is unchanged: the manifest goes in on the way out.
    expect(ctx.catalog.finetunedBytes.get(record.entry.onnxUrl)).toBe(results[0].bytes);
    const missing = await h.handle(request("GET", `/api/finetune/model/${NAME_54}/download`));
    expect(missing.status).toBe(404);
    expect(body(missing).error).toMatch(/No model finetuned in this browser/);
  });

  it("deletes a volume: its files, its layer, and the session's reference to it", async () => {
    const { ctx, h, id } = await ready();
    const other = body(await h.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_56 }))).volume_id;
    await ctx.session.update({ paintVolume: id });
    expect(body(await h.handle(request("POST", "/api/finetune/delete-volume", { volume_id: id })))).toEqual({ success: true });
    expect(await ctx.paint.list(id)).toEqual([]);
    expect(await ctx.paint.volumes()).toEqual([other]);
    expect(ctx.viewer.removed).toContain(`annotation_${id}`);
    expect(ctx.session.state.paintVolume).toBeNull();
    // another volume's deletion leaves the session's alone
    await ctx.session.update({ paintVolume: other });
    const again = body(await h.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_54 }))).volume_id;
    await ctx.session.update({ paintVolume: other });
    await h.handle(request("POST", "/api/finetune/delete-volume", { volume_id: again }));
    expect(ctx.session.state.paintVolume).toBe(other);
    expect((await h.handle(request("POST", "/api/finetune/delete-volume", { volume_id: "vol-nope" }))).status).toBe(404);
    expect((await h.handle(request("POST", "/api/finetune/delete-volume", { volume_id: "a/b" }))).status).toBe(400);
    expect((await h.handle(request("POST", "/api/finetune/delete-volume", {}))).status).toBe(400);
  });
});

describe("in the backend", () => {
  it("answers the finetune routes, and says what still needs a server", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"));
    const backend = createBackend(ctx, { finetune: fakeRun() });
    expect((await backend.handle(request("GET", "/api/finetune/models"))).status).toBe(200);
    expect((await backend.handle(request("GET", "/api/finetune/job/abc/status"))).status).toBe(404);
    const crops = await backend.handle(request("POST", "/api/finetune/load-crops"));
    expect(crops.status).toBe(503);
    expect(body(crops).error).toMatch(/needs a server .*finetune tab's crops import\)/);
    await tick();
  });
});
