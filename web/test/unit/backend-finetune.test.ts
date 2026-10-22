// The Finetune tab's routes against fake context members, the volume plan
// under numbered ids and a fake run: the JSON each answers, checked against what the tab's
// JavaScript reads (crops.js, sessions.js, models.js, job-monitor.js,
// job-card.js, log-stream.js), and what each does to the paint store, the
// session, the viewer and the catalog. A refusal is checked by its status
// (and needs_volume where the tab acts on it), not by its words.
import { describe, expect, it } from "vitest";

import { createBackend } from "../../src/backend";
import { type FinetuneDeps, finetuneRoutes, RESTART_EVENT } from "../../src/backend/routes/finetune";
import { Router } from "../../src/backend/router";
import type { ApiRequest, ApiResponse, ModelEntry } from "../../src/contracts";
import { MANIFEST_METADATA_KEY } from "../../src/contracts";
import { parseTrainGraph } from "../../src/finetune/full/graph";
import type { Finetuned, FinetuneRun, RunFinetune } from "../../src/finetune/types";
import { CHUNK_DIR, chunkKey, planVolume } from "../../src/finetune/volume";
import { entryFor, parseManifest } from "../../src/models/manifest";
import { DATASET, entry, type FakeContext, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";
import { rawManifest } from "./models-helpers";
import { readMetadata } from "./onnx-metadata";
import { portablePython } from "./pyodide";

// A volume's plan and a region's rules are the portable Python's, here in
// node's Pyodide (pyodide.ts), loaded once for the file before the tests;
// without it (no network for numpy's wheel) they skip.
const python = await portablePython();

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

const encoder = new TextEncoder();
const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

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

/** The finetune routes on their own router, with the real plan under
 * numbered volume ids and a clock one second apart per call. */
function harness(ctx: FakeContext, deps: Partial<FinetuneDeps> = {}) {
  const router = new Router();
  let volumes = 0, second = 0;
  const plan: FinetuneDeps["plan"] = (infer, dataset, e, chain, options = {}) => planVolume(infer, dataset, e, chain, { ...options, id: `vol-${String(++volumes).padStart(8, "0")}-20261015-120000` });
  const jobs = finetuneRoutes(router, { plan, now: () => new Date(2026, 9, 15, 12, 0, second++), ...deps });
  return { handle: (req: ApiRequest) => router.handle(req, ctx), jobs };
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
  };
}

/** A 3D training graph with setup_53's geometry, built here: a context of
 * 88 and block sides of 92 + 8k voxels (a valid 5³ convolution, an 8³ max
 * pool and the transposed convolution back up, then 42 voxels cropped off
 * each side of d, h and w). The route reads only its geometry; the fake run
 * trains nothing. */
const GRAPH_3D = parseTrainGraph({
  format: "cellmap-train-graph/1", layout: "NDHWC", input: "input", output: "logits", context: 88,
  params: { "c.w": { shape: [1, 1, 5, 5, 5], trainable: true }, "up.w": { shape: [1, 1, 8, 8, 8], trainable: true } },
  ops: [
    { op: "conv", x: "input", y: "a", w: "c.w", stride: [1, 1, 1] },
    { op: "maxpool", x: "a", y: "p", k: [8, 8, 8], stride: [8, 8, 8] },
    { op: "conv_t", x: "p", y: "u", w: "up.w", stride: [8, 8, 8] },
    { op: "crop", x: "u", y: "u.d", axis: "d", start: 42, end_margin: 42 },
    { op: "crop", x: "u.d", y: "u.h", axis: "h", start: 42, end_margin: 42 },
    { op: "crop", x: "u.h", y: "logits", axis: "w", start: 42, end_margin: 42 },
  ],
});

describe.skipIf(!python)("GET /api/finetune/models", () => {
  it("offers a 3D model's whole network in its files' block sides with what a step costs, one block a step, and trains the file of the block", async () => {
    // setup_53 served from its 124³ fp32 file, its manifest's full block as on the Hub since 2026-10-17.
    const file = "setup_53_mito_affinities_16nm_in124_fp32.onnx";
    const raw = rawManifest("setup_53") as { finetune: Record<string, unknown> };
    const writeBack = [124, 140, 172].map((n) => `setup_53_mito_affinities_16nm_in${n}_fp32.onnx`);
    const full = { library: "tfjs", files: ["finetune/train_graph.json"], weight_map: "finetune/weight_map.json", torch_params: "finetune/torch_params.json", initializers: "finetune/initializers.json", write_back: writeBack, frozen: [] };
    const manifest = parseManifest({ ...raw, finetune: { ...raw.finetune, full } });
    const e53 = entryFor("setup_53", manifest, manifest.variants.find((v) => v.file === file)!);
    const ctx = fakeContext();
    const { run, calls } = fakeRun();
    const h = harness(ctx, { run, fullFiles: async () => ({ graph: GRAPH_3D, initializers: {}, torchParams: {}, weightMap: {} }), webgpuMissing: async () => null });
    await serve(ctx, e53);
    const [model] = body(await h.handle(request("GET", "/api/finetune/models"))).models;
    expect(model.finetune_modes).toEqual(["head", "full"]);
    const training = model.full_training;
    expect(training).toMatchObject({ dims: 3, default_tile: 124, context: 88, unavailable: null });
    expect(training.tiles.map((t: any) => t.side)).toEqual([124, 140, 172]);
    for (const t of training.tiles) expect(t.output).toBe(t.side - 88);
    expect(training.tiles.find((t: any) => t.side === 124)).toMatchObject({ measured: true, step_ms: expect.any(Number), peak_mb: expect.any(Number) });

    // Submit: a side the graph takes, one block a step.
    await h.handle(request("POST", "/api/finetune/create-volume", { model_name: e53.name }));
    await paint(ctx, ctx.session.state.paintVolume!, [0, 0, 0]);
    const submit = (extra: Record<string, unknown>) => h.handle(request("POST", "/api/finetune/submit", { ...FORM, model_name: e53.name, finetune_mode: "full", ...extra })).then(body);
    expect((await submit({ batch_size: 2, tile_size: 124 })).success).toBe(false);
    expect((await submit({ batch_size: 1, tile_size: 125 })).success).toBe(false);
    const submitted = await submit({ batch_size: 1, tile_size: 140 });
    expect(submitted.success).toBe(true);
    expect(calls.at(-1)!.options).toMatchObject({ mode: "full", batchPatches: 1, tileSize: 140 });
    // From the 140³ file, not the 124³ one served: the finetune is served at the block it trained on,
    // and a restart on another block, whose file the trained network is not in, is a new job.
    expect(calls.at(-1)!.entry.variant.file).toBe(writeBack[1]);
    await h.jobs.get(submitted.job_id)!.settled();
    expect((await ctx.catalog.finetuned())[0].entry.variant.source?.file).toBe(writeBack[1]);
    const restart = await h.handle(request("POST", `/api/finetune/job/${submitted.job_id}/restart`, { ...FORM, finetune_mode: "full", batch_size: 1, tile_size: 172 }));
    expect(restart).toMatchObject({ status: 409 });
    expect(body(restart).error).toMatch(/new job/);
    // The job card shows the side as a block's.
    const [job] = body(await h.handle(request("GET", "/api/finetune/jobs"))).jobs;
    expect(job.params).toMatchObject({ block_size: 140 });
    expect(job.params.tile_size).toBeUndefined();
  });
});

describe.skipIf(!python)("the existing volumes", () => {
  it("resumes a volume as the session's, in place, with its painted chunks counted and its record as metadata", async () => {
    const { ctx, h, id } = await ready();
    await ctx.session.update({ paintVolume: null });
    const res = body(await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: id, output_path: "/ignored", load_id: "abc" })));
    expect(res).toMatchObject({
      success: true,
      volume_id: id,
      new_session_path: id,
      neuroglancer_url: `s3+http://localhost/paint/${id}/annotation`,
      painted_chunk_count: 1,
    });
    expect(res.metadata).toMatchObject({ id, type: "annotation_volume", model_name: NAME_54 });
    expect(ctx.session.state.paintVolume).toBe(id);
    expect(body(await h.handle(request("GET", "/api/finetune/load-existing-volume-progress?load_id=abc"))).progress).toMatchObject({ done: true });
  });

  it("refuses, with a 409, a volume painted on another dataset than the open one", async () => {
    const { ctx, h, id } = await ready();
    await ctx.session.update({ datasetUrl: "https://example.org/other.zarr", paintVolume: null });
    const res = await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: id }));
    expect(res.status).toBe(409);
    expect(body(res).success).toBe(false);
    expect(ctx.session.state.paintVolume).toBeNull();
    // the same dataset under another spelling is the same dataset
    await ctx.session.update({ datasetUrl: `${DATASET.url}/s0` });
    expect(body(await h.handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: id }))).success).toBe(true);
  });
});

describe.skipIf(!python)("POST /api/finetune/submit", () => {
  it("refuses, with needs_volume, when the session has no annotation volume, and without it when nothing is painted", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"));
    const res = await harness(ctx).handle(request("POST", "/api/finetune/submit", submitBody()));
    expect(res.status).toBe(400);
    expect(body(res)).toMatchObject({ success: false, needs_volume: true });
    const { h } = await ready({ paint: false });
    const unpainted = await h.handle(request("POST", "/api/finetune/submit", submitBody()));
    expect(unpainted.status).toBe(400);
    expect("needs_volume" in body(unpainted)).toBe(false);
    expect(h.jobs.list()).toEqual([]);
  });
});

describe.skipIf(!python)("a job's life", () => {
  it("runs the model on the volume with the form's options, and reports as job-card.js reads it once served", async () => {
    const { ctx, h, id, calls } = await ready();
    const base = ctx.infer.served().find((s) => s.entry.name === NAME_54)!.entry;
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await h.jobs.get(job_id)!.settled();
    expect(calls).toHaveLength(1);
    const run = calls[0];
    expect(run.entry).toBe(base); // the served entry, whose variant the layer shows
    expect(run.volume.id).toBe(id);
    expect(run.chain).toEqual(ctx.session.state.chain);
    expect(run.options).toMatchObject({ mode: "head", epochs: 3, learningRate: 0.01, lossType: "bce", balanceClasses: true });

    const s = await status(h, job_id);
    expect(s).toMatchObject({
      success: true,
      job_id,
      status: "WAITING_FOR_RESTART",
      model_name: NAME_54,
      current_epoch: 3,
      total_epochs: 3,
      inference_server_ready: true,
      finetuned_model_name: `${NAME_54}-ft1`,
      corrections_path: id,
      progress_percent: 100,
    });
    expect(s.loss).toBeCloseTo(0.5 / 3, 6);
    expect(s.params).toMatchObject({ model_name: NAME_54, num_epochs: 3, learning_rate: 0.01, output_type: "affinities" });

    // the log, whole lines with the byte offset after them
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`)));
    expect(logs.logs).toContain("Epoch 3/3 - Loss: 0.166667\n");
    expect(logs.offset).toBe(encoder.encode(logs.logs).byteLength);
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
    expect(body(await failing.handle(request("GET", `/api/finetune/job/${failed}/logs`))).logs).toContain("no WebGPU device");
    expect(ctx.catalog.finetunedRecords).toEqual([]);
  });
});

describe.skipIf(!python)("GET /api/finetune/job/:id/logs/stream", () => {
  it("sends the log so far with byte-offset ids, then each line as it is logged, and done when the job ends", async () => {
    const baton = new Baton();
    const { h, id } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ auto_serve: false }))));
    await baton.stopped();
    const stream = openStream(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs/stream`)));
    // Each event's id is the byte offset just past its line.
    const [, end, first] = /^id: (\d+)\ndata: (.*)\n\n$/.exec((await stream.next())!)!;
    const firstEnd = encoder.encode(first).byteLength + 1;
    expect(Number(end)).toBe(firstEnd);
    const second = `Reading the painted chunks of ${id}`;
    expect(await stream.next()).toBe(`id: ${firstEnd + encoder.encode(second).byteLength + 1}\ndata: ${second}\n\n`);
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
});

describe.skipIf(!python)("cancel, stop early, restart", () => {
  it("cancels a running job at once; the run stops at its next step and nothing of it is kept", async () => {
    const baton = new Baton();
    const { ctx, h, results } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    await baton.stopped();
    expect(body(await h.handle(request("POST", `/api/finetune/job/${job_id}/cancel`))).success).toBe(true);
    expect((await status(h, job_id)).status).toBe("CANCELLED");
    baton.go();
    await h.jobs.get(job_id)!.settled();
    expect(results).toHaveLength(1); // the fake run returned early, as a run does when cancelled
    expect(ctx.catalog.finetunedRecords).toEqual([]);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, NAME_56]);
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`))).logs as string;
    expect(logs).not.toContain("Epoch 1/3");
    // already over: nothing to cancel
    expect((await h.handle(request("POST", `/api/finetune/job/${job_id}/cancel`))).status).toBe(400);
  });

  it("stops early after the epoch now running, keeping and serving the head trained by then", async () => {
    const baton = new Baton();
    const { ctx, h } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody({ num_epochs: 5 }))));
    await baton.stopped();
    baton.go(); // epoch 1
    await baton.stopped();
    expect(body(await h.handle(request("POST", `/api/finetune/job/${job_id}/stop-early`))).success).toBe(true);
    baton.go(); // epoch 2 trains: the stop takes effect after it
    await baton.stopped();
    baton.go();
    await h.jobs.get(job_id)!.settled();
    const s = await status(h, job_id);
    expect(s).toMatchObject({ status: "WAITING_FOR_RESTART", current_epoch: 2, total_epochs: 5, inference_server_ready: true, finetuned_model_name: `${NAME_54}-ft1` });
    expect(ctx.catalog.finetunedRecords[0].epochs).toBe(2);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, `${NAME_54}-ft1`, NAME_56]);
    const logs = body(await h.handle(request("GET", `/api/finetune/job/${job_id}/logs`))).logs as string;
    expect(logs).not.toContain("Epoch 3/5");
    // not training any more: nothing to stop
    expect((await h.handle(request("POST", `/api/finetune/job/${job_id}/stop-early`))).status).toBe(400);
  });

  it("restarts from the trained head with the form's parameters, under the same job id, replacing the served model", async () => {
    const baton = new Baton();
    const { ctx, h, calls, results } = await ready({ baton });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    const job = h.jobs.get(job_id)!;
    await letRun(baton, job);
    expect(job.status).toBe("WAITING_FOR_RESTART");
    const res = body(await h.handle(request("POST", `/api/finetune/job/${job_id}/restart`, { ...FORM, num_epochs: 2, learning_rate: 0.001 })));
    expect(res).toMatchObject({ success: true, job_id });
    // the card's reset: the marker in the log, the progress from zero
    await baton.stopped();
    expect(job.toDict()).toMatchObject({ status: "RUNNING", current_epoch: 0, total_epochs: 2, latest_loss: null, inference_server_ready: false });
    expect(job.lines.some((line) => line.startsWith("RESTARTING_TRAINING"))).toBe(true);
    await letRun(baton, job);
    expect(calls).toHaveLength(2);
    expect(calls[1].initialHead).toEqual({ weight: results[0].weight, bias: results[0].bias });
    expect(calls[1].options).toMatchObject({ epochs: 2, learningRate: 0.001 });
    expect(calls[1].entry.name).toBe(NAME_54); // the base again, with the trained head
    const s = await status(h, job_id);
    expect(s).toMatchObject({ status: "WAITING_FOR_RESTART", current_epoch: 2, total_epochs: 2, finetuned_model_name: `${NAME_54}-ft2`, inference_server_ready: true });
    // ft2 took ft1's place beside the base, which stays served
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, `${NAME_54}-ft2`, NAME_56]);
    expect(ctx.infer.served().map((x) => x.entry.name).sort()).toEqual([NAME_54, `${NAME_54}-ft2`, NAME_56].sort());
    // ft1 is forgotten, record and bytes: the job keeps its latest model only.
    expect(ctx.catalog.finetunedRecords.map((r) => r.name)).toEqual([`${NAME_54}-ft2`]);
    expect([...ctx.catalog.finetunedBytes.keys()].some((url) => url.includes(`${NAME_54}-ft1`))).toBe(false);
    // ft1's layer was handed to ft2 (its place and the user's settings on
    // it), not removed and a new one appended.
    expect(ctx.viewer.replacings.at(-1)).toEqual({ [`${NAME_54}-ft2`]: `${NAME_54}-ft1` });
    expect(ctx.viewer.removed).not.toContain(`${NAME_54}-ft1`);
    expect(Object.keys(ctx.viewer.layers)).not.toContain(`${NAME_54}-ft1`);
  });
});

describe.skipIf(!python)("Shift+R over the viewer", () => {
  it("is bound once the tab asks for the jobs, and tells the page, whose job-monitor.js restarts the job; the job itself is left to it", async () => {
    const page = new EventTarget();
    const heard: string[] = [];
    page.addEventListener(RESTART_EVENT, (event) => heard.push(event.type));
    const baton = new Baton();
    const { ctx, h } = await ready({ baton, deps: { page: () => page } });
    const { job_id } = body(await h.handle(request("POST", "/api/finetune/submit", submitBody())));
    const job = h.jobs.get(job_id)!;
    await letRun(baton, job);
    expect(job.status).toBe("WAITING_FOR_RESTART");
    for (let i = 0; i < 2; i++) await h.handle(request("GET", "/api/finetune/jobs"));
    expect(ctx.viewer.bindings.map((b) => [b.key, b.action])).toEqual([["shift+keyr", "finetune-restart"]]);

    ctx.viewer.bindings[0].handler();

    expect(heard).toEqual(["cellmap-flow:restart-finetune"]);
    // The form's values are the page's: it sends the restart, as its Restart does.
    expect(job.status).toBe("WAITING_FOR_RESTART");
  });
});

describe.skipIf(!python)("downloads and deletion", () => {
  it("downloads the volume as a zip of its zarr, every file under <id>.zarr/", async () => {
    const { h, id } = await ready();
    const res = await h.handle(request("GET", `/api/finetune/volume/${id}/download`));
    expect(res.status).toBe(200);
    if (!("body" in res)) throw new Error("not a body answer");
    expect(res.type).toBe("application/zip");
    expect(res.headers).toEqual({ "content-disposition": `attachment; filename="${id}.zarr.zip"` });
    // A stored ZIP (finetune-zip.test.ts reads one back field by field) names each file in the clear.
    const text = new TextDecoder().decode(res.body as ArrayBuffer);
    for (const path of [".zattrs", ".zgroup", "annotation/.zattrs", "annotation/s0/.zarray", "annotation/s0/0.0.0"]) expect(text).toContain(`${id}.zarr/${path}`);
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
    expect((await h.handle(request("GET", `/api/finetune/model/${NAME_54}/download`))).status).toBe(404);
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
    expect((await h.handle(request("POST", "/api/finetune/delete-volume", { volume_id: "vol-nope" }))).status).toBe(404);
    expect((await h.handle(request("POST", "/api/finetune/delete-volume", { volume_id: "a/b" }))).status).toBe(400);
  });
});

describe.skipIf(!python)("in the backend", () => {
  it("answers the finetune routes", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"));
    const backend = createBackend(ctx, { finetune: fakeRun() });
    expect((await backend.handle(request("GET", "/api/finetune/models"))).status).toBe(200);
    expect((await backend.handle(request("GET", "/api/finetune/job/abc/status"))).status).toBe(404);
    await tick();
  });
});
