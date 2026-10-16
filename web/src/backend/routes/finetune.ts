// The Finetune tab's routes (dashboard/routes/finetune/annotation_core.py,
// annotation_sessions.py, overlay.py, training.py), answered in the page:
// the models to annotate for, a new annotation volume and its paint layer,
// earlier volumes resumed, the training jobs (finetune-jobs.ts) with their
// status, log and log stream, and what the server has no need of: the
// volume downloaded as a zarr (zip) and the finetuned model as .onnx, since
// here they exist only in this browser's IndexedDB, and a volume deleted.
//
// What stands in for the server's pieces: the paint store (ctx.paint,
// served to the viewer by the service worker) for MinIO and the corrections
// directory, so a "session" is a volume and its id is its path; the catalog
// for the models directory; a run in the tab (src/finetune/run.ts) for the
// cluster job, so a submit needs no output path, queue or charge group, and
// the server-only fields the form still sends (LoRA rank, the batch of
// patches, patches per epoch, rehearsal, augmentation) are ignored. The
// head trains with the server's per-voxel losses (BCE, MSE, margin), its
// distillation term (λ and scope) and the browser's own batch of voxels
// per step (readTraining); Dice needs a whole patch per sample and is
// refused (refuseLoss). The good regions' routes are routes/good-regions.ts;
// crops import and seeds keep needing a server and stay unavailable
// (index.ts). One route is the browser's own, GET
// /api/finetune/session-volume: which volume the session paints into, for
// the tab's Download and Delete buttons.
//
// The answers keep the Python routes' keys, so the tab's JavaScript
// (static/js/dashboard/finetune/*.js) reads them unchanged: a refusal is
// {success: false, error} at the Python's status, with `needs_volume` where
// requests.js's offerNewVolume keys on it.
import type { AppContext, LayerSpec, PaintVolume } from "../../contracts";
import { PAINT_ROUTE } from "../../contracts";
import { runFinetune } from "../../finetune/run";
import { HEAD_LOSSES } from "../../finetune/head-trainer";
import type { DistillationScope, FinetuneOptions, HeadLoss, RunFinetune } from "../../finetune/types";
import { DEFAULT_OPTIONS } from "../../finetune/types";
import { createVolume, hasAffinityChannels, paintedChunkIndices, planVolume, readVolume } from "../../finetune/volume";
import { volumeZip, VOLUME_ZIP_SUFFIX } from "../../finetune/zip";
import { PAINT_TAB, paintLayer, paintLayerName, paintLayerUrl } from "../../viewer/layers";
import { FinetuneJob, FinetuneJobs } from "../finetune-jobs";
import type { Router } from "../router";
import { badRequest, json, message, objectBody, text } from "../router";
import { pageBase } from "../serving";
import { done, eventStream, requestedOffset, type SseEvent } from "../sse";

/** The seams the routes run through, so they can be tested with a fake run
 * (run.ts needs onnxruntime) and a volume plan of a known shape. */
export interface FinetuneDeps {
  run: RunFinetune;
  plan: typeof planVolume;
  now: () => Date;
}

const NO_DATASET = "No dataset is open. Enter its URL in the viewer panel first; the volume is painted on it.";

/** The browser's batch of voxels per step, at most: a step's arithmetic and
 * buffers grow with it, and a million voxels is already every painted voxel
 * of any session in one step. */
const MAX_BATCH_PIXELS = 1 << 20;

/** Register the routes; returns the jobs, which live as long as the router (the page). */
export function finetuneRoutes(router: Router, deps: Partial<FinetuneDeps> = {}): FinetuneJobs {
  const d: FinetuneDeps = { run: runFinetune, plan: planVolume, now: () => new Date(), ...deps };
  const jobs = new FinetuneJobs({ run: d.run, now: d.now });

  // GET /api/finetune/models: the served models, with the geometry the tab
  // shows (write_shape in nm, the voxel size, the channels), from the
  // engine's model info as the Python asks each inference server. The one
  // way to finetune here is the head; models.js explains the word.
  router.get("/api/finetune/models", (_req, _params, ctx) => {
    const models = ctx.infer.served().map((s) => ({
      name: s.entry.name,
      write_shape: s.info.write_shape,
      output_voxel_size: s.info.output_voxel_size,
      output_channels: s.info.output_channels,
      finetune_modes: ["head"],
    }));
    return json(200, { models, selected_model: models.length === 1 ? models[0].name : null });
  });

  // POST /api/finetune/create-volume {model_name}: a new volume over the
  // open dataset on the model's output grid (planVolume), its zarr files in
  // the paint store, and the session's volume from now on. The answer's
  // URLs are the paint route's (volumeUrls); the tab then calls
  // add-to-viewer with neuroglancer_url, as it does on the server.
  router.post("/api/finetune/create-volume", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.model_name === "string" ? body.model_name.trim() : "";
    if (!name) return badRequest("model_name is required");
    const served = ctx.infer.served();
    if (!served.length) return badRequest("No models loaded");
    const model = served.find((s) => s.entry.name === name);
    if (!model) return json(404, { success: false, error: `Model ${name} not found` });
    const dataset = ctx.infer.dataset();
    if (!dataset) return badRequest(NO_DATASET);
    let volume: PaintVolume;
    try {
      volume = d.plan(dataset, model.entry, ctx.session.state.chain, { now: d.now() });
    } catch (e) {
      return badRequest(message(e));
    }
    await createVolume(ctx.paint, volume);
    await ctx.session.update({ paintVolume: volume.id });
    ctx.log(`New annotation volume ${volume.id} for ${name}: ${volume.dataset_shape_voxels.join("×")} voxels in ${volume.chunk_size.join("×")} chunks at ${volume.output_voxel_size.join("×")} nm`);
    return json(200, {
      success: true,
      volume_id: volume.id,
      zarr_path: `${PAINT_ROUTE}${volume.id}`,
      ...volumeUrls(volume.id),
      metadata: {
        dataset_shape_voxels: volume.dataset_shape_voxels,
        chunk_size: volume.chunk_size,
        output_voxel_size: volume.output_voxel_size,
        claimed_output_voxel_size: volume.claimed_output_voxel_size ?? volume.output_voxel_size,
        dataset_offset_nm: volume.dataset_offset_nm,
      },
    });
  });

  // POST /api/finetune/add-to-viewer {crop_id, minio_url}: the volume's
  // paint layer in the viewer, selected on its Draw tab (overlay.py's
  // add_annotation_layer). The URL the tab sends back is the one
  // create-volume or load-existing-volume answered; without one, the
  // volume's own.
  router.post("/api/finetune/add-to-viewer", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const id = volumeIdOf(body.crop_id);
    if (!id) return badRequest("crop_id is required: a volume id (no slash)");
    const volume = await readVolume(ctx.paint, id);
    if (!volume) return json(404, { success: false, error: `No annotation volume ${id} in this browser` });
    const name = typeof body.layer_name === "string" && body.layer_name ? body.layer_name : paintLayerName(id);
    const url = typeof body.minio_url === "string" && body.minio_url ? s3Url(body.minio_url) : paintLayerUrl(id, pageBase());
    const layer: LayerSpec = { ...paintLayer(url, volume), name };
    ctx.viewer.setLayers({ [name]: layer });
    ctx.viewer.selectLayer(name, PAINT_TAB);
    return json(200, { success: true, message: "Layer added to viewer", layer_name: name });
  });

  // POST /api/finetune/list-existing-sessions: every volume in the paint
  // store, newest first, as a session of its own (the server's sessions
  // are directories of volumes; here a volume is the unit), in the shape
  // sessions.js lists: its id as session_id and session_path, its one
  // volume, its painted chunks, the dataset it was painted on.
  router.post("/api/finetune/list-existing-sessions", async (_req, _params, ctx) => {
    const sessions: Record<string, unknown>[] = [];
    for (const id of await ctx.paint.volumes()) {
      const volume = await readVolume(ctx.paint, id);
      if (!volume) continue;
      sessions.push({
        session_id: id,
        session_path: id,
        volumes: [{ volume_id: id, path: `${PAINT_ROUTE}${id}` }],
        chunk_count: (await paintedChunkIndices(ctx.paint, id)).length,
        dataset_path: volume.dataset_path,
        model_name: volume.model_name,
        created_at: volume.created_at,
      });
    }
    sessions.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)) || String(b.session_id).localeCompare(String(a.session_id)));
    return json(200, { success: true, sessions });
  });

  // POST /api/finetune/load-existing-volume {source_session_path}: resume a
  // volume: it becomes the session's. Nothing is copied (the server copies
  // the source session into a new one, so the original stays as it was;
  // here the store is the one place and the volume is edited in place), so
  // the copy counts are zero and the progress route says done at once. A
  // volume painted on another dataset is refused as the server refuses it:
  // its voxels are positions in that dataset.
  router.post("/api/finetune/load-existing-volume", async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    const id = volumeIdOf(body.source_session_path);
    if (!id) return badRequest("source_session_path required: a volume id (no slash)");
    const volume = await readVolume(ctx.paint, id);
    if (!volume) return json(404, { success: false, error: `No annotation volume found in ${id}` });
    const open = ctx.session.state.datasetUrl;
    if (open && volume.dataset_path && !sameDataset(volume.dataset_path, open)) {
      return json(409, {
        success: false,
        error: `That session was painted on ${volume.dataset_path}, and this dashboard has ${open} open. Open ${volume.dataset_path} to resume it.`,
      });
    }
    await ctx.session.update({ paintVolume: id });
    const painted = (await paintedChunkIndices(ctx.paint, id)).length;
    ctx.log(`Resumed annotation volume ${id} (${painted} painted chunk${painted === 1 ? "" : "s"})`);
    return json(200, {
      success: true,
      volume_id: id,
      new_session_path: id,
      zarr_path: `${PAINT_ROUTE}${id}`,
      ...volumeUrls(id),
      copied_count: 0,
      copied_minio: false,
      painted_chunk_count: painted,
      skipped_chunk_extracts: 0,
      metadata: volume,
    });
  });

  router.get("/api/finetune/load-existing-volume-progress", () => json(200, { success: true, progress: { phase: "done", done: true } }));

  // GET /api/finetune/session-volume (the browser's own): the volume the
  // session paints into, if any, so that after a reload the tab's Download
  // paint and Delete volume buttons know which one; 200 either way.
  router.get("/api/finetune/session-volume", async (_req, _params, ctx) => {
    const id = ctx.session.state.paintVolume;
    const volume = id ? await readVolume(ctx.paint, id) : null;
    if (!volume) return json(200, { success: true, volume_id: null, painted_chunk_count: 0 });
    return json(200, { success: true, volume_id: volume.id, painted_chunk_count: (await paintedChunkIndices(ctx.paint, volume.id)).length, volume });
  });

  // The paint is already where training reads it (the store), so a sync
  // has nothing to move: it reports the painted chunks, which is what the
  // server's count is of. The annotated-regions boxes are not drawn here
  // (no annotation layer in the viewer contract); the count is answered.
  router.post("/api/finetune/sync-annotations", async (_req, _params, ctx) => {
    const n = await paintedCount(ctx);
    return json(200, { success: true, message: `Synced ${n} annotations`, synced_count: n });
  });
  router.post("/api/finetune/refresh-annotated-regions", async (_req, _params, ctx) => json(200, { success: true, count: await paintedCount(ctx) }));

  // POST /api/finetune/submit: a job on the session's volume. Refused, in
  // the order the tab can act on, without a volume (needs_volume: the tab
  // offers New Volume), with nothing painted, with a loss that needs a
  // server, for a model not served, and without a dataset.
  router.post("/api/finetune/submit", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.model_name === "string" ? body.model_name.trim() : "";
    if (!name) return badRequest("model_name is required");
    const volumeId = ctx.session.state.paintVolume;
    const volume = volumeId ? await readVolume(ctx.paint, volumeId) : null;
    if (!volume) {
      return json(400, { success: false, error: "No annotation volume in this session: create one (New Volume) or resume one first.", needs_volume: true });
    }
    const painted = (await paintedChunkIndices(ctx.paint, volume.id)).length;
    if (!painted) {
      return badRequest(`Nothing is painted in ${volume.id} yet. Paint background (1) and objects (2 and up) on a few slices of the annotation layer first.`);
    }
    const refused = refuseLoss(body.loss_type);
    if (refused) return badRequest(refused);
    const model = ctx.infer.served().find((s) => s.entry.name === name);
    if (!model) return json(404, { success: false, error: `Model ${name} not found` });
    if (!ctx.infer.dataset()) return badRequest(NO_DATASET);
    const training = readTraining(body);
    if ("error" in training) return badRequest(training.error);
    const autoServe = body.auto_serve !== false;
    const outputType = hasAffinityChannels(model.entry) ? "affinities" : "binary";
    const job = jobs.submit({
      ctx,
      entry: model.entry,
      volume,
      options: training.options,
      params: { model_name: name, corrections_path: volume.id, ...training.params, auto_serve: autoServe, output_type: outputType },
      autoServe,
    });
    ctx.log(`Finetune job ${job.id}: ${name} on ${painted} painted chunk${painted === 1 ? "" : "s"} of ${volume.id}`);
    return json(200, {
      success: true,
      job_id: job.id,
      lsf_job_id: null,
      output_dir: job.toDict().output_dir,
      output_type: outputType,
      message: "Finetuning job submitted successfully",
      note: `The model's head (its last layer) trains in this browser on the ${painted} painted chunk${painted === 1 ? "" : "s"}; the finetuned model is kept in this browser and listed on the Models tab.`,
    });
  });

  router.get("/api/finetune/jobs", () => json(200, { success: true, jobs: jobs.list().map((job) => job.statusRecord()) }));

  router.get("/api/finetune/job/:id/status", (_req, params) => {
    const job = jobs.get(params.id);
    if (!job) return json(404, { success: false, error: "Job not found" });
    return json(200, { success: true, ...job.statusRecord() });
  });

  // The whole log, and the byte offset after it, where the tab opens the
  // live stream (job-monitor.js's restoreActiveJob), so nothing is sent twice.
  router.get("/api/finetune/job/:id/logs", (_req, params) => {
    const job = jobs.get(params.id);
    if (!job) return json(404, { success: false, error: "Job not found" });
    return json(200, { success: true, logs: job.logText(), offset: job.logOffset() });
  });

  router.get("/api/finetune/job/:id/logs/stream", (req, params) => {
    const job = jobs.get(params.id);
    return eventStream(job ? followLog(job, requestedOffset(req)) : notFound(params.id));
  });

  router.post("/api/finetune/job/:id/cancel", (_req, params) => {
    const result = jobs.cancel(params.id);
    if (!result.ok) return json(result.status, { success: false, error: result.error });
    return json(200, { success: true, message: `Job ${params.id} cancelled` });
  });

  router.post("/api/finetune/job/:id/stop-early", (_req, params) => {
    const result = jobs.stopEarly(params.id);
    if (!result.ok) return json(result.status, { success: false, error: result.error });
    return json(200, {
      success: true,
      message: `Stop requested. Training will exit after epoch ${result.afterEpoch}; the head trained by then is kept and served, so you can restart with updated parameters.`,
    });
  });

  // POST /api/finetune/job/:id/restart: the form's parameters again (the
  // same body as submit, less the model), a new run from the trained head.
  router.post("/api/finetune/job/:id/restart", (req, params) => {
    const body = objectBody(req) ?? {};
    const refused = refuseLoss(body.loss_type);
    if (refused) return badRequest(refused);
    const training = readTraining(body);
    if ("error" in training) return badRequest(training.error);
    const result = jobs.restart(params.id, training.options, training.params);
    if (!result.ok) return json(result.status, { success: false, error: result.error });
    return json(200, {
      success: true,
      job_id: params.id,
      annotations_synced: 0,
      message: "Restart request sent. Training restarts from the trained head in this browser, on the paint as it is now.",
    });
  });

  // GET /api/finetune/volume/:id/download: the volume as a zip of its zarr,
  // for the server mode's crops import or for anyone with zarr-python.
  router.get("/api/finetune/volume/:id/download", async (_req, params, ctx) => {
    const id = volumeIdOf(params.id);
    if (!id || !(await readVolume(ctx.paint, id))) return json(404, { success: false, error: `No annotation volume ${params.id} in this browser` });
    const zip = await volumeZip(ctx.paint, id, d.now());
    return text(200, exactBuffer(zip), "application/zip", { "content-disposition": `attachment; filename="${id}${VOLUME_ZIP_SUFFIX}"` });
  });

  // GET /api/finetune/model/:name/download: a finetuned model's .onnx, by
  // its name (the layer's) or id, from the catalog's byte cache.
  router.get("/api/finetune/model/:name/download", async (_req, params, ctx) => {
    const record = (await ctx.catalog.finetuned()).find((r) => r.name === params.name || r.id === params.name);
    if (!record) return json(404, { success: false, error: `No model finetuned in this browser is named ${params.name}` });
    const bytes = await ctx.catalog.modelBytes(record.entry);
    return text(200, bytes, "application/octet-stream", { "content-disposition": `attachment; filename="${record.name}.onnx"` });
  });

  // POST /api/finetune/delete-volume {volume_id}: the volume's files out of
  // the store, its layer out of the viewer, and the session's volume
  // cleared when it was this one.
  router.post("/api/finetune/delete-volume", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const id = volumeIdOf(body.volume_id);
    if (!id) return badRequest("volume_id is required: a volume id (no slash)");
    if (!(await ctx.paint.list(id)).length) return json(404, { success: false, error: `No annotation volume ${id} in this browser` });
    await ctx.paint.removeVolume(id);
    ctx.viewer.removeLayer(paintLayerName(id));
    if (ctx.session.state.paintVolume === id) await ctx.session.update({ paintVolume: null });
    ctx.log(`Deleted annotation volume ${id}`);
    return json(200, { success: true });
  });

  return jobs;
}

/** The volume's URLs as the Python answers them: `minio_url` names the
 * volume's zarr as the viewer reaches it, `neuroglancer_url` its
 * `annotation` group, which the paint layer reads and writes. Here both are
 * the paint route's, in the fork's `s3+http(s)://` scheme (paintLayerUrl),
 * which add-to-viewer takes back as it is. */
function volumeUrls(id: string): { minio_url: string; neuroglancer_url: string } {
  const neuroglancer = paintLayerUrl(id, pageBase());
  return { minio_url: neuroglancer.replace(/\/annotation\/?$/, ""), neuroglancer_url: neuroglancer };
}

/** `value` as a volume id, or null when it is not one: a non-empty string
 * without "/" (the paint route and the store key `<volume>/<path>` split on
 * the first slash, so an id with one would name another volume's file). */
function volumeIdOf(value: unknown): string | null {
  const id = typeof value === "string" ? value.trim() : "";
  return id && !id.includes("/") ? id : null;
}

/** `url` in the fork's s3+ scheme, as add_annotation_layer prefixes the
 * MinIO URL; one already in it (the paint route's) is kept. */
function s3Url(url: string): string {
  return url.startsWith("s3+") ? url : `s3+${url}`;
}

/** Whether two dataset paths name one dataset (annotation_sessions.py's
 * _same_dataset): a trailing slash and a trailing scale level (`/s2`) aside. */
export function sameDataset(a: string, b: string): boolean {
  const bare = (path: string) => path.replace(/\/+$/, "").replace(/\/s\d+$/, "");
  return bare(a) === bare(b);
}

/** The painted chunks of the session's volume; 0 without one. */
async function paintedCount(ctx: AppContext): Promise<number> {
  const id = ctx.session.state.paintVolume;
  return id ? (await paintedChunkIndices(ctx.paint, id)).length : 0;
}

/** Why a loss the form sent cannot be trained with here, or null for one
 * the head trains with (or none sent: BCE). The head's losses are the
 * server's per-voxel ones, which a batch of voxels from any chunks defines
 * as well as a patch does; Dice, and "combined" (Dice + BCE), is an overlap
 * taken over each sample's whole patch, which the browser's shuffled
 * voxels are not, and "interval" and "flow" need targets the browser does
 * not make (a distance model's bounds, Cellpose's flows). */
function refuseLoss(loss: unknown): string | null {
  if (loss === undefined || loss === null || loss === "" || HEAD_LOSSES.includes(loss as HeadLoss)) return null;
  const why = loss === "dice" || loss === "combined"
    ? "Dice is an overlap over each training patch as a whole, and the browser trains the head on voxels drawn from every painted chunk at once, not on patches"
    : "the browser makes only the binary and affinity targets";
  return `The ${String(loss)} loss needs a server: ${why}. In the browser the model's head trains with BCE (bce), MSE (mse) or margin (margin): pick one, or run the dashboard on a server for the others.`;
}

/** The training parameters of a submit or restart body (training-form.js's
 * trainingParams, as job-monitor.js sends them): the ones the browser's
 * trainer takes, checked, as run.ts's options and as the job's params (the
 * card shows the loss, margin and distillation from them). batch_pixels is
 * the browser's own: voxels per Adam step, where the server's batch_size
 * counts patches. The margin goes with the margin loss alone, and label
 * smoothing is 0 with it, as the server sets it. The fields only a cluster
 * job uses (lora_r, batch_size, patches_per_epoch, rehearsal_fraction,
 * augment, queue, checkpoint_path) are left out of both. */
export function readTraining(body: Record<string, unknown>): { options: FinetuneOptions; params: Record<string, unknown> } | { error: string } {
  const epochs = numberOr(body.num_epochs, DEFAULT_OPTIONS.epochs);
  if (!Number.isInteger(epochs) || epochs < 1) return { error: "num_epochs: expected a whole number of epochs, 1 or more" };
  const learningRate = numberOr(body.learning_rate, DEFAULT_OPTIONS.learningRate);
  if (!Number.isFinite(learningRate) || learningRate <= 0) return { error: "learning_rate: expected a positive number" };
  const lossType = body.loss_type === undefined || body.loss_type === null || body.loss_type === "" ? DEFAULT_OPTIONS.lossType : (body.loss_type as HeadLoss);
  if (!HEAD_LOSSES.includes(lossType)) return { error: refuseLoss(lossType)! };
  const margin = numberOr(body.margin, DEFAULT_OPTIONS.margin);
  if (!(margin >= 0 && margin <= 0.5)) return { error: "margin: expected a number from 0 to 0.5" };
  let labelSmoothing = numberOr(body.label_smoothing, DEFAULT_OPTIONS.labelSmoothing);
  if (!(labelSmoothing >= 0 && labelSmoothing < 1)) return { error: "label_smoothing: expected a number from 0 up to, not including, 1" };
  if (lossType === "margin") labelSmoothing = 0;
  const balanceClasses = typeof body.balance_classes === "boolean" ? body.balance_classes : DEFAULT_OPTIONS.balanceClasses;
  const batchPixels = numberOr(body.batch_pixels, DEFAULT_OPTIONS.batchPixels);
  if (!Number.isInteger(batchPixels) || batchPixels < 1 || batchPixels > MAX_BATCH_PIXELS) return { error: `batch_pixels: expected a whole number of voxels per step, from 1 to ${MAX_BATCH_PIXELS}` };
  const distillationLambda = numberOr(body.distillation_lambda, DEFAULT_OPTIONS.distillationLambda);
  if (!Number.isFinite(distillationLambda) || distillationLambda < 0) return { error: "distillation_lambda: expected a number, 0 or more (0 switches distillation off)" };
  const scope = body.distillation_scope === undefined || body.distillation_scope === null || body.distillation_scope === "" ? DEFAULT_OPTIONS.distillationScope : String(body.distillation_scope).toLowerCase();
  if (scope !== "unlabeled" && scope !== "all") return { error: "distillation_scope: expected unlabeled or all" };
  const distillationScope = scope as DistillationScope;
  return {
    options: { ...DEFAULT_OPTIONS, epochs, learningRate, lossType, margin, balanceClasses, labelSmoothing, batchPixels, distillationLambda, distillationScope },
    params: {
      num_epochs: epochs,
      learning_rate: learningRate,
      loss_type: lossType,
      ...(lossType === "margin" ? { margin } : {}),
      balance_classes: balanceClasses,
      label_smoothing: labelSmoothing,
      batch_pixels: batchPixels,
      distillation_lambda: distillationLambda,
      distillation_scope: distillationScope,
      finetune_mode: "head",
    },
  };
}

/** A body field as a number (the form sends numbers; a string is read), or `fallback` when absent. */
function numberOr(value: unknown, fallback: number): number {
  if (value === undefined || value === null || value === "") return fallback;
  return typeof value === "number" ? value : Number(value);
}

/** The job's log as server-sent events (training.py's stream_job_logs): the
 * lines after `offset`, each with the log's byte count after it as its id,
 * then each new line as the run logs it, and `done` with the final status
 * once the job has one. A job waiting for a restart keeps the stream open,
 * as the Python keeps it while the trainer lives: a restarted run's lines
 * (the RESTARTING_TRAINING marker first, which resets the card) come down
 * the same stream, the card is not told the training ended when it has
 * not, and job-monitor.js only closes the stream itself on a final status.
 * Empty lines count toward the offset and are not sent (logEvents). No
 * heartbeat: the stream crosses no proxy, only the service worker's relay.
 *
 * A stream the client dropped is woken by the job's next change and then
 * returns; until then it holds one pending promise, nothing more. */
export async function* followLog(job: FinetuneJob, offset: number): AsyncGenerator<SseEvent> {
  let next = job.offsets.findIndex((end) => end > offset);
  if (next === -1) next = job.lines.length;
  for (;;) {
    while (next < job.lines.length) {
      const line = job.lines[next];
      const id = job.offsets[next];
      next++;
      if (line !== "") yield { id, data: line };
    }
    if (job.isFinal()) {
      yield done(job.status);
      return;
    }
    await job.changed();
  }
}

/** The stream for a job the page does not have, as the Python answers it: a line saying so, then done NOT_FOUND. */
function* notFound(id: string): Generator<SseEvent> {
  yield { data: `Job ${id} not found` };
  yield done("NOT_FOUND");
}

/** `bytes` as an ArrayBuffer of exactly its length (the relay transfers the
 * whole buffer, so a view on a larger one would send more than the file). */
function exactBuffer(bytes: Uint8Array): ArrayBuffer {
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) return bytes.buffer as ArrayBuffer;
  return bytes.slice().buffer as ArrayBuffer;
}
