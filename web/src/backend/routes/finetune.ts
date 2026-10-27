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
// directory, so a "session" is a volume, the one it trains on now, and its
// id is its path (a volume Use this model superseded is kept in the store
// but neither listed nor resumed: volume.ts supersedeVolume); the catalog
// for the models directory; a run in the tab (src/finetune/run.ts) for the
// cluster job, so a submit needs no output path, queue or charge group, and
// the server-only fields the form still sends (LoRA rank, patches per
// epoch, rehearsal) are ignored. What trains is what the model's manifest
// offers (finetune/modes.ts, answered as finetune_modes) and the form's
// Training field picks, with the server's per-voxel losses (BCE, MSE,
// margin), smoothing, class balance and distillation term (λ and scope) in
// both (readTraining), Dice refused since it needs a whole patch per sample
// (refuseLoss): the head, with the browser's own batch of voxels per step;
// or the whole network (finetune/full), with the server's Batch Size in
// tiles (a 3D model's one block a step), a tile or block side of its own,
// which the models route offers with what a step costs (fullTraining), and
// augmentation. Both train one channel of a model whose several channels
// are not affinities (target-call.ts channelChoice, which the models route
// answers as channel_choice), the one the form's Channel picks
// (select_channel), and submit refuses such a model without the pick
// (refuseChannel); POST /api/finetune/channel shows the channel picked on
// the model's layer, and on each layer built for it or its finetunes after
// (the session's shownChannels). The good
// regions' routes are routes/good-regions.ts, and the patch tools' Label it
// and AI annotation have routes of their own
// (view-labels.ts, ai-annotate.ts); crops import keeps needing a server and
// stays unavailable (index.ts). One route is the browser's own, GET
// /api/finetune/session-volume: which volume the session paints into, for
// the tab's Download and Delete buttons. A finetuned model downloads with its
// manifest in the file's metadata_props (downloadableOnnx), so that the
// server's onnx model type can serve the file on its own.
//
// The answers keep the Python routes' keys, so the tab's JavaScript
// (static/js/dashboard/finetune/*.js) reads them unchanged: a refusal is
// {success: false, error} at the Python's status, with `needs_volume` where
// requests.js's offerNewVolume keys on it.
//
// Shift+R over the viewer restarts the job the Training Status card shows,
// as its Restart does without the confirm (the server's
// routes/finetune/restart_key.py). The server's key runs in its Python
// process, so it counts the press in the job's status answer
// (`restart_requests`) for the tab to act on at its next poll; here the key
// runs in the page, so it tells the tab at once (RESTART_EVENT), and the
// status answers carry no count. job-monitor.js decides from its card
// whether the job can take the restart, and says why not in the tab's logs:
// this Neuroglancer's status bar is not reachable from the page.
import { volumeCornerNm } from "../../ai-annotate/fill";
import type { AppContext, FinetunedRecord, LayerSpec, ModelEntry, PaintVolume } from "../../contracts";
import { MANIFEST_METADATA_KEY, PAINT_ROUTE } from "../../contracts";
import { type FetchFullFiles, fetchFullFiles } from "../../finetune/full/files";
import { defaultTile, is3d, tileChoices } from "../../finetune/full/graph";
import { BLOCK_COSTS, checkTileSide, writableBlocks } from "../../finetune/full/run-full";
import { webgpuMissing } from "../../finetune/full/tfjs";
import { loadGoodRegions, saveGoodRegions } from "../../finetune/good-regions";
import { finetuneModes, MODE_NOUN, readsOneSlice, refuseMode, sourceOf } from "../../finetune/modes";
import { withMetadata } from "../../finetune/onnx-proto";
import { runFinetune } from "../../finetune/run";
import { HEAD_LOSSES } from "../../finetune/head-trainer";
import { channelChoice } from "../../finetune/target-call";
import type { DistillationScope, FinetuneOptions, HeadLoss, RunFinetune } from "../../finetune/types";
import { DEFAULT_OPTIONS, FULL_LEARNING_RATE } from "../../finetune/types";
import { CHUNK_DIR, chunkKey, createVolume, currentVolume, hasAffinityChannels, paintedChunkIndices, planPaintVolume, planVolume, readPaintedChunk, readVolume, supersedeVolume, trainingRefusal, viewPixelNm, VOLUME_CALLS } from "../../finetune/volume";
import { volumeZip, VOLUME_ZIP_SUFFIX } from "../../finetune/zip";
import { PAINT_TAB, paintLayer, paintLayerName, paintLayerUrl } from "../../viewer/layers";
import { FinetuneJob, FinetuneJobs } from "../finetune-jobs";
import type { Router } from "../router";
import { badRequest, json, message, objectBody, text } from "../router";
import { pageBase, showChannel } from "../serving";
import { done, eventStream, requestedOffset, type SseEvent } from "../sse";

/** The seams the routes run through, so they can be tested with a fake run
 * (run.ts needs onnxruntime), a volume plan of a known shape, and where
 * Shift+R's event goes (the page's document, which job-monitor.js listens
 * on; null where there is none, as in the other routes' tests). */
export interface FinetuneDeps {
  run: RunFinetune;
  plan: typeof planVolume;
  now: () => Date;
  page: () => EventTarget | null;
  /** A whole-network model's training graph and offsets (Hugging Face). */
  fullFiles: FetchFullFiles;
  /** Why the whole network cannot train here (no GPU for TF.js), or null. */
  webgpuMissing: () => Promise<string | null>;
}

/** Shift+R over the viewer (restart_key.py's KEY and ACTION), and the event
 * it sends the page, which job-monitor.js's RESTART_EVENT names too. */
export const RESTART_KEY = "shift+keyr";
export const RESTART_ACTION = "finetune-restart";
export const RESTART_EVENT = "cellmap-flow:restart-finetune";

const NO_DATASET = "No dataset is open. Enter its URL in the viewer panel first; the volume is painted on it.";

/** The browser's batch of voxels per step, at most: a step's arithmetic and
 * buffers grow with it, and a million voxels is already every painted voxel
 * of any session in one step. */
const MAX_BATCH_PIXELS = 1 << 20;

/** The whole network's batch of tiles per step, at most (the form offers up to 32, as the server's). */
const MAX_BATCH_PATCHES = 32;

/** Register the routes; returns the jobs, which live as long as the router (the page). */
export function finetuneRoutes(router: Router, deps: Partial<FinetuneDeps> = {}): FinetuneJobs {
  const d: FinetuneDeps = {
    run: runFinetune,
    plan: planVolume,
    now: () => new Date(),
    page: () => (typeof document === "undefined" ? null : document),
    fullFiles: fetchFullFiles,
    webgpuMissing,
    ...deps,
  };

  /** What the tab shows for a model's whole-network training: the tile
   * sides it offers with their cost per step (full/graph.ts tileChoices),
   * the default, and the measurement's setup, or why the graph could not
   * be had (the Hub unreachable, say), in which case no side is offered. A
   * 3D model's are block sides (dims 3), whose costs were measured on one
   * GPU (run-full.ts BLOCK_COSTS), which `measured_gpu` names. */
  async function fullTraining(entry: ModelEntry): Promise<Record<string, unknown>> {
    const spec = entry.manifest.finetune!.full!;
    try {
      const { graph } = await d.fullFiles(sourceOf(entry), spec);
      const unavailable = await refuseGpu();
      const blocks = is3d(graph), costs = blocks ? BLOCK_COSTS.architectures[entry.manifest.architecture] : spec.spike?.full;
      // A 3D model's blocks are its files' (writableBlocks), its default the
      // one it is served at: the finetune is served from the file it trains.
      const tiles = tileChoices(graph, costs, blocks ? [...writableBlocks(entry).keys()] : []);
      const measured = blocks ? { measured_on: costs ? BLOCK_COSTS.setup : null, measured_gpu: costs ? BLOCK_COSTS.gpu : null } : { measured_on: spec.spike?.setup ?? null };
      return { dims: blocks ? 3 : 2, tiles, default_tile: defaultTile(tiles, blocks ? entry.variant.input_size[0] : null), context: graph.context, ...measured, unavailable };
    } catch (e) {
      return { tiles: [], default_tile: null, context: null, measured_on: null, error: `The training graph could not be fetched: ${message(e)}` };
    }
  }

  /** Why the whole network cannot train in this browser, or null. */
  async function refuseGpu(): Promise<string | null> {
    const missing = await d.webgpuMissing();
    return missing && `The whole network trains on a GPU with WebGPU, and ${missing}. Train the last layer, or open the page in Chrome or Edge on a machine with a GPU.`;
  }

  /** A whole-network run checked: `entry`'s manifest offering it, a GPU to
   * train on, and the options' tile or block side against `entry`'s graph,
   * and for a 3D model a batch of one block. Null when it can run (or the
   * run is the head's), else why not. */
  async function refuseFull(entry: ModelEntry, options: FinetuneOptions): Promise<string | null> {
    if (options.mode !== "full") return null;
    const refused = refuseMode(entry, "full") ?? (await refuseGpu());
    if (refused) return refused;
    try {
      const { graph } = await d.fullFiles(sourceOf(entry), entry.manifest.finetune!.full!);
      const side = checkTileSide(graph, options.tileSize, entry.name);
      if (is3d(graph) && options.batchPatches !== 1) {
        return `${entry.name} is a 3D model, whose whole network trains one block a step (a block of ${side}³ is a batch's memory already): set Batch Size to 1.`;
      }
      return null;
    } catch (e) {
      return message(e);
    }
  }

  /** The form's training fields read (readTraining) and checked against
   * `entry`, its channel pick among them (refuseChannel); a 3D model's
   * whole-network side is kept in the job's params as block_size, which the
   * job card shows as a cube. */
  async function checkedTraining(body: Record<string, unknown>, mode: unknown, entry: ModelEntry | undefined): Promise<ReturnType<typeof readTraining>> {
    const lossRefused = mode === "full" ? null : refuseLoss(body.loss_type);
    if (lossRefused) return { error: lossRefused };
    const training = readTraining({ ...body, finetune_mode: mode });
    if ("error" in training) return training;
    const refused = entry ? refuseChannel(entry, training.options.selectChannel) ?? (await refuseFull(entry, training.options)) : null;
    if (refused) return { error: refused };
    if (entry && training.options.mode === "full" && !readsOneSlice(entry) && "tile_size" in training.params) {
      const { tile_size: side, ...params } = training.params;
      return { options: training.options, params: { ...params, block_size: side } };
    }
    return training;
  }
  const jobs = new FinetuneJobs({ run: d.run, now: d.now });

  /** restart_key.ensure_binding: bind Shift+R in the viewer, once per
   * router; the viewer binds it again in every viewer it mounts after. The
   * press only tells the page: the form's values, and the card that says
   * whether the job can take a restart, are job-monitor.js's. */
  let restartKeyBound = false;
  function ensureRestartKey(ctx: AppContext): void {
    if (restartKeyBound) return;
    restartKeyBound = true;
    ctx.viewer.bindKey(RESTART_KEY, RESTART_ACTION, () => d.page()?.dispatchEvent(new Event(RESTART_EVENT)));
  }

  // GET /api/finetune/models: the served models, with the geometry the tab
  // shows (write_shape in nm, the voxel size, the channels), from the
  // engine's model info as the Python asks each inference server, the
  // channels' names (the manifest's) and whether the model trains one of
  // them at a time (channel_choice: target-call.ts channelChoice), for the
  // form's Channel, and the
  // ways each can be finetuned here, from its manifest (finetune/modes.ts:
  // "head", "full"), as the server answers finetune_modes; a model whose
  // manifest declares none has the head the graph is found to end in. A
  // model that offers the whole network has its tile sides with them
  // (full_training), which models.js and training-form.js show. And the
  // open dataset's levels' voxel sizes (raw_voxel_sizes, finest first, as
  // the server answers them), which New Volume offers for a volume painted
  // before any model (volume-grid.js).
  router.get("/api/finetune/models", async (_req, _params, ctx) => {
    const models = await Promise.all(ctx.infer.served().map(async (s) => {
      const modes = finetuneModes(s.entry);
      return {
        name: s.entry.name,
        write_shape: s.info.write_shape,
        output_voxel_size: s.info.output_voxel_size,
        output_channels: s.info.output_channels,
        channels: s.entry.manifest.channels.length ? [...s.entry.manifest.channels] : null,
        channel_choice: channelChoice(s.entry),
        finetune_modes: modes,
        ...(modes.includes("full") ? { full_training: await fullTraining(s.entry) } : {}),
      };
    }));
    const rawVoxelSizes = (ctx.infer.dataset()?.levels ?? []).map((level) => level.voxel.slice());
    return json(200, { models, selected_model: models.length === 1 ? models[0].name : null, raw_voxel_sizes: rawVoxelSizes });
  });

  // POST /api/finetune/channel {model_name, select_channel}: the channel
  // the session shows for the model (select_channel, its index; null clears
  // the pick), as the form's Channel picks it, so the layer shows what is
  // trained (Use this model sends the row's channel with its own request,
  // rank.ts, so that it is picked before the model is served). It is kept in
  // the session (shownChannels) and shown now on the model's layer and on
  // its finetunes' that follow its pick (pickShownChannel); a layer built
  // for them later opens on it (showLayers). Answers whether a layer was
  // there to change: none is for a model not served, an output with no
  // channel axis or a chain that changes the channel count (displayChannel),
  // and a pick cleared leaves the layers as they are. Refused (400) for a
  // pick that is not a channel's index or is past the model's channels, and
  // (404) for a model the browser does not have, which changes nothing.
  router.post("/api/finetune/channel", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.model_name === "string" ? body.model_name.trim() : "";
    if (!name) return badRequest("model_name is required");
    const pick = readSelectChannel(body.select_channel);
    if ("error" in pick) return badRequest(pick.error);
    const entry = ctx.infer.served().find((s) => s.entry.name === name)?.entry ?? (await ctx.catalog.list()).find((e) => e.name === name);
    if (!entry) return json(404, { success: false, error: `${name} is not a model the browser can run` });
    const past = pick.channel === null ? null : notAChannel(entry, pick.channel);
    if (past) return badRequest(past);
    return json(200, { success: true, shown: await pickShownChannel(ctx, name, pick.channel) });
  });

  // POST /api/finetune/create-volume {model_name} or {voxel_size}: a new
  // volume over the open dataset, its zarr files in the paint store, and the
  // session's volume from now on. For a model, on its output grid
  // (planVolume); with no model named, one painted before any model, on the
  // raw's grid at voxel_size (nm: three numbers, or one for all three; none,
  // the level the viewer shows: planPaintVolume), refused with a model as
  // the server refuses it. Both plans are the portable Python's, whose
  // refusal is a 400. The answer's URLs are the paint route's (volumeUrls);
  // the tab then calls add-to-viewer with neuroglancer_url, as it does on
  // the server.
  router.post("/api/finetune/create-volume", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.model_name === "string" ? body.model_name.trim() : "";
    const size = voxelSizeOf(body.voxel_size);
    if (size === undefined) return badRequest(`voxel_size must be one positive number or three (z, y, x), not ${JSON.stringify(body.voxel_size)}`);
    if (name && size) return badRequest(`voxel_size is for a volume made without a model; ${name}'s volume lies on its own grid`);
    const dataset = ctx.infer.dataset();
    let volume: PaintVolume;
    if (!name) {
      if (!dataset) return badRequest(NO_DATASET);
      try {
        volume = await planPaintVolume(ctx.infer, dataset, size, ctx.session.state.chain, { now: d.now(), pixelNm: viewPixelNm(ctx.viewer.viewerUrl()) });
      } catch (e) {
        return badRequest(message(e));
      }
    } else {
      const served = ctx.infer.served();
      if (!served.length) return badRequest("No models loaded");
      const model = served.find((s) => s.entry.name === name);
      if (!model) return json(404, { success: false, error: `Model ${name} not found` });
      if (!dataset) return badRequest(NO_DATASET);
      try {
        volume = await d.plan(ctx.infer, dataset, model.entry, ctx.session.state.chain, { now: d.now() });
      } catch (e) {
        return badRequest(message(e));
      }
    }
    await createVolume(ctx.paint, volume);
    await ctx.session.update({ paintVolume: volume.id });
    ctx.log(`New annotation volume ${volume.id} ${name ? `for ${name}` : "for no model yet"}: ${volume.dataset_shape_voxels.join("×")} voxels in ${volume.chunk_size.join("×")} chunks at ${volume.output_voxel_size.join("×")} nm`);
    return json(200, {
      success: true,
      volume_id: volume.id,
      zarr_path: `${PAINT_ROUTE}${volume.id}`,
      ...volumeUrls(volume.id),
      model_name: volume.model_name,
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

  // POST /api/finetune/list-existing-sessions: the sessions in the paint
  // store, newest first, each its one current volume: the server's sessions
  // are directories of volumes, listed by the one their manifest trains on;
  // here a volume is the unit, and one Use this model superseded
  // (supersedeVolume) is not listed, since its paint is the current one's
  // too. In the shape sessions.js lists: the volume's id as session_id and
  // session_path, the volume, its painted chunks, the model it is for
  // (model_name, null for one painted before any model), the dataset it was
  // painted on.
  router.post("/api/finetune/list-existing-sessions", async (_req, _params, ctx) => {
    const sessions: Record<string, unknown>[] = [];
    for (const id of await ctx.paint.volumes()) {
      const volume = await readVolume(ctx.paint, id);
      if (!volume || volume.superseded_by) continue;
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
  // session: its current volume becomes the session's, the one named or,
  // for one Use this model superseded (named by a list shown before it
  // ran), the volume that took its place (currentVolume), as the server
  // resumes only a session's current volume. Nothing is copied (the server
  // copies that volume into a new session, so the original stays as it
  // was; here the store is the one place and the volume is edited in
  // place), so the copy counts are zero and the progress route says done at
  // once. A volume painted on another dataset is refused as the server
  // refuses it: its voxels are positions in that dataset.
  router.post("/api/finetune/load-existing-volume", async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    const named = volumeIdOf(body.source_session_path);
    if (!named) return badRequest("source_session_path required: a volume id (no slash)");
    const volume = await currentVolume(ctx.paint, named);
    if (!volume) return json(404, { success: false, error: `No annotation volume found in ${named}` });
    const { id } = volume, open = ctx.session.state.datasetUrl;
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
  // offers New Volume), with nothing painted, for a model not served,
  // without a dataset, on a volume painted before any model (paint_first:
  // trainingRefusal's sentence, the server's, says to Use this model), with
  // a loss that needs a server (the head's: the whole network has its own),
  // a model of several channels trained one at a time without the channel
  // picked, or a pick it cannot take (refuseChannel), and for a mode the
  // model's manifest does not offer, a tile or block side its graph does not
  // take, or a 3D model's batch of more than one block. A model whose grid
  // is not the volume's (another model's volume, whose blocks and voxel
  // sizes are not this one's: the portable Python's carry_note says) has the
  // paint carried onto its grid first, as Use this model carries it
  // (carryOnto), after every refusal, so a refused submit changes nothing;
  // the chain and the channel stay as the user set them. The answer then
  // says so: `carried` {from_model, from_volume, volume_id, chunks}, and the
  // note names both grids.
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
    const model = ctx.infer.served().find((s) => s.entry.name === name);
    if (!model) return json(404, { success: false, error: `Model ${name} not found` });
    const dataset = ctx.infer.dataset();
    if (!dataset) return badRequest(NO_DATASET);
    const paintFirst = await trainingRefusal(ctx.infer, volume);
    if (paintFirst) return json(400, { success: false, error: paintFirst, paint_first: true });
    // The mode the form asks for; a body without one (an older tab) the
    // first the model offers.
    const mode = body.finetune_mode ?? finetuneModes(model.entry)[0] ?? "head";
    const training = await checkedTraining(body, mode, model.entry);
    if ("error" in training) return badRequest(training.error);
    const refusedMode = refuseMode(model.entry, training.options.mode);
    if (refusedMode) return badRequest(refusedMode);
    // The volume the model trains on: the session's, or, for a model on
    // another grid, a new one with the paint carried onto it.
    let trained = volume, chunks = painted, carried: Record<string, unknown> | null = null, carryNote: string | null = null;
    if (volume.model_id !== model.entry.id) {
      let to: PaintVolume;
      try {
        to = await d.plan(ctx.infer, dataset, model.entry, ctx.session.state.chain, { now: d.now() });
      } catch (e) {
        return badRequest(message(e));
      }
      const change = await ctx.infer.python<{ note?: string; error?: string }>(`${VOLUME_CALLS}:carry_note`, { from: volume, to, kept: "in this browser, marked superseded" });
      if (change.error) return badRequest(change.error);
      if (change.note) {
        const moved = await carryOnto(ctx, volume, to, name);
        [trained, chunks, carryNote] = [to, moved.chunks, change.note];
        carried = { from_model: volume.model_name, from_volume: volume.id, volume_id: to.id, chunks: moved.chunks };
      }
    }
    const autoServe = body.auto_serve !== false;
    const outputType = hasAffinityChannels(model.entry) ? "affinities" : "binary";
    const job = jobs.submit({
      ctx,
      entry: model.entry,
      volume: trained,
      options: training.options,
      params: { model_name: name, corrections_path: trained.id, ...training.params, auto_serve: autoServe, output_type: outputType },
      autoServe,
    });
    const channel = training.options.selectChannel;
    const channelWords = channel === null ? "" : `'s channel ${channel} (${model.entry.manifest.channels[channel]})`;
    const chunkWords = `${chunks} painted chunk${chunks === 1 ? "" : "s"}`;
    ctx.log(`Finetune job ${job.id}: ${name}${channelWords} on ${chunkWords} of ${trained.id}`);
    const where = training.options.mode === "full"
      ? `The model's whole network trains in this browser with TensorFlow.js on the paint of the ${chunkWords}; the finetuned model is kept in this browser and listed on the Models tab.`
      : `The model's head (its last layer) trains in this browser on the ${chunkWords}; the finetuned model is kept in this browser and listed on the Models tab.`;
    return json(200, {
      success: true,
      job_id: job.id,
      lsf_job_id: null,
      output_dir: job.toDict().output_dir,
      output_type: outputType,
      message: "Finetuning job submitted successfully",
      note: carryNote ? `${carryNote} ${where}` : where,
      ...(carried ? { carried } : {}),
    });
  });

  // The tab asks for the jobs on every page load (job-monitor.js's
  // restoreActiveJob), which is where Shift+R is bound, as the server binds
  // it: before there is a job, so that it can say there is none.
  router.get("/api/finetune/jobs", (_req, _params, ctx) => {
    ensureRestartKey(ctx);
    return json(200, { success: true, jobs: jobs.list().map((job) => job.statusRecord()) });
  });

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
    const noun = MODE_NOUN[jobs.get(params.id)!.args.options.mode];
    return json(200, {
      success: true,
      message: `Stop requested. Training will exit after epoch ${result.afterEpoch}; the ${noun} trained by then is kept and served, so you can restart with updated parameters.`,
    });
  });

  // POST /api/finetune/job/:id/restart: the form's parameters again (the
  // same body as submit, less the model), a new run from the trained head.
  // The job's channel when the body picks none (Shift+R, or an older tab):
  // a restart trains what the job trains, as the server's keeps
  // select_channel; another channel picked trains that one from there.
  router.post("/api/finetune/job/:id/restart", async (req, params) => {
    const job = jobs.get(params.id);
    const sent = objectBody(req) ?? {};
    const body = blank(sent.select_channel) && job ? { ...sent, select_channel: job.args.options.selectChannel } : sent;
    // The job's own mode unless the body names one (jobs.restart refuses a switch).
    const mode = body.finetune_mode ?? job?.args.options.mode ?? "head";
    const training = await checkedTraining(body, mode, job?.args.entry);
    if ("error" in training) return badRequest(training.error);
    const result = jobs.restart(params.id, training.options, training.params);
    if (!result.ok) return json(result.status, { success: false, error: result.error });
    return json(200, {
      success: true,
      job_id: params.id,
      annotations_synced: 0,
      message: `Restart request sent. Training restarts from the trained ${MODE_NOUN[training.options.mode]} in this browser, on the paint as it is now.`,
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
  // its name (the layer's) or id, from the catalog's byte cache, with its
  // manifest written in (downloadableOnnx).
  router.get("/api/finetune/model/:name/download", async (_req, params, ctx) => {
    const record = (await ctx.catalog.finetuned()).find((r) => r.name === params.name || r.id === params.name);
    if (!record) return json(404, { success: false, error: `No model finetuned in this browser is named ${params.name}` });
    const bytes = downloadableOnnx(record, await ctx.catalog.modelBytes(record.entry));
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

/** create-volume's voxel_size: null when not given (or blank), three
 * numbers from three or from one (every axis's), and undefined for anything
 * else. Whether they are positive and the raw's axes is the plan's to say. */
function voxelSizeOf(value: unknown): number[] | null | undefined {
  if (value === undefined || value === null || value === "" || (Array.isArray(value) && !value.length)) return null;
  const numbers = (Array.isArray(value) ? value : [value]).map((v) => (typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : NaN));
  if (numbers.some((v) => !Number.isFinite(v)) || (numbers.length !== 1 && numbers.length !== 3)) return undefined;
  return numbers.length === 1 ? [numbers[0], numbers[0], numbers[0]] : numbers;
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
 * card shows the loss, margin and distillation from them). finetune_mode
 * says what trains, "head" when absent. Both modes take the loss fields
 * (readLoss) and select_channel, the channel picked (a whole number, or
 * absent for none; whether the model has it is refuseChannel's to say,
 * with the model). For the head, batch_pixels is the browser's own: voxels per
 * Adam step, where the server's batch_size counts patches. The fields only
 * a cluster job uses (lora_r, patches_per_epoch, rehearsal_fraction, queue,
 * checkpoint_path, and for the head batch_size and augment) are left out
 * of both. For the whole network (readFullTraining) the batch is
 * batch_size, in tiles, the tile's side tile_size, and augment says
 * whether its tiles are augmented. */
export function readTraining(body: Record<string, unknown>): { options: FinetuneOptions; params: Record<string, unknown> } | { error: string } {
  const mode = body.finetune_mode === undefined || body.finetune_mode === null || body.finetune_mode === "" ? "head" : body.finetune_mode;
  if (mode !== "head" && mode !== "full") return { error: `finetune_mode: expected head or full, not ${JSON.stringify(mode)}` };
  const epochs = numberOr(body.num_epochs, DEFAULT_OPTIONS.epochs);
  if (!Number.isInteger(epochs) || epochs < 1) return { error: "num_epochs: expected a whole number of epochs, 1 or more" };
  const learningRate = numberOr(body.learning_rate, mode === "full" ? FULL_LEARNING_RATE : DEFAULT_OPTIONS.learningRate);
  if (!Number.isFinite(learningRate) || learningRate <= 0) return { error: "learning_rate: expected a positive number" };
  const loss = readLoss(body);
  if ("error" in loss) return loss;
  const pick = readSelectChannel(body.select_channel);
  if ("error" in pick) return pick;
  const selectChannel = pick.channel;
  const training = mode === "full" ? readFullTraining(body, epochs, learningRate, loss) : readHeadTraining(body, epochs, learningRate, loss);
  if ("error" in training) return training;
  return {
    options: { ...training.options, selectChannel },
    params: { ...training.params, ...(selectChannel === null ? {} : { select_channel: selectChannel }) },
  };
}

/** A body's select_channel as a channel's index (a whole number from 0, a
 * string spelling one as the form's select sends it), null when blank (no
 * channel picked), or why it is neither: whether the model has that channel
 * is for the route to say, with the model. */
export function readSelectChannel(value: unknown): { channel: number | null } | { error: string } {
  const channel = blank(value) ? null : Number(value);
  if (channel !== null && !(Number.isInteger(channel) && channel >= 0)) return { error: "select_channel: expected the index of one of the model's output channels, a whole number from 0" };
  return { channel };
}

/** readTraining's head part: the loss fields, and the batch in voxels (batch_pixels). */
function readHeadTraining(body: Record<string, unknown>, epochs: number, learningRate: number, loss: { options: LossFields; params: Record<string, unknown> }): { options: FinetuneOptions; params: Record<string, unknown> } | { error: string } {
  const batchPixels = numberOr(body.batch_pixels, DEFAULT_OPTIONS.batchPixels);
  if (!Number.isInteger(batchPixels) || batchPixels < 1 || batchPixels > MAX_BATCH_PIXELS) return { error: `batch_pixels: expected a whole number of voxels per step, from 1 to ${MAX_BATCH_PIXELS}` };
  return {
    options: { ...DEFAULT_OPTIONS, epochs, learningRate, ...loss.options, batchPixels },
    params: { num_epochs: epochs, learning_rate: learningRate, ...loss.params, batch_pixels: batchPixels, finetune_mode: "head" },
  };
}

/** Why `entry` cannot train with `channel` picked (null: none), or null
 * when it can: no pick for a model that trains one channel at a time
 * (channelChoice), in the server's words, the form's Channel named; a pick
 * for an affinity model, each of whose channels has a target of its own
 * and which train together (the server's cli refuses select_channel with
 * affinities); a pick past its channels. */
function refuseChannel(entry: ModelEntry, channel: number | null): string | null {
  const channels = entry.manifest.channels;
  if (channel === null) {
    return channelChoice(entry) ? `${entry.name} has ${channels.length} output channels and trains one at a time: pick the channel your paint is for (Channel, in the training form).` : null;
  }
  if (hasAffinityChannels(entry)) return `${entry.name} is an affinity model: its target has a channel of its own for each of its ${channels.length} channels, which train together, so it takes no channel pick (select_channel).`;
  return notAChannel(entry, channel);
}

/** Why `channel` is not one of `entry`'s output channels (it is past the
 * last), or null when it is one: a channel picked to train, to show (POST
 * /api/finetune/channel) or with the model used (rank.ts, Use this model). */
export function notAChannel(entry: ModelEntry, channel: number): string | null {
  const channels = entry.manifest.channels.length;
  return channel < channels ? null : `select_channel ${channel} is not one of ${entry.name}'s output channels: it has ${channels} (0 to ${channels - 1}).`;
}

/** `channel` made the one the session shows model `name`'s layer on (null:
 * the pick cleared), kept in the session's shownChannels and shown now on
 * the layers that open on it (serving.ts showChannel); a layer built for the
 * model later opens on it (showLayers), so a pick made before the model is
 * served is the channel its layer first shows. Answers whether a layer was
 * changed: none is for a model not served, and a pick cleared leaves the
 * layers as they are. */
export async function pickShownChannel(ctx: AppContext, name: string, channel: number | null): Promise<boolean> {
  const shownChannels = { ...ctx.session.state.shownChannels };
  if (channel === null) delete shownChannels[name];
  else shownChannels[name] = channel;
  await ctx.session.update({ shownChannels });
  return channel !== null && showChannel(ctx, name);
}

/** What carryPaint did: the new volume's voxels painted, the old volume's painted
 * voxels they came from, the new voxels left unpainted because their old voxels
 * disagreed, and the new volume's chunks written. */
export interface Carried { painted: number; source: number; conflicts: number; chunks: number }

/** `from`'s paint carried onto `to`, a volume planned over the same dataset
 * for model `name` on its grid (planVolume, not yet written): Use this
 * model's (rank.ts) and training's submit for a model on another grid. `to`
 * is written (createVolume), the paint carried onto it (carryPaint) with
 * its good regions (boxes in nm, the same on any grid), and made the
 * session's volume, its paint layer in `from`'s place; `from` is retired,
 * kept in the store but superseded (supersedeVolume), so the session has
 * one volume, the one it trains on. The chain and the channel shown are left
 * as they are: Use this model sets those itself. Answers carryPaint's
 * counts, the good regions carried and the new paint layer's name. */
export async function carryOnto(ctx: AppContext, from: PaintVolume, to: PaintVolume, name: string): Promise<Carried & { goodRegions: number; layer: string }> {
  await createVolume(ctx.paint, to);
  const carried = await carryPaint(ctx, from, to);
  const regions = await loadGoodRegions(ctx.paint, from.id);
  if (regions.length) await saveGoodRegions(ctx.paint, to.id, regions);
  await ctx.session.update({ paintVolume: to.id });
  await supersedeVolume(ctx.paint, from.id, to.id);
  const layer = paintLayerName(to.id);
  ctx.viewer.removeLayer(paintLayerName(from.id));
  ctx.viewer.setLayers({ [layer]: { ...paintLayer(paintLayerUrl(to.id, pageBase()), to), name: layer } });
  ctx.viewer.selectLayer(layer, PAINT_TAB);
  ctx.log(`Carried ${from.id}'s paint to ${to.id} for ${name}: ${carried.painted} voxels at ${to.output_voxel_size.join("×")} nm from ${carried.source} at ${from.output_voxel_size.join("×")} nm, in ${carried.chunks} chunk${carried.chunks === 1 ? "" : "s"}${carried.conflicts ? `; ${carried.conflicts} left unpainted where object and background met in one voxel` : ""}; ${from.id} is kept, superseded`);
  return { ...carried, goodRegions: regions.length, layer };
}

/** The portable Python's carry of paint from one grid to another
 * (cellmap_flow/portable/calls.py carry_paint, over portable/carry.py). */
const CARRY_PAINT = "cellmap_flow.portable.calls:carry_paint";

/** The paint of `from` copied into `to`, an empty volume over the same dataset on
 * another model's grid (createVolume'd): cellmap_flow.portable.carry, in the engine's
 * Python, with the paint store's reads and writes around it here. The old volume's
 * painted chunks go whole, both grids placed by their volume's dataset_offset_nm
 * (voxel 0's centre, fill.ts volumeCornerNm), and the new chunks that got paint come
 * back whole, to be written as they are. */
async function carryPaint(ctx: AppContext, from: PaintVolume, to: PaintVolume): Promise<Carried> {
  const grid = (v: PaintVolume) => ({ corner_nm: volumeCornerNm(v.dataset_offset_nm, v.output_voxel_size), voxel_nm: v.output_voxel_size, shape: v.dataset_shape_voxels, chunks: v.chunk_size });
  const indices: number[][] = [], labels: ArrayBuffer[] = [];
  for (const index of await paintedChunkIndices(ctx.paint, from.id)) {
    const chunk = await readPaintedChunk(ctx.paint, from, index);
    if (chunk) { indices.push(index); labels.push(bytesOf(chunk.labels)); }
  }
  const { value, arrays } = await ctx.infer.python<{ value: Carried & { indices: number[][] }; arrays: ArrayBuffer[] }>(
    CARRY_PAINT, { old: grid(from), new: grid(to), old_dtype: from.dtype, new_dtype: to.dtype, indices }, labels);
  await Promise.all(value.indices.map((index, k) => ctx.paint.put(to.id, CHUNK_DIR + chunkKey(index), new Uint8Array(arrays[k]))));
  const { painted, source, conflicts, chunks } = value;
  return { painted, source, conflicts, chunks };
}

/** A typed array's own bytes, copied: a call's arrays are handed to the worker. */
export function bytesOf(a: ArrayBufferView): ArrayBuffer {
  return a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength) as ArrayBuffer;
}

type LossFields = Pick<FinetuneOptions, "lossType" | "margin" | "balanceClasses" | "labelSmoothing" | "distillationLambda" | "distillationScope">;

/** The loss fields both modes train with, checked: the loss (Dice refused),
 * its margin, label smoothing (0 with the margin loss), class balance and
 * the distillation term's λ and scope. */
function readLoss(body: Record<string, unknown>): { options: LossFields; params: Record<string, unknown> } | { error: string } {
  const lossType = body.loss_type === undefined || body.loss_type === null || body.loss_type === "" ? DEFAULT_OPTIONS.lossType : (body.loss_type as HeadLoss);
  if (!HEAD_LOSSES.includes(lossType)) return { error: refuseLoss(lossType)! };
  const margin = numberOr(body.margin, DEFAULT_OPTIONS.margin);
  if (!(margin >= 0 && margin <= 0.5)) return { error: "margin: expected a number from 0 to 0.5" };
  let labelSmoothing = numberOr(body.label_smoothing, DEFAULT_OPTIONS.labelSmoothing);
  if (!(labelSmoothing >= 0 && labelSmoothing < 1)) return { error: "label_smoothing: expected a number from 0 up to, not including, 1" };
  if (lossType === "margin") labelSmoothing = 0;
  const balanceClasses = typeof body.balance_classes === "boolean" ? body.balance_classes : DEFAULT_OPTIONS.balanceClasses;
  const distillationLambda = numberOr(body.distillation_lambda, DEFAULT_OPTIONS.distillationLambda);
  if (!Number.isFinite(distillationLambda) || distillationLambda < 0) return { error: "distillation_lambda: expected a number, 0 or more (0 switches distillation off)" };
  const scope = body.distillation_scope === undefined || body.distillation_scope === null || body.distillation_scope === "" ? DEFAULT_OPTIONS.distillationScope : String(body.distillation_scope).toLowerCase();
  if (scope !== "unlabeled" && scope !== "all") return { error: "distillation_scope: expected unlabeled or all" };
  const distillationScope = scope as DistillationScope;
  return {
    options: { lossType, margin, balanceClasses, labelSmoothing, distillationLambda, distillationScope },
    params: {
      loss_type: lossType,
      ...(lossType === "margin" ? { margin } : {}),
      balance_classes: balanceClasses,
      label_smoothing: labelSmoothing,
      distillation_lambda: distillationLambda,
      distillation_scope: distillationScope,
    },
  };
}

/** readTraining's whole-network part: the loss fields as the head's, the
 * batch in tiles (batch_size, 1 to MAX_BATCH_PATCHES), the tile side, or a
 * 3D model's block side (tile_size, a whole number; blank for the default),
 * which submit then checks against the model's graph, and augmentation
 * (augment). */
function readFullTraining(body: Record<string, unknown>, epochs: number, learningRate: number, loss: { options: LossFields; params: Record<string, unknown> }): { options: FinetuneOptions; params: Record<string, unknown> } | { error: string } {
  const batchPatches = numberOr(body.batch_size, DEFAULT_OPTIONS.batchPatches);
  if (!Number.isInteger(batchPatches) || batchPatches < 1 || batchPatches > MAX_BATCH_PATCHES) return { error: `batch_size: expected a whole number of tiles per step, from 1 to ${MAX_BATCH_PATCHES}` };
  const tile = body.tile_size === undefined || body.tile_size === null || body.tile_size === "" ? null : Number(body.tile_size);
  if (tile !== null && (!Number.isInteger(tile) || tile < 1)) return { error: "tile_size: expected a whole number of pixels on a side" };
  const augment = typeof body.augment === "boolean" ? body.augment : DEFAULT_OPTIONS.augment;
  return {
    options: { ...DEFAULT_OPTIONS, mode: "full", epochs, learningRate, ...loss.options, batchPatches, tileSize: tile, augment },
    params: {
      num_epochs: epochs,
      learning_rate: learningRate,
      ...loss.params,
      batch_size: batchPatches,
      ...(tile === null ? {} : { tile_size: tile }),
      augment,
      finetune_mode: "full",
    },
  };
}

/** A body field as a number (the form sends numbers; a string is read), or `fallback` when absent. */
function numberOr(value: unknown, fallback: number): number {
  if (blank(value)) return fallback;
  return typeof value === "number" ? value : Number(value);
}

/** Whether a body field is absent: undefined, null, or the form's blank "". */
function blank(value: unknown): boolean {
  return value === undefined || value === null || value === "";
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

/** A finetuned model's bytes as they download: the ONNX the catalog keeps,
 * with the record's manifest (the base's under the new name, its one variant
 * this file, `<name>.onnx`) as JSON in the model's metadata_props under
 * MANIFEST_METADATA_KEY. Written here rather than when the model is made,
 * so that the kept bytes stay the ones the run wrote (served, and the base
 * of a later finetune, as they are) and a model kept before this was added
 * downloads with its manifest too. The manifest says what the file alone
 * does not: its voxel sizes, its block, its channels, what its output is
 * and the normalization it was trained behind. */
export function downloadableOnnx(record: FinetunedRecord, bytes: ArrayBuffer): ArrayBuffer {
  return withMetadata(new Uint8Array(bytes), { [MANIFEST_METADATA_KEY]: JSON.stringify(record.entry.manifest) }).buffer;
}
