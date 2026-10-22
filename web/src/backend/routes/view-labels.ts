// The Finetune tab's "Label it" tools in the browser: the routes of
// dashboard/routes/finetune/view_labels.py, answered in the page with the
// same JSON, so the tab's label-it.js drives both. They act on the box on
// screen: one model output patch (one chunk of the volume) centred where
// the viewer looks, or `box_voxels` annotation voxels when the request
// gives them.
//
// - POST .../seed: the chosen model's prediction there, segmented into
//   objects (threshold and connected components, or the mutex watershed
//   for an affinity model), each written with an id of its own (2 and up),
//   background 1. The user then cleans it up with the brush, which is far
//   quicker than painting a whole object from nothing. The seed's maths is
//   the Python the server runs (cellmap_flow/portable/label_it.py, called
//   through label_it_calls.py in the engine's Pyodide worker): the
//   request's options, the channels it reads, where the prediction lies on
//   the volume's grid, the probability and the components; only the mutex
//   watershed itself runs here (label-it/segment.ts says why).
// - POST .../background: every unannotated voxel 1, for a region the model
//   fills with false positives.
// - POST .../split: the box's foreground relabelled by connected component,
//   so a background wall painted through a merged object gives it two ids.
// - POST .../undo: the last of those taken back, or the last accepted AI
//   annotation, whichever came last.
// - GET .../sources: the models a seed can read, the one it reads when none
//   is chosen, the methods each offers, whether Undo has anything to take
//   back (the page polls it), and the box a patch is.
//
// Seed and All Background fill only unannotated voxels (a voxel the user
// painted is a decision, a seed is a guess); Split writes only foreground
// voxels whose id changes. Every write goes into the paint store
// (ai-annotate/fill.ts) and pushes an undo entry onto the volume's stack.
// The stacks are this module's, as the server's _undo_stack is
// view_labels.py's, and the AI annotation's Accept pushes onto the same one
// (ai_annotate.py imports it; createRouter hands it to routes/ai-annotate.ts),
// so one Undo takes back whichever came last, ten deep. After a write the
// viewer reads the paint layer again (refreshLayer), as the server's
// refresh_annotation_layer has it do.
//
// What differs from the server, and why:
// - The prediction is the page's own (Inference.readOutput: the model
//   stage's output, the chain's normalizers and none of its postprocessors,
//   as the server reads its model server's array with no postprocessing),
//   read by label-it/prediction.ts. The "running servers" are the served
//   models, finetuned ones among them, and the default is the volume
//   model's latest finetune served, named `<model>-ft<n>` here
//   (models/finetuned.ts) where the server's are `<model>_finetuned...`,
//   else the model itself, else the only one served.
// - The methods: components for every model, the mutex watershed for an
//   affinity model (methodsFor says why distance_watershed and instances
//   are not offered). An affinity model's offsets are the ones its channel
//   names give, by the server's rule for a model whose script says none
//   (label_it.affinity_offsets).
// - No MinIO: nothing to check is serving, nothing to pull to disk after.
// - A seed reads the dataset the engine has open, so a volume painted on
//   another dataset is refused, as an AI annotation run is; and the Python
//   runs in that dataset's engine, so with no dataset open a seed is
//   refused before anything else.
//
// The refusals keep the Python's order, statuses and words (where a word
// was the server's own, "server", the browser's), and the answers its keys:
// the page keys on `needs_volume` and `needs_confirmation`.
import { boxVoxels, paintBox, restoreBox, rewriteForeground } from "../../ai-annotate/fill";
import { relabelObjects, seedLabels } from "../../ai-annotate/labels";
import { AIAnnotateError, type Labels, type UndoEntry } from "../../ai-annotate/types";
import type { AppContext, ModelEntry, PaintVolume, ServedModel } from "../../contracts";
import { hasAffinityChannels, readVolume } from "../../finetune/volume";
import { PYTHON } from "../../generated/python-data";
import { asConnectivity, asBoolLike } from "../../infer/label";
import { type Placement, placementArgs, readRegion, servedLevel } from "../../label-it/prediction";
import { type Block, bufferOf, LABEL_IT, mutexObjects } from "../../label-it/segment";
import { paintLayerName } from "../../viewer/layers";
import type { Router } from "../router";
import { json, message, objectBody } from "../router";

const BASE = "/api/finetune/view-labels";

/** Past this many voxels (128^3) the page asks before labelling
 * (view_labels.LARGE_BOX_VOXELS): a seed of that size is a lot to clean up,
 * and all-background a lot to undo. One output patch, the default box, is
 * well under it for every model the browser runs. */
export const LARGE_BOX_VOXELS = PYTHON.finetune.large_box_voxels;

/** A request these routes will not act on (view_labels._Refused): the
 * status, a message the page shows, and extra answer fields. */
class Refused extends Error {
  constructor(
    message: string,
    readonly status = 409,
    readonly extra: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "Refused";
  }
}

/** The patch tools' state, returned for tests and for the AI annotation's
 * Accept: the undo stacks, by volume id. */
export interface ViewLabelsState {
  undo: Map<string, UndoEntry[]>;
}

/** Register the routes; returns their state, which lives as long as the router (the page). */
export function viewLabelsRoutes(router: Router): ViewLabelsState {
  const state: ViewLabelsState = { undo: new Map() };

  /** The volume's undo stack (view_labels._undo_stack); pushUndo keeps it UNDO_DEPTH deep. */
  function undoStack(volumeId: string): UndoEntry[] {
    let stack = state.undo.get(volumeId);
    if (!stack) state.undo.set(volumeId, (stack = []));
    return stack;
  }

  async function sessionVolume(ctx: AppContext): Promise<PaintVolume | null> {
    const id = ctx.session.state.paintVolume;
    return id ? readVolume(ctx.paint, id) : null;
  }

  /** The session's volume and the box's voxels in it (_target_box). */
  async function targetBox(ctx: AppContext, body: Record<string, unknown>): Promise<{ volume: PaintVolume; lo: number[]; hi: number[] }> {
    const volume = await sessionVolume(ctx);
    if (!volume) throw new Refused("No annotation volume to label. Create or resume one first.", 409, { needs_volume: true });
    let sizeNm = body.size_nm;
    if (body.box_voxels !== undefined && body.box_voxels !== null) sizeNm = boxNm(body.box_voxels, volume);
    const { centre, size } = viewBoxNm(ctx, sizeNm, volume);
    const box = boxVoxels(volume, centre.map((c, a) => c - size[a] / 2), size, volume.dataset_shape_voxels);
    if (!box) throw new Refused("The view is outside the annotation volume.", 400);
    return { volume, lo: box.lo, hi: box.hi };
  }

  /** Fill the box's unannotated voxels with `labelsFor`'s labels (_fill): the answer's counts. */
  async function fill(ctx: AppContext, volume: PaintVolume, lo: number[], hi: number[], labelsFor: (existing: Labels) => Labels) {
    const stack = undoStack(volume.id);
    const counts = await paintBox(ctx.paint, volume, lo, hi, labelsFor, { overwrite: false, undo: stack });
    const wrote = counts.n_foreground + counts.n_background > 0;
    // The viewer shows the chunks it already read until the paint layer is read again.
    const layerRefreshed = wrote ? ctx.viewer.refreshLayer(paintLayerName(volume.id)) : false;
    ctx.log(`Labelled box ${lo.join(",")}..${hi.join(",")} of ${volume.id}: ${counts.n_foreground} foreground and ${counts.n_background} background voxels filled`);
    return {
      success: true,
      offset_voxels: lo,
      shape_voxels: hi.map((h, a) => h - lo[a]),
      filled_foreground: counts.n_foreground,
      filled_background: counts.n_background,
      reload_viewer: wrote,
      layer_refreshed: layerRefreshed,
      can_undo: stack.length > 0,
    };
  }

  /** A refusal's answer, or the 500 of anything else (as the Python's catch-all). */
  function answerError(ctx: AppContext, e: unknown, what: string) {
    if (e instanceof Refused) return json(e.status, { success: false, error: e.message, ...e.extra });
    // fill's ids running out (freshIds): a ValueError there, 400.
    if (e instanceof AIAnnotateError) return json(400, { success: false, error: e.userMessage });
    ctx.log(`Error ${what}: ${message(e)}`);
    return json(500, { success: false, error: message(e) });
  }

  // GET .../sources: the models a seed can read, for the page's picker:
  // `models` (every served model, oldest first), `default`, the one a seed
  // reads when none is chosen (null when that is a choice for the user),
  // `methods`, {model: the seed methods that fit its output, best first};
  // `can_undo`, since the page polls this; and `box_voxels`, the volume's
  // output patch (its chunk, z, y, x), the box a label action covers when
  // it gives no other.
  router.get(`${BASE}/sources`, async (_req, _params, ctx) => {
    const volumeId = ctx.session.state.paintVolume;
    const volume = await sessionVolume(ctx);
    const sources = ctx.infer.served();
    const methods: Record<string, string[]> = {};
    for (const s of sources) methods[s.entry.name] = methodsFor(outputTypeOf(s.entry));
    return json(200, {
      success: true,
      models: sources.map((s) => s.entry.name),
      default: defaultSource(sources, volume?.model_name ?? null)?.entry.name ?? null,
      methods,
      can_undo: Boolean(volumeId && state.undo.get(volumeId)?.length),
      box_voxels: volume?.chunk_size ?? null,
    });
  });

  // POST .../seed: fill the unannotated voxels of the box on screen from a
  // model's prediction. JSON body, all optional: size_nm or box_voxels
  // (the box), confirm (label a large box), select_channel (the one channel
  // to threshold), model (a name from sources; the default when absent),
  // method (one the model offers; components when absent), threshold (a
  // probability, 0 to 1; the model's own boundary, 0.5, when absent),
  // min_size (objects of fewer voxels are background), connectivity (1
  // faces, the default; 2 and edges; 3 all neighbours) and per_slice
  // (components per z slice).
  router.post(`${BASE}/seed`, async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    try {
      // The Python runs in the open dataset's engine, and a seed reads that
      // dataset: without one there is nothing to seed from.
      const dataset = ctx.infer.dataset();
      if (!dataset) throw new Refused(NO_DATASET);
      // The request's options as the server reads them, its defaults and its
      // refusals (label_it.seed_options), before anything else, as there.
      const { options } = refusedOr(await ctx.infer.python<{ options: SeedOptions } | Refusal>(`${LABEL_IT}:seed_options`, body));
      const method = options.method;

      const { volume, lo, hi } = await targetBox(ctx, body);
      const confirmation = confirmationNeeded(body, lo, hi);
      if (confirmation) return confirmation;
      const base = volume.model_name;
      if (!base) throw new Refused("The annotation volume does not say which model it is for.");
      const served = predictionSource(ctx.infer.served(), base, typeof body.model === "string" && body.model ? body.model : null);
      if (!served) throw new Refused(`${base} is not served. Serve it from the Models tab, or pick a model.`);
      // The prediction is the open dataset's: a volume painted on another
      // one would get it at the wrong place.
      if (dataset.url !== volume.dataset_path) {
        throw new Refused(`The annotation volume was painted on ${volume.dataset_path}, but the viewer shows ${dataset.url}: open that dataset again, or make a new volume on this one.`);
      }
      const name = served.entry.name;
      // What the seed reads of the model (label_it.seed_channels, the
      // offsets by its channel names) and where the box lies on its level
      // (label_it.placement), by the Python.
      const plan = refusedOr(await ctx.infer.python<SeedPlan | Refusal>(`${LABEL_IT}:seed_plan`, {
        channel_names: served.entry.manifest.channels,
        select_channel: body.select_channel ?? null,
        ...placementArgs(volume, lo, hi, servedLevel(dataset, served.entry)),
      }));
      const outputType = outputTypeOf(served.entry);
      const offered = methodsFor(outputType);
      if (!offered.includes(method)) throw new Refused(`${name} cannot seed by ${method}; it offers ${offered.join(", ")}`, 400);
      const placement = plan.placement;
      if (!placement) throw new Refused("The view is outside what the model predicts.", 400);

      let region: Block;
      try {
        region = await readRegion(ctx.infer, served.view, placement);
      } catch (e) {
        ctx.log(`Could not read the prediction to seed from: ${message(e)}`);
        throw new Refused(`Could not read the model's prediction: ${message(e)}`, 502);
      }
      const objects = await seedObjects(ctx, options, region, placement, plan, served.info.output_class);
      // An instance target needs ids apart from its neighbours'; a selected
      // channel trains as a binary target. Ids counting up need room: an
      // instance target on a uint16 volume gets them, a uint8 volume reuses
      // free ids.
      const instanceTarget = outputType === "affinities" && (body.select_channel === undefined || body.select_channel === null);
      const labelsFor = (existing: Labels) => seedLabels(objects, existing, instanceTarget && existing instanceof Uint16Array);
      const answer = await fill(ctx, volume, placement.lo, placement.hi, labelsFor);
      return json(200, { ...answer, model: name, threshold: body.threshold ?? null, method });
    } catch (e) {
      return answerError(ctx, e, "seeding the view from the prediction");
    }
  });

  // POST .../background: label every unannotated voxel of the box on
  // screen background (1). JSON body, all optional: size_nm or box_voxels,
  // and confirm, as for seed.
  router.post(`${BASE}/background`, async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    try {
      const { volume, lo, hi } = await targetBox(ctx, body);
      const confirmation = confirmationNeeded(body, lo, hi);
      if (confirmation) return confirmation;
      return json(200, await fill(ctx, volume, lo, hi, (existing) => (existing instanceof Uint16Array ? new Uint16Array(existing.length) : new Uint8Array(existing.length)).fill(1)));
    } catch (e) {
      return answerError(ctx, e, "labelling the view background");
    }
  });

  // POST .../split: relabel the box on screen's foreground by connected
  // component (labels.ts relabelObjects). For a merge the model made: paint
  // a background wall through the object, in every slice it spans, and
  // click; the two sides get ids of their own. A stroke joining two objects
  // merges them. JSON body, all optional: size_nm or box_voxels, and
  // confirm, as for seed; connectivity and per_slice, as for a seed's
  // components (with per_slice, a wall in one slice is enough, and each
  // slice of an object gets an id of its own).
  router.post(`${BASE}/split`, async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    try {
      const connectivity = valueOr400(() => asConnectivity((body.connectivity ?? 1) as never, 3));
      const perSlice = asBoolLike(body.per_slice as never);
      const { volume, lo, hi } = await targetBox(ctx, body);
      const confirmation = confirmationNeeded(body, lo, hi);
      if (confirmation) return confirmation;
      const shape = hi.map((h, a) => h - lo[a]);
      const stack = undoStack(volume.id);
      const { changed, counts } = await rewriteForeground(ctx.paint, volume, lo, hi, (existing) => relabelObjects(existing, shape, { connectivity, perSlice }), { undo: stack });
      const layerRefreshed = changed ? ctx.viewer.refreshLayer(paintLayerName(volume.id)) : false;
      ctx.log(`Relabelled box ${lo.join(",")}..${hi.join(",")} of ${volume.id}: ${counts.objects} objects, ${counts.split} split off, ${counts.merged} merged, ${changed} voxels changed`);
      return json(200, {
        success: true,
        offset_voxels: lo,
        shape_voxels: shape,
        changed,
        reload_viewer: changed > 0,
        layer_refreshed: layerRefreshed,
        can_undo: stack.length > 0,
        ...counts,
      });
    } catch (e) {
      return answerError(ctx, e, "splitting the view's objects");
    }
  });

  // POST .../undo: take back the last Seed, All Background, Split or
  // accepted AI annotation of the session's volume. The box goes back to
  // what it held before, on the voxels still as that action left them: a
  // stroke painted there since is kept. 409 when there is nothing to undo.
  router.post(`${BASE}/undo`, async (_req, _params, ctx) => {
    try {
      const volume = await sessionVolume(ctx);
      if (!volume) throw new Refused("No annotation volume to undo in.");
      const stack = undoStack(volume.id);
      const entry = stack.pop();
      if (!entry) throw new Refused("Nothing to undo.");
      const restored = await restoreBox(ctx.paint, volume, entry);
      const layerRefreshed = restored ? ctx.viewer.refreshLayer(paintLayerName(volume.id)) : false;
      ctx.log(`Undid the label action on box ${entry.lo.join(",")}..${entry.hi.join(",")} of ${volume.id}: ${restored} voxels restored`);
      return json(200, { success: true, restored, reload_viewer: restored > 0, layer_refreshed: layerRefreshed, can_undo: stack.length > 0 });
    } catch (e) {
      return answerError(ctx, e, "undoing the label action");
    }
  });

  return state;
}

// ---------------------------------------------------------------------------
// The seed
// ---------------------------------------------------------------------------

const NO_DATASET = "No dataset is open. Open the one the annotation volume was painted on.";

/** A refusal the Python answers in place of its value: the message the
 * server answers 400 with (a ValueError there). */
interface Refusal {
  error: string;
}

/** `answer`, or its refusal thrown as the 400 the server answers. */
function refusedOr<T extends object>(answer: T | Refusal): T {
  if ("error" in answer) throw new Refused(answer.error, 400);
  return answer;
}

/** A seed's options, as label_it.seed_options reads them from the body. */
interface SeedOptions {
  method: string;
  threshold: number;
  min_size: number;
  connectivity: number;
  per_slice: boolean;
}

/** What a seed reads (label_it_calls.seed_plan): `channels` [start, stop)
 * (null: every one), an affinity model's `offsets`, and where the box lies
 * on the model's level (null: it does not). */
interface SeedPlan {
  channels: [number, number] | null;
  offsets: number[][] | null;
  placement: Placement | null;
}

/** What a model is trained on, as the browser tells it (the server's
 * autodetect_output_type, by its channel-name test): "affinities" for a
 * model with a channel named `*_aff*`, else "binary". */
type OutputType = "affinities" | "binary";

function outputTypeOf(entry: ModelEntry): OutputType {
  return hasAffinityChannels(entry) ? "affinities" : "binary";
}

/** The seed methods that fit a model's output, the best fit first
 * (view_labels.methods_for): the mutex watershed for affinities, then
 * components, which fits every output and so is always offered, last.
 *
 * The server's other two fit outputs no browser model has:
 * "distance_watershed" (a distance map's watershed from its h-maxima) and
 * "instances" (a model that serves integer ids, Cellpose's masks). The
 * browser's models are those of Hugging Face's cellmap/onnx-web-models,
 * affinity models (channels mito_aff_z/y/x) and binary ones (nuc, er), all
 * raw logits, so neither is ever offered, and a request for one is refused
 * as the server refuses a method the model does not offer (the page lists
 * them greyed out, saying what they need). */
function methodsFor(outputType: OutputType): string[] {
  return outputType === "affinities" ? ["mutex_watershed", "components"] : ["components"];
}

/** The objects the seed's method makes of `region` (the served box
 * `placement` reads), 0 background and 1..n, over the placement's box, z,
 * y, x: for components, the Python's (label_it_calls.seed); for the mutex
 * watershed, its input from the Python and the watershed here
 * (label-it/segment.ts mutexObjects). `region` is handed to the worker. */
async function seedObjects(ctx: AppContext, options: SeedOptions, region: Block, placement: Placement, plan: SeedPlan, outputClass: string | null | undefined): Promise<Uint32Array> {
  const args = { ...options, output_class: outputClass ?? null, channels: plan.channels, offsets: plan.offsets, indices: placement.indices, shape: region.shape };
  const out = refusedOr(await ctx.infer.python<{ value: { shape?: number[] }; arrays: ArrayBuffer[] } | Refusal>(`${LABEL_IT}:seed`, args, [bufferOf(region.data)]));
  if (options.method === "mutex_watershed") {
    return mutexObjects({ data: new Float32Array(out.arrays[0]), shape: out.value.shape! }, plan.offsets!, options.threshold, options.min_size);
  }
  return new Uint32Array(out.arrays[0]);
}

// ---------------------------------------------------------------------------
// The box and the model
// ---------------------------------------------------------------------------

/** The size in nm of a box of `boxVoxels` annotation voxels (z, y, x, or one
 * number for every axis), each at least 1 (view_labels._box_nm). Any size:
 * the box only says which of the prediction to copy, and the model reads its
 * own input around it whatever its size. */
function boxNm(boxVoxels: unknown, volume: PaintVolume): number[] {
  const refusal = new Refused(`The box is z, y, x voxels, or one number, each at least 1; got ${pyStr(boxVoxels)}`, 400);
  const voxels = (Array.isArray(boxVoxels) ? boxVoxels : [boxVoxels]).map((v) => (typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : NaN));
  if ((voxels.length !== 1 && voxels.length !== 3) || !voxels.every((v) => Number.isFinite(v) && v >= 1)) throw refusal;
  const three = voxels.length === 1 ? [voxels[0], voxels[0], voxels[0]] : voxels;
  return three.map((v, a) => v * volume.output_voxel_size[a]);
}

/** The box centred where the viewer looks (good_regions.view_box_nm): its
 * centre and size in nm. `sizeNm` defaults (when absent, 0 or empty, as
 * Python's `size_nm or default`) to one model output patch, the volume's
 * chunk; one number is the same along every axis. */
function viewBoxNm(ctx: AppContext, sizeNm: unknown, volume: PaintVolume): { centre: number[]; size: number[] } {
  const centre = ctx.viewer.position();
  if (!centre) throw new Refused("Viewer has no position", 400);
  const empty = sizeNm === undefined || sizeNm === null || sizeNm === 0 || sizeNm === "" || (Array.isArray(sizeNm) && !sizeNm.length);
  const given = empty ? volume.chunk_size.map((n, a) => n * volume.output_voxel_size[a]) : Array.isArray(sizeNm) ? sizeNm : [sizeNm];
  const size = given.map((v) => (typeof v === "number" ? v : Number(v)));
  if ((size.length !== 1 && size.length !== 3) || size.some((v) => !Number.isFinite(v))) throw new Refused(`size_nm must be one number or three (z, y, x), got ${pyStr(sizeNm)}`, 400);
  const three = size.length === 1 ? [size[0], size[0], size[0]] : size;
  if (three.some((v) => v <= 0)) throw new Refused("size_nm must be positive", 400);
  return { centre, size: three };
}

/** The 409 asking the page to confirm a large box, or null (_confirmation_needed). */
function confirmationNeeded(body: Record<string, unknown>, lo: number[], hi: number[]) {
  const voxels = hi.reduce((n, h, a) => n * (h - lo[a]), 1);
  if (voxels <= LARGE_BOX_VOXELS || body.confirm) return null;
  return json(409, { success: false, error: `The box is ${voxels.toLocaleString("en-US")} voxels. Label it all?`, needs_confirmation: true, voxels });
}

/** Whether `name` is a model finetuned in this browser from `base`, or a
 * later one of its series (models/finetuned.ts nextFinetunedName: `x-ft1`,
 * then `x-ft2` from either), and how late: its number, or null. */
function finetuneNumber(name: string, base: string): number | null {
  const series = base.replace(/-ft\d+$/, "");
  const after = /-ft(\d+)$/.exec(base);
  const baseNumber = after ? Number(after[1]) : 0;
  if (!name.startsWith(`${series}-ft`)) return null;
  const rest = name.slice(series.length + 3);
  if (!/^\d+$/.test(rest)) return null;
  const n = Number(rest);
  return n > baseNumber ? n : null;
}

/** The served model a seed reads when none is chosen (_prediction_server's
 * default): the latest finetune of the volume's model served (the last
 * served of them), the model as it is now; else the model itself; else the
 * only model served; else null, a choice for the user. */
function defaultSource(sources: ServedModel[], base: string | null): ServedModel | null {
  if (base) {
    const finetuned = sources.filter((s) => finetuneNumber(s.entry.name, base) !== null);
    const own = sources.filter((s) => s.entry.name === base);
    const pick = finetuned.length ? finetuned : own;
    if (pick.length) return pick[pick.length - 1];
  }
  return sources.length === 1 ? sources[0] : null;
}

/** The served model whose prediction a seed reads: `chosen` when the page
 * names one (refused when it is not served), else defaultSource's. */
function predictionSource(sources: ServedModel[], base: string, chosen: string | null): ServedModel | null {
  if (chosen) {
    const found = sources.find((s) => s.entry.name === chosen);
    if (!found) throw new Refused(`${chosen} is not served. Pick another model to seed from.`);
    return found;
  }
  return defaultSource(sources, base);
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/** `fn`'s value, its error a 400 with the error's message: the Python's
 * ValueError from parsing a field. */
function valueOr400<T>(fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    if (e instanceof Refused) throw e;
    throw new Refused(message(e), 400);
  }
}

/** A value as Python's str() shows it in a refusal: a list as `[1, 2]`, None for null. */
function pyStr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (Array.isArray(value)) return `[${value.map(pyStr).join(", ")}]`;
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "string") return value;
  return String(value);
}
