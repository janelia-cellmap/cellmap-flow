// The one AI annotation of the page: what `ai["job"]` is on the server
// (dashboard/routes/finetune/ai_annotate.py: _idle_job, _run_job,
// _ask_and_stage, _fail, _resend_job and the write of ai_annotate_accept),
// for the browser mode, where the page itself reads the plane, calls the
// provider and writes the paint. One job at a time, as there: a staged
// result must be accepted or rejected before the next run, since two
// previews of overlapping planes would be confusing to review and could be
// accepted out of order.
//
// The server runs the job in a thread and the routes read it under a lock;
// here it is an async function and the routes read the fields between its
// awaits, which is all the status route needs: every field it answers
// (statusAnswer) is set in order as the job moves. Staging is in memory: the
// server writes the request image, the model's image and the mask to a
// directory under the corrections dir (ai_annotate/staging.py); the job
// holds them, and the preview (three PNGs, base64, as staging.preview) is
// built once when the result is ready and answered with `?preview=1`.
//
// What the job needs of the rest of the feature comes from the ai-annotate
// modules: the plane's plan and read (geometry.ts, plane.ts), the mask and
// the labels (mask.ts, plane.ts), the write and its undo (fill.ts), the
// prompt (prompts.ts), the audit log (audit.ts) and the day's count
// (usage.ts). The provider and the image codec are the routes' deps.
import { record, type AuditEvent } from "../ai-annotate/audit";
import { paintBox, readBox } from "../ai-annotate/fill";
import { planPlane, screenShape, toScreen, writeShape } from "../ai-annotate/geometry";
import { bytesToBase64 as base64 } from "../ai-annotate/gemini";
import { maskFraction, overlay } from "../ai-annotate/mask";
import { buildRequest, labelsForBox, maskToWriteShape, rawExtentNm, readPlane } from "../ai-annotate/plane";
import { buildRecolorPrompt, editablePrompt } from "../ai-annotate/prompts";
import type {
  Credentials,
  DepthAxis,
  ImageCodec,
  JobStatus,
  Labels,
  OrganelleProfile,
  PlanePlan,
  Preview,
  Provider,
  SegmentRequest,
  SegmentResult,
  Stage,
  StatusAnswer,
  UndoEntry,
} from "../ai-annotate/types";
import { AIAnnotateError, PLANE_NAMES, STAGE_LABELS } from "../ai-annotate/types";
import type { Usage } from "../ai-annotate/usage";
import type { AppContext, PaintStore, PaintVolume } from "../contracts";
import { hasAffinityChannels, readVolume } from "../finetune/volume";
import { paintLayerName } from "../viewer/layers";
import { message } from "./router";

/** What the page is told when the job fails for a reason that is not an
 * AIAnnotateError's (a bug, a store that would not write): the server says
 * "see the server log"; the browser's log is the dashboard's status area
 * and the console (ctx.log). */
export const GENERIC_FAILURE = "AI annotation failed; see the dashboard log.";

/** Failures that happen before the model sees the request (signing in, the
 * config): the call costs nothing, so it is not counted against the day's limit. */
const NOT_SENT: ReadonlySet<string> = new Set(["auth", "config"]);

/** A request the routes will not act on (the Python's _Refused): the
 * answer's status, its message and extra answer fields (`needs_volume`,
 * `needs_acknowledgement`, ...). */
export class Refused extends Error {
  constructor(
    message: string,
    readonly status = 409,
    readonly extra: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "Refused";
  }
}

/** What the job runs with: the context, the codec for the preview's PNGs,
 * the day's count and its cap, the plane's size, the provider's credentials
 * as entered so far (read when a call is made, so a key entered after a
 * failure is used by the resend), and the clock. */
export interface JobDeps {
  ctx: AppContext;
  codec: ImageCodec;
  usage: Usage;
  dailyCallLimit: number;
  cropSizePx: number;
  credentials: (provider: string) => Credentials | null;
  now: () => Date;
}

/** What a run is of: the provider and model, the target, the plane's point
 * and axis, the volume written and the dataset read, the structure's
 * profile and the user's prompt override (null for the catalog's text). */
export interface JobArgs {
  annotateId: string;
  provider: Provider;
  providerId: string;
  model: string;
  labelName: string;
  depthAxis: DepthAxis;
  /** The viewer's display dimensions (viewer.viewDisplayDimensions): the
   * image is laid out as its panel draws the plane. */
  displayDimensions?: string[] | null;
  pointNm: number[];
  volume: PaintVolume;
  dataset: string | null;
  profile: OrganelleProfile;
  override: string | null;
}

/** What accepting wrote (ai_annotate_accept's answer, less can_undo, which the routes know). */
export interface AcceptResult {
  filled_foreground: number;
  filled_background: number;
  overwritten: number;
  reload_viewer: boolean;
  layer_refreshed: boolean;
}

/** A fresh annotation id: 32 lowercase hex characters (staging.new_annotate_id). */
export function newAnnotateId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export class AiAnnotateJob {
  status: JobStatus;
  stage: Stage | null;
  error: string | null;
  prompt: string | null;
  /** The annotation voxels written, [lo, hi), set once the plane is planned. */
  writeBox: { lo: number[]; hi: number[] } | null = null;
  maskFraction: number | null = null;
  preview: Preview | null = null;
  /** A Shift+G refused for want of an annotation volume, for the page to offer one. */
  readonly needsVolume: boolean;
  /** The user's prompt override as of the last run or resend. */
  override: string | null;
  // What the worker and the decisions need, not answered.
  plan: PlanePlan | null = null;
  request: SegmentRequest | null = null;
  result: SegmentResult | null = null;
  /** The mask's objects on the write box's in-plane voxels (plane.ts maskToWriteShape). */
  maskWrite: Int32Array | null = null;
  private current: Promise<void> = Promise.resolve();

  /** A job about to run (`args`), or, with `failure`, the record of a start
   * that was refused (the Python's _start_from_key: a failed job with no id,
   * shown by the status route until the next run). */
  constructor(
    private readonly deps: JobDeps,
    readonly args: JobArgs | null,
    failure?: { error: string; needsVolume: boolean },
  ) {
    if (failure) {
      this.status = "failed";
      this.stage = null;
      this.error = failure.error;
      this.needsVolume = failure.needsVolume;
      this.prompt = null;
      this.override = null;
      return;
    }
    if (!args) throw new Error("a job needs its arguments or a failure");
    this.status = "running";
    this.stage = "fetching_crop";
    this.error = null;
    this.needsVolume = false;
    this.override = args.override;
    this.prompt = args.override ?? editablePrompt(args.profile);
  }

  get annotateId(): string | null {
    return this.args?.annotateId ?? null;
  }

  /** Whether the job has a staged plane a resend can send again. */
  hasPlane(): boolean {
    return this.request !== null && this.plan !== null;
  }

  // -------------------------------------------------------------- status

  /** What GET …/status answers for this job (ai_annotate_status): a result
   * being written by Accept is reported as ready, its last state the page
   * can show; the preview only when asked for and ready. */
  statusAnswer(withPreview: boolean): StatusAnswer {
    const status = this.status === "accepting" ? "ready" : this.status;
    const depthAxis = this.args?.depthAxis ?? null;
    return {
      success: true,
      annotate_id: this.annotateId,
      status,
      stage: this.stage,
      stage_label: this.stage ? STAGE_LABELS[this.stage] : null,
      error: this.error,
      prompt: this.prompt,
      provider: this.args?.providerId ?? null,
      model: this.args?.model ?? null,
      label_name: this.args?.labelName ?? null,
      depth_axis: depthAxis,
      plane: depthAxis === null ? null : PLANE_NAMES[depthAxis],
      point_nm: this.args?.pointNm ?? null,
      write_box: this.writeBox,
      mask_fraction: this.maskFraction,
      preview: status === "ready" && withPreview ? this.preview : null,
      needs_volume: this.needsVolume,
    };
  }

  /** The end of the current run or resend (for whoever must wait for it, such as a test). */
  settled(): Promise<void> {
    return this.current;
  }

  // ----------------------------------------------------------------- run

  /** Start the job: it runs from here, and the routes answer meanwhile. */
  start(): void {
    this.current = this.runJob();
  }

  /** _run_job: read the plane, ask the model, stage the mask. */
  private async runJob(): Promise<void> {
    const args = this.args!;
    const { ctx } = this.deps;
    try {
      // Recorded with record() itself, not audit(): a call is not made unless it is recorded.
      await record(ctx.paint, args.volume.id, "requested", {
        annotate_id: args.annotateId, volume_id: args.volume.id, dataset: args.dataset, provider: args.providerId, model: args.model,
        label_name: args.labelName, point_nm: args.pointNm, depth_axis: args.depthAxis, prompt_override: args.override,
      }, this.deps.now);
      this.setStage("fetching_crop");
      const dataset = ctx.infer.dataset();
      if (!dataset) throw new Refused("No dataset is open in the viewer.", 400);
      const plan = planPlane(args.volume, args.volume.dataset_shape_voxels, args.pointNm, args.depthAxis, this.deps.cropSizePx, rawExtentNm(dataset, args.volume), args.displayDimensions ?? null);
      this.plan = plan;
      this.writeBox = { lo: plan.write_lo.map((v) => Math.trunc(v)), hi: plan.write_hi.map((v) => Math.trunc(v)) };
      const plane = await readPlane(ctx.infer, dataset, args.volume, plan);
      const request: SegmentRequest = buildRequest(plane, plan, args.profile, args.override);
      this.request = request;
      await this.askAndStage(request);
    } catch (e) {
      await this.fail(e);
    }
  }

  /** The resend route's worker: the staged plane again, with `override` as
   * the prompt's editable part (null for the catalog's). The route has
   * checked that the job is ready or failed with a plane, the egress and the
   * limit; the plane is not read again. */
  resend(override: string | null): void {
    const args = this.args!;
    const request: SegmentRequest = { ...this.request!, prompt: buildRecolorPrompt(args.profile, this.plan!.resolution_nm, override) };
    this.status = "running";
    this.stage = "sending";
    this.error = null;
    // The previous answer's preview and share of the plane marked go: the
    // status answered while the new call runs must not describe it.
    this.preview = null;
    this.maskFraction = null;
    this.override = override;
    this.prompt = override ?? editablePrompt(args.profile);
    this.request = request;
    this.current = this.resendJob(request);
  }

  private async resendJob(request: SegmentRequest): Promise<void> {
    const args = this.args!;
    try {
      await record(this.deps.ctx.paint, args.volume.id, "resent", {
        annotate_id: args.annotateId, volume_id: args.volume.id, provider: args.providerId, model: args.model, prompt_override: this.override,
      }, this.deps.now);
      await this.askAndStage(request);
    } catch (e) {
      await this.fail(e);
    }
  }

  /** _ask_and_stage: ask the model about `request`, and stage what it
   * answers as the job's preview. The call is counted before it is made
   * (usage.py: it may be billed even if it fails) and refunded when it never
   * reached the model. */
  private async askAndStage(request: SegmentRequest): Promise<void> {
    const args = this.args!;
    const { usage, codec } = this.deps;
    this.setStage("sending");
    usage.checkAndCount(this.deps.dailyCallLimit);
    let result: SegmentResult;
    try {
      // No outer timeout: the provider times each attempt out itself
      // (CALL_TIMEOUT_MS) and retries a busy service, so one call may take
      // several attempts' worth; a signal here would cut the retries short.
      const credentials = this.deps.credentials(args.providerId) ?? {};
      result = await args.provider.segment(request, args.model, credentials);
    } catch (e) {
      if (e instanceof AIAnnotateError && NOT_SENT.has(e.category)) usage.refund();
      throw e;
    }
    this.setStage("extracting_mask");
    const plan = this.plan!;
    const maskWrite = maskToWriteShape(result.mask, plan);
    this.setStage("staging_preview");
    const preview = await buildPreview(codec, request, result, maskWrite, plan);
    const fraction = maskFraction(maskWrite);
    this.result = result;
    this.maskWrite = maskWrite;
    this.maskFraction = fraction;
    this.preview = preview;
    this.error = null;
    this.stage = null;
    this.status = "ready";
    await this.audit("staged", {
      annotate_id: args.annotateId, volume_id: args.volume.id, provider: args.providerId, model: args.model, mask_fraction: fraction,
      write_box: this.writeBox, usage: result.usage ?? {},
    });
  }

  /** _fail: mark the job failed with a message fit for the page, and log the detail. */
  private async fail(e: unknown): Promise<void> {
    const args = this.args!;
    const { ctx } = this.deps;
    let text: string, category: string;
    if (e instanceof AIAnnotateError) {
      text = e.userMessage;
      category = e.category;
      ctx.log(`AI annotation ${args.annotateId} failed (${category}): ${text}`);
    } else if (e instanceof Refused) {
      text = e.message;
      category = "refused";
      ctx.log(`AI annotation ${args.annotateId} failed: ${text}`);
    } else {
      text = GENERIC_FAILURE;
      category = "internal";
      ctx.log(`AI annotation ${args.annotateId} failed: ${message(e)}`);
    }
    this.status = "failed";
    this.stage = null;
    this.error = text;
    this.preview = null;
    await this.audit("failed", { annotate_id: args.annotateId, volume_id: args.volume.id, category, error: text });
  }

  private setStage(stage: Stage): void {
    this.stage = stage;
  }

  /** _audit: record an event that has already happened; a log that cannot
   * be written is only logged (the events before a call go through record()
   * itself, so that a call is not made unless it is recorded). */
  async audit(event: AuditEvent, fields: Record<string, unknown>): Promise<void> {
    const volume = this.args?.volume.id;
    if (!volume) return;
    try {
      await record(this.deps.ctx.paint, volume, event, fields, this.deps.now);
    } catch (e) {
      this.deps.ctx.log(`Could not record the AI annotation's "${event}" in the audit log: ${message(e)}`);
    }
  }

  // -------------------------------------------------------------- accept

  /** The write of ai_annotate_accept: the staged mask into the annotation
   * volume, each object an id of its own, the rest background. Without
   * `overwrite` only unannotated voxels are written, as a seed writes them;
   * with it the staged labels replace what the plane held, except that the
   * model's background only fills unlabelled voxels: an object already
   * painted that the model missed is kept, not erased. `undo` is the
   * volume's stack, which gets the box as it was and as it is (fill.ts pushUndo
   * keeps UNDO_DEPTH). The route sets the status around the call and answers can_undo.
   *
   * Throws a Refused when the volume is gone, and whatever the labelling or
   * the write throws (the Python answers a ValueError's message as a 409:
   * fill's ids running out, a mask of the wrong shape). */
  async accept(overwrite: boolean, undo: UndoEntry[]): Promise<AcceptResult> {
    const args = this.args!;
    const plan = this.plan!;
    const maskWrite = this.maskWrite!;
    const { ctx } = this.deps;
    const volume = await readVolume(ctx.paint, args.volume.id);
    if (!volume) throw new Refused(`The annotation volume ${args.volume.id} is no longer open.`);
    // As a seed: an instance target on a uint16 volume counts its ids up, a
    // uint8 volume reuses free ones. The server asks what the model trains
    // on (INSTANCE_TARGETS: affinities, flows); here the channel-name test
    // stands for it, as annotationDtypeFor's does.
    const entry = (await ctx.catalog.resolve(volume.model_id)) ?? (await ctx.catalog.resolve(volume.model_name));
    const instanceTarget = entry ? hasAffinityChannels(entry) : false;
    const countUp = instanceTarget && volume.dtype === "uint16";
    const lo = plan.write_lo.map((v) => Math.trunc(v));
    const hi = plan.write_hi.map((v) => Math.trunc(v));
    // Read just before the write, so a stroke painted in them since the plane was staged counts.
    const beside = await planesBeside(ctx.paint, volume, lo, hi, plan.depth_axis);

    const labelsFor = (existing: Labels): Labels => {
      const labels = labelsForBox(maskWrite, existing, lo, hi, plan.depth_axis, countUp, beside);
      if (overwrite) {
        for (let i = 0; i < labels.length; i++) if (labels[i] === 1 && existing[i] !== 0) labels[i] = 0;
      }
      return labels;
    };
    const counts = await paintBox(ctx.paint, volume, lo, hi, labelsFor, { undo, overwrite });
    const changed = Boolean(counts.n_foreground || counts.n_background || counts.n_overwritten);
    const layerRefreshed = changed ? ctx.viewer.refreshLayer(paintLayerName(volume.id)) : false;
    return {
      filled_foreground: counts.n_foreground,
      filled_background: counts.n_background,
      overwritten: counts.n_overwritten,
      reload_viewer: changed,
      layer_refreshed: layerRefreshed,
    };
  }
}

// ---------------------------------------------------------------------------
// The plane's surroundings
// ---------------------------------------------------------------------------

/** The planes on either side of the box [lo, hi) along `depthAxis`, as the
 * volume holds them (ai_annotate.py _planes_beside), for labelsForBox; null
 * for a side past the volume's edge. */
export async function planesBeside(store: PaintStore, volume: PaintVolume, lo: number[], hi: number[], depthAxis: DepthAxis): Promise<(Labels | null)[]> {
  const planes: (Labels | null)[] = [];
  for (const step of [-1, 1]) {
    const depth = lo[depthAxis] + step;
    if (depth < 0 || depth >= volume.dataset_shape_voxels[depthAxis]) {
      planes.push(null);
      continue;
    }
    const sideLo = lo.slice(), sideHi = hi.slice();
    sideLo[depthAxis] = depth;
    sideHi[depthAxis] = depth + 1;
    planes.push(await readBox(store, volume, sideLo, sideHi));
  }
  return planes;
}

// ---------------------------------------------------------------------------
// The preview (staging.py preview and overlay)
// ---------------------------------------------------------------------------

/** The review's images, base64 PNG: the image sent, the model's image (null
 * when it drew none) and the input with the voxels the mask labels tinted
 * the target colour. The mask is the write box's, laid out as the image is
 * (toScreen) and scaled to it by nearest neighbour, so the overlay shows
 * what accepting writes, not the model's own image. Every image is encoded
 * here, so nothing the model sent reaches the page as it came. */
export async function buildPreview(codec: ImageCodec, request: SegmentRequest, result: SegmentResult, maskWrite: Int32Array, plan: PlanePlan): Promise<Preview> {
  const [a, b] = writeShape(plan);
  const [rows, cols] = screenShape([a, b], plan.depth_axis, plan.display_dimensions);
  const screen = toScreen(Uint8Array.from(maskWrite, (o) => (o > 0 ? 1 : 0)), a, b, plan.depth_axis, plan.display_dimensions).data;
  const scaled = nearestResize(screen, rows, cols, request.image.height, request.image.width);
  const [inputPng, modelPng, overlayPng] = await Promise.all([
    codec.encodePng(request.image),
    result.model_image ? codec.encodePng(result.model_image) : Promise.resolve(null),
    codec.encodePng(overlay(request.image, scaled, request.target_rgb)),
  ]);
  return { input_png: base64(inputPng), model_png: modelPng ? base64(modelPng) : null, overlay_png: base64(overlayPng) };
}

/** `mask` (rows × cols, rows first) at newRows × newCols by nearest
 * neighbour, pixel centre to pixel centre (PIL's NEAREST). */
export function nearestResize(mask: Uint8Array, rows: number, cols: number, newRows: number, newCols: number): Uint8Array {
  if (rows === newRows && cols === newCols) return mask;
  const out = new Uint8Array(newRows * newCols);
  for (let r = 0; r < newRows; r++) {
    const sr = Math.min(rows - 1, Math.floor(((r + 0.5) * rows) / newRows));
    for (let c = 0; c < newCols; c++) {
      const sc = Math.min(cols - 1, Math.floor(((c + 0.5) * cols) / newCols));
      out[r * newCols + c] = mask[sr * cols + sc];
    }
  }
  return out;
}

