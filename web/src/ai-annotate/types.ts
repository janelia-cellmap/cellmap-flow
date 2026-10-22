// AI-assisted annotation in the browser: the types its modules share. The
// port of cellmap_flow/ai_annotate (the server's feature; docs/ai_annotate.md
// says what it does for the user) for the browser mode, where the page itself
// reads the plane from the raw volume, sends it to the model, turns the
// answer into a mask and writes the accepted mask into the paint store. The
// routes keep the Python routes' JSON (dashboard/routes/finetune/ai_annotate.py),
// so the tab's ai-annotate.js drives both. See PLAN_browser_finetune.md,
// "AI-assisted annotation in the browser".
//
// Axes are z, y, x throughout, as in the Python. An image is laid out as the
// viewer's panel draws the plane (portable/ai_annotate.py to_screen): rows
// down, columns across.
//
// Who implements what: the plane, the image sent, the mask and the labels an
// accept writes are the Python both modes run (cellmap_flow/portable/
// ai_annotate.py), which plane.ts and mask.ts call in the engine's worker;
// labels.ts and fill.ts are the Label it tools' box maths and the paint
// store's box reads and writes, prompts.ts and organelles.ts the prompt and
// the catalog (no DOM); images.ts is the canvas ImageCodec; gemini.ts,
// fake.ts and providers.ts the Providers; backend/ai-annotate-job.ts and
// backend/routes/ai-annotate.ts the routes. The constants with a Python twin
// are the Python's, exported (generated/python-data.ts).
import type { Inference } from "../contracts";
import { PYTHON } from "../generated/python-data";

const AI = PYTHON.ai_annotate;

// ---------------------------------------------------------------------------
// Planes
// ---------------------------------------------------------------------------

/** The data axis a plane is normal to: 0 (z: an XY plane), 1 (y: XZ), 2 (x: YZ). */
export type DepthAxis = 0 | 1 | 2;

/** The plane each depth axis leaves, named by the data axes it shows (geometry.py PLANE_NAMES). */
export const PLANE_NAMES: readonly ["XY", "XZ", "YZ"] = ["XY", "XZ", "YZ"];

/** One plane to annotate (geometry.py PlanePlan): what is read, what is sent
 * and what is written. Every triple is z, y, x. `raw_offset_nm` and
 * `raw_shape_nm` are the world box the raw is read over; `image_shape` the
 * (rows, cols) of the image sent, laid out as on screen; `resolution_nm` its
 * pixel size (square pixels); `write_lo`/`write_hi` the annotation voxels
 * written, [lo, hi), one voxel thick along `depth_axis`;
 * `display_dimensions` the viewer's, which say how its panel draws the
 * plane (portable/ai_annotate.py screen_axes), absent (z, y, x) in a plan
 * saved before plans kept them. */
export interface PlanePlan {
  depth_axis: DepthAxis;
  point_nm: number[];
  raw_offset_nm: number[];
  raw_shape_nm: number[];
  image_shape: [number, number];
  input_voxel_size: number[];
  write_lo: number[];
  write_hi: number[];
  resolution_nm: number;
  display_dimensions?: string[];
}

/** The longest side of an image sent to a model (geometry.py MAX_IMAGE_SIDE):
 * a hosted image model bills an image by the image, not by its pixels. */
export const MAX_IMAGE_SIDE = AI.max_image_side;

/** The plane's size on a side, in annotation voxels (config.py DEFAULT_CROP_SIZE_PX). */
export const CROP_SIZE_PX = AI.crop_size_px;

// ---------------------------------------------------------------------------
// Errors (errors.py)
// ---------------------------------------------------------------------------

export const ERROR_CATEGORIES = ["config", "auth", "quota", "timeout", "bad_response", "unavailable", "refused", "limit"] as const;
export type ErrorCategory = (typeof ERROR_CATEGORIES)[number];

/** The HTTP status each category answers unless the raiser says otherwise. */
export const DEFAULT_STATUS: Record<ErrorCategory, number> = {
  config: 500,
  auth: 502,
  quota: 429,
  timeout: 504,
  bad_response: 502,
  unavailable: 503,
  refused: 403,
  limit: 429,
};

/** The one error this feature raises: a category the routes turn into a
 * status, and a short message written here, safe to show. A provider's own
 * text (a response body, which could echo a request or a key) never becomes
 * the message. */
export class AIAnnotateError extends Error {
  readonly category: ErrorCategory;
  readonly userMessage: string;
  readonly httpStatus: number;

  constructor(category: ErrorCategory, userMessage: string, httpStatus?: number) {
    super(userMessage);
    this.name = "AIAnnotateError";
    this.category = category;
    this.userMessage = userMessage;
    this.httpStatus = httpStatus ?? DEFAULT_STATUS[category];
  }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/** An RGBA image as a canvas holds one (ImageData's layout): `data` is
 * width × height × 4 bytes, rows first, alpha 255 throughout. */
export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** The most pixels a model's image may have before it is decoded (images.py MAX_PIXELS). */
export const MAX_IMAGE_PIXELS = AI.max_image_pixels;

/** The formats a model's image may be in. */
export const ALLOWED_IMAGE_TYPES: readonly string[] = ["image/png", "image/jpeg", "image/webp"];

/** Encoding and decoding images: the page's canvas (images.ts), a
 * plain-JavaScript stand-in in tests. A model's image is untrusted input, so
 * `decode` checks the type and the pixel count before decoding and raises
 * AIAnnotateError("bad_response") for anything else. Resizing is the
 * Python's (mask.ts modelMask), so a reply at another size is lined up with
 * the plane as the server lines it up. */
export interface ImageCodec {
  /** `image` as PNG bytes. */
  encodePng(image: RgbaImage): Promise<Uint8Array>;
  /** The pixels of `bytes`, a PNG, JPEG or WebP of at most MAX_IMAGE_PIXELS.
   * `mimeType` is what the model declared, which may be empty or wrong: the
   * bytes decide. */
  decode(bytes: Uint8Array, mimeType: string): Promise<RgbaImage>;
}

// ---------------------------------------------------------------------------
// Structures (organelles.py)
// ---------------------------------------------------------------------------

/** One structure the model can be asked to paint: its catalog key, its
 * plural name as the prompt uses it, the colour asked for (as words, and as
 * RGB to find it in the reply), and how it looks in EM. */
export interface OrganelleProfile {
  key: string;
  name: string;
  color_name: string;
  rgb: [number, number, number];
  description: string;
}

/** The longest editable prompt (prompts.py MAX_PROMPT_CHARS). */
export const MAX_PROMPT_CHARS = AI.max_prompt_chars;

// ---------------------------------------------------------------------------
// Providers (backends/base.py): one RGB EM plane and a prompt in, a mask out
// ---------------------------------------------------------------------------

/** One plane to segment: `image` is exactly what is sent, `target_rgb` the
 * colour the prompt asks for, `click_px` the (row, col) the user pointed at
 * in the image, or null. */
export interface SegmentRequest {
  image: RgbaImage;
  prompt: string;
  target_rgb: [number, number, number];
  click_px: [number, number] | null;
}

/** A provider's answer: `mask` has one byte per pixel of the request's image
 * (rows first), 1 where the structure was painted; `model_image` what the
 * model drew, at the image's size, for the review (null for a provider that
 * draws nothing); `usage` whatever token counts it reports. */
export interface SegmentResult {
  mask: Uint8Array;
  model_image: RgbaImage | null;
  model: string;
  usage: Record<string, number>;
}

export type ProviderType = "gemini_api" | "vertex_gemini" | "fake";

/** What a provider needs from the user to be called. */
export type CredentialField = "api_key" | "project" | "access_token";

/** The user's credentials for a provider: typed into the panel, kept in the
 * page (and in this browser's localStorage when the user asks), sent to the
 * provider and nowhere else, never logged, never in the session state or a
 * share link. */
export interface Credentials {
  api_key?: string;
  project?: string;
  access_token?: string;
  /** Vertex AI's location; "global" unless given. */
  location?: string;
}

/** A provider as the config route describes it to the page (config.py
 * ProviderConfig.destination and AIAnnotateConfig.public): its id, type,
 * the models it offers, where it sends the data, in words, for the user to
 * acknowledge, and which credentials it needs. */
export interface ProviderInfo {
  id: string;
  type: ProviderType;
  models: string[];
  destination: string;
  credentials: CredentialField[];
}

/** The engine's call into the portable Python (Inference.python). */
export type PythonCall = Inference["python"];

export interface Provider {
  readonly info: ProviderInfo;
  /** Ask `model` to paint the request's structure, with `credentials`;
   * AIAnnotateError on failure, with a message written by the provider
   * module, never the service's own text. `options.python` is the engine's
   * Python, which a provider whose model answers with an image takes the
   * mask out of it with (mask.ts modelMask); nothing else is handed to it. */
  segment(request: SegmentRequest, model: string, credentials: Credentials, options?: { signal?: AbortSignal; python?: PythonCall }): Promise<SegmentResult>;
  /** The image models `credentials` can reach, for a provider that can list
   * them (the Gemini API lists its models); null when it cannot say. */
  listModels?(credentials: Credentials, options?: { signal?: AbortSignal }): Promise<string[] | null>;
}

/** The Gemini image models as Google's docs list them (ai.google.dev/gemini-api/docs/models,
 * read 2026-10-15): what the picker offers before the key has listed the live ones. */
export const DEFAULT_GEMINI_MODELS: readonly string[] = ["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"];

/** The fake provider's one model (config.py: `fake` offers `fake-threshold`). */
export const FAKE_MODEL = "fake-threshold";

/** How long one model call may take (config.py DEFAULT_TIMEOUT_S). */
export const CALL_TIMEOUT_MS = AI.timeout_s * 1000;

// ---------------------------------------------------------------------------
// Settings, limits and the job, as the routes answer them
// ---------------------------------------------------------------------------

/** The panel's settings (the settings route's body, minus the acknowledgement). */
export interface AiSettings {
  provider: string;
  model: string;
  label_key: string | null;
  label_name: string | null;
  /** The editable prompt when the user changed it; null for the catalog's own. */
  prompt: string | null;
}

/** The most model calls one browser makes in a day (config.py DEFAULT_DAILY_CALL_LIMIT). */
export const DAILY_CALL_LIMIT = AI.daily_call_limit;

/** The key bound in the viewer, in Neuroglancer's spelling, the action it
 * names and how the page shows it (ai_annotate.py KEY, ACTION, KEYBINDING). */
export const { key: KEY, action: ACTION, keybinding: KEYBINDING } = AI;

export type JobStatus = "idle" | "running" | "ready" | "failed" | "accepting";
export type Stage = "fetching_crop" | "sending" | "extracting_mask" | "staging_preview";

/** The stages of a job as the page names them (ai_annotate.py STAGE_LABELS). */
export const STAGE_LABELS: Readonly<Record<Stage, string>> = AI.stage_labels;

/** The review's images, base64 PNG (staging.py preview). */
export interface Preview {
  input_png: string;
  model_png: string | null;
  overlay_png: string;
}

/** What GET /api/finetune/ai-annotate/status answers (ai_annotate.py _idle_job
 * and ai_annotate_status): `accepting` is reported as `ready`, and `preview`
 * only with `?preview=1` once ready. */
export interface StatusAnswer {
  success: true;
  annotate_id: string | null;
  status: Exclude<JobStatus, "accepting">;
  stage: Stage | null;
  stage_label: string | null;
  error: string | null;
  prompt: string | null;
  provider: string | null;
  model: string | null;
  label_name: string | null;
  depth_axis: DepthAxis | null;
  plane: string | null;
  point_nm: number[] | null;
  write_box: { lo: number[]; hi: number[] } | null;
  mask_fraction: number | null;
  preview: Preview | null;
  needs_volume: boolean;
}

// ---------------------------------------------------------------------------
// Labels (fill.py): boxes of the annotation volume
// ---------------------------------------------------------------------------

/** The volume's labels as its dtype holds them (PaintVolume.dtype). */
export type Labels = Uint8Array | Uint16Array;

/** What Undo takes back (fill.restore_box): the box [lo, hi) as it was
 * before an action and as the action left it. */
export interface UndoEntry {
  lo: number[];
  hi: number[];
  before: Labels;
  after: Labels;
}

/** How many one-click label actions per volume Undo can take back (view_labels.py UNDO_DEPTH). */
export const UNDO_DEPTH = PYTHON.finetune.undo_depth;
