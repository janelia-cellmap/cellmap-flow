// Gemini image models asked to paint the structure, called straight from the
// browser: the port of cellmap_flow/ai_annotate/backends/vertex_gemini.py
// over Google's REST API instead of the google-genai SDK. See
// PLAN_browser_finetune.md, "AI-assisted annotation in the browser".
//
// One module, two providers, because the two endpoints take the same request
// JSON and answer the same shape; they differ in the URL and the credential
// header:
//   - gemini_api (id "gemini"): Google's Gemini API, with the user's API key
//     (header x-goog-api-key). It can also list its models.
//   - vertex_gemini (id "vertex"): Google Cloud Vertex AI, with the user's
//     project and a short-lived access token from `gcloud auth
//     print-access-token` (header Authorization: Bearer).
// The server mode offers Vertex only, because the Gemini API's free tier lets
// Google train on what is sent (docs/ai_annotate.md). The browser cannot run
// gcloud for the user, so the key path is offered here with that said plainly
// in its destination text, and the Vertex path is kept for anyone who must
// stay on Vertex.
//
// The flow is vertex_gemini.py's: the plane padded to a square with its
// median grey (images.ts padToSquare), sent as a PNG with the prompt and
// response modalities IMAGE and TEXT; the answer's last image that is not a
// `thought` part decoded (images.ts: the checks), then, by the Python both
// modes run (mask.ts modelMask: portable/ai_annotate.py model_mask, which the
// server's backend calls too), resized to the sent size when the model
// answered at another, the padding cropped off, and the pixels of the
// asked-for colour taken as the mask.
// Only "try again later" answers (429, 503, or an error body whose status is
// UNAVAILABLE or RESOURCE_EXHAUSTED) are retried, at most MAX_ATTEMPTS
// attempts in all with the waits RETRY_WAITS_MS; everything else fails at
// once; each attempt has CALL_TIMEOUT_MS.
//
// What the user is told is written here, never copied from the service: an
// error body can echo the request, and a URL or header can carry the key or
// the token. Of a response only enumerations (the HTTP code, error.status,
// error.details[].reason, blockReason, finishReason) and the model's own
// text, stripped of control characters and cut to MAX_REPLY_CHARS, are read
// into a message, as vertex_gemini.py does. Nothing is logged.
import { padToSquare } from "./images";
import { modelMask } from "./mask";
import {
  AIAnnotateError,
  CALL_TIMEOUT_MS,
  type Credentials,
  DEFAULT_GEMINI_MODELS,
  type ImageCodec,
  type Provider,
  type ProviderInfo,
  type SegmentResult,
} from "./types";

export type GeminiType = "gemini_api" | "vertex_gemini";

/** fetch's shape (as models/hf.ts types it), so tests can hand in a stand-in. */
export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** What a provider is built over. `fetch` defaults to the page's; `sleep`
 * is how retries wait (tests record the waits instead); `now` is accepted
 * so one deps object serves every provider (providers.ts) and is not used
 * by this one. */
export interface GeminiDeps {
  fetch?: FetchLike;
  codec: ImageCodec;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
}

export const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
export const VERTEX_BASE = "https://aiplatform.googleapis.com/v1";

/** Where each provider sends the plane, in the words the user acknowledges
 * (the plan's texts; Vertex's is config.py's). */
export const GEMINI_DESTINATION =
  "Google's Gemini API (generativelanguage.googleapis.com), with your API key. " +
  "On the paid tier Google does not use what you send to train its models; on the free tier it may.";
export const VERTEX_DESTINATION = "Google Cloud Vertex AI, location global (Google may process the data in any region)";

/** What the panel says when a provider is asked to run without its credentials. */
export const MISSING_KEY_MESSAGE = "Enter your Gemini API key in the panel first.";
export const MISSING_VERTEX_MESSAGE = "Enter the Google Cloud project and an access token (gcloud auth print-access-token) first.";

export const MAX_ATTEMPTS = 3;
/** Waits before the second and third attempts (vertex_gemini.py RETRY_WAITS_S):
 * a 429 is usually a per-minute quota, so a few seconds rarely helps; these
 * are long enough to matter and short enough not to leave the user waiting minutes. */
export const RETRY_WAITS_MS: readonly number[] = [10_000, 30_000];
export const RETRY_CODES: readonly number[] = [429, 503];
export const RETRY_STATUSES: readonly string[] = ["UNAVAILABLE", "RESOURCE_EXHAUSTED"];
/** How much of the model's text reply a "no image" error may quote. */
export const MAX_REPLY_CHARS = 200;
/** The most pages of the model list followed, against a listing that never ends. */
const MAX_LIST_PAGES = 20;

/** The one Gemini provider for `type`. Its `info` is what the config route
 * shows; `segment` and `listModels` are the Provider contract (types.ts). */
export function createGeminiProvider(type: GeminiType, deps: GeminiDeps): Provider {
  const fetchImpl: FetchLike = deps.fetch ?? ((input, init) => globalThis.fetch(input, init));
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const service = type === "gemini_api" ? GEMINI_API : VERTEX;

  const info: ProviderInfo = type === "gemini_api"
    ? { id: "gemini", type, models: [...DEFAULT_GEMINI_MODELS], destination: GEMINI_DESTINATION, credentials: ["api_key"] }
    : { id: "vertex", type, models: [...DEFAULT_GEMINI_MODELS], destination: VERTEX_DESTINATION, credentials: ["project", "access_token"] };

  return {
    info,

    async segment(request, model, credentials, options = {}) {
      const call = endpoint(type, model, credentials);
      const sent = padToSquare(request.image).image;
      const png = await deps.codec.encodePng(sent);
      const body = {
        contents: [{
          role: "user",
          parts: [
            { inlineData: { mimeType: "image/png", data: bytesToBase64(png) } },
            { text: request.prompt },
          ],
        }],
        generationConfig: { responseModalities: ["IMAGE", "TEXT"] },
      };
      const context: ErrorContext = { service, model, location: call.location };

      const answer = await generateWithRetries(
        fetchImpl, sleep, call.url, { ...call.headers, "content-type": "application/json" }, JSON.stringify(body), options.signal, context,
      );

      const { image, text, mimeType } = imageAndText(answer);
      if (image === null) throw new AIAnnotateError("bad_response", noImageMessage(answer, text));
      const reply = await deps.codec.decode(imageBytes(image), mimeType);
      // A reply at another size (or shape: stretching it then is what the
      // preview shows) goes back to the sent size, then the padding comes
      // off: what is left lines up with the plane, and its pixels of the
      // target colour are the mask.
      if (!options.python) throw new Error("the Gemini provider takes its mask with the engine's Python: open a dataset first");
      const { mask, model_image } = await modelMask(options.python, reply, sent, request.image, request.target_rgb);
      return { mask, model_image, model, usage: usageOf(answer) } satisfies SegmentResult;
    },

    async listModels(credentials, options = {}) {
      if (type !== "gemini_api") return null;  // Vertex has no listing the token could be used for here
      const headers = endpoint(type, "unused", credentials).headers;
      const context: ErrorContext = { service, model: null, location: null };
      const names = new Set<string>();
      let pageToken: string | null = null;
      for (let page = 0; page < MAX_LIST_PAGES; page++) {
        const url = `${GEMINI_API_BASE}/models?pageSize=200` + (pageToken ? `&pageToken=${encodeURIComponent(pageToken)}` : "");
        const outcome = await callOnce(fetchImpl, url, { method: "GET", headers }, options.signal, context);
        if (outcome.kind === "http") throw userError(outcome, context, 1);
        for (const name of imageModelNames(outcome.json)) names.add(name);
        pageToken = nextPageToken(outcome.json);
        if (!pageToken) break;
      }
      return [...names].sort();
    },
  };
}

// ---------------------------------------------------------------------------
// The two endpoints
// ---------------------------------------------------------------------------

/** How a service is named in a message, at the start of a sentence and inside one. */
interface ServiceNames { Sentence: string; inline: string }
const GEMINI_API: ServiceNames = { Sentence: "The Gemini API", inline: "the Gemini API" };
const VERTEX: ServiceNames = { Sentence: "Vertex AI", inline: "Vertex AI" };

/** The URL and credential header of a generateContent call, or
 * AIAnnotateError("config", 400) when the credentials the provider needs
 * are not there. The model id is path-encoded; a leading "models/" (the
 * listing's spelling, which a user may paste into *Other…*) is dropped. */
/** Said when what was pasted as a Gemini API key is a Google Cloud access
 * token: `gcloud auth print-access-token` prints those (ya29.…), and they are
 * Vertex AI's credential, not the Gemini API's, which takes an AI Studio key
 * (AIza…). Google answers such a "key" with a bare refusal. */
export const TOKEN_AS_KEY_MESSAGE =
  "That is a Google Cloud access token (it starts with ya29.), not a Gemini API key: pick the vertex provider and paste it " +
  "there with the project, or paste an AI Studio key (it starts with AIza) here.";

/** Whether `key` is an OAuth access token rather than an API key. */
export function looksLikeAccessToken(key: string): boolean {
  return /^ya29\./.test(key.trim());
}

export function endpoint(type: GeminiType, model: string, credentials: Credentials): { url: string; headers: Record<string, string>; location: string | null } {
  const modelPath = encodeURIComponent(model.replace(/^models\//, ""));
  if (type === "gemini_api") {
    const key = (credentials.api_key ?? "").trim();
    if (!key) throw new AIAnnotateError("config", MISSING_KEY_MESSAGE, 400);
    if (looksLikeAccessToken(key)) throw new AIAnnotateError("config", TOKEN_AS_KEY_MESSAGE, 400);
    return { url: `${GEMINI_API_BASE}/models/${modelPath}:generateContent`, headers: { "x-goog-api-key": key }, location: null };
  }
  const project = (credentials.project ?? "").trim(), token = (credentials.access_token ?? "").trim();
  if (!project || !token) throw new AIAnnotateError("config", MISSING_VERTEX_MESSAGE, 400);
  const location = (credentials.location ?? "").trim() || "global";
  const url = `${VERTEX_BASE}/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}` +
    `/publishers/google/models/${modelPath}:generateContent`;
  return { url, headers: { authorization: `Bearer ${token}` }, location };
}

// ---------------------------------------------------------------------------
// Calling, with the timeout and the retries
// ---------------------------------------------------------------------------

/** What one call came to: the parsed JSON of a 2xx answer, or the parts of an
 * HTTP failure that may be read (enumerations only; the body is dropped). */
type Outcome =
  | { kind: "ok"; json: unknown }
  | { kind: "http"; code: number; status: string | null; reasons: string[] };

interface ErrorContext { service: ServiceNames; model: string | null; location: string | null }

/** generateContent, retried on 429/503 (or UNAVAILABLE / RESOURCE_EXHAUSTED)
 * only, at most MAX_ATTEMPTS times (vertex_gemini.py _generate). */
async function generateWithRetries(
  fetchImpl: FetchLike, sleep: (ms: number) => Promise<void>, url: string, headers: Record<string, string>, body: string,
  signal: AbortSignal | undefined, context: ErrorContext,
): Promise<unknown> {
  for (let attempt = 1; ; attempt++) {
    const outcome = await callOnce(fetchImpl, url, { method: "POST", headers, body }, signal, context);
    if (outcome.kind === "ok") return outcome.json;
    if (attempt < MAX_ATTEMPTS && isRetryable(outcome)) {
      await sleep(RETRY_WAITS_MS[Math.min(attempt - 1, RETRY_WAITS_MS.length - 1)]);
      if (signal?.aborted) throw cancelled(signal, context.service);  // the caller gave up during the wait
      continue;
    }
    throw userError(outcome, context, attempt);
  }
}

function isRetryable(outcome: Extract<Outcome, { kind: "http" }>): boolean {
  return RETRY_CODES.includes(outcome.code) || (outcome.status !== null && RETRY_STATUSES.includes(outcome.status));
}

/** One HTTP call with CALL_TIMEOUT_MS to answer, body included, and the
 * caller's `signal` honoured. Network failure, the timeout and a cancelled
 * call are AIAnnotateErrors here; an HTTP failure is returned for the caller
 * to retry or turn into one. */
async function callOnce(fetchImpl: FetchLike, url: string, init: RequestInit, signal: AbortSignal | undefined, context: ErrorContext): Promise<Outcome> {
  const { service } = context;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, CALL_TIMEOUT_MS);
  const cancel = () => controller.abort(signal?.reason);
  if (signal?.aborted) cancel();
  else signal?.addEventListener("abort", cancel, { once: true });
  try {
    const response = await fetchImpl(url, { ...init, signal: controller.signal });
    const text = await response.text();
    if (!response.ok) return httpFailure(response.status, text);
    try {
      return { kind: "ok", json: JSON.parse(text) };
    } catch {
      throw new AIAnnotateError("bad_response", `${service.Sentence} answered with something other than JSON.`);
    }
  } catch (e) {
    if (e instanceof AIAnnotateError) throw e;
    if (timedOut) throw new AIAnnotateError("timeout", `${service.Sentence} did not answer within ${CALL_TIMEOUT_MS / 1000} s.`);
    if (signal?.aborted) throw cancelled(signal, service);
    // fetch rejects with a TypeError when the request never got an answer:
    // no network, a blocked request, a dropped connection mid-body.
    if (e instanceof TypeError) throw new AIAnnotateError("unavailable", `Could not reach ${service.inline} from this browser.`);
    throw new AIAnnotateError("bad_response", `The call to ${service.inline} failed (${errorName(e)}).`);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

/** What the caller's signal aborting means: its own deadline passing
 * (AbortSignal.timeout's reason is a TimeoutError, as the job's is) or a
 * cancellation. */
function cancelled(signal: AbortSignal, service: ServiceNames): AIAnnotateError {
  const reason: unknown = signal.reason;
  if (typeof reason === "object" && reason !== null && (reason as { name?: unknown }).name === "TimeoutError") {
    return new AIAnnotateError("timeout", `${service.Sentence} did not answer in time.`);
  }
  return new AIAnnotateError("unavailable", `The call to ${service.inline} was cancelled.`);
}

/** The parts of an HTTP failure's body that may be read: Google's
 * `error.status` (a gRPC code name) and `error.details[].reason`
 * (ErrorInfo's reason, e.g. API_KEY_INVALID). Both are enumerations; the
 * message, which can echo the request, is dropped unread. A body that is
 * not JSON (a proxy's HTML page) leaves them empty. */
function httpFailure(code: number, text: string): Outcome {
  let status: string | null = null;
  const reasons: string[] = [];
  try {
    const error = (JSON.parse(text) as { error?: unknown })?.error;
    if (isRecord(error)) {
      if (typeof error.status === "string" && /^[A-Z_]+$/.test(error.status)) status = error.status;
      if (Array.isArray(error.details)) {
        for (const detail of error.details) {
          if (isRecord(detail) && typeof detail.reason === "string" && /^[A-Z_]+$/.test(detail.reason)) reasons.push(detail.reason);
        }
      }
    }
  } catch {
    // not JSON: nothing to read
  }
  return { kind: "http", code, status, reasons };
}

/** A thrown value's class name, letters only, for a message; never its text. */
function errorName(e: unknown): string {
  const name = e instanceof Error ? e.name : typeof e;
  return /^[A-Za-z]+$/.test(name) ? name : "Error";
}

/** An HTTP failure as the AIAnnotateError the user sees (vertex_gemini.py
 * _user_error), with messages adapted to each service: what to check for
 * an API key differs from what to check for a Vertex access token. */
function userError(outcome: Extract<Outcome, { kind: "http" }>, context: ErrorContext, attempts: number): AIAnnotateError {
  const { service, model } = context;
  const { code, status, reasons } = outcome;
  const authFailure = code === 401 || code === 403 || status === "UNAUTHENTICATED" || status === "PERMISSION_DENIED" || reasons.includes("API_KEY_INVALID");
  if (authFailure) {
    const why = refusalReason(reasons);
    if (why) return new AIAnnotateError("auth", `${service.Sentence} refused the credentials: ${why}`);
    return new AIAnnotateError("auth", (service === GEMINI_API
      ? "The Gemini API refused the credentials: check the API key."
      : "Vertex AI refused the credentials: check the access token (it expires after about an hour: run gcloud auth " +
        "print-access-token again), the project, that the Vertex AI API is enabled and that you have the Vertex AI User role.") +
      reasonSuffix(reasons));
  }
  if (code === 404 && model !== null) {
    return new AIAnnotateError("config", service === GEMINI_API
      ? `The Gemini API has no model ${model}.`
      : `Vertex AI has no model ${model} in location ${context.location} for this project.`);
  }
  if (code === 429 || status === "RESOURCE_EXHAUSTED") {
    return new AIAnnotateError("quota", `${service.Sentence}'s quota is used up (still refused after ${attempts} ${attempts === 1 ? "attempt" : "attempts"}); wait a minute and try again.`);
  }
  if (code >= 500 && code < 600) {
    return new AIAnnotateError("unavailable", `${service.Sentence} is unavailable (HTTP ${code}); try again later.`);
  }
  return new AIAnnotateError("bad_response", `${service.Sentence} rejected the request (HTTP ${code}).`);
}

/** What to do about a refusal whose ErrorInfo reason Google gives (an
 * enumeration, safe to show: never the message, which may echo the
 * request), or null for one not known here. */
function refusalReason(reasons: string[]): string | null {
  if (reasons.includes("API_KEY_INVALID")) return "Google says the API key is not valid (API_KEY_INVALID): check it was copied whole, from aistudio.google.com/apikey.";
  if (reasons.includes("API_KEY_HTTP_REFERRER_BLOCKED")) return "the key's website restrictions do not allow this page's address (API_KEY_HTTP_REFERRER_BLOCKED): add it, port included (e.g. http://localhost:5174/*), in the Google Cloud console.";
  if (reasons.includes("API_KEY_SERVICE_BLOCKED")) return "the key's API restrictions do not allow the Gemini API (API_KEY_SERVICE_BLOCKED): allow the Generative Language API for it in the Google Cloud console.";
  if (reasons.includes("SERVICE_DISABLED")) return "the API is not enabled in the credentials' project (SERVICE_DISABLED): enable it in the Google Cloud console, wait a minute and try again.";
  if (reasons.includes("ACCESS_TOKEN_EXPIRED") || reasons.includes("ACCESS_TOKEN_TYPE_UNSUPPORTED")) return "the access token has expired or is not one it takes: run gcloud auth print-access-token again and paste the new token.";
  if (reasons.includes("IAM_PERMISSION_DENIED")) return "your account lacks the permission in this project (IAM_PERMISSION_DENIED): ask its owner for the Vertex AI User role.";
  return null;
}

/** Google's reason codes, as " (Google: A, B)", for a refusal not explained above; empty without any. */
function reasonSuffix(reasons: string[]): string {
  return reasons.length ? ` (Google: ${reasons.join(", ")})` : "";
}

// ---------------------------------------------------------------------------
// Reading the answer
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function candidatesOf(answer: unknown): Record<string, unknown>[] {
  const candidates = isRecord(answer) ? answer.candidates : null;
  return Array.isArray(candidates) ? candidates.filter(isRecord) : [];
}

function partsOf(answer: unknown): Record<string, unknown>[] {
  const first = candidatesOf(answer)[0];
  const content = first ? first.content : null;
  const parts = isRecord(content) ? content.parts : null;
  return Array.isArray(parts) ? parts.filter(isRecord) : [];
}

/** The last image the model gave as its answer, with its declared type, and
 * its text, from the first candidate (vertex_gemini.py _image_and_text).
 * Parts marked `thought` are skipped: an image model that thinks first may
 * send draft images before its answer. The REST API spells the blob
 * `inlineData`/`mimeType`; the SDK's `inline_data`/`mime_type` are read too. */
export function imageAndText(answer: unknown): { image: string | null; mimeType: string; text: string } {
  let image: string | null = null, mimeType = "";
  const texts: string[] = [];
  for (const part of partsOf(answer)) {
    if (part.thought === true) continue;
    const blob = isRecord(part.inlineData) ? part.inlineData : isRecord(part.inline_data) ? part.inline_data : null;
    if (blob && typeof blob.data === "string" && blob.data) {
      image = blob.data;
      const declared = blob.mimeType ?? blob.mime_type;
      mimeType = typeof declared === "string" ? declared : "";
    }
    if (typeof part.text === "string" && part.text) texts.push(part.text);
  }
  return { image, mimeType, text: texts.join(" ") };
}

/** The image part's base64 as bytes; a part that is not base64 is an image
 * that cannot be decoded. */
function imageBytes(data: string): Uint8Array {
  try {
    return base64ToBytes(data);
  } catch {
    throw new AIAnnotateError("bad_response", "The model returned an image that could not be decoded.");
  }
}

const CONTROL_CHARS = /[\x00-\x08\x0b-\x1f\x7f]/g;

/** `text` with control characters blanked, trimmed and cut to `limit`
 * (vertex_gemini.py _plain), for quoting the model in a message. */
export function plain(text: unknown, limit = MAX_REPLY_CHARS): string {
  const cleaned = String(text).replace(CONTROL_CHARS, " ").trim();
  return cleaned.length <= limit ? cleaned : cleaned.slice(0, limit - 3) + "...";
}

/** Why the reply had no image, from the block or finish reason and the
 * model's own words (vertex_gemini.py _no_image_message). */
export function noImageMessage(answer: unknown, replyText: string): string {
  let message = "The model answered without an image";
  const feedback = isRecord(answer) && isRecord(answer.promptFeedback) ? answer.promptFeedback : null;
  const block = feedback && typeof feedback.blockReason === "string" ? feedback.blockReason : null;
  const first = candidatesOf(answer)[0];
  const finish = first && typeof first.finishReason === "string" ? first.finishReason : null;
  if (block) message += ` (blocked: ${plain(block, 40)})`;
  else if (finish && finish !== "STOP") message += ` (stopped: ${plain(finish, 40)})`;
  if (replyText) message += `. It said: "${plain(replyText)}"`;
  return message + ".";
}

/** Token counts the answer reports, under the Python's provider-neutral
 * names (vertex_gemini.py _usage). */
export function usageOf(answer: unknown): Record<string, number> {
  const meta = isRecord(answer) && isRecord(answer.usageMetadata) ? answer.usageMetadata : null;
  const usage: Record<string, number> = {};
  if (!meta) return usage;
  for (const [ours, theirs] of [["prompt_tokens", "promptTokenCount"], ["output_tokens", "candidatesTokenCount"], ["total_tokens", "totalTokenCount"]] as const) {
    const value = meta[theirs];
    if (typeof value === "number" && Number.isInteger(value)) usage[ours] = value;
  }
  return usage;
}

/** The image models a models-list page names: those whose name (without
 * `models/`) contains "image" and that support generateContent. */
export function imageModelNames(page: unknown): string[] {
  const models = isRecord(page) && Array.isArray(page.models) ? page.models : [];
  const names: string[] = [];
  for (const model of models) {
    if (!isRecord(model) || typeof model.name !== "string") continue;
    const name = model.name.replace(/^models\//, "");
    const methods = Array.isArray(model.supportedGenerationMethods) ? model.supportedGenerationMethods : [];
    if (name.includes("image") && methods.includes("generateContent")) names.push(name);
  }
  return names;
}

function nextPageToken(page: unknown): string | null {
  const token = isRecord(page) ? page.nextPageToken : null;
  return typeof token === "string" && token ? token : null;
}

// ---------------------------------------------------------------------------
// Base64, in chunks: a plane's PNG is around a megabyte
// ---------------------------------------------------------------------------

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

export function base64ToBytes(text: string): Uint8Array {
  const binary = atob(text.replace(/\s+/g, ""));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
