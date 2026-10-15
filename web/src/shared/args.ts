// The args blob in a prediction layer's URL, and the engine view name built
// around it: the browser side of cellmap_flow/serving/protocol.py and
// pipeline_spec.py.
//
// A layer's source is zarr3://<page>/<model>__CFLOW_ARGS__<blob>__CFLOW_ARGS__/.
// The blob is the layer's chain, {"input_norm": [...], "postprocess": [...]}
// followed by any extras (dashboard_url, digest), as minified JSON in URL-safe
// base64 without padding. Every request the viewer makes carries it, so the
// engine knows which chain to run from the view's name alone, and a layer URL
// made here decodes on a Python server (and the other way round).
//
// Parity with Python holds byte for byte for the values the dashboard sends:
// strings ("0", "255", "true"), integers, booleans and null. Two renderings
// differ between json.dumps and JSON.stringify and so would give a different
// blob and digest for the same chain: a whole float (Python writes 0.0, JS
// writes 0) and a non-ASCII character (Python escapes it as \uXXXX, JS does
// not). Neither occurs in a chain the dashboard builds.
import { ARGS_KEY, type Chain, type Step } from "../contracts";

const INPUT_NORM_KEY = "input_norm";
const POSTPROCESS_KEY = "postprocess";
const CHAIN_KEYS = [INPUT_NORM_KEY, POSTPROCESS_KEY];

/** `{input_norm, postprocess, ...extras}` in that key order, as a URL-safe
 * base64 blob without padding (protocol.encode_to_str of PipelineSpec.to_url_blob). */
export function encodeArgs(chain: Chain, extras: Record<string, unknown> = {}): string {
  const clash = Object.keys(extras).filter((k) => CHAIN_KEYS.includes(k));
  if (clash.length) throw new Error(`extras may not replace a chain: ${clash.join(", ")}`);
  const data = { [INPUT_NORM_KEY]: chain.input_norm, [POSTPROCESS_KEY]: chain.postprocess, ...extras };
  return base64UrlEncode(JSON.stringify(data));
}

/** The chain and the extras a blob carries (PipelineSpec.from_url_blob). Either
 * chain may be in the ordered list form or the older {Name: {params}} dict
 * form; a missing or null chain is empty. */
export function decodeArgs(blob: string): { chain: Chain; extras: Record<string, unknown> } {
  const parsed: unknown = JSON.parse(base64UrlDecode(blob));
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("the args blob is not a JSON object");
  }
  const data = parsed as Record<string, unknown>;
  const chain: Chain = {
    input_norm: normalizeSteps(data[INPUT_NORM_KEY]),
    postprocess: normalizeSteps(data[POSTPROCESS_KEY]),
  };
  const extras: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(data)) if (!CHAIN_KEYS.includes(k)) extras[k] = v;
  return { chain, extras };
}

/** A chain in the ordered [{name, ...params}] form (pipeline_spec.normalize_steps).
 *
 * Accepts the list (steps copied), the older {Name: {params}} dict (one step
 * per key, in the dict's order; the key names the class, so it wins over a
 * `name` inside the params), or nothing. A list element that is not an
 * object is dropped, as the Python op readers skip it. */
export function normalizeSteps(steps: unknown): Step[] {
  if (steps === null || steps === undefined) return [];
  if (Array.isArray(steps)) {
    return steps.filter((s): s is Step => s !== null && typeof s === "object" && !Array.isArray(s)).map((s) => ({ ...s }));
  }
  if (typeof steps === "object") {
    return Object.entries(steps as Record<string, unknown>).map(([name, params]) => {
      const given = params !== null && typeof params === "object" && !Array.isArray(params) ? (params as Record<string, unknown>) : {};
      return { ...(given as Omit<Step, "name">), name };
    });
  }
  throw new Error(`a chain must be a list or an object, got ${typeof steps}`);
}

/** A short hash of the chains' content (PipelineSpec.digest): the first 16 hex
 * characters of the SHA-256 of the two chains as JSON with keys sorted
 * recursively and no spaces, so two chains that differ only in the order of a
 * step's parameters share a digest while the order of the steps counts. See
 * the note at the top on the two renderings that differ from Python's. */
export async function chainDigest(chain: Chain): Promise<string> {
  const text = canonicalJson({ [INPUT_NORM_KEY]: chain.input_norm, [POSTPROCESS_KEY]: chain.postprocess });
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

/** JSON with object keys sorted at every depth and no whitespace, like
 * json.dumps(sort_keys=True, separators=(",", ":")). */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson((value as Record<string, unknown>)[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** The engine view (and URL path segment) of a model served through `blob`. */
export function viewName(modelName: string, blob: string): string {
  return `${modelName}${ARGS_KEY}${blob}${ARGS_KEY}`;
}

/** The model name and blob of a view name (protocol.split_dataset_url): the
 * blob is null when the name carries none, and any other number of markers
 * than two is a malformed name. The name is percent-decoded in case the viewer
 * encoded it on the way. */
export function parseViewName(view: string): { name: string; blob: string | null } {
  const parts = view.split(ARGS_KEY);
  if (parts.length === 1) return { name: decodeSegment(view), blob: null };
  if (parts.length !== 3) {
    throw new Error(`Invalid dataset format. Expected two occurrences of ${ARGS_KEY}, found ${parts.length - 1} in ${view}`);
  }
  return { name: decodeSegment(parts[0]), blob: parts[1] };
}

function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

// ---------------------------------------------------------------------------
// URL-safe base64 of UTF-8 text, without padding; btoa/atob take latin1 only,
// so the text goes through bytes. Both exist in browsers and in node 22.
// ---------------------------------------------------------------------------

function base64UrlEncode(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlDecode(blob: string): string {
  let b64 = blob.replace(/-/g, "+").replace(/_/g, "/").replace(/=+$/, "");
  while (b64.length % 4) b64 += "=";
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}
