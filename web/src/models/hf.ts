// Hugging Face, as the catalog reads it: a repo's file tree (the Hub API),
// its small text files (raw/) and the .onnx binaries (resolve/, which answers
// with a redirect to the LFS CDN that fetch follows by itself), and the refs
// that name a repo or one of its folders (hubRef). Nothing here knows about
// manifests or variants; models/index.ts does.
//
// Anonymous requests are rate-limited: after a burst of downloads from one
// address the Hub answers HTTP 429. Every fetch here turns a 429 into a
// RateLimitError whose message says what happened and to retry in a minute;
// callers decide whether a cache or a fallback can stand in for the answer.
//
// CORS (checked 2026-10-14): the API, raw/ and resolve/ reflect the page's
// Origin; the CDN that resolve/ redirects to answers "*" and exposes every
// header, so Content-Length is readable and download progress can be shown,
// and so is ETag, the file's identity (fetchEtag; checked 2026-10-15).

export const HF_REPO = "cellmap/onnx-web-models";
export const HF_REVISION = "main";
const HF_HOST = "https://huggingface.co";

/** fetch's shape, so tests can hand in a stand-in. */
export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** HTTP 429 from the Hub: the anonymous download quota of this address is used up for now. */
export class RateLimitError extends Error {
  readonly status = 429;
  constructor(what: string) {
    super(`Hugging Face rate-limited the anonymous download of ${what} (HTTP 429). Please retry in a minute.`);
    this.name = "RateLimitError";
  }
}

/** Any other answer from the Hub that is not OK, with its status: a caller
 * tells "there is no such repo or file" (401 and 404: the Hub answers an
 * anonymous request for a repo that does not exist, or is private, with
 * 401; checked 2026-10-26) from a server that failed. */
export class HubStatusError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "HubStatusError";
  }
}

/** Whether `error` is the Hub saying there is nothing there to read anonymously (HTTP 401 or 404). */
export function isMissing(error: unknown): error is HubStatusError {
  return error instanceof HubStatusError && (error.status === 401 || error.status === 404);
}

// ---------------------------------------------------------------------------
// Refs: what names a repo, or a folder of one
// ---------------------------------------------------------------------------

/** A Hugging Face model repo (`<owner>/<repo>`), and one of its folders when the ref names one. */
export interface HubRef {
  repo: string;
  folder: string | null;
}

// The server's patterns (cellmap_flow/models/resolve.py _HF_REPO, _ONNX_REF,
// _DOI, _looks_like_path), so that a ref one mode adds the other adds too: a
// repo is two segments, a folder a third, a revision may follow an "@"; a
// zoo DOI (10.5281/zenodo.5764892) is repo-shaped and is not one, nor is a
// relative path to a file.
const SEGMENT = "[A-Za-z0-9][\\w.-]*";
const REPO_REF = new RegExp(`^(${SEGMENT}/${SEGMENT})(?:/(${SEGMENT}))?(?:@([\\w./-]+))?$`);
const DOI = /^10\.\d{4,9}\//;
const FILE_LIKE = /\.(?:onnx|py|ts|pt|zip|ya?ml)$/i;
/** The prefixes that say a ref is a Hugging Face model's: `hf:` and its
 * long spelling, as on the server, and `onnx:`, the server's for an ONNX
 * export (`onnx:<owner>/<repo>/<folder>`); none is needed. */
const PREFIX = /^(?:hf|huggingface|onnx):(?!\/\/)/i;
const HUB_HOSTS = ["huggingface.co", "www.huggingface.co", "hf.co"];
/** The first segments of huggingface.co pages that are not a model repo's. */
const NOT_A_MODEL = ["datasets", "spaces", "models", "docs", "collections"];

/** `ref` as the catalog matches it: trimmed, its prefix (PREFIX) and trailing slashes dropped. */
export function bareRef(ref: string): string {
  return ref.trim().replace(PREFIX, "").trim().replace(/\/+$/, "");
}

/** The repo, and the folder, that `ref` names: `<owner>/<repo>`,
 * `<owner>/<repo>/<folder>`, either with `@main` after it or a prefix before
 * it, or a huggingface.co URL (`/<owner>/<repo>`, `/tree/main/<folder>`, or a
 * file in the folder, whose folder it names: the variant run is chosen for
 * the GPU). Null for a ref that names no Hugging Face repo: one word (a
 * folder of HF_REPO, or a model's name, which the catalog looks up itself), a
 * path, a zoo DOI, another site's URL. Throws for a revision other than
 * HF_REVISION, which the browser does not read (its entries and caches are of
 * one revision), and for a huggingface.co URL that is not a model repo's. */
export function hubRef(ref: string): HubRef | null {
  const key = bareRef(ref);
  if (/^https?:\/\//i.test(key)) return hubUrlRef(key);
  const match = REPO_REF.exec(key);
  if (!match || DOI.test(key) || FILE_LIKE.test(key)) return null;
  checkRevision(match[3], key);
  return { repo: match[1], folder: match[2] ?? null };
}

function hubUrlRef(url: string): HubRef | null {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  if (!HUB_HOSTS.includes(parsed.hostname.toLowerCase())) return null;
  const parts = parsed.pathname.split("/").filter(Boolean).map(decodedSegment);
  if (parts.length < 2 || NOT_A_MODEL.includes(parts[0])) throw new Error(`${url} is not a Hugging Face model repo's page (huggingface.co/<owner>/<repo>)`);
  const repo = `${parts[0]}/${parts[1]}`;
  if (parts.length > 3 && ["tree", "blob", "resolve"].includes(parts[2])) {
    checkRevision(parts[3], url);
    return { repo, folder: parts[4] ?? null };
  }
  return { repo, folder: null };
}

function decodedSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

function checkRevision(revision: string | undefined, ref: string): void {
  if (revision !== undefined && revision !== HF_REVISION) {
    throw new Error(`${ref} names the revision ${revision}: the browser reads a Hugging Face repo at its ${HF_REVISION} revision only`);
  }
}

/** The Hub API's listing of `path` in the repo (its top level when empty). */
export function treeUrl(path = "", repo = HF_REPO, revision = HF_REVISION): string {
  const suffix = path ? `/${encodePath(path)}` : "";
  return `${HF_HOST}/api/models/${repo}/tree/${revision}${suffix}`;
}

/** A small text file's bytes as committed (no LFS redirect). */
export function rawUrl(path: string, repo = HF_REPO, revision = HF_REVISION): string {
  return `${HF_HOST}/${repo}/raw/${revision}/${encodePath(path)}`;
}

/** A file's bytes, LFS files included: a redirect to the CDN for those. */
export function resolveUrl(path: string, repo = HF_REPO, revision = HF_REVISION): string {
  return `${HF_HOST}/${repo}/resolve/${revision}/${encodePath(path)}`;
}

// Each path segment URL-encoded, the slashes kept.
function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

/** One row of the Hub API's tree listing (the fields the catalog reads). */
export interface TreeEntry {
  type: "file" | "directory";
  path: string;
  size?: number;
}

/** `url`'s body as JSON; a RateLimitError on 429, an Error naming the status otherwise. */
export async function fetchJson(fetchImpl: FetchLike, url: string, what: string): Promise<unknown> {
  const response = await fetchImpl(url, { headers: { Accept: "application/json" } });
  throwForStatus(response, what);
  return response.json();
}

/** The folders and files at `path` in `repo`. A repo the Hub does not let
 * this page read (HTTP 401 or 404) is a HubStatusError that says so. */
export async function fetchTree(fetchImpl: FetchLike, repo: string, path = ""): Promise<TreeEntry[]> {
  let listing: unknown;
  try {
    listing = await fetchJson(fetchImpl, treeUrl(path, repo), `the file list of ${repo}`);
  } catch (error) {
    if (!isMissing(error)) throw error;
    throw new HubStatusError(error.status, `There is no public Hugging Face model repo ${repo} (HTTP ${error.status}): it does not exist, or it is private or gated, and the browser reads public repos only`);
  }
  if (!Array.isArray(listing)) throw new Error(`The file list of ${repo} is not a JSON array`);
  const entries: TreeEntry[] = [];
  for (const item of listing) {
    if (item && typeof item === "object" && typeof item.path === "string" && (item.type === "file" || item.type === "directory")) {
      entries.push({ type: item.type, path: item.path, ...(typeof item.size === "number" ? { size: item.size } : {}) });
    }
  }
  return entries;
}

/** `repo`'s `<folder>/manifest.json`, parsed but not yet checked (models/manifest.ts does that). */
export function fetchManifest(fetchImpl: FetchLike, repo: string, folder: string): Promise<unknown> {
  return fetchJson(fetchImpl, rawUrl(`${folder}/manifest.json`, repo), `${folder}/manifest.json`);
}

/** The identity of the file at `url`, from a HEAD request: the final
 * response's ETag without its quotes, or null when the request fails
 * (offline) or the answer is not OK or carries no readable ETag (a server
 * that does not expose it: CORS hides headers not named in
 * Access-Control-Expose-Headers). Never throws: a caller that cannot learn
 * the identity decides for itself what to serve.
 *
 * On Hugging Face (checked 2026-10-15 with curl -sIL and an Origin header)
 * resolve/ answers a HEAD with a 302 whose X-Linked-ETag is the LFS file's
 * sha256; fetch follows the redirect, and the CDN's answer, the one the
 * page sees, carries `etag: "<content hash>"` under
 * Access-Control-Expose-Headers: *, with Content-Length. A GET's final
 * response carries the same ETag. The hash changes with the bytes and only
 * with them: a file re-uploaded under the same name (a retrained model,
 * as setup_53_2d and setup_54_2d were on 2026-10-15) gets a new one, which
 * is what the model cache keys on (models/index.ts). */
export async function fetchEtag(fetchImpl: FetchLike, url: string): Promise<string | null> {
  try {
    const response = await fetchImpl(url, { method: "HEAD", cache: "no-store" });
    return response.ok ? etagOf(response) : null;
  } catch {
    return null;
  }
}

/** `response`'s ETag as a bare token: the quotes and a weak validator's
 * W/ prefix stripped; null when there is none (or it is empty). */
export function etagOf(response: Response): string | null {
  const header = response.headers.get("etag");
  if (header === null) return null;
  const bare = header.trim().replace(/^W\//i, "").replace(/^"(.*)"$/, "$1");
  return bare || null;
}

/** A downloaded file: its bytes and the ETag its response carried (null when none was readable). */
export interface FetchedFile {
  bytes: ArrayBuffer;
  etag: string | null;
}

/** Download `url` whole, reporting progress as the body streams in.
 *
 * `onProgress(loaded, total)` runs after every chunk; `total` is the
 * Content-Length when the server sent one (null otherwise, so a progress bar
 * knows to be indeterminate). The chunks are joined into one ArrayBuffer at
 * the end: onnxruntime-web takes the model as a single buffer. */
export async function fetchBytes(
  fetchImpl: FetchLike,
  url: string,
  onProgress?: (loaded: number, total: number | null) => void,
): Promise<ArrayBuffer> {
  return (await fetchFile(fetchImpl, url, onProgress)).bytes;
}

/** fetchBytes, with the response's ETag beside the bytes (the model cache
 * keys on it; the HEAD's ETag of a moment before could in principle be of
 * an older file than the GET then downloads). */
export async function fetchFile(
  fetchImpl: FetchLike,
  url: string,
  onProgress?: (loaded: number, total: number | null) => void,
): Promise<FetchedFile> {
  const what = fileNameOf(url);
  // no-store: the bytes go into this module's own IndexedDB cache; the
  // service worker's store cache (sw.ts) honours the request's cache mode and
  // would otherwise keep a second copy of every model.
  const response = await fetchImpl(url, { cache: "no-store" });
  throwForStatus(response, what);
  const etag = etagOf(response);
  const total = contentLength(response);
  if (!response.body) {
    // No streaming body (an old runtime, or a test's plain Response): one read.
    const bytes = await response.arrayBuffer();
    onProgress?.(bytes.byteLength, bytes.byteLength);
    return { bytes, etag };
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { bytes: bytes.buffer, etag };
}

function throwForStatus(response: Response, what: string): void {
  if (response.status === 429) throw new RateLimitError(what);
  if (!response.ok) throw new HubStatusError(response.status, `Could not fetch ${what}: HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`);
}

function contentLength(response: Response): number | null {
  const header = response.headers.get("content-length");
  if (header === null) return null;
  const n = Number(header);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

/** The last path segment of `url`, decoded, for messages. */
export function fileNameOf(url: string): string {
  const path = url.split(/[?#]/)[0];
  const last = path.slice(path.lastIndexOf("/") + 1);
  try {
    return decodeURIComponent(last) || url;
  } catch {
    return last || url;
  }
}
