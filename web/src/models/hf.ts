// Hugging Face, as the catalog reads it: the repo's file tree (the Hub API),
// its small text files (raw/) and the .onnx binaries (resolve/, which answers
// with a redirect to the LFS CDN that fetch follows by itself). Nothing here
// knows about manifests or variants; models/index.ts does.
//
// Anonymous requests are rate-limited: after a burst of downloads from one
// address the Hub answers HTTP 429. Every fetch here turns a 429 into a
// RateLimitError whose message says what happened and to retry in a minute;
// callers decide whether a cache or a fallback can stand in for the answer.
//
// CORS (checked 2026-10-14): the API, raw/ and resolve/ reflect the page's
// Origin; the CDN that resolve/ redirects to answers "*" and exposes every
// header, so Content-Length is readable and download progress can be shown.

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

/** The folders and files at `path` in the repo. */
export async function fetchTree(fetchImpl: FetchLike, path = ""): Promise<TreeEntry[]> {
  const listing = await fetchJson(fetchImpl, treeUrl(path), `the file list of ${HF_REPO}`);
  if (!Array.isArray(listing)) throw new Error(`The file list of ${HF_REPO} is not a JSON array`);
  const entries: TreeEntry[] = [];
  for (const item of listing) {
    if (item && typeof item === "object" && typeof item.path === "string" && (item.type === "file" || item.type === "directory")) {
      entries.push({ type: item.type, path: item.path, ...(typeof item.size === "number" ? { size: item.size } : {}) });
    }
  }
  return entries;
}

/** The repo's `<folder>/manifest.json`, parsed but not yet checked (models/manifest.ts does that). */
export function fetchManifest(fetchImpl: FetchLike, folder: string): Promise<unknown> {
  return fetchJson(fetchImpl, rawUrl(`${folder}/manifest.json`), `${folder}/manifest.json`);
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
  const what = fileNameOf(url);
  const response = await fetchImpl(url);
  throwForStatus(response, what);
  const total = contentLength(response);
  if (!response.body) {
    // No streaming body (an old runtime, or a test's plain Response): one read.
    const bytes = await response.arrayBuffer();
    onProgress?.(bytes.byteLength, bytes.byteLength);
    return bytes;
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
  return bytes.buffer;
}

function throwForStatus(response: Response, what: string): void {
  if (response.status === 429) throw new RateLimitError(what);
  if (!response.ok) throw new Error(`Could not fetch ${what}: HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`);
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
