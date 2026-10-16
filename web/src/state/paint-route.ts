// The paint route: what the service worker answers under `paint/` (src/sw.ts),
// as a function of the request and the paint store, so that the tests run it
// without a worker. It is the little of S3's object API that the Neuroglancer
// fork's `s3+http(s)://` kvstore uses (src/kvstore/s3/common.ts and
// src/kvstore/http/read.ts at the commit web/neuroglancer.lock pins), which
// MinIO answers in server mode: GET of an object, whole or one byte range
// (206); HEAD for its size (the fork's stat); PUT of a whole object (a chunk
// the brush painted, a volume's metadata); DELETE, fine whether or not the
// object was there. An object that is not there is a 404, which a zarr reader
// takes as the fill value (a chunk never painted). No listing: the fork lists
// only to complete a URL in its UI and a zarr array is read by key, so a
// request with a query (an S3 listing probe, `?list-type=2&prefix=...`) is a
// 404 like any object that is not there.
import type { PaintStore } from "../contracts";

/** A request as the route sees it, the worker's Request reduced to what the
 * route needs (and a test can make by hand). */
export interface PaintRequest {
  method: string;
  /** What follows `paint/` in the URL's path, as the URL has it
   * (percent-encoded): `<volume>/<path within the volume>`. */
  path: string;
  /** The URL's query (`URL.search`): "" or undefined for a plain object URL. */
  query?: string;
  /** The request's Range header, if any. */
  range?: string | null;
  /** The request's body (a PUT's object), read only when it is wanted. */
  body: () => Promise<ArrayBuffer>;
}

/** What the worker builds its Response from: `body` null where HTTP has
 * none (HEAD, 204, an empty 200). Header names lower-case. */
export interface PaintResponse {
  status: number;
  headers: Record<string, string>;
  body: Uint8Array | null;
}

/** The methods the route answers, as the 405's `allow` header lists them. */
export const PAINT_METHODS = "GET, HEAD, PUT, DELETE";

/** The volume is zarr v2 (finetune/volume.ts): these are its metadata files. */
const METADATA_FILE = /(^|\/)\.z(array|attrs|group)$/;

export async function servePaint(store: PaintStore, request: PaintRequest): Promise<PaintResponse> {
  if (request.query) return text(404, "nothing is listed here: an annotation volume's files are read by their path");
  const target = parsePath(request.path);
  if ("error" in target) return text(400, target.error);
  const { volume, path } = target;
  switch (request.method) {
    case "GET":
    case "HEAD": {
      const bytes = await store.get(volume, path);
      if (bytes === undefined) return request.method === "HEAD" ? { status: 404, headers: {}, body: null } : text(404, "not found");
      const response = object(bytes, path, request.range ?? null);
      return request.method === "HEAD" ? { ...response, body: null } : response;
    }
    case "PUT": {
      const bytes = new Uint8Array(await request.body());
      await store.put(volume, path, bytes);
      return { status: 200, headers: { etag: etag(bytes) }, body: null };
    }
    case "DELETE":
      await store.delete(volume, path);
      return { status: 204, headers: {}, body: null };
    default: {
      const response = text(405, `${request.method} is not a method of the paint route (${PAINT_METHODS})`);
      response.headers.allow = PAINT_METHODS;
      return response;
    }
  }
}

/** The volume segment as the viewer may spell it: the volume id, or the id
 * with a `~<n>` refresh suffix (viewer/layers.ts respelledPaintUrl). */
const VOLUME_SEGMENT = /^([^~]+)(?:~\d+)?$/;

/** The volume id and the path within the volume from what followed
 * `paint/`, or why there is none: an empty, "." or ".." segment anywhere is
 * no file of a volume (a zarr writes none, and a store is no place to walk
 * up from).
 *
 * The volume segment may carry a `~<n>` suffix (`<volume>~2`), which is
 * dropped: it is how the page has Neuroglancer re-read a paint layer whose
 * labels were written behind its back (respelledPaintUrl respells the
 * layer's source URL, which the viewer takes for a new source), the way
 * the server respells the MinIO port, and every spelling is the same
 * volume in the store. A "~" anywhere else in the segment, or one with no
 * digits after it, names no volume (a volume id has no "~"), so it is
 * refused like the other malformed segments rather than passed to the
 * store as an id. */
function parsePath(raw: string): { volume: string; path: string } | { error: string } {
  let decoded: string;
  try {
    decoded = decodeURIComponent(raw);
  } catch {
    return { error: `malformed percent-encoding in the path ${raw}` };
  }
  const [spelled, ...rest] = decoded.split("/");
  if (spelled === "") return { error: "no volume in the path: expected paint/<volume>/<path>" };
  if (rest.length === 0 || (rest.length === 1 && rest[0] === "")) return { error: `no file in the path: expected paint/${spelled}/<path>` };
  for (const segment of [spelled, ...rest]) {
    if (segment === "" || segment === "." || segment === "..") return { error: `"${segment}" is not a segment of a volume's file path: ${decoded}` };
  }
  const volume = VOLUME_SEGMENT.exec(spelled)?.[1];
  if (volume === undefined) return { error: `"${spelled}" is not a volume: a "~" may only end the volume as its refresh suffix, ~<digits>` };
  return { volume, path: rest.join("/") };
}

/** The object, whole (200) or the part its Range header asks for (206), or
 * 416 when that part lies past its end. The headers are S3's for an object:
 * its type, its length, that ranges are taken, its ETag. */
function object(bytes: Uint8Array, path: string, range: string | null): PaintResponse {
  const headers: Record<string, string> = {
    "content-type": METADATA_FILE.test(path) ? "application/json" : "application/octet-stream",
    "accept-ranges": "bytes",
    etag: etag(bytes),
  };
  const span = range === null ? "whole" : byteRange(range, bytes.byteLength);
  if (span === "whole") return { status: 200, headers: { ...headers, "content-length": String(bytes.byteLength) }, body: bytes };
  if (span === "unsatisfiable") return { status: 416, headers: { ...headers, "content-range": `bytes */${bytes.byteLength}` }, body: null };
  const [first, last] = span;
  return {
    status: 206,
    headers: { ...headers, "content-range": `bytes ${first}-${last}/${bytes.byteLength}`, "content-length": String(last - first + 1) },
    body: bytes.subarray(first, last + 1),
  };
}

/** The bytes [first, last] of an object of `length` bytes that a Range
 * header asks for: `bytes=a-b` (what the fork sends), `bytes=a-` and
 * `bytes=-n` (S3 takes both). "whole" for a header that is not one such
 * range (several ranges, a typo: HTTP lets a server ignore it and answer
 * 200), "unsatisfiable" for one past the end (416), which the fork expects
 * for its one-byte probe of an empty object. */
function byteRange(header: string, length: number): "whole" | "unsatisfiable" | [number, number] {
  const m = /^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$/.exec(header);
  if (m === null || (m[1] === "" && m[2] === "")) return "whole";
  if (m[1] === "") {
    const suffix = Number(m[2]);
    if (suffix === 0 || length === 0) return "unsatisfiable";
    return [Math.max(0, length - suffix), length - 1];
  }
  const first = Number(m[1]);
  if (first >= length) return "unsatisfiable";
  const last = m[2] === "" ? length - 1 : Math.min(Number(m[2]), length - 1);
  if (last < first) return "whole";
  return [first, last];
}

/** S3 sends an object's ETag (its MD5) with it. Nothing here reads it, so a
 * cheap hash (FNV-1a over the bytes, with the length) stands in, quoted as
 * HTTP has it. */
function etag(bytes: Uint8Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193);
  return `"${bytes.length.toString(16)}-${(h >>> 0).toString(16).padStart(8, "0")}"`;
}

function text(status: number, message: string): PaintResponse {
  const body = new TextEncoder().encode(message);
  return { status, headers: { "content-type": "text/plain; charset=utf-8", "content-length": String(body.byteLength) }, body };
}
