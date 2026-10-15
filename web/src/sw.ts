// The service worker: the one piece of the browser mode that can answer an
// HTTP request, which a page cannot. Built on chunkmirage's relay module
// (web/chunkmirage/web/src/relay.ts), as chunkmirage's own sw.ts is, with a
// route of its own beside it. For requests from pages on this origin:
//
// - /api/*: the dashboard's own requests (fetch, EventSource, the Export
//   Config download link), and
// - /virtual/*: Neuroglancer's chunk requests for the model layers,
//   are both relayed to the page: `relay` posts the request to the pages it
//   is told to ask, over a MessageChannel, and builds the Response from the
//   reply (whole, or a head followed by the body once computed, or by an
//   event stream's pieces as they come), telling the page when the client
//   gives up. On the page, chunkmirage's Engine answers its share of
//   /virtual/ and src/backend/relay.ts answers /api/ through the Backend
//   (both through routes.ts there); a page with no route for a path
//   declines it at once.
// - local/<handleKey>/<path> under the page's directory: a file of a zarr directory the user picked
//   with the File System Access API, whose handle the page stored in
//   IndexedDB (src/state/local-store.ts). From Marwan Zouinkhi's prototype
//   (web/legacy/sw.js).
// - paint/<volume>/<path> under the page's directory: a file of an annotation
//   volume the Finetune tab paints, answered from the paint store
//   (src/state/paint-store.ts, IndexedDB) by src/state/paint-route.ts: GET,
//   HEAD, PUT and DELETE of whole objects, the little of S3 that the
//   Neuroglancer fork's writable zarr source uses, which MinIO answers in
//   server mode. The page itself reads and writes the same store directly
//   (a new volume's metadata, the paint for training).
//
// Cross-origin GETs of data (the zarr chunks the reader worker and the viewer
// fetch from a bucket) go through chunkmirage's store cache (`fromStore`): a
// region is read by many (overlapping input blocks, the raw layer in the
// viewer), and one cache here serves them all. Nothing on this origin is
// cached: the page, its scripts, the dashboard's static files, Vite's dev
// files and the viewer are the network's (and the browser's HTTP cache's)
// as they are.
/// <reference lib="webworker" />
import { fromStore, metadata, relay, stats } from "../chunkmirage/web/src/relay";
import type { StoreStats } from "../chunkmirage/web/src/types";
import { PAINT_ROUTE } from "./contracts";
import { FileSystemStore, getHandle } from "./state/local-store";
import { servePaint } from "./state/paint-route";
import { openPaintStore } from "./state/paint-store";

const sw = self as unknown as ServiceWorkerGlobalScope;
/** How long a page may take to answer /api/ before it counts as declining:
 * /api/models may be downloading a model before it answers; under Chrome's
 * 5-minute limit on a fetch event. A chunk gets the relay's default, two
 * minutes. */
const API_TIMEOUT_MS = 240_000;

/** The files that describe a volume rather than hold its voxels: zarr v2's,
 * zarr v3's and N5's. Only the content type of a local read comes from this;
 * which requests the page makes for itself is chunkmirage's `metadata`. */
const METADATA_FILE = /(^|\/)(\.z(array|attrs|group)|zarr\.json|attributes\.json)$/;

sw.addEventListener("install", () => void sw.skipWaiting());
sw.addEventListener("activate", (event) => event.waitUntil(sw.clients.claim()));

sw.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (url.origin === sw.location.origin) {
    // Matched anywhere in the path, as chunkmirage matches /virtual/: the
    // page may be served under a base (/app/ on the Pages site) and the
    // dashboard's URLs are root-relative (/api/...).
    if (url.pathname.includes("/virtual/")) return event.respondWith(relayed(req, relay(req, { pick: appWindows })));
    if (url.pathname.includes("/api/")) {
      return event.respondWith(relayed(req, relay(req, { pick: (pages) => appWindows(pages, event.clientId), timeout: API_TIMEOUT_MS })));
    }
    // local/ directly under the page's directory (the worker's scope), not
    // anywhere in the path: a page module at such a path would be answered
    // 404 here (the module for this is src/local-folder/, named so that no
    // path of the page has /local/ in it, since a worker from an older build
    // that matched loosely controls the page until the new one takes over).
    const localRoot = new URL("local/", sw.registration.scope).pathname;
    if (url.pathname.startsWith(localRoot)) return event.respondWith(serveLocal(url.pathname.slice(localRoot.length), req.signal));
    // paint/ likewise: the annotation volumes' files, every method.
    const paintRoot = new URL(PAINT_ROUTE, sw.registration.scope).pathname;
    if (url.pathname.startsWith(paintRoot)) return event.respondWith(servedPaint(req, url, paintRoot));
    return; // this origin's own files: never cached here
  }
  // Data a script fetches from elsewhere (not a page, a script or an image).
  // `cache: "no-store"` on the request opts out: the model catalog fetches
  // .onnx files (up to 43 MB) that way, since it keeps them in IndexedDB
  // itself and a second copy here would only push chunks out of the store.
  // A volume's metadata (.zattrs, zarr.json, attributes.json) is left to
  // the page that asked for it, which is how Neuroglancer's own site reads
  // a volume. Chrome gates a public site's requests to a host on the user's
  // private network (an institution's 10.x server) behind a permission it
  // can ask for, once, when the page makes such a request; a request made
  // here, by a service worker, it refuses outright unless the permission
  // was given before, with nothing to ask. On the deployed site a volume on
  // such a host failed that way ("Failed to fetch") while Neuroglancer read
  // it. The metadata is the first thing read of any volume, so the page's
  // request for it is the one Chrome can ask on, if it asks at all, and the
  // chunks that follow, read here, pass once it has. A handful of small
  // files per volume, read once: nothing lost to the cache.
  if (req.method !== "GET" || req.destination !== "" || req.cache === "no-store" || metadata(url)) return;
  event.respondWith(fromStore(req));
});

sw.addEventListener("message", (event) => {
  if ((event.data as { type?: string })?.type === "store-stats") event.ports[0]?.postMessage(stats satisfies StoreStats);
});

/** The pages that may answer: this origin's windows that are not the
 * Neuroglancer iframe (served under ./ng/, it answers nothing). When the
 * request came from one of them (the dashboard's own fetch, `clientId`),
 * that one alone is asked: each open tab has a backend of its own. */
function appWindows(pages: WindowClient[], clientId?: string): WindowClient[] {
  const app = pages.filter((c) => !new URL(c.url).pathname.includes("/ng/"));
  const own = clientId ? app.find((c) => c.id === clientId) : undefined;
  return own ? [own] : app;
}

/** A relay that failed before it could answer (a body that could not be
 * read, say) as a response, rather than a network error the client cannot
 * tell from the page being closed. */
async function relayed(req: Request, response: Promise<Response>): Promise<Response> {
  try {
    return await response;
  } catch (e) {
    return new Response(`relaying ${req.method} ${new URL(req.url).pathname} failed: ${String((e as Error)?.message ?? e)}`, { status: 500 });
  }
}

// ------------------------------------------------ /local/: a zarr directory picked on this machine

/** One store per picked directory, kept for the worker's life: the handle
 * is read from IndexedDB once, not per chunk, and the store keeps the
 * directory handles it has walked to. Dropped when a read fails for a
 * reason other than a missing file (permission withdrawn, say), so the next
 * request starts afresh from the stored handle. */
const localStores = new Map<string, Promise<FileSystemStore | null>>();

function localStore(handleKey: string): Promise<FileSystemStore | null> {
  let pending = localStores.get(handleKey);
  if (!pending) {
    pending = getHandle(handleKey).then((dir) => (dir ? new FileSystemStore(dir) : null));
    pending.catch(() => localStores.delete(handleKey));
    localStores.set(handleKey, pending);
  }
  return pending;
}

/** `.../local/<handleKey>/<path>`: the file at `path` in the directory whose
 * handle the page stored under `handleKey`, the File itself as the body:
 * the browser streams it to the client and its bytes never enter this
 * worker. Read into the worker instead (arrayBuffer), hundreds of chunks
 * in flight on a slow volume (a network mount) were hundreds of megabytes
 * held here, and the worker crawled; the head of a streamed response comes
 * back before its body, which is the read's own time, not a stall. 404
 * when the handle is gone or the file is not there (a missing chunk, which
 * zarr readers take as the fill value). Permission must have been granted
 * on the page: a service worker cannot ask. A viewer cancels the chunk
 * requests it no longer needs as the view moves (`signal`); their reads
 * are not started, or the ones in flight would hold up the ones it still
 * wants. */
async function serveLocal(rest: string, signal: AbortSignal): Promise<Response> {
  const [handleKey, ...parts] = rest.split("/");
  const path = parts.join("/");
  try {
    const store = await localStore(handleKey);
    if (!store) {
      localStores.delete(handleKey);  // a handle stored after this is found next time
      return new Response(`no local directory is registered as ${handleKey}`, { status: 404 });
    }
    if (signal.aborted) return new Response(null, { status: 499, statusText: "the request was cancelled" });
    const file = await store.file(path);
    if (file === undefined) return new Response("not found", { status: 404 });
    const isMeta = METADATA_FILE.test(path);
    const type = isMeta ? "application/json" : "application/octet-stream";
    if (file instanceof Blob) return new Response(file, { headers: { "content-type": type, "content-length": String(file.size) } });
    return new Response(await file.arrayBuffer(), { headers: { "content-type": type } });  // a store without Files (a test's)
  } catch (e) {
    localStores.delete(handleKey);
    return new Response(String((e as Error)?.message ?? e), { status: 500 });
  }
}

// ------------------------------------------------ /paint/: the annotation volumes the Finetune tab paints

/** One store for the worker's life; it opens the database on its first
 * request (db.ts keeps the connection). Same-origin, so a paint request
 * never reaches `fromStore`. */
const paintStore = openPaintStore();

/** `.../paint/<volume>/<path>`: the route (src/state/paint-route.ts) on the
 * paint store, as a Response. A store that fails (IndexedDB refusing, say)
 * is a 500 with the reason, as the fork's write error shows it, rather than
 * a network error the viewer cannot tell from the worker being gone. */
async function servedPaint(req: Request, url: URL, root: string): Promise<Response> {
  try {
    const { status, headers, body } = await servePaint(paintStore, {
      method: req.method,
      path: url.pathname.slice(root.length),
      query: url.search,
      range: req.headers.get("range"),
      body: () => req.arrayBuffer(),
    });
    // The store's bytes are ArrayBuffer-backed (IndexedDB and a request's
    // body give no SharedArrayBuffer); BodyInit's type insists on it.
    return new Response(body as Uint8Array<ArrayBuffer> | null, { status, headers });
  } catch (e) {
    return new Response(`${req.method} ${url.pathname}: ${String((e as Error)?.message ?? e)}`, { status: 500 });
  }
}
