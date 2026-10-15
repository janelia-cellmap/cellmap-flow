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
// - /local/<handleKey>/<path>: a file of a zarr directory the user picked
//   with the File System Access API, whose handle the page stored in
//   IndexedDB (src/state/local-store.ts). From Marwan Zouinkhi's prototype
//   (web/legacy/sw.js).
//
// Cross-origin GETs of data (the zarr chunks the reader worker and the viewer
// fetch from a bucket) go through chunkmirage's store cache (`fromStore`): a
// region is read by many (overlapping input blocks, the raw layer in the
// viewer), and one cache here serves them all. Nothing on this origin is
// cached: the page, its scripts, the dashboard's static files, Vite's dev
// files and the viewer are the network's (and the browser's HTTP cache's)
// as they are.
/// <reference lib="webworker" />
import { fromStore, relay, stats } from "../chunkmirage/web/src/relay";
import type { StoreStats } from "../chunkmirage/web/src/types";
import { FileSystemStore, getHandle } from "./state/local-store";

const sw = self as unknown as ServiceWorkerGlobalScope;
/** How long a page may take to answer /api/ before it counts as declining:
 * /api/models may be downloading a model before it answers; under Chrome's
 * 5-minute limit on a fetch event. A chunk gets the relay's default, two
 * minutes. */
const API_TIMEOUT_MS = 240_000;

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
    const local = url.pathname.indexOf("/local/");
    if (local !== -1) return event.respondWith(serveLocal(url.pathname.slice(local + "/local/".length)));
    return; // this origin's own files: never cached here
  }
  // Data a script fetches from elsewhere (not a page, a script or an image).
  // `cache: "no-store"` on the request opts out: the model catalog fetches
  // .onnx files (up to 43 MB) that way, since it keeps them in IndexedDB
  // itself and a second copy here would only push chunks out of the store.
  if (req.method === "GET" && req.destination === "" && req.cache !== "no-store") event.respondWith(fromStore(req));
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

/** `.../local/<handleKey>/<path>`: the file at `path` in the directory whose
 * handle the page stored under `handleKey`. 404 when the handle is gone or
 * the file is not there (a missing chunk, which zarr readers take as the
 * fill value). Permission must have been granted on the page: a service
 * worker cannot ask. */
async function serveLocal(rest: string): Promise<Response> {
  const [handleKey, ...parts] = rest.split("/");
  const path = parts.join("/");
  try {
    const dir = await getHandle(handleKey);
    if (!dir) return new Response(`no local directory is registered as ${handleKey}`, { status: 404 });
    const bytes = await new FileSystemStore(dir).get(path);
    if (bytes === undefined) return new Response("not found", { status: 404 });
    const isMeta = /\.z(array|attrs|group)$|zarr\.json$/.test(path);
    // The bytes alone (a Uint8Array may be a view on a larger buffer).
    const body = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    return new Response(body, { headers: { "content-type": isMeta ? "application/json" : "application/octet-stream" } });
  } catch (e) {
    return new Response(String((e as Error)?.message ?? e), { status: 500 });
  }
}
