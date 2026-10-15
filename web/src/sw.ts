// The service worker: the one piece of the browser mode that can answer an
// HTTP request, which a page cannot. It does three things, all for requests
// from pages on this origin:
//
// - /api/*: the dashboard's own requests (fetch, EventSource, the Export
//   Config download link). Each is posted to the page over a MessageChannel
//   and the Response is built from the page's reply (src/backend/relay.ts
//   answers, through the Backend); a streamed reply becomes a streamed
//   Response, and a client that gives up is reported to the page.
// - /virtual/*: Neuroglancer's chunk requests for the model layers. These
//   belong to chunkmirage's Engine (web/chunkmirage/web/src/engine.ts), which
//   registers THIS file under the name sw.js and expects its own relay
//   protocol on the other end: {path, range} to every open page with a
//   MessagePort, a Reply back (whole, or a `stream: true` head followed by a
//   Later with the body), and {cancel: true} to the page when the client
//   stops waiting. That relay and the store cache below are ported from
//   chunkmirage's sw.ts (JaneliaSciComp/chunkmirage, the commit in
//   web/chunkmirage.lock): re-check them against it whenever the lock is
//   bumped. Credit: chunkmirage's authors.
// - /local/<handleKey>/<path>: a file of a zarr directory the user picked
//   with the File System Access API, whose handle the page stored in
//   IndexedDB (src/state/local-store.ts). From Marwan Zouinkhi's prototype
//   (web/legacy/sw.js).
//
// Cross-origin GETs of data (the zarr chunks the reader worker and the viewer
// fetch from a bucket) go through chunkmirage's store cache: a region is read
// by many (overlapping input blocks, the raw layer in the viewer), and one
// cache here serves them all. Nothing on this origin is cached: the page, its
// scripts, the dashboard's static files, Vite's dev files and the viewer are
// the network's (and the browser's HTTP cache's) as they are.
/// <reference lib="webworker" />
import type { Later, Reply } from "../chunkmirage/web/src/types";
import type { ApiAsk, ApiChunk, ApiHead } from "./backend/protocol";
import { FileSystemStore, getHandle } from "./state/local-store";

const sw = self as unknown as ServiceWorkerGlobalScope;
const STORE_BYTES = 512 * 2 ** 20; // fetched (still compressed) store chunks kept
const MAX_FETCHES = 24; // upstream at once; more pending and Chrome refuses some
const VIRTUAL_TIMEOUT_MS = 120_000; // a page that never answers a chunk request
const API_TIMEOUT_MS = 240_000; // /api/models may be downloading a model before it answers; under Chrome's 5-minute limit on a fetch event

sw.addEventListener("install", () => void sw.skipWaiting());
sw.addEventListener("activate", (event) => event.waitUntil(sw.clients.claim()));

sw.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (url.origin === sw.location.origin) {
    // Matched anywhere in the path, as chunkmirage matches /virtual/: the
    // page may be served under a base (/app/ on the Pages site) and the
    // dashboard's URLs are root-relative (/api/...).
    if (url.pathname.includes("/virtual/")) return event.respondWith(relayVirtual(url.pathname, req.headers.get("range")));
    if (url.pathname.includes("/api/")) {
      return event.respondWith(relayApi(event, url).catch((e) => new Response(`relaying ${req.method} ${url.pathname} failed: ${String((e as Error)?.message ?? e)}`, { status: 500 })));
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
  if ((event.data as { type?: string })?.type === "store-stats") event.ports[0]?.postMessage(stats);
});

// ------------------------------------------------ /api/*: the dashboard's requests, answered by the page

/** Relay one /api/ request to the page's backend and answer with its reply. */
async function relayApi(event: FetchEvent, url: URL): Promise<Response> {
  const req = event.request;
  const body = req.method === "GET" || req.method === "HEAD" ? null : await req.arrayBuffer();
  const ask: ApiAsk = { api: true, path: url.pathname, method: req.method, url: req.url, headers: [...req.headers], body };
  const pages = await appWindows(event.clientId);
  if (!pages.length) return new Response("the cellmap-flow page that answers /api/ requests is not open", { status: 503 });
  // Every candidate page is asked (each has its own backend; the first to
  // answer wins), with its own copy of the body: a buffer can be transferred
  // to one receiver only.
  const asks = pages.map((page, i) => askApi(page, i === pages.length - 1 ? ask : { ...ask, body: body ? body.slice(0) : null }));
  const head = await firstHead(asks.map((a) => a.head));
  const owner = head ? asks.find((a) => a.answered === head) : undefined;
  for (const a of asks) if (a !== owner || !head || !("stream" in head)) a.close();
  if (!head) return new Response(`no page answered ${req.method} ${url.pathname}`, { status: 504 });
  const headers = new Headers(head.headers);
  headers.set("cache-control", "no-store");
  if (!("stream" in head)) return new Response(head.body, { status: head.status, headers });
  // A streamed body (server-sent events): the chunks arrive as the page sends
  // them. If the client stops reading (an EventSource closed), the stream is
  // cancelled, and the page is told so it can stop producing.
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      owner!.later = (m: ApiChunk) => {
        if ("chunk" in m) return controller.enqueue(new Uint8Array(m.chunk));
        if ("end" in m) controller.close();
        else controller.error(new Error(m.error));
        owner!.close();
      };
    },
    cancel() {
      owner!.tell({ cancel: true });
      owner!.close();
    },
  });
  return new Response(stream, { status: head.status, headers });
}

/** The pages that can answer /api/: this origin's windows that are not the
 * Neuroglancer iframe (served under ./ng/). When the request came from one
 * of them (the dashboard's own fetch), that one alone is asked. */
async function appWindows(clientId: string): Promise<Client[]> {
  const all = await sw.clients.matchAll({ type: "window", includeUncontrolled: true });
  const pages = all.filter((c) => !new URL(c.url).pathname.includes("/ng/"));
  const own = pages.find((c) => c.id === clientId);
  return own ? [own] : pages;
}

/** The first non-null head, or null once every page has declined or timed out. */
function firstHead(heads: Promise<ApiHead | null>[]): Promise<ApiHead | null> {
  return new Promise((resolve) => {
    let left = heads.length;
    if (!left) resolve(null);
    for (const h of heads) void h.then((head) => (head ? resolve(head) : --left || resolve(null)));
  });
}

/** Ask one page to answer `ask`. Its first non-null message is the head (a
 * null is chunkmirage's listener declining, which shares the page's
 * "message" event and is ignored); later messages are a streamed body's chunks. */
function askApi(client: Client, ask: ApiAsk) {
  const channel = new MessageChannel();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const a = {
    answered: null as ApiHead | null,
    later: (_: ApiChunk) => {},
    head: null as unknown as Promise<ApiHead | null>,
    tell: (m: { cancel: true }) => channel.port1.postMessage(m),
    close: () => {
      clearTimeout(timer);
      channel.port1.close();
    },
  };
  a.head = new Promise<ApiHead | null>((resolve) => {
    timer = setTimeout(() => resolve(null), API_TIMEOUT_MS);
    channel.port1.onmessage = (e: MessageEvent<ApiHead | ApiChunk | null>) => {
      if (e.data === null || e.data === undefined) return; // chunkmirage's listener declining, whenever it gets to it
      if (a.answered) return a.later(e.data as ApiChunk);
      clearTimeout(timer);
      a.answered = e.data as ApiHead;
      resolve(a.answered);
    };
  });
  client.postMessage(ask, ask.body ? [channel.port2, ask.body] : [channel.port2]);
  return a;
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

// ------------------------------------------------ the store cache (chunkmirage's)
interface Stored {
  body: Blob;
  status: number;
  headers: [string, string][];
}
const stored = new Map<string, Promise<Stored | null>>(); // in use order; null: not kept (opaque, or an error)
const sizes = new Map<string, number>();
let storedBytes = 0;
let fetching = 0;
const waiting: (() => void)[] = [];
const stats = { requests: 0, hits: 0, fetches: 0, fetchedBytes: 0 };
const KEEP = ["content-type", "content-range", "content-length"];

async function fromStore(req: Request): Promise<Response> {
  const key = `${req.url} ${req.headers.get("range") ?? ""}`;
  stats.requests++;
  let entry = stored.get(key);
  if (entry) {
    stats.hits++;
    stored.delete(key);
    stored.set(key, entry);
  } else {
    entry = upstream(req).then(
      async (r) => {
        if (r.type === "opaque" || (!r.ok && r.status !== 404)) {
          forget(key);
          return null;
        }
        const body = await r.blob();
        const s: Stored = {
          body,
          status: r.status,
          headers: KEEP.flatMap((h) => {
            const v = r.headers.get(h);
            return v ? [[h, v] as [string, string]] : [];
          }),
        };
        stats.fetchedBytes += body.size;
        sizes.set(key, body.size);
        storedBytes += body.size;
        evict();
        return s;
      },
      (e) => {
        forget(key);
        throw e;
      },
    );
    stored.set(key, entry);
  }
  const s = await entry;
  return s ? new Response(s.body, { status: s.status, headers: s.headers }) : fetch(req); // not kept: as it is
}

/** A network fetch, at most MAX_FETCHES at once. */
async function upstream(req: Request): Promise<Response> {
  while (fetching >= MAX_FETCHES) await new Promise<void>((r) => waiting.push(r));
  fetching++;
  stats.fetches++;
  try {
    return await fetch(req);
  } finally {
    fetching--;
    waiting.shift()?.();
  }
}

function forget(key: string) {
  stored.delete(key);
  storedBytes -= sizes.get(key) ?? 0;
  sizes.delete(key);
}

function evict() {
  for (const key of stored.keys()) {
    if (storedBytes <= STORE_BYTES) break;
    if (sizes.has(key)) forget(key); // settled entries only: one in flight has no size yet
  }
}

// ------------------------------------------------ /virtual/*: chunkmirage's relay to its Engine

async function relayVirtual(path: string, range: string | null): Promise<Response> {
  // every open page on this site is asked at once: the one whose share of /virtual/ this is
  // answers, the rest (the viewer's frame, other tabs) never do
  const pages = await sw.clients.matchAll({ type: "window", includeUncontrolled: true });
  const asks = pages.map((page) => askVirtual(page, path, range));
  const reply = await firstReply(asks.map((a) => a.reply));
  const owner = reply ? asks.find((a) => a.answered === reply) : undefined;
  for (const a of asks) if (a !== owner || !reply || !("stream" in reply)) a.cancel();
  if (!reply) return new Response("the registration page that serves this volume is closed", { status: 503 });
  const headers: Record<string, string> = { "content-type": reply.type, "cache-control": "no-store" };
  if (reply.range) headers["content-range"] = reply.range; // a part of a file (mesh fragments)
  if (!("stream" in reply)) return new Response(reply.body, { status: reply.status, headers });
  // A chunk: the head now, the body when the page has computed it. A client that stops waiting
  // cancels this stream (a service worker sees no other sign of it), and the page is told, so
  // it can drop the work nobody wants any more.
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      owner!.later = (m: Later) => {
        if ("body" in m) {
          controller.enqueue(new Uint8Array(m.body));
          controller.close();
        } else controller.error(new Error(m.error));
        owner!.cancel();
      };
    },
    cancel() {
      owner!.tell({ cancel: true });
      owner!.cancel();
    },
  });
  return new Response(body, { status: reply.status, headers });
}

/** The first reply that is not null, or null once every page has declined or timed out. */
function firstReply(replies: Promise<Reply>[]): Promise<Reply> {
  return new Promise((resolve) => {
    let left = replies.length;
    if (!left) resolve(null);
    for (const r of replies) void r.then((reply) => (reply ? resolve(reply) : --left || resolve(null)));
  });
}

/** Ask one page for `path`: its first message is the reply (or its head), later ones a
 * streamed body's contents. */
function askVirtual(client: Client, path: string, range: string | null) {
  const channel = new MessageChannel();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const a = {
    answered: null as Reply,
    later: (_: Later) => {},
    reply: null as unknown as Promise<Reply>,
    tell: (m: { cancel: true }) => channel.port1.postMessage(m),
    cancel: () => {
      clearTimeout(timer);
      channel.port1.close();
    },
  };
  let got = false;
  a.reply = new Promise<Reply>((resolve) => {
    timer = setTimeout(() => resolve(null), VIRTUAL_TIMEOUT_MS); // a page that never answers
    channel.port1.onmessage = (e: MessageEvent<Reply | Later>) => {
      if (got) return a.later(e.data as Later);
      got = true;
      clearTimeout(timer);
      a.answered = e.data as Reply;
      resolve(a.answered);
    };
  });
  client.postMessage({ path, range }, [channel.port2]);
  return a;
}
