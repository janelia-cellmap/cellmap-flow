// The page side of the service worker relay for /api/* requests.
//
// registerServiceWorker() puts sw.js in control of this page before anything
// else runs: the dashboard's JavaScript asks /api/... the moment it loads,
// and a request the service worker does not see goes to the static host and
// 404s. listenForApiRequests() then answers each relayed request through the
// Backend: the service worker (src/sw.ts, on chunkmirage's relay) posts the
// request as an Ask (web/chunkmirage/web/src/types.ts) with a MessagePort,
// chunkmirage's route() (routes.ts there) hands the ones under /api/ here,
// and the answer goes back on the port as a Reply (whole) or a Reply head
// with `stream: true` and then Laters (an event stream's pieces, then
// `done`), the shapes relay.ts there turns into a Response. One listener on
// the page for every route, the engine's /virtual/ share and this one: the
// worker takes a page's first message as its answer, so two listeners each
// declining what is not theirs would answer for each other.
//
// toApiRequest and sendResponse are the pure parts, exported for the tests.
import { route } from "../../chunkmirage/web/src/routes";
import type { Ask, Later, Reply } from "../../chunkmirage/web/src/types";
import type { ApiRequest, ApiResponse, Backend } from "../contracts";

/** sessionStorage key set across the one reload that makes the service
 * worker control a first visit, so a browser that refuses to let it never
 * reloads in a loop. */
const RELOAD_FLAG = "cellmap-flow:sw-reload";

/** Register sw.js (scope ./, beside the page: the name chunkmirage's Engine
 * registers too, so both get one registration) and resolve once it controls
 * this page. On a first visit the worker activates but does not control the
 * page that registered it, so the page is reloaded once; after that every
 * request from the page is the worker's to answer. */
export async function registerServiceWorker(): Promise<void> {
  if (!("serviceWorker" in navigator)) {
    throw new Error("This browser has no service workers (a private window?): the browser mode needs one to answer the dashboard's requests.");
  }
  if (!window.isSecureContext) {
    throw new Error("This page needs a secure context: open it over https, or through localhost.");
  }
  // updateViaCache: "none": a deployed sw.js is fetched afresh on each check,
  // so a new build's worker is not held back by the HTTP cache.
  await navigator.serviceWorker.register("sw.js", { scope: "./", updateViaCache: "none" });
  await navigator.serviceWorker.ready;
  if (navigator.serviceWorker.controller) {
    flag(null);
    return;
  }
  if (flag()) {
    flag(null);
    throw new Error("The service worker does not control this page even after a reload. A reload that bypasses it (shift-reload) does this: load the page again normally.");
  }
  flag("1");
  location.reload();
  // The page is going away; never resolve, so boot does not go on.
  await new Promise<never>(() => {});
}

/** The reload flag: read it, or set it (null clears). sessionStorage can be
 * unavailable (storage blocked), in which case there is no flag and a loop
 * is not detected, but the page still works once a reload has made the
 * worker its controller. */
function flag(): string | null;
function flag(value: string | null): void;
function flag(value?: string | null): string | null | void {
  try {
    if (value === undefined) return sessionStorage.getItem(RELOAD_FLAG);
    if (value === null) sessionStorage.removeItem(RELOAD_FLAG);
    else sessionStorage.setItem(RELOAD_FLAG, value);
  } catch {
    return null;
  }
}

/** Answer the /api/* requests the service worker relays, through `backend`:
 * a route for the paths holding "/api/" (matched anywhere, as the worker
 * matches: the page may be served under a base). Returns what removes it. */
export function listenForApiRequests(backend: Backend): () => void {
  return route("/api/", (ask, port) => void answer(backend, ask, port));
}

async function answer(backend: Backend, ask: Ask, port: MessagePort): Promise<void> {
  let response: ApiResponse;
  try {
    response = await backend.handle(toApiRequest(ask));
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`[cellmap-flow] ${ask.method} ${ask.path} failed:`, e);
    response = { status: 500, json: { success: false, error: message } };
  }
  await sendResponse(response, port);
}

/** The ApiRequest for a relayed request: the path from "/api/" on (the page
 * may be served under a base, and the dashboard's URLs are root-relative),
 * the headers lower-cased, and the body parsed as JSON when it is JSON (null
 * otherwise, as Flask's get_json(silent=True) gives). The query is the
 * request's search (`?offset=12`, which sse.ts reads), which the relay
 * passes beside the path. */
export function toApiRequest(ask: Pick<Ask, "method" | "path" | "search" | "headers" | "body">): ApiRequest {
  const url = new URL(ask.path + ask.search, "http://localhost/");
  const at = url.pathname.indexOf("/api/");
  const path = at === -1 ? url.pathname : url.pathname.slice(at);
  const headers: Record<string, string> = {};
  for (const [name, value] of ask.headers) headers[name.toLowerCase()] = value;
  let body: unknown = null;
  if (ask.body && ask.body.byteLength) {
    try {
      body = JSON.parse(new TextDecoder().decode(ask.body));
    } catch {
      body = null;
    }
  }
  return { method: ask.method.toUpperCase(), path, query: url.searchParams, body, headers };
}

/** Post `response` on `port` in the relay's shapes: a Reply, whole; or a
 * `stream: true` head, then for a streamed body its pieces as Laters
 * (`chunk`) and `done` (or the `error` that stopped it). A {cancel: true}
 * from the service worker (the client stopped reading) cancels the stream.
 * The port is closed when everything has been sent. */
export async function sendResponse(response: ApiResponse, port: MessagePort): Promise<void> {
  const post = (m: Reply | Later, transfer: Transferable[] = []) => port.postMessage(m, transfer);
  if ("json" in response) {
    post({ status: response.status, type: "application/json", body: JSON.stringify(response.json) });
    port.close();
    return;
  }
  if ("body" in response) {
    const body = response.body;
    const extra = Object.entries(response.headers ?? {});
    post({ status: response.status, type: response.type, body, ...(extra.length ? { headers: extra } : {}) }, body instanceof ArrayBuffer ? [body] : []);
    port.close();
    return;
  }
  post({ status: response.status, type: response.type, stream: true });
  const reader = response.stream.getReader();
  port.onmessage = (m: MessageEvent<{ cancel?: boolean }>) => {
    if (m.data?.cancel) void reader.cancel("the client stopped reading");
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      // A copy when the chunk is a view on a larger buffer (a TextEncoder's,
      // say), so only the bytes of this chunk cross and nothing else is detached.
      const chunk = value.byteOffset === 0 && value.byteLength === value.buffer.byteLength ? (value.buffer as ArrayBuffer) : value.slice().buffer;
      post({ chunk }, [chunk]);
    }
    post({ done: true });
  } catch (e) {
    post({ error: e instanceof Error ? e.message : String(e) });
  } finally {
    port.close();
  }
}
