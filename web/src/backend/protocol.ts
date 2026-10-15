// What the service worker and the page say to each other about a relayed
// /api/* request. The service worker (src/sw.ts) sees the dashboard's
// fetch("/api/..."), its EventSource streams and the Export Config download
// as HTTP requests; it cannot answer them itself (the backend, the engine and
// the session live in the page), so it posts each one to the page over a
// MessageChannel and builds the Response from what comes back
// (src/backend/relay.ts answers, through the Backend).
//
// Both sides import these types: sw.ts is compiled with the WebWorker lib and
// relay.ts with the DOM lib, so nothing here may name a type that only one of
// them has.

/** The service worker's message to the page: one /api/ request, with a
 * MessagePort for the answer. `api: true` tells it apart from chunkmirage's
 * /virtual/ asks, which arrive on the same `navigator.serviceWorker`
 * "message" event; `path` is there for chunkmirage's own listener, which
 * reads `e.data.path` of every message and answers null (not its share of
 * /virtual/) rather than throwing on a message without one. */
export interface ApiAsk {
  api: true;
  path: string;
  method: string;
  url: string;
  headers: [string, string][];
  /** The request body, or null for a GET or an empty body. */
  body: ArrayBuffer | null;
}

/** The page's first message on the port: the whole response, or the head of
 * one whose body follows as ApiChunk messages (a server-sent event stream).
 * null means "not mine", which is what chunkmirage's listener answers; the
 * service worker waits for a non-null head. */
export type ApiHead =
  | { status: number; headers: [string, string][]; body: ArrayBuffer | string | null }
  | { status: number; headers: [string, string][]; stream: true };

/** A streamed body's messages, after a `stream: true` head. */
export type ApiChunk = { chunk: ArrayBuffer } | { end: true } | { error: string };

/** The service worker's one message back to the page: the client stopped
 * reading a streamed response (an EventSource closed, a tab went away). */
export type ApiCancel = { cancel: true };
