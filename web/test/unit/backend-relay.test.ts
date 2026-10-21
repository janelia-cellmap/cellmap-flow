// The page side of the relay, its pure parts: a relayed Ask to an
// ApiRequest, and an ApiResponse to the Reply and Later messages chunkmirage's
// relay (web/chunkmirage/web/src/relay.ts) turns into a Response.
import { describe, expect, it } from "vitest";

import type { Later, Reply } from "../../chunkmirage/web/src/types";
import { sendResponse, toApiRequest } from "../../src/backend/relay";

const encode = (s: string) => new TextEncoder().encode(s).buffer as ArrayBuffer;

/** A MessagePort stand-in that records what is posted on it. */
function fakePort() {
  const posted: (Reply | Later)[] = [];
  let closed = false;
  const port = {
    postMessage: (m: Reply | Later) => void posted.push(m),
    close: () => void (closed = true),
    onmessage: null as ((m: MessageEvent) => void) | null,
  };
  return { port: port as unknown as MessagePort, posted, isClosed: () => closed, raw: port };
}

describe("toApiRequest", () => {
  it("takes the path from /api/ on, whatever base the page has", () => {
    const req = toApiRequest({ method: "get", path: "/cellmap-flow/app/api/finetune/job/7/logs/stream", search: "", headers: [], body: null });
    expect(req.method).toBe("GET");
    expect(req.path).toBe("/api/finetune/job/7/logs/stream");
    expect([...req.query.keys()]).toEqual([]);
    expect(toApiRequest({ method: "GET", path: "/api/models", search: "", headers: [], body: null }).path).toBe("/api/models");
  });

  it("reads the query the relay passes beside the path, and the headers by their lower-cased names", () => {
    const req = toApiRequest({ method: "GET", path: "/api/finetune/job/7/logs/stream", search: "?offset=12", headers: [["Last-Event-ID", "40"]], body: null });
    expect(req.path).toBe("/api/finetune/job/7/logs/stream");
    expect(req.query.get("offset")).toBe("12");
    expect(req.headers).toEqual({ "last-event-id": "40" });
  });

  it("parses a JSON body, and gives null for none or for one that is not JSON", () => {
    const base = { method: "POST", path: "/api/x", search: "", headers: [] as [string, string][] };
    expect(toApiRequest({ ...base, body: encode('{"dataset_path": "s3://b/k"}') }).body).toEqual({ dataset_path: "s3://b/k" });
    expect(toApiRequest({ ...base, body: encode("not json") }).body).toBeNull();
    expect(toApiRequest({ ...base, body: new ArrayBuffer(0) }).body).toBeNull();
    expect(toApiRequest({ ...base, body: null }).body).toBeNull();
  });
});

describe("sendResponse", () => {
  it("posts a JSON answer as one Reply with the JSON body, and closes the port", async () => {
    const { port, posted, isClosed } = fakePort();
    await sendResponse({ status: 400, json: { success: false, error: "no" } }, port);
    expect(posted).toEqual([{ status: 400, type: "application/json", body: '{"success":false,"error":"no"}' }]);
    expect(isClosed()).toBe(true);
  });

  it("posts a text answer with its type, and its headers beyond it", async () => {
    const { port, posted } = fakePort();
    await sendResponse({ status: 200, body: "a: 1\n", type: "text/yaml", headers: { "content-disposition": "attachment; filename=x.yaml" } }, port);
    expect(posted).toEqual([{ status: 200, type: "text/yaml", body: "a: 1\n", headers: [["content-disposition", "attachment; filename=x.yaml"]] }]);
  });

  it("posts a streamed answer as a head, its chunks and done", async () => {
    const { port, posted, isClosed } = fakePort();
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode("data: a\n\n"));
        c.enqueue(new TextEncoder().encode("data: b\n\n"));
        c.close();
      },
    });
    await sendResponse({ status: 200, stream, type: "text/event-stream" }, port);
    expect(posted[0]).toEqual({ status: 200, type: "text/event-stream", stream: true });
    const chunks = posted.slice(1, -1).map((m) => new TextDecoder().decode((m as { chunk: ArrayBuffer }).chunk));
    expect(chunks).toEqual(["data: a\n\n", "data: b\n\n"]);
    expect(posted[posted.length - 1]).toEqual({ done: true });
    expect(isClosed()).toBe(true);
  });

  it("cancels the stream when the service worker says the client stopped reading", async () => {
    const { port, posted, raw } = fakePort();
    let cancelled = false;
    let pulls = 0;
    const stream = new ReadableStream<Uint8Array>({
      async pull(c) {
        pulls++;
        if (pulls === 2) raw.onmessage!({ data: { cancel: true } } as MessageEvent);
        await new Promise((r) => setTimeout(r, 1));
        c.enqueue(new Uint8Array([pulls]));
      },
      cancel() {
        cancelled = true;
      },
    });
    await sendResponse({ status: 200, stream, type: "text/event-stream" }, port);
    expect(cancelled).toBe(true);
    expect(posted[posted.length - 1]).toEqual({ done: true });
  });
});
