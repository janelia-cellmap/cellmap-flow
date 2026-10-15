// The server-sent events helper: the wire format, the byte-offset ids of the
// log protocol, and the stream a route answers with.
import { describe, expect, it } from "vitest";

import { done, eventStream, formatEvent, logEvents, requestedOffset } from "../../src/backend/sse";
import { request } from "./backend-fakes";

async function readAll(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  let out = "";
  const decoder = new TextDecoder();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return out;
    out += decoder.decode(value, { stream: true });
  }
}

describe("formatEvent", () => {
  it("writes id, event and data lines and ends with a blank line", () => {
    expect(formatEvent({ data: "hello" })).toBe("data: hello\n\n");
    expect(formatEvent({ id: 42, data: "a\nb" })).toBe("id: 42\ndata: a\ndata: b\n\n");
    expect(formatEvent({ event: "done", data: "COMPLETED" })).toBe("event: done\ndata: COMPLETED\n\n");
    expect(formatEvent({ comment: "ping" })).toBe(": ping\n\n");
    expect(formatEvent({ id: 10 })).toBe("id: 10\n\n"); // how far the log was read, with nothing new to show
  });
});

describe("requestedOffset", () => {
  it("takes Last-Event-ID first, then ?offset=, else 0, ignoring what is not a byte count", () => {
    expect(requestedOffset(request("GET", "/api/x/logs/stream", null, { "last-event-id": "120" }))).toBe(120);
    expect(requestedOffset(request("GET", "/api/x/logs/stream?offset=33", null, { "last-event-id": "120" }))).toBe(120);
    expect(requestedOffset(request("GET", "/api/x/logs/stream?offset=33"))).toBe(33);
    expect(requestedOffset(request("GET", "/api/x/logs/stream?offset=abc"))).toBe(0);
    expect(requestedOffset(request("GET", "/api/x/logs/stream", null, { "last-event-id": "-4" }))).toBe(0);
    expect(requestedOffset(request("GET", "/api/x/logs/stream"))).toBe(0);
  });
});

describe("logEvents", () => {
  const LINES = ["EPOCH_LOSS 1 0.5", "", "EPOCH_LOSS 2 0.4", "héllo"];

  it("gives each line an id that is the byte offset after it, skipping empty lines", async () => {
    const events = [];
    for await (const e of logEvents(LINES)) events.push(e);
    // "EPOCH_LOSS 1 0.5\n" is 17 bytes, the empty line 1, the next 17, "héllo\n" 7
    expect(events).toEqual([
      { id: 17, data: "EPOCH_LOSS 1 0.5" },
      { id: 35, data: "EPOCH_LOSS 2 0.4" },
      { id: 42, data: "héllo" },
    ]);
  });

  it("skips the lines a client already has when it reconnects with an offset", async () => {
    const events = [];
    for await (const e of logEvents(LINES, 35)) events.push(e);
    expect(events).toEqual([{ id: 42, data: "héllo" }]);
  });
});

describe("eventStream", () => {
  it("is a text/event-stream response whose bytes are the events in order", async () => {
    async function* source() {
      yield* logEvents(["a", "b"]);
      yield done("COMPLETED");
    }
    const res = eventStream(source());
    expect(res.status).toBe(200);
    if (!("stream" in res)) throw new Error("not a stream");
    expect(res.type).toBe("text/event-stream");
    expect(await readAll(res.stream)).toBe("id: 2\ndata: a\n\nid: 4\ndata: b\n\nevent: done\ndata: COMPLETED\n\n");
  });

  it("returns from the generator when the client stops reading", async () => {
    let finished = false;
    async function* source() {
      try {
        for (let i = 0; ; i++) yield { data: String(i) };
      } finally {
        finished = true;
      }
    }
    const res = eventStream(source());
    if (!("stream" in res)) throw new Error("not a stream");
    const reader = res.stream.getReader();
    await reader.read();
    await reader.cancel();
    expect(finished).toBe(true);
  });
});
