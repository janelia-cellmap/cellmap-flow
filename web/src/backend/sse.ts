// Server-sent events for the routes that stream (the finetune log streams,
// in PR 3): the text/event-stream wire format, as a ReadableStream the relay
// sends chunk by chunk, following the dashboard's log protocol (lib/sse.js,
// routes/finetune/training.py):
//
// - each event's `id` is the byte offset in the log just after its lines,
//   and the browser sends the last one back as Last-Event-ID when it
//   reconnects by itself, so nothing is replayed;
// - `?offset=N` starts a new stream part-way through;
// - a finished log ends with an `event: done` whose data is the final
//   status, which closes the stream on the page.
//
// Nothing in PR 2 streams; this is the ready, tested helper for PR 3.
import type { ApiRequest, ApiResponse } from "../contracts";

export interface SseEvent {
  /** The event's data; several lines become several `data:` lines. */
  data?: string;
  /** A named event ("done"), for addEventListener on the page; unnamed is "message". */
  event?: string;
  /** The event id (a byte offset, in the log protocol). */
  id?: string | number;
  /** A comment line (": ping"), which keeps a connection alive and says nothing. */
  comment?: string;
}

/** One event as text/event-stream writes it, ending in the blank line. */
export function formatEvent(e: SseEvent): string {
  let out = "";
  if (e.comment !== undefined) out += `: ${e.comment}\n`;
  if (e.id !== undefined) out += `id: ${e.id}\n`;
  if (e.event !== undefined) out += `event: ${e.event}\n`;
  if (e.data !== undefined) for (const line of e.data.split("\n")) out += `data: ${line}\n`;
  return out + "\n";
}

/** Where the client wants a log from: its Last-Event-ID (a reconnect), else
 * `?offset=`, else the start. */
export function requestedOffset(req: ApiRequest): number {
  for (const raw of [req.headers["last-event-id"], req.query.get("offset")]) {
    if (raw === undefined || raw === null || raw === "") continue;
    const n = Number(raw);
    if (Number.isInteger(n) && n >= 0) return n;
  }
  return 0;
}

/** The `done` event that ends a log stream with the job's final status. */
export function done(status: string): SseEvent {
  return { event: "done", data: status };
}

/** `lines` of a log as events with byte-offset ids: each event's id is the
 * UTF-8 byte count of the log through that line (and its newline), and the
 * lines the client already has (those ending at or before `offset`) are
 * skipped. Empty lines count toward the offset but are not sent. */
export async function* logEvents(lines: AsyncIterable<string> | Iterable<string>, offset = 0): AsyncGenerator<SseEvent> {
  const encoder = new TextEncoder();
  let position = 0;
  for await (const line of lines) {
    position += encoder.encode(line).byteLength + 1;
    if (position <= offset || line === "") continue;
    yield { id: position, data: line };
  }
}

/** A text/event-stream response of `events`, written as they come. The
 * stream is cancelled (and `events` returned from, when it is a generator)
 * if the client stops reading. */
export function eventStream(events: AsyncIterable<SseEvent> | Iterable<SseEvent>, status = 200): ApiResponse {
  const encoder = new TextEncoder();
  const iterator = Symbol.asyncIterator in events ? events[Symbol.asyncIterator]() : (events as Iterable<SseEvent>)[Symbol.iterator]();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) return controller.close();
      controller.enqueue(encoder.encode(formatEvent(value)));
    },
    async cancel() {
      await iterator.return?.();
    },
  });
  return { status, stream, type: "text/event-stream" };
}
