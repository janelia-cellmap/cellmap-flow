// A record of every AI annotation: what was sent where, and what became of
// it. The port of cellmap_flow/ai_annotate/audit.py: each event is one JSON
// line appended to `ai_annotate_log.jsonl` at the volume's root in the paint
// store, next to the labels it concerns, so whoever reviews a volume's
// annotations can see which labels a hosted model drew, with which prompt,
// and that data left the browser for it. The file travels with the volume's
// zip download (finetune/zip.ts takes every file of the volume).
//
// A line holds the time, the user, the event and the fields the caller
// gives, nothing else: the routes pass ids, the provider and model, the
// plane and counts, never a credential. The server's second guard is a
// redaction pass over the line; here the guard is on the field names: a
// field named like a credential is refused outright, since nothing of the
// kind belongs in the log.
//
// The store has no append: the old bytes are read, the line added and the
// file put back whole. Writes are serialised through one promise chain, as
// the Python's lock serialises them, so two requests finishing together do
// not lose each other's lines.
import type { PaintStore } from "../contracts";

export const LOG_NAME = "ai_annotate_log.jsonl";

export const EVENTS = ["requested", "staged", "failed", "resent", "accepted", "rejected"] as const;
export type AuditEvent = (typeof EVENTS)[number];

/** The `user` of every line: the server writes the login name; the browser has none. */
export const USER = "browser";

/** Field names a line may never carry: the three the entry itself sets, and anything that could hold a secret. */
const FORBIDDEN_FIELDS: ReadonlySet<string> = new Set(["time", "user", "event", "api_key", "access_token", "authorization", "credentials", "key", "token"]);

let chain: Promise<void> = Promise.resolve();

/** Append `{time, user, event, ...fields}` to the audit log of `volume`.
 *
 * Rejects for an unknown event or a forbidden field name (a bug in the
 * caller), and with the store's error when the log cannot be written: the
 * caller decides whether that stops the action (the routes record
 * "requested" and "resent" before a call is made, so a call is not made
 * unless it is recorded; the events after the fact are only logged). */
export function record(store: PaintStore, volume: string, event: AuditEvent, fields: Record<string, unknown>, now: () => Date = () => new Date()): Promise<void> {
  if (!EVENTS.includes(event)) return Promise.reject(new Error(`Unknown audit event ${JSON.stringify(event)}; expected one of ${EVENTS.join(", ")}`));
  const clash = Object.keys(fields).filter((name) => FORBIDDEN_FIELDS.has(name));
  if (clash.length) return Promise.reject(new Error(`Audit fields may not be named ${clash.sort().join(", ")}`));
  const entry = { time: localIso(now()), user: USER, event, ...fields };
  const line = `${JSON.stringify(entry, jsonSafe)}\n`;
  const write = chain.then(async () => {
    const old = await store.get(volume, LOG_NAME);
    const added = new TextEncoder().encode(line);
    const bytes = new Uint8Array((old?.byteLength ?? 0) + added.byteLength);
    if (old) bytes.set(old, 0);
    bytes.set(added, old?.byteLength ?? 0);
    await store.put(volume, LOG_NAME, bytes);
  });
  // The chain goes on whether or not this write failed; the caller sees the failure.
  chain = write.catch(() => undefined);
  return write;
}

/** The log's lines, parsed, oldest first; empty without a log. For tests and whoever reviews a volume. */
export async function readLog(store: PaintStore, volume: string): Promise<Record<string, unknown>[]> {
  const bytes = await store.get(volume, LOG_NAME);
  if (!bytes) return [];
  return new TextDecoder()
    .decode(bytes)
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

/** JSON.stringify's replacer for values that are not JSON (the Python writes
 * their str): typed arrays as plain arrays, a bigint as a number, anything
 * undefined left out as JSON does. */
function jsonSafe(_key: string, value: unknown): unknown {
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) return Array.from(value as unknown as ArrayLike<number>);
  if (typeof value === "bigint") return Number(value);
  return value;
}

/** `date` as the Python's `datetime.now().astimezone().isoformat(timespec="seconds")`:
 * local time to the second with the zone's offset, "2026-10-15T12:00:00-04:00". */
export function localIso(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const zone = `${sign}${pad(Math.floor(Math.abs(offset) / 60))}:${pad(Math.abs(offset) % 60)}`;
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}${zone}`;
}
