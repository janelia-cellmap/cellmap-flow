// The dashboard's log panel (cellmap_flow/dashboard/static/js/shared/log-panel.js),
// on the Flask dashboard's pages: after an error answer, which the browser
// does not reconnect after, it opens one new stream, from the last record it
// shows, so the server sends only what it has not (routes/logging_routes.py);
// and it keeps the last 1,000 records.
import { fileURLToPath } from "node:url";
import { afterAll, expect, it, vi } from "vitest";

import { FakeElement } from "./dashboard-dom";

class FakeEventSource {
  static CLOSED = 2;
  static opened: FakeEventSource[] = [];
  readyState = 1;
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: string; lastEventId: string }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;

  constructor(public url: string) {
    FakeEventSource.opened.push(this);
  }
  addEventListener(): void {}
  close(): void {
    this.readyState = FakeEventSource.CLOSED;
  }
  // The server's next record, numbered `id`.
  send(id: number, data: string): void {
    this.onmessage!({ data, lastEventId: String(id) });
  }
}

const LOG_PANEL = fileURLToPath(new URL("../../../cellmap_flow/dashboard/static/js/shared/log-panel.js", import.meta.url));

afterAll(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("reopens its one stream from the last record shown, and keeps the last 1,000", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("document", { getElementById: () => null, createElement: (tag: string) => new FakeElement(tag) });
  vi.stubGlobal("location", { pathname: "/" });
  vi.stubGlobal("EventSource", FakeEventSource);
  vi.stubGlobal("requestAnimationFrame", () => 0);
  const { mountLogPanel } = (await import(/* @vite-ignore */ LOG_PANEL)) as { mountLogPanel: (panel: FakeElement) => void };

  // templates/_log_panel.html's parts.
  const panel = new FakeElement("section");
  for (const part of ["log-header", "log-latest", "log-status", "log-content", "log-output"]) {
    const el = new FakeElement("div");
    el.className = part;
    panel.append(el);
  }
  const output = panel.querySelector(".log-output")!;
  const shown = () => output.children.map((line) => line.textContent);
  mountLogPanel(panel);
  expect(FakeEventSource.opened.map((s) => s.url)).toEqual(["/api/logs/stream"]);

  const first = FakeEventSource.opened[0];
  first.send(101, "2026-10-27 21:03:05 INFO cellmap_flow.x: one");
  first.send(102, "2026-10-27 21:03:06 ERROR cellmap_flow.x: two");
  expect(panel.querySelector(".log-latest")!.textContent).toBe("2026-10-27 21:03:06 ERROR cellmap_flow.x: two");

  // An error answer: the browser closes the stream for good.
  first.readyState = FakeEventSource.CLOSED;
  first.onerror!({});
  expect(FakeEventSource.opened).toHaveLength(1);
  vi.advanceTimersByTime(5000);
  expect(FakeEventSource.opened.map((s) => s.url)).toEqual(["/api/logs/stream", "/api/logs/stream?offset=102"]);
  const second = FakeEventSource.opened[1];
  second.send(103, "2026-10-27 21:03:09 INFO cellmap_flow.x: three");
  expect(shown()).toEqual([
    "2026-10-27 21:03:05 INFO cellmap_flow.x: one",
    "2026-10-27 21:03:06 ERROR cellmap_flow.x: two",
    "2026-10-27 21:03:09 INFO cellmap_flow.x: three",
  ]);

  for (let id = 104; id < 1104; id += 1) second.send(id, `line ${id}`);
  expect(shown()).toHaveLength(1000);
  expect(shown()[0]).toBe("line 104");
});
