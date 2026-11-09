// The visit count (src/count.ts): on the public site it sends the app's path
// and nothing of the address after it; anywhere else it sends nothing.
import { afterEach, expect, it, vi } from "vitest";

import { countVisit } from "../../src/count";

afterEach(() => vi.unstubAllGlobals());

function visit(href: string, referrer = "") {
  const sent: string[] = [];
  const url = new URL(href);
  vi.stubGlobal("location", { hostname: url.hostname, href });
  vi.stubGlobal("navigator", { webdriver: false, sendBeacon: (u: string) => (sent.push(u), true) });
  vi.stubGlobal("document", { referrer });
  vi.stubGlobal("screen", { width: 1440 });
  countVisit();
  return sent;
}

it("counts a visit to the public app by its path alone, whatever the link holds", () => {
  const [url, ...more] = visit(
    "https://janelia-cellmap.github.io/cellmap-flow/app/?dataset=https://fileglancer.example/private.zarr&model=setup_54#!%7B%22layers%22%7D",
    "https://huggingface.co/some/private-repo",
  );
  expect(more).toEqual([]);
  const sent = new URL(url);
  expect(sent.origin).toBe("https://davidackerman.goatcounter.com");
  expect(sent.searchParams.get("p")).toBe("/cellmap-flow/app/");
  expect(sent.searchParams.get("r")).toBe("https://huggingface.co");
  expect(url).not.toContain("fileglancer");
  expect(url).not.toContain("private-repo");
});

it("counts nothing from another host", () => {
  expect(visit("http://localhost:4173/?dataset=x")).toEqual([]);
});
