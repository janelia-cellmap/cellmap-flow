// The visit count (src/count.ts): on the public site it sends the app's path
// and nothing of the address after it; anywhere else, or from a browser
// that turned it off with #toggle-goatcounter, it sends nothing.
import { afterEach, expect, it, vi } from "vitest";

import { countVisit } from "../../src/count";

afterEach(() => vi.unstubAllGlobals());

const stored = new Map<string, string>();
let address = "";

function visit(href: string, referrer = "") {
  const sent: string[] = [];
  const url = new URL(href);
  address = href;
  vi.stubGlobal("location", { hostname: url.hostname, href, hash: url.hash });
  vi.stubGlobal("navigator", { webdriver: false, sendBeacon: (u: string) => (sent.push(u), true) });
  vi.stubGlobal("document", { referrer });
  vi.stubGlobal("screen", { width: 1440 });
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => stored.get(k) ?? null,
    setItem: (k: string, v: string) => void stored.set(k, v),
    removeItem: (k: string) => void stored.delete(k),
  });
  vi.stubGlobal("history", { replaceState: (_s: unknown, _t: string, u: URL) => void (address = String(u)) });
  vi.stubGlobal("alert", () => {});
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

it("stops counting this browser at #toggle-goatcounter, and starts again at the next", () => {
  const app = "https://janelia-cellmap.github.io/cellmap-flow/app/";
  expect(visit(`${app}?dataset=x#toggle-goatcounter`)).toEqual([]);
  expect(address).toBe(`${app}?dataset=x`);
  expect(visit(app)).toEqual([]);
  expect(visit(`${app}#toggle-goatcounter`)).toHaveLength(1);
  expect(visit(app)).toHaveLength(1);
});
