// createCatalog's list(), resolve() and refresh(): the Hub listing, its
// day-long cache, the fallback when the Hub rate-limits the tree listing and
// the failure when it rate-limits everything, and the references an entry is
// found by.
import { describe, expect, it } from "vitest";

import { createCatalog, LISTING_KEY } from "../../src/models";
import { LISTING_TTL_MS, MemoryKeyValueStore } from "../../src/models/listing-cache";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { fakeHub, fakeLog, FOLDERS, type HubOptions } from "./models-helpers";

const T0 = Date.UTC(2026, 9, 14, 12, 0, 0);

function catalogWith(options: HubOptions = {}, storage = new MemoryKeyValueStore(), now = () => T0) {
  const hub = fakeHub(options);
  const { log } = fakeLog();
  const catalog = createCatalog({ log, fetch: hub.fetch, storage, bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry(), now });
  return { catalog, hub, storage };
}

const treeCalls = (calls: string[]) => calls.filter((u) => u.includes("/tree/main"));
const manifestCalls = (calls: string[]) => calls.filter((u) => u.endsWith("/manifest.json"));

describe("list()", () => {
  it("lists the eight models from the Hub's tree and manifests, conv3d left out", async () => {
    const { catalog, hub } = catalogWith();
    const entries = await catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS);
    expect(entries.map((e) => e.name)).toEqual([
      "mito_aff_setup53_16nm",
      "mito_aff_setup53-2d_16nm",
      "mito_aff_setup54_16nm",
      "mito_aff_setup54-2d_16nm",
      "nuc_setup55_128nm",
      "nuc_setup56_128nm",
      "nuc_setup61_256nm",
      "er_setup67_8nm",
    ]);
    expect(entries.every((e) => e.id === `cellmap/onnx-web-models/${e.folder}`)).toBe(true);
    expect(treeCalls(hub.calls)).toEqual(["https://huggingface.co/api/models/cellmap/onnx-web-models/tree/main"]);
    expect(manifestCalls(hub.calls)).toEqual(FOLDERS.map((f) => `https://huggingface.co/cellmap/onnx-web-models/raw/main/${f}/manifest.json`));
    expect(hub.calls.some((u) => u.includes("conv3d"))).toBe(false);
  });

  it("caches the listing in storage and answers from it for a day", async () => {
    const storage = new MemoryKeyValueStore();
    const first = catalogWith({}, storage);
    await first.catalog.list();
    expect(storage.getItem(LISTING_KEY)).not.toBeNull();

    // A new page 23 hours later: nothing fetched.
    const later = catalogWith({ treeStatus: 500, manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 500])) }, storage, () => T0 + LISTING_TTL_MS - 3600_000);
    const entries = await later.catalog.list();
    expect(entries).toHaveLength(8);
    expect(later.hub.calls).toHaveLength(0);

    // A day later the listing is fetched again.
    const nextDay = catalogWith({}, storage, () => T0 + LISTING_TTL_MS);
    await nextDay.catalog.list();
    expect(treeCalls(nextDay.hub.calls)).toHaveLength(1);
  });

  it("uses the built-in folder list when the tree listing is rate-limited", async () => {
    const { catalog, hub } = catalogWith({ treeStatus: 429 });
    const entries = await catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS);
    expect(manifestCalls(hub.calls)).toHaveLength(FOLDERS.length);
  });

  it("fails with the rate-limit message when there is nothing to fall back on", async () => {
    const blocked = { treeStatus: 429, manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 429])) };
    const { catalog, hub } = catalogWith(blocked);
    await expect(catalog.list()).rejects.toThrow(/rate-limited/);
    // The failure is not memoised: the next list() asks again.
    const before = hub.calls.length;
    await expect(catalog.list()).rejects.toThrow();
    expect(hub.calls.length).toBeGreaterThan(before);
  });
});

describe("resolve()", () => {
  it("finds an entry by id, repo/folder, folder or name", async () => {
    const { catalog } = catalogWith();
    const byId = await catalog.resolve("cellmap/onnx-web-models/setup_54");
    expect(byId?.folder).toBe("setup_54");
    expect(await catalog.resolve("setup_54")).toBe(byId);
    expect(await catalog.resolve("mito_aff_setup54_16nm")).toBe(byId);
    expect(await catalog.resolve(" cellmap/onnx-web-models/setup_54/ ")).toBe(byId);
    expect((await catalog.resolve("er_setup67_8nm"))?.folder).toBe("setup_67");
  });
});

describe("refresh()", () => {
  it("fetches the listing again although the cache is fresh, and list() then answers the new one", async () => {
    const storage = new MemoryKeyValueStore();
    await catalogWith({}, storage).catalog.list();
    const { catalog, hub } = catalogWith({}, storage);
    const cached = await catalog.list();
    expect(hub.calls).toHaveLength(0);
    const refreshed = await catalog.refresh();
    expect(treeCalls(hub.calls)).toHaveLength(1);
    expect(refreshed).toHaveLength(8);
    expect(refreshed).not.toBe(cached);
    expect(await catalog.list()).toBe(refreshed);
  });
});
