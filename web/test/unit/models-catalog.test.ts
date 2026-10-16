// createCatalog's list(), resolve() and refresh(): the Hub listing, its
// day-long cache, the fallbacks when the Hub rate-limits, and the default
// variant of each entry.
import { describe, expect, it } from "vitest";

import { createCatalog, LISTING_KEY } from "../../src/models";
import { LISTING_TTL_MS, MemoryKeyValueStore } from "../../src/models/listing-cache";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { fakeHub, fakeLog, FOLDERS, type HubOptions } from "./models-helpers";

const T0 = Date.UTC(2026, 9, 14, 12, 0, 0);

function catalogWith(options: HubOptions = {}, storage = new MemoryKeyValueStore(), now = () => T0) {
  const hub = fakeHub(options);
  const { log, lines } = fakeLog();
  const catalog = createCatalog({ log, fetch: hub.fetch, storage, bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry(), now });
  return { catalog, hub, lines, storage };
}

const treeCalls = (calls: string[]) => calls.filter((u) => u.includes("/tree/main"));
const manifestCalls = (calls: string[]) => calls.filter((u) => u.endsWith("/manifest.json"));

describe("list()", () => {
  it("lists the eight models from the Hub's tree and manifests, conv3d left out", async () => {
    const { catalog, hub, lines } = catalogWith();
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
    expect(lines).toContain("model catalog: 8 models from Hugging Face");
  });

  it("gives each entry fp32 and the smallest block as its default variant (retuned to the GPU before serving)", async () => {
    const { catalog } = catalogWith();
    const entries = await catalog.list();
    const by = Object.fromEntries(entries.map((e) => [e.folder, e]));
    expect(by.setup_54!.variant).toMatchObject({ variant: "fp32", input_size: [116, 116, 116], output_size: [28, 28, 28] });
    expect(by.setup_53!.variant).toMatchObject({ variant: "fp32", input_size: [124, 124, 124], output_size: [36, 36, 36] });
    expect(by.setup_54!.onnxUrl).toBe("https://huggingface.co/cellmap/onnx-web-models/resolve/main/setup_54/setup_54_mito_affinities_16nm_in116_fp32.onnx");
  });

  it("asks the Hub once per page: a second list() answers from memory", async () => {
    const { catalog, hub } = catalogWith();
    const first = await catalog.list();
    const second = await catalog.list();
    expect(second).toBe(first);
    expect(hub.calls).toHaveLength(1 + FOLDERS.length);
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
    expect(later.lines).toContain("model catalog: 8 models from the browser's cache");

    // A day later the listing is fetched again.
    const nextDay = catalogWith({}, storage, () => T0 + LISTING_TTL_MS);
    await nextDay.catalog.list();
    expect(treeCalls(nextDay.hub.calls)).toHaveLength(1);
  });

  it("uses the built-in folder list when the tree listing is rate-limited", async () => {
    const { catalog, hub, lines } = catalogWith({ treeStatus: 429 });
    const entries = await catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS);
    expect(manifestCalls(hub.calls)).toHaveLength(FOLDERS.length);
    expect(lines.some((l) => l.includes("could not list cellmap/onnx-web-models") && l.includes("HTTP 429") && l.includes("built-in list of 8 folders"))).toBe(true);
  });

  it("serves the stale cache when no manifest can be fetched", async () => {
    const storage = new MemoryKeyValueStore();
    await catalogWith({}, storage).catalog.list();
    const blocked = { treeStatus: 429, manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 429])) };
    const stale = catalogWith(blocked, storage, () => T0 + 3 * LISTING_TTL_MS);
    const entries = await stale.catalog.list();
    expect(entries).toHaveLength(8);
    expect(stale.lines.some((l) => l.includes("no manifest could be fetched") && l.includes("2026-10-14T12:00:00.000Z"))).toBe(true);
    expect(stale.lines).toContain("model catalog: 8 models from the browser's cache (stale)");
  });

  it("fails with the rate-limit message when there is nothing to fall back on", async () => {
    const blocked = { treeStatus: 429, manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 429])) };
    const { catalog, hub } = catalogWith(blocked);
    await expect(catalog.list()).rejects.toThrow(/Hugging Face rate-limited the anonymous requests .*HTTP 429.*retry in a minute/);
    // The failure is not memoised: the next list() asks again.
    const before = hub.calls.length;
    await expect(catalog.list()).rejects.toThrow();
    expect(hub.calls.length).toBeGreaterThan(before);
  });

  it("names the first failure when the manifests fail for another reason", async () => {
    const { catalog } = catalogWith({ manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 500])) });
    await expect(catalog.list()).rejects.toThrow("Could not fetch any model manifest from cellmap/onnx-web-models: setup_53: Could not fetch setup_53/manifest.json: HTTP 500");
  });

  it("lists what it could when one manifest fails, without caching the gap", async () => {
    const storage = new MemoryKeyValueStore();
    const { catalog, lines } = catalogWith({ manifestStatus: { setup_61: 500 } }, storage);
    const entries = await catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS.filter((f) => f !== "setup_61"));
    expect(lines).toContain("model catalog: setup_61: Could not fetch setup_61/manifest.json: HTTP 500");
    expect(storage.getItem(LISTING_KEY)).toBeNull();
  });

  it("fills a failed manifest from the cache when it has one", async () => {
    const storage = new MemoryKeyValueStore();
    await catalogWith({}, storage).catalog.list();
    const gap = catalogWith({ manifestStatus: { setup_61: 500 } }, storage, () => T0 + 2 * LISTING_TTL_MS);
    const entries = await gap.catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS);
    expect(gap.lines).toContain("model catalog: 8 models from Hugging Face");
  });

  it("skips a folder whose manifest is malformed and says so", async () => {
    const hub = fakeHub();
    const broken = async (url: string) => {
      if (url.endsWith("setup_55/manifest.json")) return new Response(JSON.stringify({ model: "setup_55" }), { status: 200 });
      return hub.fetch(url);
    };
    const { log, lines } = fakeLog();
    const catalog = createCatalog({ log, fetch: broken, storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry(), now: () => T0 });
    const entries = await catalog.list();
    expect(entries.map((e) => e.folder)).toEqual(FOLDERS.filter((f) => f !== "setup_55"));
    expect(lines.some((l) => l.startsWith("model catalog: skipping setup_55: setup_55/manifest.json:"))).toBe(true);
  });

  it("tolerates a corrupt cache entry", async () => {
    const storage = new MemoryKeyValueStore();
    storage.setItem(LISTING_KEY, "{not json");
    const { catalog, hub } = catalogWith({}, storage);
    expect(await catalog.list()).toHaveLength(8);
    expect(treeCalls(hub.calls)).toHaveLength(1);
    storage.setItem(LISTING_KEY, JSON.stringify({ v: 7, manifests: {} }));
    const other = catalogWith({}, storage);
    expect(await other.catalog.list()).toHaveLength(8);
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

  it("is null for an unknown reference or an empty one", async () => {
    const { catalog } = catalogWith();
    expect(await catalog.resolve("setup_99")).toBeNull();
    expect(await catalog.resolve("cellmap/mito-aff-unet-setup-16")).toBeNull();
    expect(await catalog.resolve("")).toBeNull();
    expect(await catalog.resolve("   ")).toBeNull();
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

  it("answers the cached list when the Hub will not answer a refresh", async () => {
    const storage = new MemoryKeyValueStore();
    await catalogWith({}, storage).catalog.list();
    const blocked = { treeStatus: 429, manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 500])) };
    const { catalog, lines } = catalogWith(blocked, storage);
    const before = await catalog.list();
    const after = await catalog.refresh();
    expect(after.map((e) => e.folder)).toEqual(before.map((e) => e.folder));
    expect(lines).toContain("model catalog: 8 models from the browser's cache (stale)");
  });

  it("keeps the list in memory when a refresh fails and nothing could be cached", async () => {
    // Storage that refuses writes (private mode, quota): the first listing
    // lives in memory only, so a failed refresh has no cache to fall back on.
    const refusing = new MemoryKeyValueStore();
    refusing.setItem = () => {
      throw new Error("QuotaExceededError");
    };
    const hub = fakeHub();
    let blocked = false;
    const flaky = async (url: string) => (blocked ? new Response("", { status: 429 }) : hub.fetch(url));
    const { log, lines } = fakeLog();
    const catalog = createCatalog({ log, fetch: flaky, storage: refusing, bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry(), now: () => T0 });
    const before = await catalog.list();
    expect(lines).toContain("model catalog: could not cache the model list (storage blocked or full)");
    blocked = true;
    await expect(catalog.refresh()).rejects.toThrow(/rate-limited/);
    expect(await catalog.list()).toBe(before);
  });

  it("surfaces a refresh failure when there is no cache at all, and list() still works afterwards", async () => {
    const blocked = { treeStatus: 429, manifestStatus: Object.fromEntries(FOLDERS.map((f) => [f, 500])) };
    const { catalog } = catalogWith(blocked);
    await expect(catalog.refresh()).rejects.toThrow(/Could not fetch any model manifest/);
    await expect(catalog.list()).rejects.toThrow(/Could not fetch any model manifest/);
  });
});

describe("chooseVariant() and metadataFor() on the catalog", () => {
  it("are the module functions", async () => {
    const { catalog } = catalogWith();
    const [entry] = await catalog.list();
    const v = catalog.chooseVariant(entry!.manifest, { maxBufferSize: 4 * 1024 ** 3, maxStorageBufferBindingSize: null, shaderF16: true });
    expect(v.variant).toBe("fp16");
    expect(catalog.metadataFor(entry!)).toMatchObject({ framework: "onnx", model_name: entry!.name });
  });
});
