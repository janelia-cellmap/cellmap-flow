// modelBytes(): the streamed download with progress, the IndexedDB cache
// (hit and miss, with the tiny in-memory IndexedDB of models-helpers.ts), the
// shared in-flight download, and the rate-limit error.
import { describe, expect, it } from "vitest";

import { createCatalog } from "../../src/models";
import { type ByteCache, IndexedDbByteCache, MemoryByteCache, MODEL_DB_NAME, openByteCache } from "../../src/models/byte-cache";
import { MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { fetchBytes, fileNameOf, RateLimitError } from "../../src/models/hf";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFixture, fakeHub, fakeIndexedDb, fakeLog, type HubOptions, streamedResponse } from "./models-helpers";

const FILE = "setup_54_mito_affinities_16nm_in132_fp32.onnx";
const MODEL_BYTES = Uint8Array.from({ length: 100 }, (_, i) => i);

describe("IndexedDbByteCache", () => {
  it("creates the database and store on first open and round-trips bytes", async () => {
    const idb = fakeIndexedDb();
    const cache = new IndexedDbByteCache(idb.factory);
    const key = "https://huggingface.co/x/y/resolve/main/a.onnx";
    expect(await cache.get(key)).toBeUndefined();
    await cache.put(key, MODEL_BYTES.buffer);
    const stores = idb.databases.get(MODEL_DB_NAME)!;
    expect(stores).toBeDefined();
    expect(stores.get("bytes")!.get(key)).toBe(MODEL_BYTES.buffer);
    const back = await cache.get(key);
    expect(back).toBe(MODEL_BYTES.buffer);
    await cache.delete(key);
    expect(await cache.get(key)).toBeUndefined();
    // One connection for all of it.
    expect(idb.opens()).toBe(1);
  });

  it("reopens after another tab's version change closed the connection", async () => {
    const idb = fakeIndexedDb();
    const cache = new IndexedDbByteCache(idb.factory);
    await cache.put("k", MODEL_BYTES.buffer);
    idb.connections[0]!.onversionchange?.({});
    expect(idb.connections[0]!.closed).toBe(true);
    expect(await cache.get("k")).toBe(MODEL_BYTES.buffer);
    expect(idb.opens()).toBe(2);
  });

  it("reports a refused open and tries again next time", async () => {
    const idb = fakeIndexedDb({ failOpen: true });
    const cache = new IndexedDbByteCache(idb.factory);
    await expect(cache.get("k")).rejects.toThrow("open refused");
    await expect(cache.get("k")).rejects.toThrow("open refused");
    expect(idb.opens()).toBe(2);
  });

  it("openByteCache falls back to memory without IndexedDB and says so", async () => {
    const { log, lines } = fakeLog();
    const cache = openByteCache(undefined, log);
    expect(cache).toBeInstanceOf(MemoryByteCache);
    expect(lines[0]).toContain("no IndexedDB");
    expect(openByteCache(fakeIndexedDb().factory, log)).toBeInstanceOf(IndexedDbByteCache);
  });
});

describe("fetchBytes", () => {
  it("streams the body, reporting progress against Content-Length", async () => {
    const progress: [number, number | null][] = [];
    const bytes = await fetchBytes(async () => streamedResponse(MODEL_BYTES, 30, true), "https://x/a.onnx", (l, t) => progress.push([l, t]));
    expect(new Uint8Array(bytes)).toEqual(MODEL_BYTES);
    expect(progress).toEqual([
      [30, 100],
      [60, 100],
      [90, 100],
      [100, 100],
    ]);
  });

  it("reports a null total without Content-Length", async () => {
    const progress: [number, number | null][] = [];
    await fetchBytes(async () => streamedResponse(MODEL_BYTES, 50, false), "https://x/a.onnx", (l, t) => progress.push([l, t]));
    expect(progress).toEqual([
      [50, null],
      [100, null],
    ]);
  });

  it("reads a body that does not stream in one go", async () => {
    const progress: [number, number | null][] = [];
    const response = new Response(MODEL_BYTES, { status: 200 });
    Object.defineProperty(response, "body", { value: null });
    const bytes = await fetchBytes(async () => response, "https://x/a.onnx", (l, t) => progress.push([l, t]));
    expect(new Uint8Array(bytes)).toEqual(MODEL_BYTES);
    expect(progress).toEqual([[100, 100]]);
  });

  it("names the file in the rate-limit error and other failures", async () => {
    const url = "https://huggingface.co/cellmap/onnx-web-models/resolve/main/setup_54/" + FILE;
    const limited = fetchBytes(async () => new Response("", { status: 429 }), url);
    await expect(limited).rejects.toBeInstanceOf(RateLimitError);
    await expect(limited).rejects.toThrow(`Hugging Face rate-limited the anonymous download of ${FILE} (HTTP 429). Please retry in a minute.`);
    await expect(fetchBytes(async () => new Response("", { status: 404, statusText: "Not Found" }), url)).rejects.toThrow(`Could not fetch ${FILE}: HTTP 404 Not Found`);
    expect(fileNameOf("https://x/a/b%20c.onnx?download=true")).toBe("b c.onnx");
  });
});

describe("Catalog.modelBytes", () => {
  function catalogWith(options: HubOptions, bytes: ByteCache = new MemoryByteCache()) {
    const hub = fakeHub({ onnx: { [FILE]: MODEL_BYTES }, chunkSize: 40, ...options });
    const { log, lines } = fakeLog();
    const catalog = createCatalog({ log, fetch: hub.fetch, storage: new MemoryKeyValueStore(), bytes, finetuned: new MemoryFinetunedRegistry() });
    return { catalog, hub, lines, bytes };
  }
  const entry = () => entryFixture("setup_54");

  it("downloads on a miss, with progress, and caches under the model's URL", async () => {
    const { catalog, hub, lines, bytes } = catalogWith({});
    const progress: [number, number | null][] = [];
    const result = await catalog.modelBytes(entry(), (l, t) => progress.push([l, t]));
    expect(new Uint8Array(result)).toEqual(MODEL_BYTES);
    expect(progress).toEqual([
      [40, 100],
      [80, 100],
      [100, 100],
    ]);
    expect(hub.calls).toEqual([entry().onnxUrl]);
    expect(await bytes.get(entry().onnxUrl)).toBe(result);
    expect(lines).toEqual([
      `mito_aff_setup54_16nm: downloading ${FILE} (42.9 MB) from Hugging Face`,
      "mito_aff_setup54_16nm: downloaded 0.0 MB",
    ]);
  });

  it("serves a hit from the cache without fetching", async () => {
    const bytes = new MemoryByteCache();
    await bytes.put(entry().onnxUrl, MODEL_BYTES.buffer);
    const { catalog, hub, lines } = catalogWith({ onnxStatus: 500 }, bytes);
    const progress: [number, number | null][] = [];
    const result = await catalog.modelBytes(entry(), (l, t) => progress.push([l, t]));
    expect(result).toBe(MODEL_BYTES.buffer);
    expect(hub.calls).toEqual([]);
    expect(progress).toEqual([[100, 100]]);
    expect(lines).toEqual(["mito_aff_setup54_16nm: 0.0 MB from the browser's cache"]);
  });

  it("works with the IndexedDB cache end to end", async () => {
    const idb = fakeIndexedDb();
    const first = catalogWith({}, new IndexedDbByteCache(idb.factory));
    await first.catalog.modelBytes(entry());
    expect(first.hub.calls).toHaveLength(1);
    const second = catalogWith({ onnxStatus: 500 }, new IndexedDbByteCache(idb.factory));
    expect(new Uint8Array(await second.catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(second.hub.calls).toHaveLength(0);
  });

  it("says Hugging Face rate-limited the download on a 429", async () => {
    const { catalog, bytes } = catalogWith({ onnxStatus: 429 });
    await expect(catalog.modelBytes(entry())).rejects.toThrow(
      `Hugging Face rate-limited the anonymous download of ${FILE} (HTTP 429). Please retry in a minute.`,
    );
    expect(await bytes.get(entry().onnxUrl)).toBeUndefined();
  });

  it("shares one download between callers asking at the same time", async () => {
    const { catalog, hub } = catalogWith({});
    const [a, b] = await Promise.all([catalog.modelBytes(entry()), catalog.modelBytes(entry())]);
    expect(a).toBe(b);
    expect(hub.calls).toHaveLength(1);
    // Once done, the cache answers; no second download either.
    await catalog.modelBytes(entry());
    expect(hub.calls).toHaveLength(1);
  });

  it("still returns the bytes when the cache cannot keep them", async () => {
    const broken = new MemoryByteCache();
    broken.put = async () => {
      throw new Error("QuotaExceededError");
    };
    const { catalog, lines } = catalogWith({}, broken);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(lines.some((l) => l.includes("could not cache the model (QuotaExceededError)"))).toBe(true);
  });

  it("downloads when the cache cannot be read", async () => {
    const broken = new MemoryByteCache();
    broken.get = async () => {
      throw new Error("InvalidStateError");
    };
    const { catalog, hub, lines } = catalogWith({}, broken);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(hub.calls).toHaveLength(1);
    expect(lines.some((l) => l.includes("could not read the model cache (InvalidStateError)"))).toBe(true);
  });
});
