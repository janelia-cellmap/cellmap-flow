// modelBytes(): the streamed download with progress, the IndexedDB cache
// (hit and miss, with the tiny in-memory IndexedDB of models-helpers.ts), the
// ETag check against Hugging Face before a cached copy is served (a model
// re-exported under the same name, the Hub unreachable, a server without
// ETags, a copy from before the ETag was in the key), the shared in-flight
// download, and the rate-limit error.
import { describe, expect, it } from "vitest";

import { createCatalog } from "../../src/models";
import { type ByteCache, IndexedDbByteCache, MemoryByteCache, MODEL_DB_NAME, modelBytesKey, openByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { etagOf, fetchBytes, fetchEtag, fetchFile, fileNameOf, RateLimitError } from "../../src/models/hf";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFixture, fakeEtag, fakeHub, fakeIndexedDb, fakeLog, headResponse, type HubOptions, streamedResponse } from "./models-helpers";

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

describe("ByteCache keys and deleteWhere", () => {
  const caches: [string, () => ByteCache][] = [
    ["MemoryByteCache", () => new MemoryByteCache()],
    ["IndexedDbByteCache", () => new IndexedDbByteCache(fakeIndexedDb().factory)],
  ];
  for (const [name, make] of caches) {
    it(`${name} lists the keys under a prefix in key order and deletes all of them but one`, async () => {
      const cache = make();
      const url = "https://huggingface.co/x/y/resolve/main/a.onnx";
      await cache.put(url, MODEL_BYTES.buffer);
      await cache.put(`${url}#bbb`, MODEL_BYTES.buffer);
      await cache.put(`${url}#aaa`, MODEL_BYTES.buffer);
      await cache.put("https://huggingface.co/x/y/resolve/main/b.onnx#ccc", MODEL_BYTES.buffer);
      await cache.put("finetuned:finetuned/m-ft1", MODEL_BYTES.buffer);
      expect(await cache.keys(`${url}#`)).toEqual([`${url}#aaa`, `${url}#bbb`]);
      expect(await cache.keys(url)).toEqual([url, `${url}#aaa`, `${url}#bbb`]);
      expect(await cache.keys("finetuned:")).toEqual(["finetuned:finetuned/m-ft1"]);
      expect(await cache.keys("nothing")).toEqual([]);
      await cache.deleteWhere(`${url}#`, `${url}#bbb`);
      expect(await cache.keys(url)).toEqual([url, `${url}#bbb`]);
      expect(await cache.get(`${url}#bbb`)).toBe(MODEL_BYTES.buffer);
      await cache.deleteWhere(url);
      expect(await cache.keys(url)).toEqual([]);
      expect(await cache.keys("")).toEqual(["finetuned:finetuned/m-ft1", "https://huggingface.co/x/y/resolve/main/b.onnx#ccc"]);
    });
  }

  it("modelBytesKey is the URL and ETag parted by #, the bare URL without an ETag", () => {
    expect(modelBytesKey("https://x/a.onnx", "7ff35ac3")).toBe("https://x/a.onnx#7ff35ac3");
    expect(modelBytesKey("https://x/a.onnx", null)).toBe("https://x/a.onnx");
  });
});

describe("fetchEtag", () => {
  const url = "https://huggingface.co/cellmap/onnx-web-models/resolve/main/setup_54/" + FILE;

  it("is the HEAD response's ETag without its quotes, asked for with a HEAD that bypasses the service worker's cache", async () => {
    const inits: (RequestInit | undefined)[] = [];
    const tag = await fetchEtag(async (_u, init) => {
      inits.push(init);
      return headResponse(100, "7ff35ac3d927d201");
    }, url);
    expect(tag).toBe("7ff35ac3d927d201");
    expect(inits).toEqual([{ method: "HEAD", cache: "no-store" }]);
  });

  it("is null when the HEAD fails (offline), is not OK, or exposes no ETag", async () => {
    expect(
      await fetchEtag(async () => {
        throw new TypeError("Failed to fetch");
      }, url),
    ).toBeNull();
    expect(await fetchEtag(async () => new Response("", { status: 429 }), url)).toBeNull();
    expect(await fetchEtag(async () => headResponse(100, null), url)).toBeNull();
  });

  it("etagOf strips the quotes and a weak validator's W/, and is null for none or an empty one", () => {
    expect(etagOf(new Response("", { headers: { etag: '"abc"' } }))).toBe("abc");
    expect(etagOf(new Response("", { headers: { etag: 'W/"abc"' } }))).toBe("abc");
    expect(etagOf(new Response("", { headers: { etag: "abc" } }))).toBe("abc");
    expect(etagOf(new Response("", { headers: { etag: '""' } }))).toBeNull();
    expect(etagOf(new Response(""))).toBeNull();
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

  it("fetchFile carries the response's ETag beside the bytes", async () => {
    const got = await fetchFile(async () => streamedResponse(MODEL_BYTES, 50, true, "abc"), "https://x/a.onnx");
    expect(new Uint8Array(got.bytes)).toEqual(MODEL_BYTES);
    expect(got.etag).toBe("abc");
    expect((await fetchFile(async () => streamedResponse(MODEL_BYTES, 50, true), "https://x/a.onnx")).etag).toBeNull();
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
  const NAME = "mito_aff_setup54_16nm";
  const FILE_URL = entry().onnxUrl;
  const ETAG = fakeEtag(MODEL_BYTES);
  const KEY = modelBytesKey(FILE_URL, ETAG);
  const HEAD = `HEAD ${FILE_URL}`;
  const UNCHECKED = `${NAME}: 0.0 MB from the browser's cache (could not check Hugging Face for a newer file)`;

  it("downloads on a miss with progress, without a HEAD first (the GET carries the ETag), and caches under the URL and ETag", async () => {
    const { catalog, hub, lines, bytes } = catalogWith({});
    const progress: [number, number | null][] = [];
    const result = await catalog.modelBytes(entry(), (l, t) => progress.push([l, t]));
    expect(new Uint8Array(result)).toEqual(MODEL_BYTES);
    expect(progress).toEqual([
      [40, 100],
      [80, 100],
      [100, 100],
    ]);
    expect(hub.calls).toEqual([FILE_URL]);
    expect(KEY).toBe(`${FILE_URL}#${ETAG}`);
    expect(await bytes.get(KEY)).toBe(result);
    expect(await bytes.get(FILE_URL)).toBeUndefined();
    expect(lines).toEqual([`${NAME}: downloading ${FILE} (42.9 MB) from Hugging Face`, `${NAME}: downloaded 0.0 MB`]);
  });

  it("serves a hit from the cache after the HEAD alone, without downloading", async () => {
    const bytes = new MemoryByteCache();
    await bytes.put(KEY, MODEL_BYTES.buffer);
    const { catalog, hub, lines } = catalogWith({ getStatus: 500 }, bytes);
    const progress: [number, number | null][] = [];
    const result = await catalog.modelBytes(entry(), (l, t) => progress.push([l, t]));
    expect(result).toBe(MODEL_BYTES.buffer);
    expect(hub.calls).toEqual([HEAD]);
    expect(progress).toEqual([[100, 100]]);
    expect(lines).toEqual([`${NAME}: 0.0 MB from the browser's cache`]);
  });

  it("downloads again when the file changed on Hugging Face, and drops the old copy", async () => {
    const bytes = new MemoryByteCache();
    await catalogWith({}, bytes).catalog.modelBytes(entry());
    // The model retrained and re-exported under the same file name: new bytes, so a new ETag.
    const retrained = Uint8Array.from({ length: 100 }, (_, i) => 99 - i);
    const newTag = fakeEtag(retrained);
    expect(newTag).not.toBe(ETAG);
    const { catalog, hub, lines } = catalogWith({ onnx: { [FILE]: retrained } }, bytes);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(retrained);
    expect(hub.calls).toEqual([HEAD, FILE_URL]);
    expect(lines).toEqual([`${NAME}: ${FILE} updated on Hugging Face (new ${newTag.slice(0, 8)}…); downloading again`, `${NAME}: downloaded 0.0 MB`]);
    expect(await bytes.keys(FILE_URL)).toEqual([modelBytesKey(FILE_URL, newTag)]);
  });

  it("serves the cached copy when Hugging Face cannot be asked (offline), saying it could not check", async () => {
    const bytes = new MemoryByteCache();
    await bytes.put(KEY, MODEL_BYTES.buffer);
    const { catalog, hub, lines } = catalogWith({ headFails: true, getStatus: 500 }, bytes);
    expect(await catalog.modelBytes(entry())).toBe(MODEL_BYTES.buffer);
    expect(hub.calls).toEqual([HEAD]);
    expect(lines).toEqual([UNCHECKED]);
  });

  it("asks for no ETag when nothing is cached, so a HEAD that would fail is never made", async () => {
    const { catalog, hub, lines, bytes } = catalogWith({ headFails: true });
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(hub.calls).toEqual([FILE_URL]);
    expect(lines[0]).toBe(`${NAME}: downloading ${FILE} (42.9 MB) from Hugging Face`);
    expect(await bytes.keys(FILE_URL)).toEqual([KEY]);
  });

  it("keys the copy by the HEAD's ETag when the download exposes none, once there is a copy to check", async () => {
    // The first download has no ETag to key by (no HEAD is made for it), so it is kept under the bare URL.
    const bytes = new MemoryByteCache();
    await catalogWith({ noGetEtag: true }, bytes).catalog.modelBytes(entry());
    expect(await bytes.keys(FILE_URL)).toEqual([FILE_URL]);
    // The next load checks that copy: unkeyed, so downloaded once more and kept under the HEAD's ETag; the one after serves it.
    await catalogWith({ noGetEtag: true }, bytes).catalog.modelBytes(entry());
    expect(await bytes.keys(FILE_URL)).toEqual([KEY]);
    const third = catalogWith({ noGetEtag: true, getStatus: 500 }, bytes);
    expect(new Uint8Array(await third.catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(third.hub.calls).toEqual([HEAD]);
  });

  it("keys by the bare URL for a server that exposes no ETag, and serves that copy unchecked next time", async () => {
    const bytes = new MemoryByteCache();
    await catalogWith({ noEtag: true }, bytes).catalog.modelBytes(entry());
    expect(await bytes.keys(FILE_URL)).toEqual([FILE_URL]);
    const { catalog, hub, lines } = catalogWith({ noEtag: true, getStatus: 500 }, bytes);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(hub.calls).toEqual([HEAD]);
    expect(lines).toEqual([UNCHECKED]);
  });

  it("replaces a copy cached under the bare URL (from before the ETag was in the key) with a checked download", async () => {
    // What a browser holds from the first release: the bytes of the time under the URL alone.
    const old = Uint8Array.from({ length: 100 }, () => 7);
    const bytes = new MemoryByteCache();
    await bytes.put(FILE_URL, old.buffer);
    const { catalog, hub, lines } = catalogWith({}, bytes);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(hub.calls).toEqual([HEAD, FILE_URL]);
    expect(lines[0]).toBe(`${NAME}: the cached ${FILE} predates the version check; downloading it again (42.9 MB) from Hugging Face`);
    expect(await bytes.keys(FILE_URL)).toEqual([KEY]);
    // Offline, that old copy is still served rather than nothing.
    const offline = new MemoryByteCache();
    await offline.put(FILE_URL, old.buffer);
    const unchecked = catalogWith({ headFails: true, getStatus: 500 }, offline);
    expect(await unchecked.catalog.modelBytes(entry())).toBe(old.buffer);
    expect(unchecked.lines).toEqual([UNCHECKED]);
  });

  it("works with the IndexedDB cache end to end", async () => {
    const idb = fakeIndexedDb();
    const first = catalogWith({}, new IndexedDbByteCache(idb.factory));
    await first.catalog.modelBytes(entry());
    expect(first.hub.calls).toEqual([FILE_URL]);
    expect([...idb.databases.get(MODEL_DB_NAME)!.get("bytes")!.keys()]).toEqual([KEY]);
    const second = catalogWith({ getStatus: 500 }, new IndexedDbByteCache(idb.factory));
    expect(new Uint8Array(await second.catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(second.hub.calls).toEqual([HEAD]);
  });

  it("says Hugging Face rate-limited the download on a 429", async () => {
    const { catalog, bytes } = catalogWith({ onnxStatus: 429 });
    await expect(catalog.modelBytes(entry())).rejects.toThrow(
      `Hugging Face rate-limited the anonymous download of ${FILE} (HTTP 429). Please retry in a minute.`,
    );
    expect(await bytes.keys(FILE_URL)).toEqual([]);
  });

  it("shares one download between callers asking at the same time", async () => {
    const { catalog, hub } = catalogWith({});
    const [a, b] = await Promise.all([catalog.modelBytes(entry()), catalog.modelBytes(entry())]);
    expect(a).toBe(b);
    expect(hub.calls).toEqual([FILE_URL]);
    // Once done, the cache answers after a HEAD; no second download.
    await catalog.modelBytes(entry());
    expect(hub.calls).toEqual([FILE_URL, HEAD]);
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
    broken.keys = async () => {
      throw new Error("InvalidStateError");
    };
    const { catalog, hub, lines } = catalogWith({}, broken);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(hub.calls).toEqual([FILE_URL]);  // nothing known to be cached, so nothing to check
    expect(lines.some((l) => l.includes("could not read the model cache (InvalidStateError)"))).toBe(true);
  });

  it("leaves a finetuned model's bytes alone: no request to the Hub, its key as it is", async () => {
    // models-finetuned.test.ts covers the finetuned entries; this is the one
    // point of the ETag check: it must not reach them.
    const finetuned = finetunedEntry(entry(), "mito_aff_setup54_16nm-ft1", 0.1);
    const bytes = new MemoryByteCache();
    await bytes.put(finetuned.onnxUrl, MODEL_BYTES.buffer);
    const { catalog, hub } = catalogWith({ headFails: true }, bytes);
    expect(await catalog.modelBytes(finetuned)).toBe(MODEL_BYTES.buffer);
    expect(hub.calls).toEqual([]);
    expect(await bytes.keys("")).toEqual([finetuned.onnxUrl]);
  });
});
