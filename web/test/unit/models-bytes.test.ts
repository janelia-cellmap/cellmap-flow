// Catalog.modelBytes(): the streamed download with progress, the byte cache
// (hit and miss), the ETag check against Hugging Face before a cached copy
// is served (a model re-exported under the same name, the Hub unreachable),
// the rate-limit error, and a cache that cannot keep the bytes (a full
// quota). A finetuned model's bytes, which never reach the Hub, are
// models-finetuned.test.ts's.
import { describe, expect, it } from "vitest";

import { createCatalog } from "../../src/models";
import { type ByteCache, MemoryByteCache, modelBytesKey } from "../../src/models/byte-cache";
import { MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { RateLimitError } from "../../src/models/hf";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFixture, fakeEtag, fakeHub, fakeLog, type HubOptions } from "./models-helpers";

const FILE = "setup_54_mito_affinities_16nm_in132_fp32.onnx";
const MODEL_BYTES = Uint8Array.from({ length: 100 }, (_, i) => i);

describe("Catalog.modelBytes", () => {
  function catalogWith(options: HubOptions, bytes: ByteCache = new MemoryByteCache()) {
    const hub = fakeHub({ onnx: { [FILE]: MODEL_BYTES }, chunkSize: 40, ...options });
    const { log, lines } = fakeLog();
    const catalog = createCatalog({ log, fetch: hub.fetch, storage: new MemoryKeyValueStore(), bytes, finetuned: new MemoryFinetunedRegistry() });
    return { catalog, hub, lines, bytes };
  }
  const entry = () => entryFixture("setup_54");
  const FILE_URL = entry().onnxUrl;
  const ETAG = fakeEtag(MODEL_BYTES);
  const KEY = modelBytesKey(FILE_URL, ETAG);
  const HEAD = `HEAD ${FILE_URL}`;

  it("downloads on a miss with progress, without a HEAD first (the GET carries the ETag), and caches under the URL and ETag", async () => {
    const { catalog, hub, bytes } = catalogWith({});
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
  });

  it("serves a hit from the cache after the HEAD alone, without downloading", async () => {
    const bytes = new MemoryByteCache();
    await bytes.put(KEY, MODEL_BYTES.buffer);
    const { catalog, hub } = catalogWith({ getStatus: 500 }, bytes);
    const progress: [number, number | null][] = [];
    const result = await catalog.modelBytes(entry(), (l, t) => progress.push([l, t]));
    expect(result).toBe(MODEL_BYTES.buffer);
    expect(hub.calls).toEqual([HEAD]);
    expect(progress).toEqual([[100, 100]]);
  });

  it("downloads again when the file changed on Hugging Face, and drops the old copy", async () => {
    const bytes = new MemoryByteCache();
    await catalogWith({}, bytes).catalog.modelBytes(entry());
    // The model retrained and re-exported under the same file name: new bytes, so a new ETag.
    const retrained = Uint8Array.from({ length: 100 }, (_, i) => 99 - i);
    const newTag = fakeEtag(retrained);
    expect(newTag).not.toBe(ETAG);
    const { catalog, hub } = catalogWith({ onnx: { [FILE]: retrained } }, bytes);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(retrained);
    expect(hub.calls).toEqual([HEAD, FILE_URL]);
    expect(await bytes.keys(FILE_URL)).toEqual([modelBytesKey(FILE_URL, newTag)]);
  });

  it("serves the cached copy when Hugging Face cannot be asked (offline), saying it could not check", async () => {
    const bytes = new MemoryByteCache();
    await bytes.put(KEY, MODEL_BYTES.buffer);
    const { catalog, hub, lines } = catalogWith({ headFails: true, getStatus: 500 }, bytes);
    expect(await catalog.modelBytes(entry())).toBe(MODEL_BYTES.buffer);
    expect(hub.calls).toEqual([HEAD]);
    expect(lines.some((l) => l.includes("could not check"))).toBe(true);
  });

  it("says Hugging Face rate-limited the download on a 429, and caches nothing", async () => {
    const { catalog, bytes } = catalogWith({ onnxStatus: 429 });
    await expect(catalog.modelBytes(entry())).rejects.toBeInstanceOf(RateLimitError);
    expect(await bytes.keys(FILE_URL)).toEqual([]);
  });

  it("still returns the bytes when the cache cannot keep them", async () => {
    const broken = new MemoryByteCache();
    broken.put = async () => {
      throw new Error("QuotaExceededError");
    };
    const { catalog, lines } = catalogWith({}, broken);
    expect(new Uint8Array(await catalog.modelBytes(entry()))).toEqual(MODEL_BYTES);
    expect(lines.some((l) => l.includes("could not cache"))).toBe(true);
  });
});
