// Models finetuned in this browser, in the catalog: the name the next one
// gets, its one variant kept through serving's re-tuning, and
// createCatalog's addFinetuned / finetuned / removeFinetuned with list(),
// resolve() and modelBytes() on the entries, including a fresh catalog over
// the same stores (the next visit, in memory and through IndexedDB) and
// bytes that are gone.
import { describe, expect, it } from "vitest";

import type { FinetunedRecord, GpuLimits, ModelEntry } from "../../src/contracts";
import { createCatalog } from "../../src/models";
import { type ByteCache, IndexedDbByteCache, MemoryByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, type FinetunedRegistry, IndexedDbFinetunedRegistry, MemoryFinetunedRegistry, nextFinetunedName } from "../../src/models/finetuned";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFor } from "../../src/models/manifest";
import { chooseVariant, UNKNOWN_LIMITS } from "../../src/models/variant";
import { entryFixture, fakeHub, fakeIndexedDb, fakeLog, FOLDERS, type HubOptions } from "./models-helpers";

const BYTES = Uint8Array.from({ length: 64 }, (_, i) => 255 - i);
const VOLUME = "vol-1a2b3c4d-20261015-101500";
const T1 = "2026-10-15T10:15:00.000Z";
const T2 = "2026-10-15T11:30:00.000Z";
const BIG_GPU: GpuLimits = { maxBufferSize: 4 * 1024 ** 3, maxStorageBufferBindingSize: null, shaderF16: true };
const SMALL_GPU: GpuLimits = { maxBufferSize: 64 * 1024 ** 2, maxStorageBufferBindingSize: null, shaderF16: false };

/** A record as the finetune run would make one: the entry from finetunedEntry, the base's id, name and file, one volume, 20 epochs. */
function recordFor(base: ModelEntry, name: string, overrides: Partial<FinetunedRecord> = {}): FinetunedRecord {
  const entry = finetunedEntry(base, name, BYTES.byteLength / 1e6);
  return {
    id: entry.id,
    name,
    base: { id: base.id, name: base.name, file: base.variant.file },
    volume: VOLUME,
    created_at: T1,
    params: { epochs: 20, learning_rate: 0.001 },
    epochs: 20,
    final_loss: 0.31234,
    entry,
    ...overrides,
  };
}

function catalogWith(registry: FinetunedRegistry = new MemoryFinetunedRegistry(), bytes: ByteCache = new MemoryByteCache(), options: HubOptions = {}) {
  const hub = fakeHub(options);
  const { log } = fakeLog();
  const catalog = createCatalog({ log, fetch: hub.fetch, storage: new MemoryKeyValueStore(), bytes, finetuned: registry });
  return { catalog, hub, registry, bytes };
}

const HUB_IDS = FOLDERS.map((f) => `cellmap/onnx-web-models/${f}`);

describe("nextFinetunedName", () => {
  const base = entryFixture("setup_54");

  it("starts at -ft1 and counts up past the highest kept", () => {
    expect(nextFinetunedName(base, [])).toBe("mito_aff_setup54_16nm-ft1");
    const ft1 = recordFor(base, "mito_aff_setup54_16nm-ft1");
    expect(nextFinetunedName(base, [ft1])).toBe("mito_aff_setup54_16nm-ft2");
    const ft2 = recordFor(base, "mito_aff_setup54_16nm-ft2", { created_at: T2 });
    expect(nextFinetunedName(base, [ft2, ft1])).toBe("mito_aff_setup54_16nm-ft3");
    // A gap (ft1 deleted) is not refilled: the next is past the highest.
    expect(nextFinetunedName(base, [ft2])).toBe("mito_aff_setup54_16nm-ft3");
  });
});

describe("entryFor on a finetuned entry", () => {
  it("round-trips through serving's re-tuning, whatever the GPU", () => {
    const entry = finetunedEntry(entryFixture("setup_54"), "mito_aff_setup54_16nm-ft1", 42.9);
    for (const limits of [UNKNOWN_LIMITS, BIG_GPU, SMALL_GPU]) {
      const variant = chooseVariant(entry.manifest, limits);
      expect(variant).toBe(entry.variant);
      expect(entryFor(entry.folder, entry.manifest, variant, entry.repo)).toEqual(entry);
    }
    // An fp16 model too, on a GPU without shader-f16: the one variant is still the one.
    const fp16 = finetunedEntry(entryFixture("setup_54", BIG_GPU), "mito_aff_setup54_16nm-ft2");
    expect(entryFor(fp16.folder, fp16.manifest, chooseVariant(fp16.manifest, SMALL_GPU), fp16.repo)).toEqual(fp16);
  });
});

describe("the catalog's finetuned models", () => {
  it("lists an added model after the Hub's, resolves it three ways and serves its bytes without a download", async () => {
    const { catalog, hub } = catalogWith();
    const base = (await catalog.resolve("mito_aff_setup54_16nm"))!;
    const name = nextFinetunedName(base, await catalog.finetuned());
    expect(name).toBe("mito_aff_setup54_16nm-ft1");
    const record = recordFor(base, name);
    await catalog.addFinetuned(record, BYTES.buffer);

    const entries = await catalog.list();
    expect(entries.map((e) => e.id)).toEqual([...HUB_IDS, "finetuned/mito_aff_setup54_16nm-ft1"]);
    expect(entries.at(-1)).toBe(record.entry);
    expect(await catalog.resolve("finetuned/mito_aff_setup54_16nm-ft1")).toBe(record.entry);
    expect(await catalog.resolve("mito_aff_setup54_16nm-ft1")).toBe(record.entry);
    expect(await catalog.resolve(" finetuned/mito_aff_setup54_16nm-ft1/ ")).toBe(record.entry);
    expect(await catalog.finetuned()).toEqual([record]);

    const before = hub.calls.length;
    const progress: [number, number | null][] = [];
    const bytes = await catalog.modelBytes(record.entry, (l, t) => progress.push([l, t]));
    expect(bytes).toBe(BYTES.buffer);
    expect(hub.calls).toHaveLength(before);
    expect(hub.calls.some((u) => u.endsWith(".onnx"))).toBe(false);
    expect(progress).toEqual([[64, 64]]);
  });

  it("removeFinetuned forgets the entry and its bytes", async () => {
    const { catalog, bytes, registry } = catalogWith();
    const base = (await catalog.resolve("mito_aff_setup54_16nm"))!;
    const record = recordFor(base, "mito_aff_setup54_16nm-ft1");
    await catalog.addFinetuned(record, BYTES.buffer);
    expect(await bytes.get(record.entry.onnxUrl)).toBe(BYTES.buffer);

    await catalog.removeFinetuned(record.id);
    expect((await catalog.list()).map((e) => e.id)).toEqual(HUB_IDS);
    expect(await catalog.resolve("mito_aff_setup54_16nm-ft1")).toBeNull();
    expect(await catalog.finetuned()).toEqual([]);
    expect(await registry.list()).toEqual([]);
    expect(await bytes.get(record.entry.onnxUrl)).toBeUndefined();
    // Forgetting it again, or something that never was, is nothing.
    await catalog.removeFinetuned(record.id);
    await catalog.removeFinetuned("finetuned/never-was");
  });

  it("says to train again when the bytes are gone, and never downloads", async () => {
    const registry = new MemoryFinetunedRegistry();
    const record = recordFor(entryFixture("setup_54"), "mito_aff_setup54_16nm-ft1");
    await registry.put(record); // the record survived, the bytes did not
    const { catalog, hub } = catalogWith(registry, new MemoryByteCache(), { onnx: { "mito_aff_setup54_16nm-ft1.onnx": BYTES } });
    expect((await catalog.list()).at(-1)!.id).toBe(record.id);
    await expect(catalog.modelBytes(record.entry)).rejects.toThrow(/Train it again/);
    expect(hub.calls.some((u) => u.endsWith(".onnx"))).toBe(false);
  });

  it("lists a finetuned model on a fresh catalog's first list(): the next visit", async () => {
    const first = catalogWith();
    const base = (await first.catalog.resolve("mito_aff_setup54_16nm"))!;
    const record = recordFor(base, "mito_aff_setup54_16nm-ft1");
    await first.catalog.addFinetuned(record, BYTES.buffer);

    const next = catalogWith(first.registry, first.bytes);
    const entries = await next.catalog.list();
    expect(entries.map((e) => e.id)).toEqual([...HUB_IDS, record.id]);
    expect(entries.at(-1)).toEqual(record.entry);
    expect(await next.catalog.modelBytes(record.entry)).toBe(BYTES.buffer);
    expect(await next.catalog.finetuned()).toEqual([record]);
    // Through IndexedDB too: the bytes and the record in one database.
    const idb = fakeIndexedDb();
    const trained = catalogWith(new IndexedDbFinetunedRegistry(idb.factory), new IndexedDbByteCache(idb.factory));
    await trained.catalog.addFinetuned(record, BYTES.buffer);
    const revisited = catalogWith(new IndexedDbFinetunedRegistry(idb.factory), new IndexedDbByteCache(idb.factory));
    expect((await revisited.catalog.list()).at(-1)!.id).toBe(record.id);
    expect(new Uint8Array(await revisited.catalog.modelBytes(record.entry))).toEqual(BYTES);
  });
});
