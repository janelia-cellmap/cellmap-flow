// Models finetuned in this browser, in the catalog: the entry built for one
// and its name, the registry (memory and IndexedDB, in the byte cache's
// database), and createCatalog's addFinetuned / finetuned / removeFinetuned
// with list(), resolve(), modelBytes() and metadataFor() on the entries,
// including a fresh catalog over the same stores (the next visit).
import { describe, expect, it } from "vitest";

import { ARGS_KEY, FINETUNED_REPO } from "../../src/contracts";
import type { FinetunedRecord, GpuLimits, ModelEntry } from "../../src/contracts";
import { createCatalog } from "../../src/models";
import { type ByteCache, IndexedDbByteCache, MemoryByteCache, MODEL_DB_NAME } from "../../src/models/byte-cache";
import {
  finetunedEntry,
  type FinetunedRegistry,
  IndexedDbFinetunedRegistry,
  MemoryFinetunedRegistry,
  nextFinetunedName,
  openFinetunedRegistry,
} from "../../src/models/finetuned";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFor, finetunedBytesKey } from "../../src/models/manifest";
import { describe as describeEntry, FINETUNED_AUTHOR, metadataFor } from "../../src/models/metadata";
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
  const { log, lines } = fakeLog();
  const catalog = createCatalog({ log, fetch: hub.fetch, storage: new MemoryKeyValueStore(), bytes, finetuned: registry });
  return { catalog, hub, lines, registry, bytes };
}

const HUB_IDS = FOLDERS.map((f) => `cellmap/onnx-web-models/${f}`);
const isFinetuned = (e: ModelEntry) => e.repo === FINETUNED_REPO;

describe("finetunedEntry", () => {
  const base = entryFixture("setup_54"); // fp32, 132³

  it("is the base's manifest under the new name with one variant, keyed into the byte cache", () => {
    const entry = finetunedEntry(base, "mito_aff_setup54_16nm-ft1", 21.555);
    expect(entry.id).toBe("finetuned/mito_aff_setup54_16nm-ft1");
    expect(entry.name).toBe("mito_aff_setup54_16nm-ft1");
    expect(entry.repo).toBe("finetuned");
    expect(entry.folder).toBe("mito_aff_setup54_16nm-ft1");
    expect(entry.onnxUrl).toBe("finetuned:finetuned/mito_aff_setup54_16nm-ft1");
    expect(entry.manifest.model).toBe("mito_aff_setup54_16nm-ft1");
    expect(entry.manifest.checkpoint).toBe("finetuned in the browser from model_checkpoint_260000");
    expect(entry.manifest.variants).toHaveLength(1);
    expect(entry.manifest.variants[0]).toBe(entry.variant);
    // The base's block and precision, the file named after the model, the size from the bytes; no fidelity numbers.
    expect(entry.variant).toEqual({ variant: "fp32", input_size: [132, 132, 132], output_size: [44, 44, 44], file: "mito_aff_setup54_16nm-ft1.onnx", file_mb: 21.56 });
    // The rest is the base's.
    for (const key of ["task", "architecture", "params", "channels", "input_voxel_size_nm", "output_voxel_size_nm", "output", "input_norm", "io"] as const) {
      expect(entry.manifest[key], key).toEqual(base.manifest[key]);
    }
    // The base is untouched.
    expect(base.manifest.model).toBe("setup_54");
    expect(base.manifest.variants).toHaveLength(12);
  });

  it("keeps the base's file size when the bytes are not given", () => {
    expect(finetunedEntry(base, "x-ft1").variant.file_mb).toBe(base.variant.file_mb);
  });

  it("follows the base's variant: an fp16 base gives an fp16 model", () => {
    const fp16 = entryFixture("setup_54", BIG_GPU);
    expect(fp16.variant.variant).toBe("fp16");
    const entry = finetunedEntry(fp16, "mito_aff_setup54_16nm-ft1");
    expect(entry.variant.variant).toBe("fp16");
    expect(entry.variant.input_size).toEqual(fp16.variant.input_size);
  });

  it("names the base when its manifest has no checkpoint", () => {
    const bare = { ...base, manifest: { ...base.manifest, checkpoint: "" } };
    expect(finetunedEntry(bare, "x-ft1").manifest.checkpoint).toBe("finetuned in the browser from mito_aff_setup54_16nm");
  });
});

describe("nextFinetunedName", () => {
  const base = entryFixture("setup_54");
  const other = entryFixture("setup_53");

  it("starts at -ft1 and counts up past the highest kept", () => {
    expect(nextFinetunedName(base, [])).toBe("mito_aff_setup54_16nm-ft1");
    const ft1 = recordFor(base, "mito_aff_setup54_16nm-ft1");
    expect(nextFinetunedName(base, [ft1])).toBe("mito_aff_setup54_16nm-ft2");
    const ft2 = recordFor(base, "mito_aff_setup54_16nm-ft2", { created_at: T2 });
    expect(nextFinetunedName(base, [ft2, ft1])).toBe("mito_aff_setup54_16nm-ft3");
    // A gap (ft1 deleted) is not refilled: the next is past the highest.
    expect(nextFinetunedName(base, [ft2])).toBe("mito_aff_setup54_16nm-ft3");
  });

  it("ignores another base's records", () => {
    const others = [1, 2, 3, 4, 5].map((n) => recordFor(other, `mito_aff_setup53_16nm-ft${n}`));
    expect(nextFinetunedName(base, others)).toBe("mito_aff_setup54_16nm-ft1");
    expect(nextFinetunedName(other, others)).toBe("mito_aff_setup53_16nm-ft6");
  });

  it("counts a record of this base whatever it was named", () => {
    const renamed = recordFor(base, "my-model-ft7");
    expect(nextFinetunedName(base, [renamed])).toBe("mito_aff_setup54_16nm-ft8");
  });

  it("continues the series when the base is itself finetuned", () => {
    const ft1 = recordFor(base, "mito_aff_setup54_16nm-ft1");
    expect(nextFinetunedName(ft1.entry, [ft1])).toBe("mito_aff_setup54_16nm-ft2");
    const ft3 = recordFor(base, "mito_aff_setup54_16nm-ft3");
    expect(nextFinetunedName(ft1.entry, [ft1, ft3])).toBe("mito_aff_setup54_16nm-ft4");
  });

  it("is a URL path segment and a layer name", () => {
    const name = nextFinetunedName(base, []);
    expect(encodeURIComponent(name)).toBe(name);
    expect(name).not.toContain(ARGS_KEY);
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

  it("keys the bytes by the id, apart from the Hub's URLs", () => {
    expect(finetunedBytesKey("finetuned/x-ft1")).toBe("finetuned:finetuned/x-ft1");
    expect(entryFor("x-ft1", entryFixture().manifest, entryFixture().variant, FINETUNED_REPO).onnxUrl).toBe("finetuned:finetuned/x-ft1");
  });
});

describe("the registries", () => {
  const base = entryFixture("setup_54");
  const older = recordFor(base, "mito_aff_setup54_16nm-ft1", { created_at: T1 });
  const newer = recordFor(base, "mito_aff_setup54_16nm-ft2", { created_at: T2 });

  it("MemoryFinetunedRegistry lists newest first, replaces by id and forgets", async () => {
    const registry = new MemoryFinetunedRegistry();
    await registry.put(older);
    await registry.put(newer);
    expect((await registry.list()).map((r) => r.name)).toEqual(["mito_aff_setup54_16nm-ft2", "mito_aff_setup54_16nm-ft1"]);
    await registry.put({ ...older, epochs: 40 });
    expect((await registry.list()).map((r) => r.epochs)).toEqual([20, 40]);
    await registry.remove(newer.id);
    await registry.remove("finetuned/never-was");
    expect((await registry.list()).map((r) => r.id)).toEqual([older.id]);
  });

  it("IndexedDbFinetunedRegistry keeps the records in the byte cache's database, beside the bytes", async () => {
    const idb = fakeIndexedDb();
    const registry = new IndexedDbFinetunedRegistry(idb.factory);
    const cache = new IndexedDbByteCache(idb.factory);
    expect(await registry.list()).toEqual([]);
    await registry.put(older);
    await registry.put(newer);
    await cache.put(newer.entry.onnxUrl, BYTES.buffer);
    const stores = idb.databases.get(MODEL_DB_NAME)!;
    expect([...stores.keys()].sort()).toEqual(["bytes", "finetuned"]);
    expect(stores.get("finetuned")!.get(newer.id)).toBe(newer);
    expect(stores.get("bytes")!.get(newer.entry.onnxUrl)).toBe(BYTES.buffer);
    expect((await registry.list()).map((r) => r.name)).toEqual(["mito_aff_setup54_16nm-ft2", "mito_aff_setup54_16nm-ft1"]);
    await registry.remove(older.id);
    expect((await registry.list()).map((r) => r.id)).toEqual([newer.id]);
    // Whichever opens first creates both stores: a registry alone, then a cache.
    const second = fakeIndexedDb();
    await new IndexedDbFinetunedRegistry(second.factory).list();
    await new IndexedDbByteCache(second.factory).put("k", BYTES.buffer);
    expect([...second.databases.get(MODEL_DB_NAME)!.keys()].sort()).toEqual(["bytes", "finetuned"]);
  });

  it("leaves a row that is not a record out of the listing", async () => {
    const idb = fakeIndexedDb();
    const registry = new IndexedDbFinetunedRegistry(idb.factory);
    await registry.put(older);
    idb.databases.get(MODEL_DB_NAME)!.get("finetuned")!.set("junk", { nope: true });
    expect((await registry.list()).map((r) => r.id)).toEqual([older.id]);
  });

  it("openFinetunedRegistry falls back to memory without IndexedDB, leaving the note to the byte cache", () => {
    expect(openFinetunedRegistry(undefined)).toBeInstanceOf(MemoryFinetunedRegistry);
    expect(openFinetunedRegistry(fakeIndexedDb().factory)).toBeInstanceOf(IndexedDbFinetunedRegistry);
    // A catalog left to find its own storage, here (no IndexedDB): one note, from the byte cache.
    const { log, lines } = fakeLog();
    createCatalog({ log, fetch: fakeHub().fetch, storage: new MemoryKeyValueStore() });
    expect(lines).toEqual(["model cache: no IndexedDB here; model files will be downloaded on every visit, and a model finetuned in this browser is forgotten when the page closes"]);
  });
});

describe("the catalog's finetuned models", () => {
  it("lists an added model after the Hub's, resolves it three ways and serves its bytes without a download", async () => {
    const { catalog, hub, lines } = catalogWith();
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
    expect(lines).toContain("mito_aff_setup54_16nm-ft1: 0.0 MB from the browser's storage");
  });

  it("describes a finetuned model from its record in metadataFor", async () => {
    const { catalog } = catalogWith();
    const base = (await catalog.resolve("mito_aff_setup54_16nm"))!;
    const record = recordFor(base, "mito_aff_setup54_16nm-ft1");
    await catalog.addFinetuned(record, BYTES.buffer);
    const metadata = catalog.metadataFor(record.entry);
    expect(metadata.description).toBe(
      "finetuned in this browser from mito_aff_setup54_16nm on vol-1a2b3c4d-20261015-101500, 2026-10-15; 20 epochs, final loss 0.312; ONNX fp32, block 116³ → 28³, 0.0 MB; raw logits (apply sigmoid / threshold at 0)",
    );
    expect(metadata).toMatchObject({
      model_name: "mito_aff_setup54_16nm-ft1",
      model_type: "ResUNet3DValid",
      framework: "onnx",
      author: FINETUNED_AUTHOR,
      version: T1,
      repo: "finetuned",
      folder: "mito_aff_setup54_16nm-ft1",
      checkpoint: "finetuned in the browser from model_checkpoint_260000",
      channels_names: base.manifest.channels,
      file: "mito_aff_setup54_16nm-ft1.onnx",
      onnx_url: "finetuned:finetuned/mito_aff_setup54_16nm-ft1",
      finetuned: { base: record.base, volume: VOLUME, created_at: T1, epochs: 20, final_loss: 0.31234, params: record.params },
    });
    expect(metadata.author).not.toBe("Marwan Zouinkhi");
    // The Hub's entries describe themselves as before.
    expect(catalog.metadataFor(base).description).toMatch(/^mito affinities at 16 nm/);
    expect(JSON.parse(JSON.stringify(metadata))).toEqual(metadata);
  });

  it("describes without a record from the manifest, and words the loss and epochs plainly", () => {
    const base = entryFixture("setup_54");
    const record = recordFor(base, "mito_aff_setup54_16nm-ft1");
    expect(describeEntry(record.entry)).toBe("finetuned in the browser from model_checkpoint_260000; ONNX fp32, block 132³ → 44³, 0.0 MB; raw logits (apply sigmoid / threshold at 0)");
    expect(metadataFor(record.entry).description).toBe(describeEntry(record.entry));
    expect(describeEntry(record.entry, { ...record, epochs: 1, final_loss: 0.0004 })).toContain("; 1 epoch, final loss 0.0004;");
    expect(describeEntry(record.entry, { ...record, final_loss: null })).toContain("; 20 epochs, final loss unknown;");
    expect(metadataFor(record.entry).version).toBeNull();
  });

  it("lists the finetuned models newest first, whatever the order they were added in", async () => {
    const { catalog } = catalogWith();
    const base = (await catalog.resolve("mito_aff_setup54_16nm"))!;
    const ft1 = recordFor(base, "mito_aff_setup54_16nm-ft1", { created_at: T1 });
    const ft2 = recordFor(base, "mito_aff_setup54_16nm-ft2", { created_at: T2 });
    await catalog.addFinetuned(ft2, BYTES.buffer);
    await catalog.addFinetuned(ft1, BYTES.buffer);
    expect((await catalog.finetuned()).map((r) => r.name)).toEqual(["mito_aff_setup54_16nm-ft2", "mito_aff_setup54_16nm-ft1"]);
    expect((await catalog.list()).filter(isFinetuned).map((e) => e.name)).toEqual(["mito_aff_setup54_16nm-ft2", "mito_aff_setup54_16nm-ft1"]);
    // Adding again under the same id replaces the record.
    await catalog.addFinetuned({ ...ft1, epochs: 40 }, BYTES.buffer);
    expect((await catalog.finetuned()).map((r) => [r.name, r.epochs])).toEqual([
      ["mito_aff_setup54_16nm-ft2", 20],
      ["mito_aff_setup54_16nm-ft1", 40],
    ]);
    expect((await catalog.list()).filter(isFinetuned)).toHaveLength(2);
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
    await expect(catalog.modelBytes(record.entry)).rejects.toThrow(
      "mito_aff_setup54_16nm-ft1 was finetuned in this browser and its bytes are no longer in the browser's storage (site data cleared, or another browser or profile). Train it again to serve it.",
    );
    expect(hub.calls.some((u) => u.endsWith(".onnx"))).toBe(false);
  });

  it("says so when the storage cannot be read", async () => {
    const broken = new MemoryByteCache();
    broken.get = async () => {
      throw new Error("InvalidStateError");
    };
    const { catalog } = catalogWith(new MemoryFinetunedRegistry(), broken);
    const record = recordFor(entryFixture("setup_54"), "mito_aff_setup54_16nm-ft1");
    await expect(catalog.modelBytes(record.entry)).rejects.toThrow("mito_aff_setup54_16nm-ft1 was finetuned in this browser and its storage could not be read (InvalidStateError).");
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
    expect(next.lines).toContain("model catalog: 1 model finetuned in this browser");
    expect(await next.catalog.modelBytes(record.entry)).toBe(BYTES.buffer);
    expect(next.catalog.metadataFor(entries.at(-1)!).description).toMatch(/^finetuned in this browser from mito_aff_setup54_16nm on vol-/);
    expect(await next.catalog.finetuned()).toEqual([record]);
    // Through IndexedDB too: the bytes and the record in one database.
    const idb = fakeIndexedDb();
    const trained = catalogWith(new IndexedDbFinetunedRegistry(idb.factory), new IndexedDbByteCache(idb.factory));
    await trained.catalog.addFinetuned(record, BYTES.buffer);
    const revisited = catalogWith(new IndexedDbFinetunedRegistry(idb.factory), new IndexedDbByteCache(idb.factory));
    expect((await revisited.catalog.list()).at(-1)!.id).toBe(record.id);
    expect(new Uint8Array(await revisited.catalog.modelBytes(record.entry))).toEqual(BYTES);
  });

  it("shows a model added before the first list(), or while it loads, exactly once", async () => {
    const base = entryFixture("setup_54");
    const early = catalogWith();
    await early.catalog.addFinetuned(recordFor(base, "mito_aff_setup54_16nm-ft1"), BYTES.buffer);
    expect((await early.catalog.list()).filter(isFinetuned).map((e) => e.name)).toEqual(["mito_aff_setup54_16nm-ft1"]);

    const during = catalogWith();
    const pending = during.catalog.list();
    await during.catalog.addFinetuned(recordFor(base, "mito_aff_setup54_16nm-ft1"), BYTES.buffer);
    await pending;
    expect((await during.catalog.list()).filter(isFinetuned).map((e) => e.name)).toEqual(["mito_aff_setup54_16nm-ft1"]);
  });

  it("refresh() keeps the finetuned entries and picks up a record another tab kept", async () => {
    const { catalog, registry } = catalogWith();
    const base = (await catalog.resolve("mito_aff_setup54_16nm"))!;
    const mine = recordFor(base, "mito_aff_setup54_16nm-ft1", { created_at: T1 });
    await catalog.addFinetuned(mine, BYTES.buffer);
    const theirs = recordFor(base, "mito_aff_setup54_16nm-ft2", { created_at: T2 });
    await registry.put(theirs);
    const refreshed = await catalog.refresh();
    expect(refreshed.filter(isFinetuned).map((e) => e.name)).toEqual(["mito_aff_setup54_16nm-ft2", "mito_aff_setup54_16nm-ft1"]);
    expect(await catalog.list()).toBe(refreshed);
    expect((await catalog.finetuned()).map((r) => r.name)).toEqual(["mito_aff_setup54_16nm-ft2", "mito_aff_setup54_16nm-ft1"]);
  });

  it("still lists the Hub's models when the registry cannot be read", async () => {
    const registry = new MemoryFinetunedRegistry();
    registry.list = async () => {
      throw new Error("open refused");
    };
    const { catalog, lines } = catalogWith(registry);
    expect((await catalog.list()).map((e) => e.id)).toEqual(HUB_IDS);
    expect(lines).toContain("model catalog: could not read the models finetuned in this browser (open refused)");
    expect(await catalog.finetuned()).toEqual([]);
  });

  it("takes the bytes back when the record cannot be written, and refuses a record that is not finetuned", async () => {
    const registry = new MemoryFinetunedRegistry();
    registry.put = async () => {
      throw new Error("QuotaExceededError");
    };
    const { catalog, bytes } = catalogWith(registry);
    const base = (await catalog.resolve("mito_aff_setup54_16nm"))!;
    const record = recordFor(base, "mito_aff_setup54_16nm-ft1");
    await expect(catalog.addFinetuned(record, BYTES.buffer)).rejects.toThrow("QuotaExceededError");
    expect(await bytes.get(record.entry.onnxUrl)).toBeUndefined();
    expect((await catalog.list()).map((e) => e.id)).toEqual(HUB_IDS);

    const notFinetuned = { ...record, entry: base };
    await expect(catalogWith().catalog.addFinetuned(notFinetuned, BYTES.buffer)).rejects.toThrow(/not a model finetuned in this browser/);
  });
});
