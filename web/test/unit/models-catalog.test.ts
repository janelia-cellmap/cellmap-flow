// createCatalog's list(), resolve() and refresh(): the Hub listing, its
// day-long cache, the fallback when the Hub rate-limits the tree listing and
// the failure when it rate-limits everything, the references an entry is
// found by, and the repos added beside the built-in one by a ref of theirs.
import { describe, expect, it } from "vitest";

import type { FinetunedRecord } from "../../src/contracts";
import { createCatalog, LISTING_KEY } from "../../src/models";
import { LISTING_TTL_MS, MemoryKeyValueStore } from "../../src/models/listing-cache";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { entryFixture, fakeHub, fakeLog, FOLDERS, type HubOptions, publishedManifest, rawManifest } from "./models-helpers";

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

describe("another Hugging Face repo", () => {
  const FT = "mito_aff_setup54_16nm-ft1";
  const ALICE = `alice/${FT}`;
  const repos: HubOptions["repos"] = {
    // A model published from a browser, and a folder that holds no model.
    [ALICE]: { manifests: { [FT]: publishedManifest(entryFixture("setup_54"), FT) }, dirs: ["images"] },
    // Two models: setup_54's export copied, so its name is the built-in's, and setup_56's.
    "bob/models": { manifests: { copy_54: rawManifest("setup_54"), copy_56: rawManifest("setup_56") } },
    "carol/private": { status: 401 },
  };

  it("resolves every form of ref the server takes, adds the repo after the built-in one and lists it on the next visit", async () => {
    const storage = new MemoryKeyValueStore();
    const { catalog, hub } = catalogWith({ repos }, storage);
    const entry = await catalog.resolve(`${ALICE}/${FT}`);
    expect(entry).toMatchObject({ id: `${ALICE}/${FT}`, repo: ALICE, folder: FT, name: FT });
    expect(entry!.onnxUrl).toBe(`https://huggingface.co/${ALICE}/resolve/main/${FT}/${FT}.onnx`);
    for (const ref of [`hf:${ALICE}/${FT}`, `${ALICE}/${FT}@main`, ALICE, `https://huggingface.co/${ALICE}`, `https://huggingface.co/${ALICE}/tree/main/${FT}`, `https://huggingface.co/${ALICE}/blob/main/${FT}/${FT}.onnx`, FT]) {
      expect(await catalog.resolve(ref)).toBe(entry);
    }
    expect((await catalog.list()).map((e) => e.repo)).toEqual([...FOLDERS.map(() => "cellmap/onnx-web-models"), ALICE]);
    expect(treeCalls(hub.calls).filter((u) => u.includes(ALICE))).toHaveLength(1);

    // Refused, saying why: several models, a private or missing repo, a folder it does not have, another revision.
    await expect(catalog.resolve("bob/models")).rejects.toThrow(/bob\/models\/copy_54, bob\/models\/copy_56/);
    await expect(catalog.resolve("carol/private/x")).rejects.toThrow(/no public Hugging Face model repo carol\/private \(HTTP 401\)/);
    await expect(catalog.resolve(`${ALICE}/images`)).rejects.toThrow(/no model folder images/);
    await expect(catalog.resolve(`${ALICE}/${FT}@dev`)).rejects.toThrow(/revision dev/);
    // Not a Hugging Face model: a zoo DOI, a path, a word.
    for (const ref of ["10.5281/zenodo.5764892", "runs/model.onnx", "nothing_here"]) expect(await catalog.resolve(ref)).toBeNull();

    // The next visit, within the day: listed from the cache, the Hub not asked.
    const next = catalogWith({ repos }, storage);
    expect((await next.catalog.list()).map((e) => e.id)).toContain(`${ALICE}/${FT}`);
    expect(next.hub.calls.filter((u) => u.includes(ALICE))).toEqual([]);
  });

  it("names a published model by its published name, and breaks a clash with _2 where the built-in and finetuned names stay", async () => {
    const registry = new MemoryFinetunedRegistry();
    const { log } = fakeLog();
    const catalog = createCatalog({ log, fetch: fakeHub({ repos }).fetch, storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache(), finetuned: registry, now: () => T0 });
    expect((await catalog.resolve("bob/models/copy_54"))!.name).toBe("mito_aff_setup54_16nm_2");
    expect((await catalog.resolve("setup_54"))!.name).toBe("mito_aff_setup54_16nm");
    expect((await catalog.resolve(`${ALICE}/${FT}`))!.name).toBe(FT);

    // This browser's own -ft1 keeps its name, and the published one moves on.
    const own = finetunedEntry(entryFixture("setup_54"), FT);
    const record: FinetunedRecord = { id: own.id, name: FT, base: { id: "cellmap/onnx-web-models/setup_54", name: "mito_aff_setup54_16nm", file: own.variant.file }, volume: "vol-x", created_at: "2026-10-26T10:00:00.000Z", params: {}, epochs: 20, final_loss: 0.2, entry: own };
    await catalog.addFinetuned(record, new ArrayBuffer(8));
    const names = (await catalog.list()).map((e) => e.name);
    expect(names.slice(-4)).toEqual(["mito_aff_setup54_16nm_2", "nuc_setup56_128nm_2", `${FT}_2`, FT]);
    expect(new Set(names).size).toBe(names.length);
    expect((await catalog.resolve(FT))!.id).toBe(own.id);
  });
});
