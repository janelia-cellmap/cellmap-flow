// The Share button's link (src/share-link.ts): the volume and the models in
// the query, the view in the hash; and a finetuned model's ref built from its
// bytes and its lineage's records.
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { describe, expect, it } from "vitest";

import type { FinetunedRecord, ModelEntry } from "../../src/contracts";
import { headOf, withHead } from "../../src/finetune/onnx-head";
import { parseFinetunedRef, parseLink, resolveLinkedModels, sameHead } from "../../src/link";
import { createCatalog } from "../../src/models";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { linkRefs, shareLink } from "../../src/share-link";
import { encodeState } from "../../src/viewer/share";
import { entryFixture, fakeHub, publishedManifest } from "./models-helpers";

const PAGE = "https://janelia-cellmap.github.io/cellmap-flow/app/";
const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
const MODELS = ["setup_53", "setup_54"];

describe("shareLink", () => {
  it("names the volume and the models readably, and parseLink reads them back", () => {
    const link = shareLink(PAGE, DATASET, MODELS, null)!;
    expect(link).toBe(`${PAGE}?dataset=${DATASET}&model=setup_53&model=setup_54`);
    expect(parseLink(new URL(link).search)).toEqual({ datasetUrl: DATASET, models: ["setup_53", "setup_54"] });
  });

  it("carries the viewer's state hash, and drops the page's own query and hash", () => {
    const hash = encodeState({ position: [1, 2, 3], layers: [] });
    expect(shareLink(`${PAGE}?x=1#!old`, DATASET, [], `${PAGE}ng/index.html${hash}`)).toBe(`${PAGE}?dataset=${DATASET}${hash}`);
    expect(shareLink(PAGE, DATASET, [], `${PAGE}ng/index.html`)).toBe(`${PAGE}?dataset=${DATASET}`);
  });
});

describe("linkRefs", () => {
  const HEAD_ONNX = new Uint8Array(readFileSync(resolvePath(import.meta.dirname, "../fixtures/head.onnx")));
  const hub = entryFixture("setup_53");
  const heads = {
    ft1: { weight: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), bias: Float32Array.from([0.5, -0.5]) },
    ft2: { weight: Float32Array.from([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]), bias: Float32Array.from([0.125, 0.25]) },
  };
  const record = (base: ModelEntry, name: string): FinetunedRecord => {
    const entry = finetunedEntry(base, name);
    return { id: entry.id, name, base: { id: base.id, name: base.name, file: base.variant.file }, volume: "vol-x", created_at: "2026-10-16T08:00:00.000Z", params: {}, epochs: 20, final_loss: 0.2, entry };
  };
  const ft1 = record(hub, "mito_aff_setup53_16nm-ft1");
  // Finetuned from ft1: its base is a finetuned model, whose base is the Hub's.
  const ft2 = record(ft1.entry, "mito_aff_setup53_16nm-ft2");
  const bytes: Record<string, ArrayBuffer> = {
    [ft1.id]: withHead(HEAD_ONNX, headOf(HEAD_ONNX), heads.ft1.weight, heads.ft1.bias),
    [ft2.id]: withHead(HEAD_ONNX, headOf(HEAD_ONNX), heads.ft2.weight, heads.ft2.bias),
  };
  const catalog = {
    finetuned: async () => [ft2, ft1],
    modelBytes: async (entry: ModelEntry) => {
      const kept = bytes[entry.id];
      if (!kept) throw new Error(`${entry.name}'s bytes are no longer in the browser's storage`);
      return kept;
    },
    resolve: async (ref: string) => (ref === hub.id ? hub : null),
  };

  it("names a Hugging Face model by its folder and a finetuned one by its head, on the root of its lineage", async () => {
    const linked = await linkRefs([hub, ft2.entry, ft1.entry], catalog);
    expect(linked.refs[0]).toBe("setup_53");
    const two = parseFinetunedRef(linked.refs[1]);
    expect({ name: two.name, base: two.base, file: two.file }).toEqual({ name: ft2.name, base: "setup_53", file: hub.variant.file });
    expect(sameHead(two, heads.ft2)).toBe(true);
    expect(sameHead(parseFinetunedRef(linked.refs[2]), heads.ft1)).toBe(true);
    expect(linked.rebuilt).toEqual([{ name: ft2.name, base: hub.name }, { name: ft1.name, base: hub.name }]);
    expect(linked.unlinked).toEqual([]);
  });

  it("names another repo's model by its id, a published finetune where it was published and a finetune of another repo's model on it, and another browser opens all three", async () => {
    const ALICE = "alice/mito-ft1", A_FT = "mito_aff_setup54_16nm-ft1";
    const CAROL = "carol/mito-full", C_FT = "mito_aff_setup53_16nm-ft1";
    const repos = {
      [ALICE]: { manifests: { [A_FT]: publishedManifest(entryFixture("setup_54"), A_FT) } },
      [CAROL]: { manifests: { [C_FT]: publishedManifest(hub, C_FT) } },
    };
    const onnx = { [`${A_FT}.onnx`]: HEAD_ONNX, [`${C_FT}.onnx`]: HEAD_ONNX };
    const browser = () => createCatalog({ log: () => {}, fetch: fakeHub({ repos, onnx }).fetch, storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry() });

    // The sender: alice's model, a last-layer finetune of it, and a whole network it trained and published to carol's repo.
    const sender = browser();
    const alice = (await sender.resolve(`${ALICE}/${A_FT}`))!;
    const onAlice = record(alice, "mito_aff_setup54_16nm-ft2");
    await sender.addFinetuned(onAlice, withHead(HEAD_ONNX, headOf(HEAD_ONNX), heads.ft2.weight, heads.ft2.bias));
    const full: FinetunedRecord = { ...record(hub, C_FT), params: { finetune_mode: "full" }, published: { repo: CAROL, folder: C_FT, at: "2026-10-26T12:00:00.000Z" } };
    await sender.addFinetuned(full, HEAD_ONNX.slice().buffer);
    const linked = await linkRefs([alice, onAlice.entry, full.entry], sender);
    expect(linked.unlinked).toEqual([]);
    expect([linked.refs[0], parseFinetunedRef(linked.refs[1]).base, linked.refs[2]]).toEqual([`${ALICE}/${A_FT}`, `${ALICE}/${A_FT}`, `${CAROL}/${C_FT}`]);

    const { models } = parseLink(new URL(shareLink(PAGE, DATASET, linked.refs, null)!).search);
    const recipient = browser();
    const opened = await resolveLinkedModels(recipient, models, () => {});
    expect(opened.leftOut).toEqual([]);
    expect(opened.entries.map((e) => [e.repo, e.name])).toEqual([[ALICE, A_FT], ["finetuned", onAlice.name], [CAROL, C_FT]]);
    expect(sameHead(headOf(await recipient.modelBytes(opened.entries[1])), heads.ft2)).toBe(true);
  });
});
