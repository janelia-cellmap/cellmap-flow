// The Share button's link (src/share-link.ts): the volume and the models in
// the query, the view in the hash, nothing without a volume; a finetuned
// model's ref built from its bytes and its lineage's records; and what the
// sender is told the link leaves behind.
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { describe, expect, it } from "vitest";

import type { FinetunedRecord, ModelEntry } from "../../src/contracts";
import { headOf, withHead } from "../../src/finetune/onnx-head";
import { finetunedRef, parseFinetunedRef, parseLink, sameHead } from "../../src/link";
import { finetunedEntry } from "../../src/models/finetuned";
import { linkRefs, shareLink, shareNotes } from "../../src/share-link";
import { encodeState } from "../../src/viewer/share";
import { entryFixture } from "./models-helpers";

const PAGE = "https://janelia-cellmap.github.io/cellmap-flow/app/";
const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
const MODELS = ["setup_53", "setup_54"];

describe("shareLink", () => {
  it("is nothing without a volume", () => {
    expect(shareLink(PAGE, null, MODELS, null)).toBeNull();
  });

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

  it("escapes what would end or split the query, and reads back the same", () => {
    const odd = "https://h/a b.zarr/x&y#z+w";
    const link = shareLink(PAGE, odd, [], null)!;
    expect(link).not.toMatch(/[ #+]/);
    expect(parseLink(new URL(link).search).datasetUrl).toBe(odd);
  });

  it("puts a finetuned model's ref in its place among the others, as it is, and parseLink reads it back the same", () => {
    const ref = finetunedRef({ name: "mito_aff_setup53_16nm-ft1", base: "setup_53", file: "x_fp32.onnx", weight: Float32Array.from([0.1, 0.2]), bias: Float32Array.from([0.3]) });
    const link = shareLink(PAGE, DATASET, ["setup_54", ref, "setup_53"], null)!;
    expect(link).toBe(`${PAGE}?dataset=${DATASET}&model=setup_54&model=${ref}&model=setup_53`);
    expect(parseLink(new URL(link).search).models).toEqual(["setup_54", ref, "setup_53"]);
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
  // Finetuned from a model since removed: the lineage is broken.
  const orphan = record(finetunedEntry(hub, "mito_aff_setup53_16nm-ft7"), "mito_aff_setup53_16nm-ft8");
  const bytes: Record<string, ArrayBuffer> = {
    [ft1.id]: withHead(HEAD_ONNX, headOf(HEAD_ONNX), heads.ft1.weight, heads.ft1.bias),
    [ft2.id]: withHead(HEAD_ONNX, headOf(HEAD_ONNX), heads.ft2.weight, heads.ft2.bias),
  };
  const catalog = {
    finetuned: async () => [ft2, ft1, orphan],
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

  it("names a finetuned model it cannot carry as it is, and says why", async () => {
    const gone = record(hub, "mito_aff_setup53_16nm-ft9");
    const linked = await linkRefs([orphan.entry, gone.entry], { ...catalog, finetuned: async () => [orphan, gone] });
    expect(linked.refs).toEqual([orphan.name, gone.name]);
    expect(linked.rebuilt).toEqual([]);
    expect(linked.unlinked).toEqual([
      { name: orphan.name, why: "mito_aff_setup53_16nm-ft7, the model it was finetuned from, is no longer kept in this browser" },
      { name: gone.name, why: "mito_aff_setup53_16nm-ft9's bytes are no longer in the browser's storage" },
    ]);
  });
});

describe("shareNotes", () => {
  it("says the paint stays here and how to take it along, and where a finetuned model is rebuilt", () => {
    const notes = shareNotes(["vol-0badcafe-20261016-101500"], {
      rebuilt: [{ name: "mito_aff_setup53_16nm-ft1", base: "mito_aff_setup53_16nm" }],
      unlinked: [{ name: "mito_aff_setup53_16nm-ft8", why: "its record is not in this browser's catalog" }],
    });
    expect(notes).toEqual([
      "The paint in vol-0badcafe-20261016-101500 stays in this browser: someone else opening the link will not see it. The Finetune tab's Download paint (zarr) saves it as a .zarr.zip to send with the link.",
      "mito_aff_setup53_16nm-ft1's last layer is in the link: it is rebuilt from mito_aff_setup53_16nm where the link opens.",
      "mito_aff_setup53_16nm-ft8 could not be put in the link as a finetuned model (its record is not in this browser's catalog): it opens only in this browser.",
    ]);
  });

  it("says nothing of a link that opens elsewhere as it is here", () => {
    expect(shareNotes([], { rebuilt: [], unlinked: [] })).toEqual([]);
  });
});
