// A model opened from a file (src/model-file.ts): its manifest read from the
// .onnx's metadata_props, or from a manifest.json picked with it; the model
// kept in the catalog as one of this browser's, on the next visit too and
// once however often it is opened; and a link, which cannot name it, saying so.
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { describe, expect, it } from "vitest";

import { MANIFEST_METADATA_KEY } from "../../src/contracts";
import { withMetadata } from "../../src/finetune/onnx-proto";
import { openModelFile } from "../../src/model-file";
import { createCatalog } from "../../src/models";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { linkRefs } from "../../src/share-link";
import { entryFixture, fakeHub, rawManifest } from "./models-helpers";

/** A 4 → 2 head on a small network (make_onnx_fixture.py): an ONNX model with no manifest in it. */
const HEAD_ONNX = new Uint8Array(readFileSync(resolvePath(import.meta.dirname, "../fixtures/head.onnx")));
const NOW = () => new Date("2026-10-26T12:00:00.000Z");

function browser(kept = { storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry() }) {
  return { catalog: createCatalog({ log: () => {}, fetch: fakeHub().fetch, ...kept }), kept };
}

describe("openModelFile", () => {
  it("keeps an .onnx that carries its manifest as a model of this browser, on the next visit too, once; a link leaves it here", async () => {
    // What Download finetuned model saves: the bytes, with the entry's manifest in them.
    const downloaded = finetunedEntry(entryFixture("setup_54"), "mito_aff_setup54_16nm-ft3");
    const file = { name: "mito_aff_setup54_16nm-ft3.onnx", bytes: withMetadata(HEAD_ONNX, { [MANIFEST_METADATA_KEY]: JSON.stringify(downloaded.manifest) }).slice().buffer };
    const { catalog, kept } = browser();
    const entry = await openModelFile(catalog, file, null, NOW);
    expect(entry).toMatchObject({ id: "finetuned/mito_aff_setup54_16nm-ft3", name: "mito_aff_setup54_16nm-ft3", repo: "finetuned" });
    // The Hugging Face file it was made from, so that it finetunes again on that file's finetune block.
    expect(entry.variant.source).toEqual(downloaded.variant.source);
    expect(await openModelFile(catalog, file, null, NOW)).toEqual(entry);
    expect(await catalog.finetuned()).toHaveLength(1);

    const next = browser(kept).catalog;
    const listed = await next.resolve(entry.name);
    expect(listed?.id).toBe(entry.id);
    expect(new Uint8Array(await next.modelBytes(listed!))).toEqual(new Uint8Array(file.bytes));
    const linked = await linkRefs([listed!], next);
    expect(linked.refs).toEqual([entry.folder]);
    expect(linked.unlinked.map((u) => u.name)).toEqual([entry.name]);
  });

  it("goes by a manifest.json picked with an .onnx that carries none, and refuses the .onnx alone", async () => {
    const { catalog } = browser();
    const variant = entryFixture("setup_54").variant;
    const onnx = { name: variant.file, bytes: HEAD_ONNX.slice().buffer };
    await expect(openModelFile(catalog, onnx, null, NOW)).rejects.toThrow();
    expect(await catalog.finetuned()).toEqual([]);
    const manifest = { name: "manifest.json", bytes: new TextEncoder().encode(JSON.stringify(rawManifest("setup_54"))).buffer };
    const entry = await openModelFile(catalog, onnx, manifest, NOW);
    // Named after the file, as the manifest's variant of that file.
    expect(entry.name).toBe(variant.file.replace(/\.onnx$/, ""));
    expect(entry.manifest.variants).toHaveLength(1);
    expect([entry.variant.variant, entry.variant.input_size]).toEqual([variant.variant, variant.input_size]);
  });
});
