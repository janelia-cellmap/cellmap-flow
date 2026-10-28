// A link that opens a volume and runs models (src/link.ts): its query
// parsed and its models resolved through the catalog; a finetuned model's
// ref (its base and its head's numbers) written, read back and rebuilt in a
// browser that never trained it, without replacing a model of that name kept
// there; and a linked view without the layers this browser cannot show
// (paint painted elsewhere, models not served here).
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { describe, expect, it } from "vitest";

import { createBackend, restoreSession } from "../../src/backend";
import { ARGS_KEY, type FinetunedRecord, type ModelEntry } from "../../src/contracts";
import { headOf, withHead } from "../../src/finetune/onnx-head";
import { chainFromHash, finetunedRef, type LinkCatalog, type LinkedFinetuned, linkedView, parseLink, resolveLinkedModels, sameHead } from "../../src/link";
import { createCatalog, NoBrowserCopyError } from "../../src/models";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFor } from "../../src/models/manifest";
import { encodeArgs } from "../../src/shared/args";
import { paintLayerUrl } from "../../src/viewer/layers";
import { encodeState } from "../../src/viewer/share";
import type { NgState } from "../../src/viewer/state";
import { fakeContext, request } from "./backend-fakes";
import { entryFixture, fakeHub, fakeLog, type FakeRepo, type HubOptions, publishedManifest, rawManifest } from "./models-helpers";

const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";

describe("parseLink", () => {
  it("reads the dataset and the models, repeated or comma-separated, each once", () => {
    expect(parseLink(`?dataset=${DATASET}&model=setup_53`)).toEqual({ datasetUrl: DATASET, models: ["setup_53"] });
    expect(parseLink("?model=setup_53&model=setup_54,setup_53&dataset=")).toEqual({ datasetUrl: null, models: ["setup_53", "setup_54"] });
    expect(parseLink("")).toEqual({ datasetUrl: null, models: [] });
    expect(parseLink("?other=1")).toEqual({ datasetUrl: null, models: [] });
  });
});

describe("a link that names a model and no volume", () => {
  it("puts another repo's model in the session, which is served once a volume is opened", async () => {
    const ALICE = "alice/mito-ft1", FT = "mito_aff_setup54_16nm-ft1";
    const { catalog } = browser({ repos: { [ALICE]: { manifests: { [FT]: publishedManifest(entryFixture("setup_54"), FT) } } }, onnx: { [`${FT}.onnx`]: HEAD_ONNX } });
    // "Try it in the browser", as a published model's card links it.
    const link = parseLink(`?model=${ALICE}/${FT}`);
    expect(link.datasetUrl).toBeNull();
    const linked = await resolveLinkedModels(catalog, link.models, () => {});
    // As main.ts does: the models into the session; then Connect, and the session's restore after its reload.
    const ctx = { ...fakeContext({ dataset: null, state: { models: linked.entries } }), catalog };
    expect((await createBackend(ctx).handle(request("POST", "/api/set-data", { dataset_path: DATASET }))).status).toBe(200);
    await restoreSession(ctx);
    expect(ctx.infer.served().map((s) => s.entry.id)).toEqual([`${ALICE}/${FT}`]);
  }, 60_000);  // the serve reads what the network ends in in node's Pyodide, which loads the first time
});

describe("a link to a model a server published without a copy for the browser", () => {
  it("leaves it out saying why, and names its smaller copy once the repo has it", async () => {
    // The server form at the root (finetune/export_web.py), and no folder:
    // the network was too big for an .onnx file. Its smaller copy comes later.
    const REPO = "ada/mito-ft", FOLDER = "mito_aff_unet_setup_16_finetuned", SMALL = `${FOLDER}-small`;
    const server = ["metadata.json", "model.ts", "model.pt2", "export_record.json"];
    const catalogOf = (repo: FakeRepo) => browser({ repos: { [REPO]: repo } }).catalog;
    const refs = [REPO, `${REPO}/${FOLDER}`, `${REPO}/${SMALL}`];

    // While the smaller copy trains: each ref is left out, each saying why itself.
    const training = catalogOf({ files: server });
    for (const ref of refs) await expect(training.resolve(ref)).rejects.toBeInstanceOf(NoBrowserCopyError);
    const linked = await resolveLinkedModels(training, refs, () => {});
    expect(linked.entries).toEqual([]);
    expect(linked.leftOut.map((m) => m.name)).toEqual(refs);

    // Once it is uploaded: the repo and the smaller copy open it, and the model's folder points to it.
    const done = catalogOf({ files: server, manifests: { [SMALL]: rawManifest("setup_54") } });
    expect((await done.resolve(REPO))!.id).toBe(`${REPO}/${SMALL}`);
    expect((await done.resolve(`${REPO}/${SMALL}`))!.id).toBe(`${REPO}/${SMALL}`);
    const [leftOut] = (await resolveLinkedModels(done, [`${REPO}/${FOLDER}`], () => {})).leftOut;
    expect(leftOut!.why).toContain(`${REPO}/${SMALL}`);
  });
});

// ---------------------------------------------------------------------------
// A finetuned model in the link
// ---------------------------------------------------------------------------

const FIXTURES = resolvePath(import.meta.dirname, "../fixtures");
/** A 4 → 2 head on a small network (make_onnx_fixture.py), as float32 and as float16. */
const HEAD_ONNX = new Uint8Array(readFileSync(resolvePath(FIXTURES, "head.onnx")));
const HEAD_FP16_ONNX = new Uint8Array(readFileSync(resolvePath(FIXTURES, "head_fp16.onnx")));
/** setup_53's files the tests serve: two the links name, and the one the catalog picks without a GPU's limits. */
const FP32_140 = "setup_53_mito_affinities_16nm_in140_fp32.onnx";
const FP16_140 = "setup_53_mito_affinities_16nm_in140_fp16.onnx";
const DEFAULT_FILE = "setup_53_mito_affinities_16nm_in124_fp32.onnx";
const BASE_NAME = "mito_aff_setup53_16nm";
const FT = (n: number) => `${BASE_NAME}-ft${n}`;
const NOW = () => new Date("2026-10-16T09:00:00.000Z");

/** A trained head: values that decimals would round, a negative zero, a tiny one and a huge one. */
const TRAINED = { weight: Float32Array.from([0.1, -2.5, 3e-7, 1234.5678, -0, 0.75, Math.PI, -1e-38]), bias: Float32Array.from([0.25, -0.125]) };
/** Another browser's head of the same shape. */
const OTHER = { weight: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), bias: Float32Array.from([0.5, -0.5]) };

function linkedModel(name = FT(1), file = FP32_140, head: { weight: Float32Array; bias: Float32Array } = TRAINED): LinkedFinetuned {
  return { name, base: "setup_53", file, weight: head.weight, bias: head.bias };
}

/** A browser that opens links: the real catalog over the Hub's fixtures, serving the head fixture as setup_53's files. */
function browser(options: HubOptions = { onnx: { [FP32_140]: HEAD_ONNX, [DEFAULT_FILE]: HEAD_ONNX, [FP16_140]: HEAD_FP16_ONNX } }) {
  const hub = fakeHub(options);
  const { log } = fakeLog();
  const catalog = createCatalog({ log, fetch: hub.fetch, storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry() });
  return { catalog, log };
}

/** Keep a model finetuned in `catalog`'s browser itself, on setup_53's `file`, with `head`. */
async function keepOwn(catalog: LinkCatalog, name: string, head: { weight: Float32Array; bias: Float32Array }, file = FP32_140): Promise<FinetunedRecord> {
  const hub = (await catalog.resolve("setup_53"))!;
  const source = entryFor(hub.folder, hub.manifest, hub.manifest.variants.find((v) => v.file === file)!, hub.repo);
  const entry = finetunedEntry(source, name);
  const record: FinetunedRecord = {
    id: entry.id, name, base: { id: source.id, name: source.name, file }, volume: "vol-0badcafe-20261015-101500",
    created_at: "2026-10-15T10:15:00.000Z", params: { epochs: 20 }, epochs: 20, final_loss: 0.3, entry,
  };
  await catalog.addFinetuned(record, withHead(HEAD_ONNX, headOf(HEAD_ONNX), head.weight, head.bias));
  return record;
}

/** The head in a kept model's bytes. */
async function headIn(catalog: LinkCatalog, entry: ModelEntry) {
  return headOf(await catalog.modelBytes(entry));
}

describe("resolveLinkedModels with a finetuned ref", () => {
  it("rebuilds it on the base's file the link names, keeps it with where it came from, and serves it in the link's order", async () => {
    const { catalog, log } = browser();
    const linked = await resolveLinkedModels(catalog, ["setup_54", finetunedRef(linkedModel()), "setup_53"], log, { now: NOW });
    expect(linked.entries.map((e) => e.name)).toEqual(["mito_aff_setup54_16nm", FT(1), BASE_NAME]);
    expect(linked.renamed).toEqual({});
    expect(linked.leftOut).toEqual([]);
    const entry = linked.entries[1];
    expect(entry).toMatchObject({ id: `finetuned/${FT(1)}`, repo: "finetuned", variant: { variant: "fp32", input_size: [140, 140, 140] } });
    expect(sameHead(await headIn(catalog, entry), TRAINED)).toBe(true);
    const [record] = await catalog.finetuned();
    expect(record).toMatchObject({
      name: FT(1), base: { id: "cellmap/onnx-web-models/setup_53", name: BASE_NAME, file: FP32_140 }, volume: "", epochs: 0, final_loss: null,
      created_at: "2026-10-16T09:00:00.000Z", from_link: { name: FT(1) },
    });
  });

  it("never replaces this browser's own model of that name: the link's takes the series' next name, once", async () => {
    const { catalog, log } = browser();
    const own = await keepOwn(catalog, FT(1), OTHER);
    const ref = finetunedRef(linkedModel());
    const linked = await resolveLinkedModels(catalog, [ref], log, { now: NOW });
    expect(linked.entries.map((e) => e.name)).toEqual([FT(2)]);
    expect(linked.renamed).toEqual({ [FT(1)]: FT(2) });
    expect(sameHead(await headIn(catalog, own.entry), OTHER)).toBe(true);
    expect(sameHead(await headIn(catalog, linked.entries[0]), TRAINED)).toBe(true);
    const kept = (await catalog.finetuned()).find((r) => r.name === FT(2))!;
    expect(kept.from_link).toEqual({ name: FT(1) });
    // Opened again, the link finds the model it was kept as.
    const again = await resolveLinkedModels(catalog, [ref], log);
    expect(again.entries.map((e) => e.name)).toEqual([FT(2)]);
    expect(again.renamed).toEqual({ [FT(1)]: FT(2) });
    expect((await catalog.finetuned()).map((r) => r.name).sort()).toEqual([FT(1), FT(2)]);
  });

  it("leaves out what cannot be rebuilt, saying why, keeps nothing of it, and runs the rest", async () => {
    const { catalog, log } = browser({ onnx: { [FP32_140]: HEAD_ONNX } });
    const wide = { weight: new Float32Array(12), bias: new Float32Array(3) };
    const refs = [
      finetunedRef({ ...linkedModel(FT(1)), base: "setup_99" }),
      finetunedRef(linkedModel(FT(2), FP32_140, wide)),
      finetunedRef(linkedModel(FT(3), FP16_140)),  // the Hub answers 404 for it
      `${FT(4)}~setup_53~cut-short`,
      "setup_53",
    ];
    const linked = await resolveLinkedModels(catalog, refs, log);
    expect(linked.entries.map((e) => e.name)).toEqual([BASE_NAME]);
    expect(linked.leftOut.map((l) => l.name)).toEqual([FT(1), FT(2), FT(3), FT(4)]);
    // Each says why: the base it names, the head's shape, the Hub's answer, the ref that cannot be read.
    expect(linked.leftOut.map((l) => l.why)).toEqual([expect.stringContaining("setup_99"), expect.stringContaining("3×4"), expect.stringContaining("404"), expect.any(String)]);
    expect(await catalog.finetuned()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// The link's view in this browser
// ---------------------------------------------------------------------------

describe("linkedView", () => {
  const PAGE = "https://janelia-cellmap.github.io/cellmap-flow/app/";
  const HERE = "vol-0badcafe-20261015-101500", ELSEWHERE = "vol-1a2b3c4d-20261016-083000";
  const view = (name: string) => `zarr3://${PAGE}virtual/a30d4ff4/${name}${ARGS_KEY}${encodeArgs({ input_norm: [], postprocess: [] })}${ARGS_KEY}/`;
  const raw = { type: "image", name: "data", source: "zarr://https://bucket.s3.amazonaws.com/paint/raw.zarr/s0" };
  const paintHere = { type: "segmentation", name: `annotation_${HERE}`, source: { url: paintLayerUrl(HERE, PAGE), subsources: { default: { writingEnabled: true } } } };
  const paintElsewhere = { type: "segmentation", name: `annotation_${ELSEWHERE}`, source: `${paintLayerUrl(ELSEWHERE, PAGE).replace(ELSEWHERE, `${ELSEWHERE}~2`)}` };
  const served = { type: "image", name: BASE_NAME, source: view(BASE_NAME), shader: "#uicontrol vec3 color" };
  const renamed = { type: "image", name: FT(1), source: { url: view(FT(1)), transform: { outputDimensions: { z: [1.6e-8, "m"] } } }, visible: false };
  const notServed = { type: "image", name: "mito_aff_setup54_16nm", source: view("mito_aff_setup54_16nm") };
  const regions = { type: "annotation", name: "good_regions", source: [{ url: "local://annotations" }], annotations: [] };
  const state: NgState = {
    position: [1, 2, 3],
    layers: [raw, paintHere, paintElsewhere, served, renamed, notServed, regions],
    selectedLayer: { layer: FT(1), visible: true, side: "left" },
  };
  const here = { served: new Set([BASE_NAME, FT(2)]), renamed: { [FT(1)]: FT(2) } };

  it("leaves out paint painted elsewhere and models not served here, renames a renamed model's layer, and keeps the rest", () => {
    const { state: shown, dropped } = linkedView(state, { ...here, volumes: new Set([HERE]) });
    expect(dropped).toEqual([{ name: `annotation_${ELSEWHERE}`, why: "paint", volume: ELSEWHERE }, { name: "mito_aff_setup54_16nm", why: "model" }]);
    expect(shown.layers).toEqual([raw, paintHere, served, { ...renamed, name: FT(2), source: { ...renamed.source, url: view(FT(2)) } }, regions]);
    expect(shown.selectedLayer).toEqual({ layer: FT(2), visible: true, side: "left" });
    expect(shown.position).toEqual([1, 2, 3]);
    // The renamed layer still names the chain it was drawn through.
    expect(chainFromHash(encodeState(shown))).toEqual({ input_norm: [], postprocess: [] });
  });
});
