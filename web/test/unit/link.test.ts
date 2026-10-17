// A link that opens a volume and runs models (src/link.ts): its query
// parsed, taken off the URL, and its models resolved through the catalog; a
// finetuned model's ref (its base and its head's numbers) written, read back
// and rebuilt in a browser that never trained it, without replacing a model
// of that name kept there; and a linked view without the layers this
// browser cannot show (paint painted elsewhere, models not served here).
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { describe, expect, it } from "vitest";

import { ARGS_KEY, type FinetunedRecord, type ModelEntry } from "../../src/contracts";
import { float16ArrayToFloat32, float32ArrayToFloat16, headOf, withHead } from "../../src/finetune/onnx-head";
import {
  chainFromHash, finetunedRef, isFinetunedRef, type LinkCatalog, type LinkedFinetuned, linkedHash, linkedView, linkNotes, paintVolumesIn,
  parseFinetunedRef, parseLink, resolveLinkedModels, sameHead, withoutLink,
} from "../../src/link";
import { createCatalog } from "../../src/models";
import { MemoryByteCache } from "../../src/models/byte-cache";
import { finetunedEntry, MemoryFinetunedRegistry } from "../../src/models/finetuned";
import { MemoryKeyValueStore } from "../../src/models/listing-cache";
import { entryFor } from "../../src/models/manifest";
import { encodeArgs } from "../../src/shared/args";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { paintLayerUrl } from "../../src/viewer/layers";
import { decodeState, encodeState } from "../../src/viewer/share";
import type { NgState } from "../../src/viewer/state";
import { fakeHub, fakeLog, type HubOptions } from "./models-helpers";

const DATASET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";

describe("parseLink", () => {
  it("reads the dataset and the models, repeated or comma-separated, each once", () => {
    expect(parseLink(`?dataset=${DATASET}&model=setup_53`)).toEqual({ datasetUrl: DATASET, models: ["setup_53"] });
    expect(parseLink("?model=setup_53&model=setup_54,setup_53&dataset=")).toEqual({ datasetUrl: null, models: ["setup_53", "setup_54"] });
    expect(parseLink("")).toEqual({ datasetUrl: null, models: [] });
    expect(parseLink("?other=1")).toEqual({ datasetUrl: null, models: [] });
  });

  it("takes the dataset URL encoded or plain", () => {
    expect(parseLink(`?dataset=${encodeURIComponent(DATASET)}`).datasetUrl).toBe(DATASET);
    expect(parseLink(`?dataset=${DATASET}`).datasetUrl).toBe(DATASET);
  });
});

describe("withoutLink", () => {
  it("drops the link's parameters and keeps the rest of the URL", () => {
    expect(withoutLink(`https://h/app/?dataset=${encodeURIComponent(DATASET)}&model=setup_53`)).toBe("https://h/app/");
    expect(withoutLink("https://h/app/?dataset=x&keep=1&model=a#!state")).toBe("https://h/app/?keep=1#!state");
    expect(withoutLink("https://h/app/")).toBe("https://h/app/");
  });
});

describe("chainFromHash", () => {
  const chain = {
    input_norm: [{ name: "MinMaxNormalizer", min_value: "0", max_value: "255" }],
    postprocess: [{ name: "ThresholdPostprocessor", threshold: "0.5" }],
  };
  const raw = { type: "image", source: "zarr://https://x/data.zarr/", name: "data" };

  it("reads the chain a model layer's view was drawn through", () => {
    const view = `mito_aff_setup53_16nm${ARGS_KEY}${encodeArgs(chain, { digest: "abc" })}${ARGS_KEY}`;
    const model = { type: "image", source: `zarr3://https://h/app/virtual/a30d4ff4/${view}/`, name: "mito_aff_setup53_16nm" };
    expect(chainFromHash(encodeState({ layers: [raw, model] }))).toEqual(chain);
  });

  it("is null without such a layer, or without a state", () => {
    expect(chainFromHash(encodeState({ layers: [raw] }))).toBeNull();
    expect(chainFromHash("")).toBeNull();
    expect(chainFromHash("#!not-a-state")).toBeNull();
  });
});

/** A catalog that only looks refs up: what a link of plain refs needs. */
function plainCatalog(resolve: (ref: string) => Promise<ModelEntry | null>): LinkCatalog {
  const never = () => Promise.reject(new Error("a plain ref needs no bytes"));
  return { resolve, modelBytes: never, finetuned: async () => [], addFinetuned: never };
}

describe("resolveLinkedModels with plain refs", () => {
  const setup53 = { id: "cellmap/onnx-web-models/setup_53", name: "setup_53" } as unknown as ModelEntry;

  it("resolves what the catalog knows, logs what it does not, and keeps each model once", async () => {
    const { log, lines } = fakeLog();
    const catalog = plainCatalog(async (ref) => (ref.endsWith("setup_53") ? setup53 : null));
    const linked = await resolveLinkedModels(catalog, ["setup_53", "nope", "cellmap/onnx-web-models/setup_53"], log);
    expect(linked).toEqual({ entries: [setup53], renamed: {}, leftOut: [{ name: "nope", why: "it is not a model this browser can run" }] });
    expect(lines).toEqual([expect.stringContaining("nope")]);
  });

  it("logs a lookup that fails and goes on", async () => {
    const { log, lines } = fakeLog();
    const catalog = plainCatalog(async (ref) => { if (ref === "bad") throw new Error("offline"); return setup53; });
    const linked = await resolveLinkedModels(catalog, ["bad", "setup_53"], log);
    expect(linked.entries).toEqual([setup53]);
    expect(linked.leftOut).toEqual([{ name: "bad", why: "it could not be looked up (offline)" }]);
    expect(lines).toEqual([expect.stringMatching(/bad.*offline/)]);
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
  const { log, lines } = fakeLog();
  const catalog = createCatalog({ log, fetch: hub.fetch, storage: new MemoryKeyValueStore(), bytes: new MemoryByteCache(), finetuned: new MemoryFinetunedRegistry() });
  const onnxCalls = () => hub.calls.filter((call) => call.endsWith(".onnx")).length;
  return { catalog, hub, log, lines, onnxCalls };
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

describe("finetunedRef and parseFinetunedRef", () => {
  it("round-trips a head exactly, every bit of every number", () => {
    const model = linkedModel();
    const ref = finetunedRef(model);
    expect(ref.startsWith(`${FT(1)}~setup_53~${FP32_140}~2x4~`)).toBe(true);
    const parsed = parseFinetunedRef(ref);
    expect({ name: parsed.name, base: parsed.base, file: parsed.file }).toEqual({ name: FT(1), base: "setup_53", file: FP32_140 });
    expect(Array.from(new Uint32Array(parsed.weight.buffer))).toEqual(Array.from(new Uint32Array(TRAINED.weight.buffer)));
    expect(Array.from(new Uint32Array(parsed.bias.buffer))).toEqual(Array.from(new Uint32Array(TRAINED.bias.buffer)));
    expect(sameHead(parsed, model)).toBe(true);
    expect(Object.is(parsed.weight[4], -0)).toBe(true);
  });

  it("needs no escaping in a URL, and is short: 10 numbers in 54 characters", () => {
    const ref = finetunedRef(linkedModel());
    expect(encodeURIComponent(ref)).toBe(ref);
    expect(ref.split("~")[4]).toHaveLength(54);
    expect(parseLink(`?dataset=${DATASET}&model=setup_54&model=${ref}`).models).toEqual(["setup_54", ref]);
  });

  it("leaves plain refs plain", () => {
    for (const ref of ["setup_53", "cellmap/onnx-web-models/setup_53", BASE_NAME, FT(1)]) expect(isFinetunedRef(ref)).toBe(false);
    expect(isFinetunedRef(finetunedRef(linkedModel()))).toBe(true);
  });

  it("refuses a ref cut short or edited", () => {
    const [name, base, file, shape, numbers] = finetunedRef(linkedModel()).split("~");
    const ref = (...parts: string[]) => parts.join("~");
    expect(() => parseFinetunedRef(ref(name, base, file, shape))).toThrow();
    // A shape the numbers do not fill, and numbers that are not base64.
    expect(() => parseFinetunedRef(ref(name, base, file, "3x4", numbers))).toThrow();
    expect(() => parseFinetunedRef(ref(name, base, file, shape, `${numbers.slice(0, -1)}+`))).toThrow();
  });

  it("refuses to write what could not be read back", () => {
    expect(() => finetunedRef({ ...linkedModel(), name: "a~b" })).toThrow(/cannot go in a link/);
    expect(() => finetunedRef({ ...linkedModel(), weight: new Float32Array(7) })).toThrow(/not \[out, in\]/);
    expect(() => finetunedRef({ ...linkedModel(), bias: new Float32Array(0) })).toThrow(/not \[out, in\]/);
  });
});

describe("resolveLinkedModels with a finetuned ref", () => {
  it("rebuilds it on the base's file the link names, keeps it with where it came from, and serves it in the link's order", async () => {
    const { catalog, log, lines } = browser();
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
    expect(catalog.metadataFor(entry).description).toMatch(/^from a shared link: mito_aff_setup53_16nm with the link's last layer, 2026-10-16; ONNX fp32/);
    expect(lines).toContain(`The link's ${FT(1)} is kept in this browser, rebuilt from ${BASE_NAME}`);
  });

  it("rebuilds an fp16 model exactly: every float16 value is a float32", async () => {
    const { catalog, log } = browser();
    const half = { weight: float16ArrayToFloat32(float32ArrayToFloat16(TRAINED.weight)), bias: float16ArrayToFloat32(float32ArrayToFloat16(TRAINED.bias)) };
    const linked = await resolveLinkedModels(catalog, [finetunedRef(linkedModel(FT(1), FP16_140, half))], log);
    const head = await headIn(catalog, linked.entries[0]);
    expect(head.dtype).toBe("float16");
    expect(sameHead(head, half)).toBe(true);
    expect(linked.entries[0].variant.variant).toBe("fp16");
  });

  it("serves the model already kept with the link's weights: opening a link twice keeps one, and downloads nothing", async () => {
    const { catalog, log, onnxCalls } = browser();
    const ref = finetunedRef(linkedModel());
    const first = await resolveLinkedModels(catalog, [ref], log);
    const calls = onnxCalls();
    const again = await resolveLinkedModels(catalog, [ref], log);
    expect(again.entries).toEqual(first.entries);
    expect(again.renamed).toEqual({});
    expect(await catalog.finetuned()).toHaveLength(1);
    expect(onnxCalls()).toBe(calls);
  });

  it("serves this browser's own model of that name when it has the link's weights", async () => {
    const { catalog, log } = browser();
    const own = await keepOwn(catalog, FT(1), TRAINED);
    const linked = await resolveLinkedModels(catalog, [finetunedRef(linkedModel())], log);
    expect(linked.entries).toEqual([own.entry]);
    expect(await catalog.finetuned()).toEqual([own]);
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
    expect(catalog.metadataFor(kept.entry).description).toMatch(/^from a shared link \(as mito_aff_setup53_16nm-ft1\): /);
    // Opened again, the link finds the model it was kept as.
    const again = await resolveLinkedModels(catalog, [ref], log);
    expect(again.entries.map((e) => e.name)).toEqual([FT(2)]);
    expect(again.renamed).toEqual({ [FT(1)]: FT(2) });
    expect((await catalog.finetuned()).map((r) => r.name).sort()).toEqual([FT(1), FT(2)]);
  });

  it("passes over the link's own names when it renames one, so the others keep theirs", async () => {
    const { catalog, log } = browser();
    await keepOwn(catalog, FT(1), OTHER);
    const second = { weight: TRAINED.weight.map((w) => w * 2), bias: TRAINED.bias };
    const linked = await resolveLinkedModels(catalog, [finetunedRef(linkedModel(FT(1))), finetunedRef(linkedModel(FT(2), FP32_140, second))], log);
    expect(linked.entries.map((e) => e.name)).toEqual([FT(3), FT(2)]);
    expect(linked.renamed).toEqual({ [FT(1)]: FT(3) });
    expect(sameHead(await headIn(catalog, linked.entries[1]), second)).toBe(true);
  });

  it("rebuilds on the base's default file when its manifest no longer has the link's, and says so", async () => {
    const { catalog, log, lines } = browser();
    const linked = await resolveLinkedModels(catalog, [finetunedRef(linkedModel(FT(1), "setup_53_mito_affinities_16nm_in999_fp32.onnx"))], log);
    expect(linked.entries[0].variant.input_size).toEqual([124, 124, 124]);
    expect((await catalog.finetuned())[0].base.file).toBe(DEFAULT_FILE);
    expect(lines).toContainEqual(expect.stringMatching(/in999_fp32\.onnx, which its manifest no longer lists; rebuilding it on .*in124_fp32\.onnx/));
  });

  it("leaves out what cannot be rebuilt, saying why, keeps nothing of it, and runs the rest", async () => {
    const { catalog, log, lines } = browser({ onnx: { [FP32_140]: HEAD_ONNX } });
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
    expect(lines.filter((l) => l.includes("is left out"))).toHaveLength(4);
  });
});

// ---------------------------------------------------------------------------
// The link's view in this browser
// ---------------------------------------------------------------------------

describe("linkedView, linkedHash and linkNotes", () => {
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

  it("names the paint volumes a view paints into", () => {
    expect(paintVolumesIn(state)).toEqual([HERE, ELSEWHERE]);
    expect(paintVolumesIn(null)).toEqual([]);
  });

  it("leaves out paint painted elsewhere and models not served here, renames a renamed model's layer, and keeps the rest", () => {
    const { state: shown, dropped } = linkedView(state, { ...here, volumes: new Set([HERE]) });
    expect(dropped).toEqual([{ name: `annotation_${ELSEWHERE}`, why: "paint", volume: ELSEWHERE }, { name: "mito_aff_setup54_16nm", why: "model" }]);
    expect(shown.layers).toEqual([raw, paintHere, served, { ...renamed, name: FT(2), source: { ...renamed.source, url: view(FT(2)) } }, regions]);
    expect(shown.selectedLayer).toEqual({ layer: FT(2), visible: true, side: "left" });
    expect(shown.position).toEqual([1, 2, 3]);
    // The renamed layer still names the chain it was drawn through.
    expect(chainFromHash(encodeState(shown))).toEqual({ input_norm: [], postprocess: [] });
  });

  it("leaves a view this browser can show whole as it is", () => {
    const whole: NgState = { layers: [raw, paintHere, served] };
    expect(linkedView(whole, { served: new Set([BASE_NAME]), renamed: {}, volumes: new Set([HERE]) })).toEqual({ state: whole, dropped: [] });
    expect(linkedView(whole, { served: new Set([BASE_NAME]), renamed: {}, volumes: new Set([HERE]) }).state).toBe(whole);
  });

  it("leaves out a second layer that a rename would give the name of the first", () => {
    const twice: NgState = { layers: [served, { ...served, name: FT(1), source: view(FT(1)) }] };
    const { state: shown, dropped } = linkedView(twice, { served: new Set([BASE_NAME]), renamed: { [FT(1)]: BASE_NAME }, volumes: new Set() });
    expect(shown.layers).toEqual([served]);
    expect(dropped).toEqual([{ name: FT(1), why: "model" }]);
  });

  it("asks the paint store which volumes this browser holds, and keeps the paint of one it does", async () => {
    const paint = new MemoryPaintStore();
    await paint.put(HERE, ".zattrs", new Uint8Array([123, 125]));
    const hash = encodeState(state);
    const { log, lines } = fakeLog();
    const result = await linkedHash(hash, { ...here, paint, log });
    expect(result.dropped.map((d) => d.name)).toEqual([`annotation_${ELSEWHERE}`, "mito_aff_setup54_16nm"]);
    expect(decodeState(result.hash)).toEqual(linkedView(state, { ...here, volumes: new Set([HERE]) }).state);
    expect(lines).toEqual(linkNotes({ renamed: {}, leftOut: [] }, result.dropped));
    // The sender's own tab: both volumes are here.
    await paint.put(ELSEWHERE, "annotation/s0/0.0.0", new Uint8Array(1));
    const own = await linkedHash(hash, { served: new Set([BASE_NAME, FT(1), "mito_aff_setup54_16nm"]), renamed: {}, paint, log });
    expect(own).toEqual({ hash, dropped: [] });
    // A hash without a state is not a view to trim.
    expect(await linkedHash("", { ...here, paint, log })).toEqual({ hash: "", dropped: [] });
    expect(await linkedHash("#!https://h/state.json", { ...here, paint, log })).toEqual({ hash: "#!https://h/state.json", dropped: [] });
    expect(lines).toHaveLength(2);
  });

  it("says what was left out and renamed, and nothing for a link that opened whole", () => {
    const { dropped } = linkedView(state, { ...here, volumes: new Set([HERE]) });
    // A note each, naming it: the model left out, the name it is served as, the paint's layer, the model's layer.
    expect(linkNotes({ renamed: here.renamed, leftOut: [{ name: "setup_99", why: "it is not a model this browser can run" }] }, dropped)).toEqual(
      ["setup_99", FT(2), `annotation_${ELSEWHERE}`, "mito_aff_setup54_16nm"].map((name) => expect.stringContaining(name)),
    );
    expect(linkNotes({ renamed: {}, leftOut: [] }, [])).toEqual([]);
  });
});
