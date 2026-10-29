// The /api/* routes against fake context members: the JSON each answers,
// checked against what the dashboard's JavaScript reads (models-tab.js,
// main.js, connect.js, model-advice.js).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { createBackend, restoreSession } from "../../src/backend";
import { toYaml } from "../../src/backend/yaml";
import { ARGS_KEY } from "../../src/contracts";
import type { ApiResponse, FinetunedRecord, Step } from "../../src/contracts";
import { applyPostprocessors } from "../../src/infer/post";
import { finetunedEntry } from "../../src/models/finetuned";
import { decodeArgs } from "../../src/shared/args";
import { defaultChain } from "../../src/state";
import { rawLayer } from "../../src/viewer/layers";
import { DATASET, entry, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";
import { portablePython } from "./pyodide";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

/** An exported config without its comments (the header's lines, a path
 * line's note): the config itself, as toYaml writes it. */
function uncommented(yaml: string): string {
  return yaml.split("\n").filter((line) => !line.startsWith("#")).map((line) => line.replace(/ +#.*$/, "")).join("\n");
}

/** setup_54 as the server's onnx entry, running `file`. */
const SERVER_54 = (file: string) => ({ type: "onnx", name: NAME_54, repo: "cellmap/onnx-web-models", folder: "setup_54", file });

const HF54 = "cellmap/onnx-web-models/setup_54";
const HF56 = "cellmap/onnx-web-models/setup_56";

function submit(models: string[], extra: Record<string, unknown> = {}) {
  return request("POST", "/api/models", {
    selected_models: [],
    selected_hf_models: models,
    selected_bioimage_models: [],
    selected_cellpose_models: [],
    resample: false,
    ...extra,
  });
}

describe("POST /api/set-data", () => {
  it("opens the URL in the engine, prepares the viewer for the reload, keeps it, and answers the viewer's URL", async () => {
    const ctx = fakeContext({ dataset: null });
    const res = await createBackend(ctx).handle(request("POST", "/api/set-data", { dataset_path: " https://example.org/em.zarr " }));
    expect(res.status).toBe(200);
    expect(body(res)).toEqual({ success: true, neuroglancer_url: "http://localhost/ng/#!state" });
    expect(ctx.infer.opened).toEqual(["https://example.org/em.zarr"]);
    expect(ctx.viewer.prepared[0].url).toBe("https://example.org/em.zarr");
    expect(ctx.viewer.prepared[0].info?.url).toBe("https://example.org/em.zarr"); // the volume's info, for the viewer's dimensions and position
    expect(ctx.viewer.openedWith).toEqual([]); // not shown before connect.js reloads the page
    expect(ctx.session.state.datasetUrl).toBe("https://example.org/em.zarr");
  });

  it("answers a 500 {error} when the dataset cannot be opened, as the Python route does", async () => {
    const ctx = fakeContext({ dataset: null });
    ctx.infer.failOpen = "no zarr.json at that URL";
    const res = await createBackend(ctx).handle(request("POST", "/api/set-data", { dataset_path: "https://example.org/nothing" }));
    expect(res.status).toBe(500);
    expect(body(res)).toEqual({ error: "no zarr.json at that URL" });
    expect(ctx.session.state.datasetUrl).toBeNull();
  });

  it("explains a fetch that got no answer: unreachable, CORS, or a private network the site may not use", async () => {
    const ctx = fakeContext({ dataset: null });
    ctx.infer.failOpen = "Failed to fetch";
    const res = await createBackend(ctx).handle(request("POST", "/api/set-data", { dataset_path: "https://data.example.org/em.zarr" }));
    expect(res.status).toBe(500);
    expect(body(res).error).toMatch(/CORS/);
  });
});

describe("POST /api/models", () => {
  it("serves the ticked models through the session's chain and echoes the selection", async () => {
    const ctx = fakeContext();
    const res = await createBackend(ctx).handle(submit([HF54, HF56]));
    expect(res.status).toBe(200);
    expect(body(res)).toEqual({ message: "Data received successfully", models: [], hf_models: [HF54, HF56], bioimage_models: [], cellpose_models: [] });
    expect(ctx.infer.serves.map((s) => s.name)).toEqual([NAME_54, NAME_56]);
    // the blob is the session's chain with the dashboard's address and the digest
    const decoded = decodeArgs(ctx.infer.serves[0].blob);
    expect(decoded.chain).toEqual(defaultChain());
    expect(decoded.extras.dashboard_url).toBe("http://localhost/");
    expect(typeof decoded.extras.digest).toBe("string");
    // one layer per model, and the models kept in the session for the next load
    expect(Object.keys(ctx.viewer.layers)).toEqual([NAME_54, NAME_56]);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_54, NAME_56]);
  });

  it("disposes of a model that is unticked and removes its layer, keeping the others as they are", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54, HF56]));
    await backend.handle(submit([HF56]));
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56]);
    expect(ctx.infer.disposed).toHaveLength(1);
    expect(ctx.infer.disposed[0].startsWith(`${NAME_54}${ARGS_KEY}`)).toBe(true);
    expect(ctx.viewer.removed).toEqual([NAME_54]);
    expect(ctx.infer.serves).toHaveLength(2); // setup_56 was not served again
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_56]);
  });

  it("serves the variant the catalog chooses for this GPU, keeping the catalog's entry in the session", async () => {
    const ctx = fakeContext();
    ctx.infer.limits = { maxBufferSize: 256 * 2 ** 20, maxStorageBufferBindingSize: null, shaderF16: false };
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    expect(ctx.infer.serves).toEqual([expect.objectContaining({ name: NAME_54, variant: "int8w" })]);
    expect(ctx.infer.served()[0].entry.onnxUrl).toMatch(/setup_54_int8w\.onnx$/);
    expect(ctx.session.state.models[0].variant.variant).toBe("fp32"); // the catalog's default, tuned again on each load
    // a new chain serves the same variant again (the session is shared), without asking the GPU again
    ctx.infer.limits = { maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false };
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: [], postprocess: [] }));
    expect(ctx.infer.serves[1]).toEqual(expect.objectContaining({ name: NAME_54, variant: "int8w" }));
  });

  // The manifests of the fakes declare MinMaxNormalizer then LambdaNormalizer.
  it.each([
    ["no input normalization", []],
    ["the user's", [{ name: "LambdaNormalizer", expression: "x/255" }]],
  ])("sets the input normalization a model starting declares only when the chain has none: %s", async (_label, inputNorm) => {
    const postprocess = [{ name: "SigmoidPostprocessor" }];
    const ctx = fakeContext({ state: { chain: { input_norm: inputNorm, postprocess } } });
    const backend = createBackend(ctx);
    expect((await backend.handle(submit([HF54]))).status).toBe(200);
    const expected = { input_norm: inputNorm.length ? inputNorm : entry("setup_54").manifest.input_norm, postprocess };
    expect(ctx.session.state.chain).toEqual(expected);
    // the model is served, and its layer drawn, through it alone
    expect(ctx.infer.served().map((s) => decodeArgs(s.view.split(ARGS_KEY)[1]).chain)).toEqual([expected]);
    // the page reads the chain back with the model whose declaration it is
    const loaded = body(await backend.handle(request("GET", "/api/models/loaded")));
    expect(loaded.pipeline).toEqual(expected);
    expect(loaded.input_norm_from).toBe(inputNorm.length ? null : NAME_54);
    // a model added after it finds input normalizers there, and leaves them
    await backend.handle(request("POST", "/api/models/add", { entry: { type: "onnx", repo: "cellmap/onnx-web-models", folder: "setup_56", name: NAME_56 } }));
    expect(ctx.session.state.chain).toEqual(expected);
  });

  it("sets the postprocessing a model starting needs, from its manifest's output, only when the chain has none", async (t) => {
    // The rule is the portable Python's (rank.needed_postprocess), in node's Pyodide.
    if (!(await portablePython())) t.skip();
    const inputNorm = entry("setup_54").manifest.input_norm;
    const ctx = fakeContext({ state: { chain: { input_norm: inputNorm, postprocess: [] } } });
    const backend = createBackend(ctx);
    // setup_54's affinities are logits too: the sigmoid, before it is served.
    expect((await backend.handle(submit([HF54]))).status).toBe(200);
    const expected = { input_norm: inputNorm, postprocess: [{ name: "SigmoidPostprocessor" }] };
    expect(ctx.session.state.chain).toEqual(expected);
    expect(ctx.infer.served().map((s) => decodeArgs(s.view.split(ARGS_KEY)[1]).chain)).toEqual([expected]);
    expect(body(await backend.handle(request("GET", "/api/models/loaded")))).toMatchObject({ postprocess_from: NAME_54, postprocess_fitted: false });
    // The user's steps since are theirs: the next start leaves them, and the list says nothing set them.
    const theirs = [{ name: "ThresholdPostprocessor", threshold: 0.3 }];
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: inputNorm, postprocess: theirs }));
    await backend.handle(request("POST", "/api/models/add", { entry: { type: "onnx", repo: "cellmap/onnx-web-models", folder: "setup_56", name: NAME_56 } }));
    expect(ctx.session.state.chain.postprocess).toEqual(theirs);
    expect(body(await backend.handle(request("GET", "/api/models/loaded")))).toMatchObject({ postprocess_from: null, postprocess_fitted: null });
  }, 120_000);  // Pyodide and numpy load the first time, a few seconds

  it("reads what a model starting needs off its network, over a manifest that says otherwise", async (t) => {
    if (!(await portablePython())) t.skip();
    const ctx = fakeContext({ state: { chain: { input_norm: entry("setup_54").manifest.input_norm, postprocess: [] } } });
    // setup_54's manifest says it has its sigmoid, but the bytes it runs end in a Conv: logits.
    const model = ctx.catalog.entries[0];
    model.manifest.output = "probabilities (sigmoid applied)";
    const head = readFileSync(resolve(import.meta.dirname, "../fixtures/head.onnx"));
    ctx.catalog.finetunedBytes.set(model.onnxUrl, head.buffer.slice(head.byteOffset, head.byteOffset + head.byteLength));
    const backend = createBackend(ctx);
    expect((await backend.handle(submit([HF54]))).status).toBe(200);
    expect(ctx.session.state.chain.postprocess).toEqual([{ name: "SigmoidPostprocessor" }]);
    // and the advice says where the class came from
    const m = body(await backend.handle(request("GET", "/api/model_advice"))).models[0];
    expect(m).toMatchObject({ output_class: "unbounded", output_class_from: "graph", output_op: "Conv" });
  }, 120_000);

  it("refuses a reference that is not a model the browser runs, changing nothing", async () => {
    const ctx = fakeContext();
    const res = await createBackend(ctx).handle(submit(["cellmap/some-pytorch-model"]));
    expect(res.status).toBe(400);
    expect(body(res).success).toBe(false);
    expect(body(res).error).toContain("cellmap/some-pytorch-model");
    expect(ctx.infer.serves).toEqual([]);
  });

  it("runs a ticked model at the voxel size typed beside it, which the listing and the export say, until a Submit leaves it blank", async () => {
    const ctx = fakeContext({ state: { datasetUrl: "https://example.org/em.zarr" } });
    const backend = createBackend(ctx);
    const listed = async () => body(await backend.handle(request("GET", "/api/huggingface-models")));
    expect((await backend.handle(submit([HF54, HF56], { run_voxel_sizes: { [HF54]: "32", [HF56]: "" } }))).status).toBe(200);
    const [at32, own] = ctx.infer.served();
    expect(at32.entry.runVoxelSize).toEqual([32, 32, 32]);
    expect(own.entry.runVoxelSize).toBeUndefined();
    // Kept with the session, so a reload serves it there again.
    expect(ctx.session.state.models.map((m) => m.runVoxelSize)).toEqual([[32, 32, 32], undefined]);
    const listing = await listed();
    expect(listing[HF54]).toMatchObject({ selected: true, run_voxel_size: [32, 32, 32] });
    expect(listing[HF56]).not.toHaveProperty("run_voxel_size");
    const exported = await backend.handle(request("GET", "/api/export-config"));
    if (!("body" in exported)) throw new Error("not a body answer");
    expect(uncommented(exported.body as string)).toContain(toYaml({ models: [{ ...SERVER_54("setup_54_fp32.onnx"), run_voxel_size: [32, 32, 32] }] }).replace(/^models:\n/, ""));

    // Another size is another view (another URL, so no chunk of the other is
    // shown), and a blank one its own size again.
    await backend.handle(submit([HF54, HF56], { run_voxel_sizes: { [HF54]: [32, 16, 16] } }));
    const served54 = () => ctx.infer.served().find((s) => s.entry.name === NAME_54)!;
    const at321616 = served54();
    expect(at321616.entry.runVoxelSize).toEqual([32, 16, 16]);
    expect(at321616.view).not.toBe(at32.view);
    await backend.handle(submit([HF54, HF56], { run_voxel_sizes: { [HF54]: " " } }));
    expect(ctx.infer.served().map((s) => s.entry.runVoxelSize)).toEqual([undefined, undefined]);
    expect(served54().view).not.toBe(at321616.view);
    expect(ctx.infer.serves.map((s) => s.name)).toEqual([NAME_54, NAME_56, NAME_54, NAME_54]);  // setup_56 was not served again
    expect(await listed()).not.toHaveProperty([HF54, "run_voxel_size"]);

    // A size that is none is refused, naming the model, and changes nothing.
    const refused = await backend.handle(submit([HF54], { run_voxel_sizes: { [HF54]: "16,8" } }));
    expect(refused.status).toBe(400);
    expect(body(refused).error).toContain(HF54);
    expect(ctx.infer.served()).toHaveLength(2);
  });
});

describe("the Loaded models list: GET /api/models/loaded and POST /api/models/remove", () => {
  function loaded(name: string, folder: string, id: string) {
    return { name, source: "huggingface", detail: folder, status: "running", removable: true, note: null, hf_id: id, zoo_id: null, cellpose: null };
  }

  it("lists the served models in order, each with where it came from and the Hugging Face row that ticks it", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    expect(body(await backend.handle(request("GET", "/api/models/loaded")))).toEqual({
      success: true, models: [], pipeline: defaultChain(), input_norm_from: null, postprocess_from: null, postprocess_fitted: null,
    });
    await backend.handle(submit([HF56, HF54]));
    expect(body(await backend.handle(request("GET", "/api/models/loaded")))).toEqual({
      success: true,
      models: [loaded(NAME_56, "setup_56", HF56), loaded(NAME_54, "setup_54", HF54)],
      // the chain they run through; the default's input normalization is what both declare
      pipeline: defaultChain(),
      input_norm_from: NAME_56,
      // the default's sigmoid was there before either started
      postprocess_from: null,
      postprocess_fitted: null,
    });
  });

  it("stops serving the one model removed; the others stay served, their layers not redrawn, and nothing else in the request is read", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54, HF56]));
    const serves = ctx.infer.serves.length;
    const redraws = ctx.viewer.setLayersCalls;
    const layer56 = ctx.viewer.layers[NAME_56];
    // The tab has setup_56 unticked and nothing submitted: a remove names one model, and only it changes.
    const res = await backend.handle(request("POST", "/api/models/remove", { name: NAME_54, selected_hf_models: [] }));
    expect(res.status).toBe(200);
    expect(body(res)).toEqual({ success: true, removed: NAME_54, models: [loaded(NAME_56, "setup_56", HF56)] });
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56]);
    expect(ctx.infer.disposed).toHaveLength(1);
    expect(ctx.infer.disposed[0].startsWith(`${NAME_54}${ARGS_KEY}`)).toBe(true);
    expect(ctx.viewer.removed).toEqual([NAME_54]);
    expect(ctx.viewer.layers[NAME_56]).toBe(layer56);
    expect(ctx.infer.serves).toHaveLength(serves);
    expect(ctx.viewer.setLayersCalls).toBe(redraws);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_56]);
    // Ticked again and submitted, it is served again beside the other.
    await backend.handle(submit([HF54, HF56]));
    expect(ctx.infer.served().map((s) => s.entry.name).sort()).toEqual([NAME_54, NAME_56].sort());
  });

  it("unserves a finetuned model but keeps it, listed and stored, to serve again", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    const base = entry("setup_54");
    const fentry = finetunedEntry(base, `${NAME_54}-ft1`, 0.1);
    const record: FinetunedRecord = {
      id: fentry.id, name: fentry.name, base: { id: base.id, name: base.name, file: base.variant.file },
      volume: "vol-1", created_at: "2026-10-16T12:00:00Z", params: {}, epochs: 1, final_loss: null, entry: fentry,
    };
    await ctx.catalog.addFinetuned(record, new ArrayBuffer(16));
    await backend.handle(submit([HF54, fentry.id]));
    const listed = body(await backend.handle(request("GET", "/api/models/loaded"))).models;
    expect(listed[1]).toMatchObject({ name: fentry.name, source: "finetuned", detail: `from ${NAME_54}`, removable: true, hf_id: fentry.id });

    const res = await backend.handle(request("POST", "/api/models/remove", { name: fentry.name }));
    expect(res.status).toBe(200);
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_54]);
    expect(ctx.viewer.removed).toEqual([fentry.name]);
    // Its record and bytes stay: the Hugging Face list still has it, and ticking it serves it again.
    expect(ctx.catalog.finetunedRecords.map((r) => r.id)).toEqual([fentry.id]);
    expect(ctx.catalog.finetunedBytes.has(fentry.onnxUrl)).toBe(true);
    const listing = body(await backend.handle(request("GET", "/api/huggingface-models")));
    expect(listing[fentry.id]).toBeDefined();
    expect(listing[fentry.id].selected).toBeUndefined();
    await backend.handle(submit([HF54, fentry.id]));
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_54, fentry.name]);
  });
});

describe("the model lists: GET /api/huggingface-models and /api/finetune/rank/candidates", () => {
  it("tag each model alike, with its format, its ways to finetune, its size and each precision's", async () => {
    const backend = createBackend(fakeContext());
    const listing = body(await backend.handle(request("GET", "/api/huggingface-models")));
    const { candidates } = body(await backend.handle(request("GET", "/api/finetune/rank/candidates")));
    const e = entry("setup_54");
    const facts = {
      formats: ["ONNX"], finetune_modes: ["head"], size_mb: 42.9, browser_copy: e.id,
      variants: [{ variant: "fp32", file_mb: 42.9 }, { variant: "int8w", file_mb: 10.9 }],
    };
    expect(listing[e.id]).toMatchObject(facts);
    expect(candidates.find((c: { name: string }) => c.name === NAME_54)).toMatchObject(facts);
  });
});

describe("PUT /api/pipeline", () => {
  const CHAIN = {
    input_norm: [
      { name: "MinMaxNormalizer", min_value: "0", max_value: "255" },
      { name: "LambdaNormalizer", expression: "2*x-1" },
    ],
    postprocess: [{ name: "SigmoidPostprocessor" }, { name: "ThresholdPostprocessor", threshold: "0.5" }],
  };

  it("keeps the chain, serves every model anew through it, and answers the chain, its digest and the layers", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54, HF56]));
    const before = ctx.infer.served().map((s) => s.view);
    const res = await backend.handle(request("PUT", "/api/pipeline", CHAIN));
    expect(res.status).toBe(200);
    const answer = body(res);
    expect(answer.success).toBe(true);
    expect(answer.pipeline).toEqual(CHAIN); // the steps as given, values uncoerced
    expect(answer.digest).toMatch(/^[0-9a-f]{16}$/);
    expect(answer.layers).toEqual([NAME_54, NAME_56]);
    expect(ctx.session.state.chain).toEqual(CHAIN);
    // new views carrying the new chain; the old ones disposed of after
    const after = ctx.infer.served().map((s) => s.view);
    expect(after).toHaveLength(2);
    expect(after.every((v) => !before.includes(v))).toBe(true);
    expect(ctx.infer.disposed.sort()).toEqual(before.sort());
    expect(decodeArgs(after[0].split(ARGS_KEY)[1]).chain).toEqual(CHAIN);
    expect(decodeArgs(after[0].split(ARGS_KEY)[1]).extras.digest).toBe(answer.digest);
  });

  it("keeps the user's shader across a normalization change, and only its colour across a postprocessing change", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const shader = '#uicontrol invlerp normalized(range=[0, 1]);\n#uicontrol vec3 color color(default="magenta");\nvoid main(){emitRGB(color * normalized());}';
    ctx.viewer.shaders[NAME_54] = { shader, shaderControls: { normalized: { range: [0.2, 0.9] } } };
    // input normalization changed only: the layer keeps the shader as it is
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: [], postprocess: CHAIN.postprocess.slice(0, 1) }));
    expect(ctx.viewer.layers[NAME_54].shader).toBe(shader);
    expect(ctx.viewer.layers[NAME_54].shaderControls).toEqual({ normalized: { range: [0.2, 0.9] } });
    // postprocessing changed: a fresh shader over the new range, in the colour the layer had
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: [], postprocess: [] }));
    const fresh = String(ctx.viewer.layers[NAME_54].shader);
    expect(fresh).not.toBe(shader);
    expect(fresh).toContain('color(default="magenta")');
    expect(ctx.viewer.layers[NAME_54].shaderControls).toBeUndefined();
  });

  it("refuses a step the browser cannot run, naming it, and keeps the chain as it was", async () => {
    const ctx = fakeContext();
    const res = await createBackend(ctx).handle(
      request("PUT", "/api/pipeline", { input_norm: [], postprocess: [{ name: "MutexWatershedPostprocessor" }] }),
    );
    expect(res.status).toBe(400);
    expect(body(res).success).toBe(false);
    expect(body(res).error).toContain("MutexWatershedPostprocessor");
    expect(ctx.session.state.chain).toEqual(defaultChain());
  });
});

describe("the raw's channels", () => {
  it("answers a volume's channels, takes the Channel's pick into the chain, and shows that channel in the raw layer", async () => {
    const dataset = { ...DATASET, url: "https://example.org/lm.zarr", channels: 2, channelLabels: ["DAPI", "GFP"] };
    const ctx = fakeContext({ dataset });
    ctx.viewer.layers.data = rawLayer(dataset.url, dataset, 0);
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    expect(body(await backend.handle(request("GET", "/api/raw/channels")))).toEqual({ success: true, channels: 2, labels: ["DAPI", "GFP"], selected: 0 });

    const pick = { name: "ChannelSelector", channel: "1" };
    const res = await backend.handle(request("PUT", "/api/pipeline", { ...defaultChain(), input_norm: [pick, ...defaultChain().input_norm] }));
    expect(res.status).toBe(200);
    expect(ctx.infer.serves.at(-1)!.chain.input_norm[0]).toEqual(pick);
    expect(ctx.viewer.layers.data.localPosition).toEqual([1]);
    expect(body(await backend.handle(request("GET", "/api/raw/channels"))).selected).toBe(1);
    // The pick is the dataset's, not the model's: its declared normalization still matches.
    expect(body(await backend.handle(request("GET", "/api/model_advice"))).models[0].input_norm_matches).toBe(true);
    // A channel the volume has not got changes nothing.
    expect((await backend.handle(request("PUT", "/api/pipeline", { input_norm: [{ name: "ChannelSelector", channel: 2 }], postprocess: [] }))).status).toBe(400);
    expect(ctx.session.state.chain.input_norm[0]).toEqual(pick);
  });
});

describe("adding a model by reference", () => {
  it("resolves a catalog model to the server's onnx entry, which the form can submit", async () => {
    const res = await createBackend(fakeContext()).handle(request("POST", "/api/models/resolve", { ref: HF54 }));
    expect(res.status).toBe(200);
    const d = body(res);
    expect(d).toMatchObject({ success: true, type: "onnx", params: { repo: "cellmap/onnx-web-models", folder: "setup_54" }, name: NAME_54, env: null, needs: [] });
    expect(d.params).not.toHaveProperty("file");  // the one run is chosen for this GPU
  });

  it("adds the entry to the served models, and refuses a name already running", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF56]));
    const res = await backend.handle(request("POST", "/api/models/add", { entry: { type: "onnx", repo: "cellmap/onnx-web-models", folder: "setup_54", name: NAME_54 } }));
    expect(body(res)).toEqual({ success: true, name: NAME_54 });
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56, NAME_54]);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_56, NAME_54]);
    const again = await backend.handle(request("POST", "/api/models/add", { entry: { name: NAME_54 } }));
    expect(again.status).toBe(409);
    expect((await backend.handle(request("POST", "/api/models/add", { entry: {} }))).status).toBe(400);
  });
});

describe("GET /api/model_advice", () => {
  it("flags a chain that does not match the manifest's normalization", async () => {
    const ctx = fakeContext({ state: { chain: { input_norm: [{ name: "LambdaNormalizer", expression: "x" }], postprocess: [] } } });
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const m = body(await backend.handle(request("GET", "/api/model_advice"))).models[0];
    expect(m.input_norm_matches).toBe(false);
    expect(m.configured_input_norm).toEqual(["LambdaNormalizer"]);
    expect(m.input_norm_differences).toEqual([]);
  });

  it("holds a declared value to its number, not its spelling, and names a value that differs", async () => {
    // The manifest declares MinMax 0..255 then 2*x-1 (backend-fakes.ts); this
    // is that chain as the Processing tab's form sends it.
    const asForm = [
      { name: "MinMaxNormalizer", min_value: "0.0", max_value: "255.0", invert: "False" },
      { name: "LambdaNormalizer", expression: "2*x-1" },
    ];
    const ctx = fakeContext({ state: { chain: { input_norm: asForm, postprocess: [] } } });
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    let m = body(await backend.handle(request("GET", "/api/model_advice"))).models[0];
    expect(m.input_norm_matches).toBe(true);
    expect(m.input_norm_differences).toEqual([]);
    // The same map written another way is the declared one.
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: [asForm[0], { name: "LambdaNormalizer", expression: "x*2 - 1" }], postprocess: [] }));
    m = body(await backend.handle(request("GET", "/api/model_advice"))).models[0];
    expect(m.input_norm_differences).toEqual([]);
    // The same steps with the expression edited rated "looks right" before.
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: [asForm[0], { name: "LambdaNormalizer", expression: "2*x" }], postprocess: [] }));
    m = body(await backend.handle(request("GET", "/api/model_advice"))).models[0];
    expect(m.input_norm_matches).toBe(false);
    expect(m.input_norm_differences).toEqual([{ step: "LambdaNormalizer", param: "expression", configured: "2*x", declared: "2*x-1" }]);
  });

  it("offers an affinity model its chain into a segmentation cut where its ranking fitted the paint, which finds a 2D model's objects on one slice, and warns of a threshold ahead of it", async () => {
    // A 2D model's three channels (z, y, x) on the one slice it is run on,
    // which has no neighbour along z: the z channel is the model's guess at
    // one, here "apart" everywhere. Two 4×4 squares whose neighbours'
    // affinities are faint, a logit of -2 (0.12) along y and x inside each
    // and -6 (0.0025) everywhere else, which a ranking fits at 0.076.
    const square = (y: number, x: number) => (y >= 1 && y < 5 && x >= 1 && x < 5 ? 1 : y >= 6 && y < 10 && x >= 6 && x < 10 ? 2 : 0);
    const logits = new Float32Array(3 * 12 * 12).fill(-6);
    for (let y = 0; y < 12; y++) {
      for (let x = 0; x < 12; x++) {
        const here = square(y, x);
        if (here && square(y + 1, x) === here) logits[144 + y * 12 + x] = -2;
        if (here && square(y, x + 1) === here) logits[288 + y * 12 + x] = -2;
      }
    }
    const objects = (steps: Step[]) => new Set(Array.from(applyPostprocessors({ data: logits, shape: [3, 1, 12, 12] }, steps).data as BigUint64Array)).size - 1;
    const applied = (review: any): Step[] => review.suggest.map((name: string) => ({ name, ...review.params[name] }));
    const review = async () => body(await backend.handle(request("GET", "/api/model_advice"))).models[0].postprocess_review;

    // The row the ranking set, the sigmoid and that threshold: the chain into
    // a segmentation is offered in its place, joining neighbours above the
    // same cut-off, where it finds the squares; at the 0.5 it is offered at
    // without a fit, none.
    const ranked: Step[] = [{ name: "SigmoidPostprocessor" }, { name: "ThresholdPostprocessor", threshold: 0.076 }];
    const fits = { [NAME_54]: { input_norm: [], postprocess: ranked, threshold: 0.076 } };
    const ctx = fakeContext({ state: { chain: { input_norm: [], postprocess: ranked }, rankFits: fits } });
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const offered = await review();
    expect(offered).toMatchObject({ level: "optional", suggest: ["SigmoidPostprocessor", "DefaultPostprocessor", "AffinityPostprocessor"] });
    expect(offered.params.AffinityPostprocessor.bias).toBe(0.076);
    expect(objects(applied(offered))).toBe(2);
    await ctx.session.update({ rankFits: {} });
    expect(objects(applied(await review()))).toBe(0);

    // The threshold ahead of the watershed left it 0s and 1s, and no
    // objects: the advice says to take it out.
    await ctx.session.update({ rankFits: fits });
    const thresholded = [...ranked, { name: "AffinityPostprocessor" }];
    expect(objects(thresholded)).toBe(0);
    await backend.handle(request("PUT", "/api/pipeline", { input_norm: [], postprocess: thresholded }));
    const warned = await review();
    expect(warned.level).toBe("warn");
    expect(objects(applied(warned))).toBe(2);
  });
});

describe("the settings routes", () => {
  it("has no jobs, no queues and an empty server config, in the shapes the page reads", async () => {
    const backend = createBackend(fakeContext());
    expect(body(await backend.handle(request("GET", "/api/job-logs")))).toEqual({ success: true, jobs: [] });
    const queues = body(await backend.handle(request("GET", "/api/gpu-queues")));
    expect(queues.queues).toEqual([]);
    expect(queues.available).toBe(false);
    expect(body(await backend.handle(request("GET", "/api/server-config")))).toEqual({ cached: true });
  });

  it("keeps what the server config form saves and reads it back", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    const saved = body(await backend.handle(request("POST", "/api/server-config", { queue: "gpu_h100", charge_group: "cellmap", nb_workers: 4, bogus: 1 })));
    expect(saved).toEqual({ success: true, config: { queue: "gpu_h100", charge_group: "cellmap", nb_workers: 4 } });
    expect(body(await backend.handle(request("GET", "/api/server-config")))).toEqual({ queue: "gpu_h100", charge_group: "cellmap", nb_workers: 4, cached: true });
  });

  it("exports the session as YAML to download", async () => {
    const ctx = fakeContext({ state: { datasetUrl: "https://example.org/em.zarr" } });
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const res = await backend.handle(request("GET", "/api/export-config"));
    expect(res.status).toBe(200);
    if (!("body" in res)) throw new Error("not a body answer");
    expect(res.type).toBe("text/yaml");
    expect(res.headers?.["content-disposition"]).toMatch(/^attachment; filename=cellmap_flow_config_\d{8}_\d{6}\.yaml$/);
    expect(uncommented(res.body as string)).toBe(toYaml({
      data_path: "https://example.org/em.zarr",
      resample: false,
      json_data: {
        input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }, { name: "LambdaNormalizer", expression: "2*x-1" }],
        postprocess: [{ name: "SigmoidPostprocessor" }],
      },
      models: [SERVER_54("setup_54_fp32.onnx")],
    }));
  });

  it("exports a model finetuned here by the path its download saves, saying where to put it", async () => {
    const ctx = fakeContext({ state: { datasetUrl: "https://example.org/em.zarr" } });
    const base = entry("setup_54");
    const finetuned = finetunedEntry(base, `${NAME_54}-ft1`);
    await ctx.session.update({ models: [base, finetuned] });
    const res = await createBackend(ctx).handle(request("GET", "/api/export-config"));
    if (!("body" in res)) throw new Error("not a body answer");
    const yaml = res.body as string;
    expect(uncommented(yaml)).toContain(toYaml({ models: [SERVER_54("setup_54_fp32.onnx"), { type: "onnx", name: `${NAME_54}-ft1`, path: `${NAME_54}-ft1.onnx` }] }));
    // The path line carries a note of where the file goes.
    expect(yaml.split("\n").find((line) => line.includes(`path: ${NAME_54}-ft1.onnx`))).toContain("#");
  });
});

describe("everything else under /api/", () => {
  it("is unavailable, with the route named", async () => {
    const backend = createBackend(fakeContext());
    for (const [method, path] of [
      ["POST", "/api/finetune/load-crops"],
      ["GET", "/api/review/status"],
      ["POST", "/api/blockwise/validate"],
      ["GET", "/api/logs/stream"],
    ]) {
      const res = await backend.handle(request(method, path));
      expect(res.status).toBe(503);
      expect(body(res)).toMatchObject({ success: false, available: false });
      expect(body(res).error).toContain(`${method} ${path}`);
    }
  });

  it("answers a wrong method on a known path with a 405", async () => {
    const res = await createBackend(fakeContext()).handle(request("GET", "/api/set-data"));
    expect(res.status).toBe(405);
  });
});

describe("restoreSession", () => {
  it("opens the session's dataset again and serves its models through its chain", async () => {
    const ctx = fakeContext({ dataset: null, state: { datasetUrl: "https://example.org/em.zarr", models: [entry("setup_54")] } });
    await restoreSession(ctx);
    expect(ctx.infer.opened).toEqual(["https://example.org/em.zarr"]);
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_54]);
    expect(decodeArgs(ctx.infer.serves[0].blob).chain).toEqual(defaultChain());
    expect(Object.keys(ctx.viewer.layers)).toEqual([NAME_54]);
    // The viewer, waiting for the volume's geometry, is given it before the model's layer comes in.
    expect(ctx.viewer.datasetsOpened.map(({ url, info }) => [url, info?.url])).toEqual([["https://example.org/em.zarr", "https://example.org/em.zarr"]]);
  });

  it("does nothing without a dataset, and logs rather than throws when the dataset is gone", async () => {
    const empty = fakeContext({ dataset: null });
    await restoreSession(empty);
    expect(empty.infer.opened).toEqual([]);
    const gone = fakeContext({ dataset: null, state: { datasetUrl: "https://example.org/moved.zarr", models: [entry("setup_54")] } });
    gone.infer.failOpen = "404";
    await restoreSession(gone);
    expect(gone.infer.serves).toEqual([]);
    expect(gone.logged.join("\n")).toContain("https://example.org/moved.zarr");
    // A volume that cannot be opened: the viewer is mounted without its geometry, for Neuroglancer to try.
    expect(gone.viewer.datasetsOpened).toEqual([{ url: "https://example.org/moved.zarr", info: null }]);
  });

  it("leaves no model served that was removed from the Loaded models list while it was being served again", async () => {
    const models = [entry("setup_54"), entry("setup_56", "nuc", ["nuc"], 128)];
    const ctx = fakeContext({ dataset: null, state: { datasetUrl: "https://example.org/em.zarr", models } });
    // The volume opens slowly, so the ✕ comes while the models are still to be served ("starting").
    let opened!: () => void;
    const slow = new Promise<void>((resolve) => (opened = resolve));
    const open = ctx.infer.openDataset.bind(ctx.infer);
    ctx.infer.openDataset = async (url) => {
      await slow;
      return open(url);
    };
    const restoring = restoreSession(ctx);
    const removed = await createBackend(ctx).handle(request("POST", "/api/models/remove", { name: NAME_54 }));
    expect(removed.status).toBe(200);
    expect(ctx.infer.serves).toEqual([]);
    opened();
    await restoring;
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56]);
    expect(Object.keys(ctx.viewer.layers)).toEqual([NAME_56]);
    expect(ctx.session.state.models.map((m) => m.name)).toEqual([NAME_56]);
  });
});
