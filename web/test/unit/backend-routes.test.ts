// The /api/* routes against fake context members: the JSON each answers,
// checked against what the dashboard's JavaScript reads (models-tab.js,
// main.js, connect.js, model-advice.js).
import { describe, expect, it } from "vitest";

import { createBackend, restoreSession } from "../../src/backend";
import { explainOpenError } from "../../src/backend/routes/data";
import { ARGS_KEY } from "../../src/contracts";
import type { ApiResponse, FinetunedRecord } from "../../src/contracts";
import { finetunedEntry } from "../../src/models/finetuned";
import { decodeArgs } from "../../src/shared/args";
import { defaultChain } from "../../src/state";
import { entry, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

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

  it("lets a precomputed:// URL through to the engine as it was typed", async () => {
    const ctx = fakeContext({ dataset: null });
    const res = await createBackend(ctx).handle(request("POST", "/api/set-data", { dataset_path: "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg" }));
    expect(res.status).toBe(200);
    expect(ctx.infer.opened).toEqual(["precomputed://gs://flyem-male-cns/em/em-clahe-jpeg"]);
  });

  it("refuses a missing path and a path that is not a URL with a 400 the connect panel shows", async () => {
    const backend = createBackend(fakeContext({ dataset: null }));
    expect(body(await backend.handle(request("POST", "/api/set-data", { dataset_path: "" })))).toEqual({ success: false, error: "dataset_path is required" });
    expect((await backend.handle(request("POST", "/api/set-data", null))).status).toBe(400);
    const res = await backend.handle(request("POST", "/api/set-data", { dataset_path: "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr" }));
    expect(res.status).toBe(400);
    expect(body(res).error).toMatch(/not a URL/);
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
    const error = body(res).error as string;
    expect(error).toMatch(/^Failed to fetch: the browser got no answer from data.example.org\./);
    expect(error).toMatch(/CORS/);
    expect(error).toMatch(/Local network access/); // the test runs on no page, so not on localhost
  });
});

describe("explainOpenError", () => {
  it("passes any other error through as it is", () => {
    expect(explainOpenError("https://example.org/em.zarr", new Error("no zarr.json at that URL"), "localhost")).toBe("no zarr.json at that URL");
    expect(explainOpenError("https://example.org/em.zarr", "a string", "localhost")).toBe("a string");
  });

  it("names the host and, on localhost, leaves out the private-network reason", () => {
    const error = explainOpenError("https://cellmap-vm1.int.janelia.org/nrs/data/em.zarr/", new TypeError("Failed to fetch"), "localhost");
    expect(error).toBe("Failed to fetch: the browser got no answer from cellmap-vm1.int.janelia.org. Either it cannot be reached from here, or its server does not allow this page to read it (CORS).");
  });

  it("knows every browser's wording, and a bucket URL's bucket", () => {
    expect(explainOpenError("s3://bucket/em.zarr", new TypeError("NetworkError when attempting to fetch resource."), "janelia-cellmap.github.io")).toMatch(/no answer from bucket\. .*Local network access/);
    expect(explainOpenError("https://example.org/em.zarr", new TypeError("Load failed"), "janelia-cellmap.github.io")).toMatch(/^Load failed: the browser got no answer from example.org\./);
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

  it("takes a model by folder or name too (selected_models, a model added by hand), and the resample setting", async () => {
    const ctx = fakeContext();
    const res = await createBackend(ctx).handle(submit([], { selected_models: ["setup_54", NAME_56], resample: true }));
    expect(body(res).models).toEqual(["setup_54", NAME_56]); // echoed as sent
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_54, NAME_56]);
    expect(ctx.session.state.resample).toBe(true);
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

  it("refuses a reference that is not a model the browser runs, changing nothing", async () => {
    const ctx = fakeContext();
    const res = await createBackend(ctx).handle(submit(["cellmap/some-pytorch-model"]));
    expect(res.status).toBe(400);
    expect(body(res).success).toBe(false);
    expect(body(res).error).toMatch(/cellmap\/some-pytorch-model is not a model the browser can run/);
    expect(ctx.infer.serves).toEqual([]);
  });

  it("refuses to serve without a dataset", async () => {
    const ctx = fakeContext({ dataset: null });
    const res = await createBackend(ctx).handle(submit([HF54]));
    expect(res.status).toBe(400);
    expect(body(res).error).toMatch(/No dataset is open/);
  });

  it("ignores zoo and Cellpose selections with a note, and answers them empty", async () => {
    const ctx = fakeContext();
    const res = await createBackend(ctx).handle(
      submit([HF54], { selected_bioimage_models: [{ id: "affable-shark", voxel_size: null }], selected_cellpose_models: [{ model: "cpsam", voxel_size: "8" }] }),
    );
    expect(res.status).toBe(200);
    expect(body(res).bioimage_models).toEqual([]);
    expect(body(res).cellpose_models).toEqual([]);
    expect(body(res).note).toMatch(/BioImage Model Zoo and Cellpose/);
  });
});

describe("the Loaded models list: GET /api/models/loaded and POST /api/models/remove", () => {
  function loaded(name: string, folder: string, id: string) {
    return { name, source: "huggingface", detail: folder, status: "running", removable: true, note: null, hf_id: id, zoo_id: null, cellpose: null };
  }

  it("lists the served models in order, each with where it came from and the Hugging Face row that ticks it", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    expect(body(await backend.handle(request("GET", "/api/models/loaded")))).toEqual({ success: true, models: [] });
    await backend.handle(submit([HF56, HF54]));
    expect(body(await backend.handle(request("GET", "/api/models/loaded")))).toEqual({
      success: true,
      models: [loaded(NAME_56, "setup_56", HF56), loaded(NAME_54, "setup_54", HF54)],
    });
  });

  it("says a model the session names but has not served again yet (a reload's restore) is starting", async () => {
    const ctx = fakeContext({ state: { models: [entry("setup_54")] } });
    const models = body(await createBackend(ctx).handle(request("GET", "/api/models/loaded"))).models;
    expect(models.map((m: any) => [m.name, m.status])).toEqual([[NAME_54, "starting"]]);
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

  it("answers a 404 for a model not served and a 400 without a name, changing nothing", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const gone = await backend.handle(request("POST", "/api/models/remove", { name: NAME_56 }));
    expect(gone.status).toBe(404);
    expect(body(gone)).toEqual({ success: false, error: `No model named ${NAME_56} is running` });
    const nameless = await backend.handle(request("POST", "/api/models/remove", { name: " " }));
    expect(nameless.status).toBe(400);
    expect(body(nameless).error).toBe("name is required");
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_54]);
    expect(ctx.infer.disposed).toEqual([]);
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
    expect(listed[1].note).toMatch(/Kept in this browser/);

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

  it("changes nothing when the same chain is sent again and the layers carry it", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    await backend.handle(request("PUT", "/api/pipeline", CHAIN));
    const serves = ctx.infer.serves.length;
    const calls = ctx.viewer.setLayersCalls;
    const res = await backend.handle(request("PUT", "/api/pipeline", CHAIN));
    expect(body(res).layers).toEqual([NAME_54]);
    expect(ctx.infer.serves).toHaveLength(serves);
    expect(ctx.viewer.setLayersCalls).toBe(calls);
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
    expect(body(res).error).toMatch(/MutexWatershedPostprocessor is not available in the browser/);
    expect(ctx.session.state.chain).toEqual(defaultChain());
  });

  it("refuses a body that is not two lists of named steps", async () => {
    const backend = createBackend(fakeContext());
    expect(body(await backend.handle(request("PUT", "/api/pipeline", { input_norm: {}, postprocess: [] }))).error).toBe("input_norm: expected a list of steps");
    expect(body(await backend.handle(request("PUT", "/api/pipeline", { input_norm: [], postprocess: [{ threshold: 1 }] }))).error).toBe("postprocess[0]: a step needs a name");
    expect((await backend.handle(request("PUT", "/api/pipeline", [1, 2]))).status).toBe(400);
  });

  it("keeps the chain and draws nothing when no dataset is open", async () => {
    const ctx = fakeContext({ dataset: null });
    const res = await createBackend(ctx).handle(request("PUT", "/api/pipeline", CHAIN));
    expect(res.status).toBe(200);
    expect(body(res).layers).toEqual([]);
    expect(ctx.session.state.chain).toEqual(CHAIN);
  });
});

describe("the model lists", () => {
  it("lists the catalog by model id with each model's metadata, on GET and on refresh (which reads the repository again)", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    for (const req of [request("GET", "/api/huggingface-models"), request("POST", "/api/huggingface-models/refresh")]) {
      const res = await backend.handle(req);
      expect(res.status).toBe(200);
      expect(Object.keys(body(res))).toEqual([HF54, HF56]);
      expect(body(res)[HF54].description).toBe("mito_affinities");
    }
    expect(ctx.catalog.refreshed).toBe(1);
  });

  it("marks the models the session is serving as selected, and no other", async () => {
    const ctx = fakeContext({ state: { models: [entry("setup_54")] } });
    const listing = body(await createBackend(ctx).handle(request("GET", "/api/huggingface-models")));
    expect(listing[HF54].selected).toBe(true);
    expect("selected" in listing[HF56]).toBe(false);
  });

  it("answers a failed listing with {error} and a 500, which the tab shows in the list", async () => {
    const ctx = fakeContext();
    ctx.catalog.failList = "huggingface.co: 429 Too Many Requests";
    const res = await createBackend(ctx).handle(request("GET", "/api/huggingface-models"));
    expect(res.status).toBe(500);
    expect(body(res)).toEqual({ error: "huggingface.co: 429 Too Many Requests" });
  });

  it("has no zoo models", async () => {
    const backend = createBackend(fakeContext());
    expect(body(await backend.handle(request("GET", "/api/bioimage-models")))).toEqual({ models: [], fetched: null, stale: false });
    expect(body(await backend.handle(request("POST", "/api/bioimage-models/refresh")))).toEqual({ models: [], fetched: null, stale: false });
  });
});

describe("adding a model by reference", () => {
  it("resolves a catalog model to the server's onnx entry, which the form can submit", async () => {
    const res = await createBackend(fakeContext()).handle(request("POST", "/api/models/resolve", { ref: HF54 }));
    expect(res.status).toBe(200);
    const d = body(res);
    expect(d).toMatchObject({ success: true, type: "onnx", params: { repo: "cellmap/onnx-web-models", folder: "setup_54" }, name: NAME_54, env: null, needs: [] });
    expect(d.params).not.toHaveProperty("file");  // the one run is chosen for this GPU
    expect(d.how).toMatch(/browser/);
    expect(d.notes.join(" ")).toMatch(/fp32, int8w/);
  });

  it("says a reference it cannot run is unavailable in the browser", async () => {
    const backend = createBackend(fakeContext());
    const res = await backend.handle(request("POST", "/api/models/resolve", { ref: "/nrs/cellmap/models/run07" }));
    expect(res.status).toBe(503);
    expect(body(res)).toMatchObject({ success: false, available: false });
    expect(body(await backend.handle(request("POST", "/api/models/resolve", { ref: " " }))).error).toBe("ref is required");
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
    expect(body(await backend.handle(request("POST", "/api/models/add", { entry: {} }))).error).toBe("The model needs a name");
  });
});

describe("GET /api/model_advice", () => {
  it("answers one entry per served model that the banner shows nothing alarming for", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const d = body(await backend.handle(request("GET", "/api/model_advice")));
    expect(d.success).toBe(true);
    expect(d.models).toHaveLength(1);
    const m = d.models[0];
    expect(m.model).toBe(NAME_54);
    expect(m.probe_available).toBe(false);
    expect(m.out_channels).toBe(3);
    expect(m.postprocess_review.level).toBe("unknown");
    expect(m.postprocess_review.suggest).toEqual([]);
    expect(m.configured_postprocess).toEqual(["SigmoidPostprocessor"]);
    expect(m.configured_input_norm).toEqual(["MinMaxNormalizer", "LambdaNormalizer"]);
    // the manifest's normalization is the suggestion, and the default chain matches it
    expect(m.input_norm_suggestion.order).toEqual(["MinMaxNormalizer", "LambdaNormalizer"]);
    expect(m.input_norm_suggestion.input_norm.MinMaxNormalizer).toEqual({ min_value: 0, max_value: 255 });
    expect(m.input_norm_suggestion.confidence).toBe("high");
    expect(m.input_norm_matches).toBe(true);
    expect(m.input_relabelled_from).toBeNull();
  });

  it("flags a chain that does not match the manifest's normalization", async () => {
    const ctx = fakeContext({ state: { chain: { input_norm: [{ name: "LambdaNormalizer", expression: "x" }], postprocess: [] } } });
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    const m = body(await backend.handle(request("GET", "/api/model_advice"))).models[0];
    expect(m.input_norm_matches).toBe(false);
    expect(m.configured_input_norm).toEqual(["LambdaNormalizer"]);
  });
});

describe("the settings routes", () => {
  it("has no jobs, no queues and an empty server config, in the shapes the page reads", async () => {
    const backend = createBackend(fakeContext());
    expect(body(await backend.handle(request("GET", "/api/job-logs")))).toEqual({ success: true, jobs: [] });
    const queues = body(await backend.handle(request("GET", "/api/gpu-queues")));
    expect(queues.queues).toEqual([]);
    expect(queues.available).toBe(false);
    expect(queues.reason).toMatch(/browser/);
    expect(body(await backend.handle(request("GET", "/api/server-config")))).toEqual({ cached: true });
  });

  it("keeps what the server config form saves and reads it back", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    const saved = body(await backend.handle(request("POST", "/api/server-config", { queue: "gpu_h100", charge_group: "cellmap", nb_workers: 4, bogus: 1 })));
    expect(saved).toEqual({ success: true, config: { queue: "gpu_h100", charge_group: "cellmap", nb_workers: 4 } });
    expect(body(await backend.handle(request("GET", "/api/server-config")))).toEqual({ queue: "gpu_h100", charge_group: "cellmap", nb_workers: 4, cached: true });
  });

  it("keeps the finetune tab's preferences in the session", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx);
    expect(body(await backend.handle(request("GET", "/api/finetune/user-prefs")))).toEqual({ success: true, prefs: {} });
    await backend.handle(request("POST", "/api/finetune/user-prefs", { outputPath: "/tmp/corrections", other: null }));
    expect(body(await backend.handle(request("GET", "/api/finetune/user-prefs")))).toEqual({ success: true, prefs: { outputPath: "/tmp/corrections" } });
    expect(ctx.session.state.prefs.user).toEqual({ outputPath: "/tmp/corrections" });
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
    const yaml = res.body as string;
    expect(yaml).toContain("data_path: https://example.org/em.zarr\n");
    expect(yaml).toContain("resample: false\n");
    expect(yaml).toContain("json_data:\n  input_norm:\n  - name: MinMaxNormalizer\n    min_value: 0\n    max_value: 255\n  - name: LambdaNormalizer\n    expression: 2*x-1\n  postprocess:\n  - name: SigmoidPostprocessor\n");
    expect(yaml).toContain(`models:\n- type: onnx\n  name: ${NAME_54}\n  repo: cellmap/onnx-web-models\n  folder: setup_54\n  file: setup_54_fp32.onnx\n`);
    expect(yaml).toMatch(/^# Exported from the browser mode\. Each model is an ONNX export, which the\n# server serves with ONNX Runtime \(type: onnx/m);
    expect(yaml).not.toMatch(/PyTorch/);
  });

  it("exports the variant the page serves on this GPU, not the catalog's default the session keeps", async () => {
    const ctx = fakeContext({ state: { datasetUrl: "https://example.org/em.zarr" } });
    ctx.infer.limits = { maxBufferSize: 256 * 2 ** 20, maxStorageBufferBindingSize: null, shaderF16: false };
    const backend = createBackend(ctx);
    await backend.handle(submit([HF54]));
    expect(ctx.session.state.models[0].variant.file).toBe("setup_54_fp32.onnx");
    const res = await backend.handle(request("GET", "/api/export-config"));
    if (!("body" in res)) throw new Error("not a body answer");
    expect(res.body as string).toContain("  folder: setup_54\n  file: setup_54_int8w.onnx\n");
  });

  it("exports a model finetuned here by the path its download saves, saying where to put it", async () => {
    const ctx = fakeContext({ state: { datasetUrl: "https://example.org/em.zarr" } });
    const base = entry("setup_54");
    const finetuned = finetunedEntry(base, `${NAME_54}-ft1`);
    await ctx.session.update({ models: [base, finetuned] });
    const res = await createBackend(ctx).handle(request("GET", "/api/export-config"));
    if (!("body" in res)) throw new Error("not a body answer");
    expect(res.body as string).toContain(
      `models:\n- type: onnx\n  name: ${NAME_54}\n  repo: cellmap/onnx-web-models\n  folder: setup_54\n  file: setup_54_fp32.onnx\n` +
      `- type: onnx\n  name: ${NAME_54}-ft1\n  path: ${NAME_54}-ft1.onnx  # the file Download finetuned model saves: put it beside this YAML\n`,
    );
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
    expect(body(res).error).toMatch(/POST/);
  });

  it("answers a handler that throws with a 500 (through the relay) rather than nothing", async () => {
    const ctx = fakeContext();
    ctx.infer.serve = async () => {
      throw new Error("WebGPU is gone");
    };
    await expect(createBackend(ctx).handle(submit([HF54]))).rejects.toThrow("WebGPU is gone");
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
    expect(gone.logged.join("\n")).toMatch(/Could not open https:\/\/example.org\/moved.zarr again: 404/);
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
