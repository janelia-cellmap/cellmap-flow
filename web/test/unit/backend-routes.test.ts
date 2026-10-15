// The /api/* routes against fake context members: the JSON each answers,
// checked against what the dashboard's JavaScript reads (models-tab.js,
// main.js, connect.js, model-advice.js).
import { describe, expect, it } from "vitest";

import { createBackend, restoreSession } from "../../src/backend";
import { ARGS_KEY } from "../../src/contracts";
import type { ApiResponse } from "../../src/contracts";
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

  it("gives the raw layer the contrast the engine sampled, when it sampled one", async () => {
    const ctx = fakeContext({ dataset: null });
    ctx.infer.dtype = "int16";  // uint8's shader would clip a sampled range to what the dtype holds
    ctx.infer.sampledRange = [5490, 5793];
    await createBackend(ctx).handle(request("POST", "/api/set-data", { dataset_path: "https://example.org/em.zarr" }));
    expect(ctx.viewer.prepared[0].info?.displayRange).toEqual([5490, 5793]);
    expect(String(ctx.viewer.prepared[0].raw.shader)).toContain("range=[5490, 5793]");
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
  it("resolves a catalog model to a huggingface entry the form can submit", async () => {
    const res = await createBackend(fakeContext()).handle(request("POST", "/api/models/resolve", { ref: HF54 }));
    expect(res.status).toBe(200);
    const d = body(res);
    expect(d).toMatchObject({ success: true, type: "huggingface", params: { repo: "cellmap/onnx-web-models", folder: "setup_54" }, name: NAME_54, env: null, needs: [] });
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
    const res = await backend.handle(request("POST", "/api/models/add", { entry: { type: "huggingface", repo: "cellmap/onnx-web-models", folder: "setup_54", name: NAME_54 } }));
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
    expect(yaml).toContain(`models:\n- type: huggingface\n  repo: cellmap/onnx-web-models\n  name: ${NAME_54}\n  folder: setup_54\n  onnx: setup_54_fp32.onnx\n`);
  });
});

describe("everything else under /api/", () => {
  it("is unavailable, with the route named", async () => {
    const backend = createBackend(fakeContext());
    for (const [method, path] of [
      ["GET", "/api/finetune/models"],
      ["POST", "/api/finetune/submit"],
      ["GET", "/api/finetune/job/abc/status"],
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
  });
});
