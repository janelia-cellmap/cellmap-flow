// The Models tab's routes (routes/models.py, model_resolve.py,
// model_advice.py): which models there are, which to serve, which are
// served (the Loaded models list) and how to stop one, and advice about the
// chain they are served through.
//
// In the browser the models are the ONNX exports of cellmap/onnx-web-models
// and of the Hugging Face repos added beside it, and the models finetuned
// in this browser or opened from a file (the Catalog); "serving" one is an
// Engine view with an onnxruntime-web session (Inference.serve), so Submit
// serves and disposes views where the Python route starts and kills
// inference servers. The BioImage Model Zoo and Cellpose have no web
// exports: their lists are empty and their selections ignored (the template
// hides their panels in browser mode).
import type { AppContext, Chain, FinetunedRecord, ModelEntry, ServedModel, Step } from "../../contracts";
import { HF_REPO, isFinetuned } from "../../models";
import type { Router } from "../router";
import { badRequest, json, message, objectBody, unavailable } from "../router";
import { inputNormDeclaredBy, serveModels, setDeclaredInputNorm, stepNames, unserveModel } from "../serving";
import { serverParams } from "./config";

/** The catalog as /api/huggingface-models lists it: {model id: metadata},
 * the id "cellmap/onnx-web-models/setup_54" (models-tab.js shows its last
 * segment and the metadata's description). A model the session is serving
 * (after a reload, or from a link that named it) has `selected: true`, and
 * the tab ticks it, so the ticks say what is running; the Flask listing
 * has no such field, so its rows start unticked as before. `refresh` reads
 * the repository again instead of the cached listing. A listing that fails
 * is {error} with a 500, as the Flask route answers; the tab shows the
 * message. */
async function listModels(ctx: AppContext, refresh: boolean) {
  try {
    const entries = refresh ? await ctx.catalog.refresh() : await ctx.catalog.list();
    const serving = new Set(ctx.session.state.models.map((m) => m.id));
    return json(200, Object.fromEntries(entries.map((e) => [e.id, { ...ctx.catalog.metadataFor(e), ...(serving.has(e.id) ? { selected: true } : {}) }])));
  } catch (e) {
    return json(500, { error: message(e) });
  }
}

/** A list of strings from a body field, or null when it is not one. */
function strings(value: unknown, name: string): string[] | { error: string } {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) return { error: `${name}: expected a list of strings` };
  return value as string[];
}

/** What the browser can add, for the answer to a ref that is none of it. */
const CAN_ADD =
  `the browser runs the ONNX models of ${HF_REPO} (a folder, such as setup_54) and those of any public Hugging Face model repo laid out as that one is ` +
  "(<owner>/<repo>/<folder>, a folder holding a manifest.json and the .onnx files it names)";

/** The catalog's entry for `ref`, or an error that says why there is none:
 * the catalog's own (no such public repo, no such folder, ...) or CAN_ADD. */
async function resolveOne(ctx: AppContext, ref: string): Promise<{ entry: ModelEntry } | { error: string }> {
  try {
    const entry = await ctx.catalog.resolve(ref);
    return entry ? { entry } : { error: `${ref} is not a model the browser can run: ${CAN_ADD}.` };
  } catch (e) {
    return { error: message(e) };
  }
}

/** The catalog entries for the references the dashboard sent (repo ids from
 * the Hugging Face list, names of models added by hand), one per model, or
 * the first reference that is not a model the browser can run. */
async function resolveAll(ctx: AppContext, refs: string[]): Promise<{ entries: ModelEntry[] } | { error: string }> {
  const entries: ModelEntry[] = [];
  for (const ref of refs) {
    const resolved = await resolveOne(ctx, ref);
    if ("error" in resolved) return resolved;
    if (!entries.some((e) => e.id === resolved.entry.id)) entries.push(resolved.entry);
  }
  return { entries };
}

/** One model of the Loaded models list, as services/launch.py's
 * loaded_models gives it (models-tab.js reads both): its name, where it came
 * from ("huggingface", or "finetuned" for one finetuned in this browser),
 * what it is (the Hugging Face list's folder, or the base it was finetuned
 * from), whether it is served yet ("starting" while the session's restore
 * serves it after a reload), and the Hugging Face row that ticks it (the
 * list's value, the entry's id), which the tab unticks on a remove. Every
 * model can be removed: nothing runs outside the page, and a finetuned one
 * stays kept. */
export interface LoadedModel {
  name: string;
  source: "huggingface" | "finetuned";
  detail: string | null;
  status: "running" | "starting";
  removable: true;
  note: string | null;
  hf_id: string;
  zoo_id: null;
  cellpose: null;
}

/** The session's models, in order, as the Loaded models list shows them. */
export async function loadedModels(ctx: AppContext): Promise<LoadedModel[]> {
  const served = ctx.infer.served();
  let records: FinetunedRecord[] = [];
  if (ctx.session.state.models.some(isFinetuned)) {
    try {
      records = await ctx.catalog.finetuned();
    } catch {
      // Without its record a finetuned model is listed without its base.
    }
  }
  return ctx.session.state.models.map((m) => {
    const finetuned = isFinetuned(m);
    const base = finetuned ? records.find((r) => r.id === m.id)?.base.name : undefined;
    return {
      name: m.name,
      source: finetuned ? "finetuned" : "huggingface",
      detail: finetuned ? (base ? `from ${base}` : null) : m.folder,
      status: served.some((s) => s.entry.id === m.id) ? "running" : "starting",
      removable: true,
      note: finetuned ? "Kept in this browser: ✕ stops serving it; tick it under Hugging Face and Submit to serve it again." : null,
      hf_id: m.id,
      zoo_id: null,
      cellpose: null,
    };
  });
}

const NO_DATASET = "No dataset is open. Enter its URL in the viewer panel first; the models are then run on it.";

export function modelRoutes(router: Router): void {
  router.get("/api/huggingface-models", (_req, _params, ctx) => listModels(ctx, false));
  router.post("/api/huggingface-models/refresh", (_req, _params, ctx) => listModels(ctx, true));

  // The zoo's weights are not web exports: an empty, never stale list, in
  // the shape bioimage_catalog.list_bioimage_models answers.
  const zoo = () => json(200, { models: [], fetched: null, stale: false });
  router.get("/api/bioimage-models", zoo);
  router.post("/api/bioimage-models/refresh", zoo);

  router.post("/api/models", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const selected = strings(body.selected_models, "selected_models");
    if (!Array.isArray(selected)) return badRequest(selected.error);
    const hf = strings(body.selected_hf_models, "selected_hf_models");
    if (!Array.isArray(hf)) return badRequest(hf.error);
    const ignored: string[] = [];
    if (Array.isArray(body.selected_bioimage_models) && body.selected_bioimage_models.length) ignored.push("BioImage Model Zoo");
    if (Array.isArray(body.selected_cellpose_models) && body.selected_cellpose_models.length) ignored.push("Cellpose");
    if (typeof body.resample === "boolean" && body.resample !== ctx.session.state.resample) await ctx.session.update({ resample: body.resample });
    const resolved = await resolveAll(ctx, [...selected, ...hf]);
    if ("error" in resolved) return badRequest(resolved.error);
    if (resolved.entries.length && !ctx.infer.dataset()) return badRequest(NO_DATASET);
    // A chain with no input normalizers takes the ones the first model
    // started declares, before it is served (serving.ts setDeclaredInputNorm).
    await setDeclaredInputNorm(ctx, resolved.entries.filter((e) => !ctx.session.state.models.some((m) => m.id === e.id)));
    await serveModels(ctx, resolved.entries, ctx.session.state.chain);
    await ctx.session.update({ models: resolved.entries });
    return json(200, {
      message: "Data received successfully",
      models: selected,
      hf_models: hf,
      bioimage_models: [],
      cellpose_models: [],
      ...(ignored.length ? { note: `${ignored.join(" and ")} models are not available in the browser and were not started.` } : {}),
    });
  });

  // The Loaded models list (routes/models.py's /api/models/loaded), and its
  // ✕: stop serving that one model, whatever else is ticked on the tab.
  // Beside it, the chain the models run through and the model whose
  // declared input normalization its input part is: the page reads them
  // after every model change, since a model's start can set that
  // normalization (serving.ts setDeclaredInputNorm).
  router.get("/api/models/loaded", async (_req, _params, ctx) => {
    const { chain, models } = ctx.session.state;
    return json(200, { success: true, models: await loadedModels(ctx), pipeline: chain, input_norm_from: inputNormDeclaredBy(chain, models) });
  });

  router.post("/api/models/remove", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.name === "string" ? body.name.trim() : "";
    if (!name) return badRequest("name is required");
    const loaded = ctx.session.state.models.some((m) => m.name === name) || ctx.infer.served().some((s) => s.entry.name === name);
    if (!loaded) return json(404, { success: false, error: `No model named ${name} is running` });
    await unserveModel(ctx, name);
    return json(200, { success: true, removed: name, models: await loadedModels(ctx) });
  });

  // Add a model by reference: what a Hugging Face ONNX model resolves to (a
  // folder of the built-in catalog, or of any repo laid out as it is, which
  // the catalog then lists: Catalog.resolve), or a model kept in this
  // browser, in the shape models/resolve.py's Resolved gives, as the
  // server's onnx type's entry (routes/config.ts serverParams, without the
  // file: the one run is chosen for this GPU). A ref the catalog finds
  // wanting (no such public repo, no such folder) is a 400 with its reason,
  // as the server answers resolve's ValueError; anything else (a path, a
  // checkpoint, a zoo id) needs a server.
  router.post("/api/models/resolve", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const ref = typeof body.ref === "string" ? body.ref.trim() : "";
    if (!ref) return badRequest("ref is required");
    let entry: ModelEntry | null;
    try {
      entry = await ctx.catalog.resolve(ref);
    } catch (e) {
      return badRequest(message(e));
    }
    if (!entry) return unavailable(`${ref} is not a model the browser can add: ${CAN_ADD}.`);
    // The catalog's default variant; the one served is chosen for the GPU when it is.
    const v = entry.variant;
    return json(200, {
      success: true,
      type: "onnx",
      params: serverParams(entry, false),
      name: entry.name,
      env: null,
      how: "an ONNX export run in the browser (onnxruntime-web)",
      needs: [],
      notes: [
        `${entry.manifest.task}: ${entry.manifest.channels.join(", ")} at ${entry.manifest.output_voxel_size_nm.join("x")} nm.`,
        `Variants ${entry.manifest.variants.map((x) => x.variant).filter((x, i, a) => a.indexOf(x) === i).join(", ")}; ${v.variant} at ${sizeText(v.input_size)} voxels is ${v.file_mb} MB. The one run is chosen for this GPU.`,
        ...(entry.repo !== HF_REPO && !isFinetuned(entry) ? [`${entry.repo}'s models are listed under Hugging Face from now on.`] : []),
      ],
    });
  });

  router.post("/api/models/add", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const entry = body.entry;
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return badRequest("entry: expected a model entry");
    const e = entry as Record<string, unknown>;
    const name = typeof e.name === "string" ? e.name.trim() : "";
    if (!name) return badRequest("The model needs a name");
    if (ctx.infer.served().some((s) => s.entry.name === name)) return json(409, { success: false, error: `A model named ${name} is already running` });
    // The entry as resolve answered it (repo and folder), else by its name.
    const ref = typeof e.repo === "string" && typeof e.folder === "string" ? `${e.repo}/${e.folder}` : name;
    const resolved = await resolveOne(ctx, ref);
    const model = "entry" in resolved ? resolved.entry : await ctx.catalog.resolve(name).catch(() => null);
    if (!model) return badRequest("error" in resolved ? resolved.error : `${name} is not a model the browser can run`);
    if (!ctx.infer.dataset()) return badRequest(NO_DATASET);
    const entries = [...ctx.session.state.models.filter((m) => m.id !== model.id), model];
    // As Submit: a chain with no input normalizers takes the ones it declares.
    if (!ctx.session.state.models.some((m) => m.id === model.id)) await setDeclaredInputNorm(ctx, [model]);
    await serveModels(ctx, entries, ctx.session.state.chain);
    await ctx.session.update({ models: entries });
    return json(200, { success: true, name: model.name });
  });

  router.get("/api/model_advice", (_req, _params, ctx) => {
    return json(200, { success: true, models: ctx.infer.served().map((s) => advice(s, ctx.session.state.chain)) });
  });
}

function sizeText(size: number | number[]): string {
  return Array.isArray(size) ? size.join("x") : String(size);
}

/** A value's number, when it is one: a number, or a string spelling one
 * (the form sends "255.0" for a manifest's 255); null otherwise ("" and
 * null are not numbers). */
function numeric(value: unknown): number | null {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value))) return Number(value);
  return null;
}

/** Whether a configured parameter value is the declared one: numbers by
 * value, anything else as text, trimmed and without regard to case (a
 * form's "False", Python's False and JSON's false are one value; an
 * expression is compared as written). probe.py's same_value. */
export function sameValue(configured: unknown, declared: unknown): boolean {
  const a = numeric(configured), b = numeric(declared);
  if (a !== null && b !== null) return a === b;
  return String(configured).trim().toLowerCase() === String(declared).trim().toLowerCase();
}

export interface InputNormDifference {
  step: string;
  param: string;
  configured: unknown;
  declared: unknown;
}

/** Where the configured input normalization departs from what a model
 * declares, when it has the declared steps in the declared order: one
 * {step, param, configured, declared} per declared parameter whose value is
 * not the configured one (sameValue), in step order; [] when every value is
 * the declared one, and when the steps themselves differ, which the step
 * names already say. A parameter the declaration does not name (a
 * MinMaxNormalizer's invert) is not compared. `declared` is {name: params}
 * in order, as advice builds it from the manifest. probe.py's
 * input_norm_differences. */
export function inputNormDifferences(configured: Step[], declared: Record<string, Record<string, unknown>>): InputNormDifference[] {
  const order = Object.keys(declared);
  if (!order.length || JSON.stringify(stepNames(configured)) !== JSON.stringify(order)) return [];
  const differences: InputNormDifference[] = [];
  for (const step of configured) {
    for (const [param, value] of Object.entries(declared[step.name] ?? {})) {
      if (!sameValue(step[param], value)) differences.push({ step: step.name, param, configured: step[param] ?? null, declared: value });
    }
  }
  return differences;
}

/** One model's entry of /api/model_advice. There is no output probe in the
 * browser (probe_available: false, and a review that says so, which the
 * banner does not show); the input normalization is the one the model's
 * manifest declares, which suggest_input_norm treats as declared metadata:
 * high confidence, and a match when the configured steps are those in
 * that order with the declared values (inputNormDifferences says where
 * a value is not). */
export function advice(served: ServedModel, chain: Chain) {
  const manifest = served.entry.manifest;
  const declared: Record<string, Record<string, unknown>> = {};
  for (const step of manifest.input_norm ?? []) {
    const { name, ...params } = step as Step;
    declared[name] = params;
  }
  const order = Object.keys(declared);
  const configuredNorm = stepNames(chain.input_norm);
  const differences = inputNormDifferences(chain.input_norm, declared);
  const output = manifest.output ?? "";
  return {
    model: served.entry.name,
    probe_available: false,
    out_channels: served.info.output_channels,
    output_class: /logit|unbounded|distance/i.test(output) ? "unbounded" : /sigmoid|probab|\[0, ?1\]/i.test(output) ? "unit" : null,
    output_min: null,
    output_max: null,
    configured_postprocess: stepNames(chain.postprocess),
    configured_input_norm: configuredNorm,
    input_voxel_size: served.info.input_voxel_size,
    input_resampled_from: null,
    input_relabelled_from: null,
    postprocess_review: {
      level: "unknown",
      message: `No output probe in the browser. The model's manifest says its output is: ${output || "not stated"}.`,
      suggest: [],
    },
    input_norm_suggestion: order.length
      ? { input_norm: declared, order, reason: "Declared in the model's manifest (cellmap/onnx-web-models)", confidence: "high" }
      : { input_norm: null, order: [], reason: "The model's manifest does not say", confidence: "low" },
    input_norm_differences: differences,
    input_norm_matches: order.length > 0 && JSON.stringify(configuredNorm) === JSON.stringify(order) && differences.length === 0,
  };
}
