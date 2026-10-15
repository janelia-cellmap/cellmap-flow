// The Models tab's routes (routes/models.py, model_resolve.py,
// model_advice.py): which models there are, which to serve, and advice about
// the chain they are served through.
//
// In the browser the models are the ONNX exports of cellmap/onnx-web-models
// (the Catalog); "serving" one is an Engine view with an onnxruntime-web
// session (Inference.serve), so Submit serves and disposes views where the
// Python route starts and kills inference servers. The BioImage Model Zoo
// and Cellpose have no web exports: their lists are empty and their
// selections ignored (the template hides their panels in browser mode).
import type { AppContext, ModelEntry, ServedModel, Step } from "../../contracts";
import type { Router } from "../router";
import { badRequest, json, message, objectBody, unavailable } from "../router";
import { serveModels, stepNames } from "../serving";

/** The catalog as /api/huggingface-models lists it: {model id: metadata},
 * the id "cellmap/onnx-web-models/setup_54" (models-tab.js shows its last
 * segment and the metadata's description). `refresh` reads the repository
 * again instead of the cached listing. A listing that fails is {error}
 * with a 500, as the Flask route answers; the tab shows the message. */
async function listModels(ctx: AppContext, refresh: boolean) {
  try {
    const entries = refresh ? await ctx.catalog.refresh() : await ctx.catalog.list();
    return json(200, Object.fromEntries(entries.map((e) => [e.id, ctx.catalog.metadataFor(e)])));
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

/** The catalog entries for the references the dashboard sent (repo ids from
 * the Hugging Face list, names of models added by hand), one per model, or
 * the first reference that is not a model the browser can run. */
async function resolveAll(ctx: AppContext, refs: string[]): Promise<{ entries: ModelEntry[] } | { error: string }> {
  const entries: ModelEntry[] = [];
  for (const ref of refs) {
    const entry = await ctx.catalog.resolve(ref);
    if (!entry) return { error: `${ref} is not a model the browser can run: only the ONNX exports listed under Hugging Face are.` };
    if (!entries.some((e) => e.id === entry.id)) entries.push(entry);
  }
  return { entries };
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

  // Add a model by reference: what a Hugging Face ONNX model resolves to, in
  // the shape models/resolve.py's Resolved gives; anything else (a path, a
  // checkpoint, a zoo id) needs a server.
  router.post("/api/models/resolve", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const ref = typeof body.ref === "string" ? body.ref.trim() : "";
    if (!ref) return badRequest("ref is required");
    const entry = await ctx.catalog.resolve(ref);
    if (!entry) {
      return unavailable(`Only the ONNX models of Hugging Face repositories such as cellmap/onnx-web-models can be added in the browser; ${ref} is not one of them.`);
    }
    // The catalog's default variant; the one served is chosen for the GPU when it is.
    const v = entry.variant;
    return json(200, {
      success: true,
      type: "huggingface",
      params: { repo: entry.repo, folder: entry.folder },
      name: entry.name,
      env: null,
      how: "an ONNX export run in the browser (onnxruntime-web)",
      needs: [],
      notes: [
        `${entry.manifest.task}: ${entry.manifest.channels.join(", ")} at ${entry.manifest.output_voxel_size_nm.join("x")} nm.`,
        `Variants ${entry.manifest.variants.map((x) => x.variant).filter((x, i, a) => a.indexOf(x) === i).join(", ")}; ${v.variant} at ${sizeText(v.input_size)} voxels is ${v.file_mb} MB. The one run is chosen for this GPU.`,
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
    const model = (await ctx.catalog.resolve(ref)) ?? (await ctx.catalog.resolve(name));
    if (!model) return badRequest(`${ref} is not a model the browser can run`);
    if (!ctx.infer.dataset()) return badRequest(NO_DATASET);
    const entries = [...ctx.session.state.models.filter((m) => m.id !== model.id), model];
    await serveModels(ctx, entries, ctx.session.state.chain);
    await ctx.session.update({ models: entries });
    return json(200, { success: true, name: model.name });
  });

  router.get("/api/model_advice", (_req, _params, ctx) => {
    const { chain } = ctx.session.state;
    const configuredNorm = stepNames(chain.input_norm);
    const configuredPost = stepNames(chain.postprocess);
    return json(200, { success: true, models: ctx.infer.served().map((s) => advice(s, configuredNorm, configuredPost)) });
  });
}

function sizeText(size: number | number[]): string {
  return Array.isArray(size) ? size.join("x") : String(size);
}

/** One model's entry of /api/model_advice. There is no output probe in the
 * browser (probe_available: false, and a review that says so, which the
 * banner does not show); the input normalization is the one the model's
 * manifest declares, which suggest_input_norm treats as declared metadata:
 * high confidence, and a match when the configured steps are those in
 * that order. */
export function advice(served: ServedModel, configuredNorm: string[], configuredPost: string[]) {
  const manifest = served.entry.manifest;
  const declared: Record<string, Record<string, unknown>> = {};
  for (const step of manifest.input_norm ?? []) {
    const { name, ...params } = step as Step;
    declared[name] = params;
  }
  const order = Object.keys(declared);
  const output = manifest.output ?? "";
  return {
    model: served.entry.name,
    probe_available: false,
    out_channels: served.info.output_channels,
    output_class: /logit|unbounded|distance/i.test(output) ? "unbounded" : /sigmoid|probab|\[0, ?1\]/i.test(output) ? "unit" : null,
    output_min: null,
    output_max: null,
    configured_postprocess: configuredPost,
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
    input_norm_matches: order.length > 0 && JSON.stringify(configuredNorm) === JSON.stringify(order),
  };
}
