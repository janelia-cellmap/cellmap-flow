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
import type { ApiRequest, ApiResponse, AppContext, Chain, FinetunedRecord, ModelEntry, RankFit, ServedModel, SessionState, Step } from "../../contracts";
import { hasAffinityChannels } from "../../finetune/volume";
import { PYTHON } from "../../generated/python-data";
import { compileExpression } from "../../infer/expression";
import { HF_REPO, isFinetuned, listFacts } from "../../models";
import type { Router } from "../router";
import { badRequest, json, message, objectBody, unavailable } from "../router";
import { inputNormDeclaredBy, serveModels, setDeclaredInputNorm, setNeededPostprocess, stepNames, unserveModel } from "../serving";
import { serverParams } from "./config";

/** The catalog as /api/huggingface-models lists it: {model id: metadata},
 * the id "cellmap/onnx-web-models/setup_54" (huggingface.js shows its last
 * segment and the metadata's description), with what its row tags it with
 * and sorts it by (models/metadata.ts listFacts: its format, its ways to
 * finetune, its size and precisions), as the ranking's candidates have
 * them and the server's listing names them. A model the session is serving
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
    return json(200, Object.fromEntries(entries.map((e) => [e.id, { ...ctx.catalog.metadataFor(e), ...listFacts(e), ...(serving.has(e.id) ? { selected: true } : {}) }])));
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
  `the browser runs the ONNX models of ${HF_REPO} (a folder, such as setup_54), those of any public Hugging Face model repo laid out as that one is ` +
  "(<owner>/<repo>/<folder>, a folder holding a manifest.json and the .onnx files it names), and an .onnx file opened with Open an .onnx file…";

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
 * serves it after a reload, "waiting" while no dataset is open to serve it
 * on, as after a link with no dataset), and the Hugging Face row that ticks it (the
 * list's value, the entry's id), which the tab unticks on a remove. Every
 * model can be removed: nothing runs outside the page, and a finetuned one
 * stays kept. */
export interface LoadedModel {
  name: string;
  source: "huggingface" | "finetuned";
  detail: string | null;
  status: "running" | "starting" | "waiting";
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
      status: served.some((s) => s.entry.id === m.id) ? "running" : ctx.session.state.datasetUrl ? "starting" : "waiting",
      removable: true,
      note: finetuned ? "Kept in this browser: ✕ stops serving it; tick it under Hugging Face and Submit to serve it again." : null,
      hf_id: m.id,
      zoo_id: null,
      cellpose: null,
    };
  });
}

/** The catalog's entry for the model a `{ref}` body names, as Add a model
 * resolves it: the Models tab's (POST /api/models/resolve) and Which model
 * to start from's (POST /api/finetune/rank/add, rank.ts), so the two take
 * the same refs and refuse the same ones. Or the answer that refuses it:
 * a 400 for a body with no ref, and for a ref that is not a model the
 * browser can run (anything but the catalog's ONNX models: a path, a
 * checkpoint, a zoo id) the 503 that says it needs a server. */
export async function resolveRef(req: ApiRequest, ctx: AppContext): Promise<{ entry: ModelEntry } | { refused: ApiResponse }> {
  const body = objectBody(req);
  if (!body) return { refused: badRequest("expected a JSON object") };
  const ref = typeof body.ref === "string" ? body.ref.trim() : "";
  if (!ref) return { refused: badRequest("ref is required") };
  let entry: ModelEntry | null;
  try {
    entry = await ctx.catalog.resolve(ref);
  } catch (e) {
    return { refused: badRequest(message(e)) };
  }
  if (!entry) return { refused: unavailable(`${ref} is not a model the browser can add: ${CAN_ADD}.`) };
  return { entry };
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
    // started declares, and one with no postprocessing what the first
    // model's output needs, before it is served (serving.ts
    // setDeclaredInputNorm, setNeededPostprocess).
    const starting = resolved.entries.filter((e) => !ctx.session.state.models.some((m) => m.id === e.id));
    await setDeclaredInputNorm(ctx, starting);
    await setNeededPostprocess(ctx, starting, resolved.entries);
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
  // Beside it, the chain the models run through, the model whose declared
  // input normalization its input part is, and the model whose settings its
  // postprocessing is (postprocess_from) with whether the ranking fitted
  // them to the paint (postprocess_fitted; false: the model's start set
  // what its output needs, a sigmoid for logits; both null when no loaded
  // model's: postprocessSetBy). The page reads them after every model
  // change, since a model's start can set either part (serving.ts
  // setDeclaredInputNorm, setNeededPostprocess).
  router.get("/api/models/loaded", async (_req, _params, ctx) => {
    const { chain, models } = ctx.session.state;
    const post = postprocessSetBy(ctx.session.state);
    return json(200, {
      success: true, models: await loadedModels(ctx), pipeline: chain, input_norm_from: inputNormDeclaredBy(chain, models),
      postprocess_from: post?.model ?? null, postprocess_fitted: post?.fitted ?? null,
    });
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

  // Add a model by reference: what a Hugging Face ONNX model resolves to
  // (resolveRef), in the shape models/resolve.py's Resolved gives, as the
  // server's onnx type's entry (routes/config.ts serverParams, without the
  // file: the one run is chosen for this GPU).
  router.post("/api/models/resolve", async (req, _params, ctx) => {
    const resolved = await resolveRef(req, ctx);
    if ("refused" in resolved) return resolved.refused;
    const { entry } = resolved;
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
    // As Submit: a chain with no input normalizers takes the ones it
    // declares, one with no postprocessing what its output needs.
    if (!ctx.session.state.models.some((m) => m.id === model.id)) {
      await setDeclaredInputNorm(ctx, [model]);
      await setNeededPostprocess(ctx, [model], entries);
    }
    await serveModels(ctx, entries, ctx.session.state.chain);
    await ctx.session.update({ models: entries });
    return json(200, { success: true, name: model.name });
  });

  // Without model_advice.py's `starting` and `digest`: a model's start here
  // sets the chain before its Submit answers, and the page reads the chain
  // again after every model change, so it has no later change to look for.
  router.get("/api/model_advice", (_req, _params, ctx) => {
    const { chain, rankFits } = ctx.session.state;
    return json(200, { success: true, models: ctx.infer.served().map((s) => advice(s, chain, rankFits[s.entry.name])) });
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

/** Where two expressions are compared: a declared "x*2-1" and a typed
 * "2*x-1" are one map, which the text alone does not say. probe.py's
 * EXPRESSION_SAMPLES. */
const EXPRESSION_SAMPLES = [-2, -1, -0.5, 0, 0.25, 0.5, 1, 2, 255];

/** Whether a configured parameter value is the declared one: numbers by
 * value, anything else as text, trimmed and without regard to case (a
 * form's "False", Python's False and JSON's false are one value), and two
 * expressions by what they give at EXPRESSION_SAMPLES, each through the
 * whitelist LambdaNormalizer runs it through (compileExpression).
 * probe.py's same_value. */
export function sameValue(configured: unknown, declared: unknown): boolean {
  const a = numeric(configured), b = numeric(declared);
  if (a !== null && b !== null) return a === b;
  if (String(configured).trim().toLowerCase() === String(declared).trim().toLowerCase()) return true;
  try {
    const f = compileExpression(String(configured)), g = compileExpression(String(declared));
    return EXPRESSION_SAMPLES.every((x) => {
      const fx = f(x), gx = g(x);
      return Math.abs(fx - gx) <= 1e-8 + 1e-5 * Math.abs(gx);
    });
  } catch {
    return false;
  }
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

/** A model's postprocess_review, as probe.py's review_postprocess answers
 * it: a level ("unknown" when there is nothing to go on), a message, the
 * chain suggested (step names) and the parameters its steps need. */
interface PostprocessReview {
  level: "ok" | "optional" | "warn" | "unknown";
  message: string;
  suggest: string[];
  params?: Record<string, Record<string, number | string>>;
}

/** probe.py's NEIGHBORHOOD_OFFSETS: the mutex watershed's offsets,
 * longest-range last, cut to a model's channels. */
const NEIGHBORHOOD_OFFSETS = PYTHON.advice.neighborhood_offsets;

/** probe.py's suggest_affinity_chain: an affinity model's chain into a
 * segmentation, the sigmoid first for logits, then DefaultPostprocessor's
 * 0-255 into AffinityPostprocessor. */
function affinityChain(outputClass: string): string[] {
  return [...(outputClass === "unbounded" ? ["SigmoidPostprocessor"] : []), "DefaultPostprocessor", "AffinityPostprocessor"];
}

/** probe.py's suggest_postprocess_params: the values the steps of `chain`
 * need, from the range that reaches each (a DefaultPostprocessor after a
 * sigmoid maps [0, 1] onto 0-255, where its defaults map [-1, 1]), and an
 * AffinityPostprocessor's bias, `bias` (the cut-off the ranking fitted to
 * the paint, its row's threshold) or the midpoint 0.5 when null, and its
 * neighborhood, cut to the model's `channels` and written as Python writes
 * a list; only the steps whose defaults are wrong for the chain, and the
 * AffinityPostprocessor's bias and neighborhood always. */
function postprocessParams(chain: string[], outputClass: string, channels: number, bias: number | null): Record<string, Record<string, number | string>> {
  const params: Record<string, Record<string, number | string>> = {};
  let range: number[] | null = outputClass === "unit" ? [0, 1] : outputClass === "signed_unit" ? [-1, 1] : null;
  for (const name of chain) {
    if (name === "SigmoidPostprocessor") {
      range = [0, 1];
    } else if (name === "DefaultPostprocessor") {
      if (range) params[name] = { clip_min: range[0], clip_max: range[1], bias: -range[0] + 0, multiplier: 255 / (range[1] - range[0]) };
      range = [0, 255];
    } else if (name === "ThresholdPostprocessor") {
      // On logits the boundary is 0, not a sigmoid's 0.5.
      params[name] = { threshold: range ? (range[0] + range[1]) / 2 : 0 };
      range = [0, 1];
    } else if (name === "AffinityPostprocessor") {
      // Its bias the cut-off fitted, where the score called a painted pair
      // joined, so the objects are cut where the score read the model.
      const offsets = NEIGHBORHOOD_OFFSETS.slice(0, channels);
      params[name] = { bias: bias ?? 0.5, ...(channels ? { neighborhood: `[${offsets.map((o) => `[${o.join(", ")}]`).join(", ")}]` } : {}) };
      range = null;
    }
  }
  return params;
}

/** The review of the postprocessing `served` is shown through (`names`,
 * the chain's step names), as probe.py's review_postprocess gives an
 * affinity model's, from the output class its served info reads off the
 * model (there is no output probe in the browser): a threshold ahead of
 * the watershed warned of, as is an output nothing puts in [0, 1] for it;
 * and, once its affinities show (as probabilities, the sigmoid its start
 * or the ranking set on logits, with the ranking's threshold), the chain
 * into a segmentation offered as "optional", one quiet line with its
 * Apply. The chains suggested join neighbours above `threshold`, the one
 * the ranking fitted to the paint for the model (its rankFits'), or 0.5
 * when it has none, and say so. Any other model, or an affinity model
 * whose activation is still to set, is "unknown", which the banner does
 * not show. */
function postprocessReview(served: ServedModel, names: string[], threshold: number | null): PostprocessReview {
  const outputClass = served.info.output_class ?? null;
  const channels = served.info.output_channels;
  if (outputClass && hasAffinityChannels(served.entry)) {
    const verdict = (level: PostprocessReview["level"], message: string, suggest: string[]): PostprocessReview =>
      ({ level, message, suggest, params: postprocessParams(suggest, outputClass, channels, threshold) });
    // The affinity the watershed joins neighbours above, as the messages say it.
    const cut = threshold === null ? "0.5" : `${threshold} (fitted to your paint)`;
    if (names.includes("AffinityPostprocessor")) {
      // Whether the affinities reach it in [0, 1], or DefaultPostprocessor's 0-255.
      const scaled = names.includes("DefaultPostprocessor") || names.includes("SigmoidPostprocessor") || outputClass === "unit";
      // A threshold ahead of it leaves 0s and 1s, which it reads as 0-255
      // and divides by 255: no edge reaches its bias. The chain is
      // suggested back without it, or as the whole affinity chain when
      // nothing else ahead puts the affinities in [0, 1].
      const ahead = names.indexOf("AffinityPostprocessor");
      if (names.slice(0, ahead).includes("ThresholdPostprocessor")) {
        return verdict("warn",
          `A ThresholdPostprocessor comes before the AffinityPostprocessor, so the watershed gets 0s and 1s instead of affinities and finds no objects: take the threshold out, and neighbours join above ${cut}.`,
          scaled ? names.filter((n, i) => i > ahead || n !== "ThresholdPostprocessor") : affinityChain(outputClass));
      }
      if (!scaled) {
        return verdict("warn",
          "AffinityPostprocessor needs affinities in [0, 1] (or the 0-255 that DefaultPostprocessor produces), but this model's output is not bounded to [0, 1] and nothing ahead of it rescales it, so the watershed works on the wrong values.",
          affinityChain(outputClass));
      }
      return verdict("ok", "Affinity chain looks complete.", []);
    }
    // The chain into a segmentation, offered over whatever shows the
    // affinities: the ranking's row among them, the sigmoid and the
    // threshold fitted to the paint (only the threshold on a tanh's or a
    // sigmoid's output). It replaces the threshold, which would leave the
    // watershed 0s and 1s, and joins neighbours above the same cut-off.
    // Unless the activation needs setting first: the sigmoid logits with
    // nothing set need, or a second one to take away.
    const unset = (outputClass === "unbounded" && !names.length) || ((outputClass === "unit" || outputClass === "signed_unit") && names.includes("SigmoidPostprocessor"));
    if (!unset) return verdict("optional", `its ${channels} channels look like affinities, which this chain turns into a segmentation, joined above ${cut}.`, affinityChain(outputClass));
  }
  const { output_class, output_class_from, output_op } = served.info;
  const from = output_class_from === "graph"
    ? `Its network ends in a ${output_op}, so its output is ${output_class}.`
    : `Its network's last op does not say what its output is; the model's manifest says: ${served.entry.manifest.output || "not stated"}.`;
  return { level: "unknown", message: `No output probe in the browser. ${from}`, suggest: [] };
}

/** One model's entry of /api/model_advice. There is no output probe in the
 * browser (probe_available: false): the output class is the served model's
 * (ModelInfo.output_class: its network's last op, else its manifest's
 * text), with where it was read (output_class_from, output_op) as the
 * server's advice gives it. The postprocessing review is an affinity
 * model's (postprocessReview, its chain cut at the threshold `fit` holds),
 * and "unknown" for any other, which the banner does not show, saying
 * where the class was read. The input normalization is the one the model's
 * manifest declares, which suggest_input_norm treats as declared metadata:
 * high confidence, and a match when the configured steps are those in
 * that order with the declared values (inputNormDifferences says where
 * a value is not).
 *
 * The input normalization the ranking fitted this model to (`fit`, the
 * session's rankFits for it: the steps its last score was read through,
 * which Use this model, Fit to Paint and Use these settings set, stretched
 * to the raw under the paint, say) matches as it is, while it is the
 * chain's (input_norm_fitted): suggested the declared ones back, a second
 * Apply undid the ranking. */
export function advice(served: ServedModel, chain: Chain, fit?: RankFit) {
  const manifest = served.entry.manifest;
  const declared: Record<string, Record<string, unknown>> = {};
  for (const step of manifest.input_norm ?? []) {
    const { name, ...params } = step as Step;
    declared[name] = params;
  }
  const order = Object.keys(declared);
  const configuredNorm = stepNames(chain.input_norm);
  const fitted = !!fit?.input_norm.length && sameSteps(fit.input_norm, chain.input_norm);
  const differences = fitted ? [] : inputNormDifferences(chain.input_norm, declared);
  const { output_class, output_class_from, output_op } = served.info;
  const configuredPost = stepNames(chain.postprocess);
  return {
    model: served.entry.name,
    probe_available: false,
    out_channels: served.info.output_channels,
    output_class: output_class ?? null,
    output_class_from: output_class_from ?? null,
    output_op: output_op ?? null,
    output_min: null,
    output_max: null,
    configured_postprocess: configuredPost,
    configured_input_norm: configuredNorm,
    input_voxel_size: served.info.input_voxel_size,
    input_resampled_from: null,
    input_relabelled_from: null,
    postprocess_review: postprocessReview(served, configuredPost, fit?.threshold ?? null),
    input_norm_suggestion: order.length
      ? { input_norm: declared, order, reason: "Declared in the model's manifest (cellmap/onnx-web-models)", confidence: "high" }
      : { input_norm: null, order: [], reason: "The model's manifest does not say", confidence: "low" },
    input_norm_fitted: fitted,
    input_norm_differences: differences,
    input_norm_matches: fitted || (order.length > 0 && JSON.stringify(configuredNorm) === JSON.stringify(order) && differences.length === 0),
  };
}

/** Whether two chains are the same steps with the same parameters, each
 * value compared as sameValue compares it ("0" is 0): portable/rank.py's
 * same_steps, which the server's advice and postprocess_set_by compare
 * with. */
function sameSteps(a: readonly Step[], b: readonly Step[]): boolean {
  return a.length === b.length && a.every((x, i) => {
    const y = b[i], keys = Object.keys(x);
    return keys.length === Object.keys(y).length && keys.every((k) => k in y && sameValue(x[k], y[k]));
  });
}

/** The loaded model whose settings the session chain's postprocessing is,
 * step for step (sameSteps), with whether the ranking fitted them to the
 * paint (routes/pipeline.py's postprocess_set_by): the ranking's first (a
 * model's rankFits, what Use this model, Fit to Paint and Use these
 * settings set, the sigmoid and the threshold the paint is fitted best
 * at), else a model's start's (startPostprocess, serving.ts
 * setNeededPostprocess: the sigmoid its logits need); null when it is no
 * loaded model's, or the chain has none. An Apply of other steps since is
 * the user's, and nothing has to forget it. */
function postprocessSetBy(state: SessionState): { model: string; fitted: boolean } | null {
  const live = state.chain.postprocess;
  if (!live.length) return null;
  const names = state.models.map((m) => m.name);
  const ranked = names.find((name) => {
    const fitted = state.rankFits[name]?.postprocess;
    return fitted && sameSteps(fitted, live);
  });
  if (ranked) return { model: ranked, fitted: true };
  const started = state.startPostprocess;
  return started && names.includes(started.model) && sameSteps(started.postprocess, live) ? { model: started.model, fitted: false } : null;
}
