// What the routes share about serving models: the chain's URL blob, which
// names each model's view (and so its layer source: the same chain gives the
// same source, so a resubmitted chain leaves Neuroglancer's chunks alone);
// serving a set of models through a chain and disposing of what is no longer
// wanted; and the layers the viewer shows for them.
//
// In server mode this is services/launch.py's update_run_models (start and
// stop inference servers) and routes/pipeline.py's _set_chain_and_redraw
// (re-point every layer at the new chain). Here a "server" is an Engine
// view with an onnxruntime-web session behind it (src/infer), so starting
// one is Inference.serve and stopping one is ServedModel.dispose. A model's
// start sets the input normalization it declares when the chain has none
// (setDeclaredInputNorm, routes/pipeline.py's set_declared_input_norm).
import { ARGS_KEY } from "../contracts";
import type { AppContext, Chain, ModelEntry, ServedModel, Step } from "../contracts";
import { SUPPORTED_NORMALIZERS } from "../infer/norms";
import { withVariant } from "../models";
import { chainDigest, encodeArgs } from "../shared/args";
import { shaderColor } from "../viewer/display";
import { PREDICTION_COLORS, predictionLayer } from "../viewer/layers";

/** The dashboard's address as the Python routes put it in the blob
 * (request.host_url: the scheme, host and the page's base, with the
 * trailing slash). An inference server posts back to it in server mode;
 * here it says which page a layer came from. */
export function pageBase(): string {
  if (typeof location === "undefined") return "http://localhost/";
  return new URL(".", location.href).href;
}

/** The chain's URL blob (shared/args.ts: minified JSON in URL-safe base64)
 * with the extras the Python dashboard adds, and its digest. */
export async function blobFor(chain: Chain): Promise<{ blob: string; digest: string }> {
  const digest = await chainDigest(chain);
  const blob = encodeArgs(chain, { dashboard_url: pageBase(), digest });
  return { blob, digest };
}

/** The blob a served model's view carries: `${name}__CFLOW_ARGS__${blob}__CFLOW_ARGS__`. */
export function blobOf(served: ServedModel): string {
  return served.view.split(ARGS_KEY)[1] ?? "";
}

/** Whether every one of `entries` is served through `blob` already: then a
 * redraw would show nothing new (pipeline.py's _shows). */
export function allServed(ctx: AppContext, entries: ModelEntry[], blob: string): boolean {
  const served = ctx.infer.served();
  return entries.every((e) => served.some((s) => s.entry.id === e.id && blobOf(s) === blob));
}

/** Make `chain` the session's and draw its models' layers through it
 * (pipeline.py's _set_chain_and_redraw): the models drawn, or null when no
 * dataset is open (the chain is kept and nothing drawn). The same chain
 * again over layers that carry it already draws nothing (it would reset
 * what the user set in the viewer, a layer hidden, an opacity). */
export async function setChain(ctx: AppContext, chain: Chain): Promise<ServedModel[] | null> {
  const previous = ctx.session.state.chain;
  const [{ blob, digest }, previousDigest] = await Promise.all([blobFor(chain), blobFor(previous).then((b) => b.digest)]);
  const entries = ctx.session.state.models;
  await ctx.session.update({ chain });
  if (!ctx.infer.dataset()) return null;
  if (digest === previousDigest && allServed(ctx, entries, blob)) return ctx.infer.served().filter((s) => entries.some((e) => e.id === s.entry.id));
  const postprocessChanged = JSON.stringify(previous.postprocess) !== JSON.stringify(chain.postprocess);
  const served = await serveModels(ctx, entries, chain, { blob, postprocessChanged });
  ctx.log(`Chain ${digest}: ${served.length ? served.map((s) => s.entry.name).join(", ") : "no model"} drawn through it`);
  return served;
}

/** The input normalization `entry`'s manifest declares (the steps the
 * browser feeds the model through), each step copied; [] for none.
 * pipeline.py's _declared_steps. */
export function declaredInputNorm(entry: ModelEntry): Step[] {
  return (entry.manifest.input_norm ?? []).map((step) => ({ ...step }));
}

/** Set the input normalization a model declares as the chain's, as part of
 * starting it, when the chain has none (pipeline.py's
 * set_declared_input_norm). Returns the name of the model whose declaration
 * it is, or null when nothing changed.
 *
 * The Models tab's Submit and Add (routes/models.ts) call this with the
 * models they are about to serve that are not served yet, before serving
 * them. The first of them whose manifest declares steps gives them, and the
 * postprocessing stays as it is; the chain is set through setChain, as PUT
 * /api/pipeline sets it, so the models served already are drawn through it
 * too. A chain that has input normalizers already is left alone, and after
 * this only the page's Apply changes it: the Input list shows the chain as
 * set (GET /api/models/loaded's `pipeline`, which the page reads after every
 * model change) and says where it came from. A model that declares nothing
 * changes nothing, and a declaration naming a step the browser cannot run
 * is skipped, saying so, and the next model's is tried. */
export async function setDeclaredInputNorm(ctx: AppContext, starting: readonly ModelEntry[]): Promise<string | null> {
  const { chain } = ctx.session.state;
  if (chain.input_norm.length) return null;
  for (const entry of starting) {
    const steps = declaredInputNorm(entry);
    if (!steps.length) continue;
    const unsupported = steps.find((step) => !SUPPORTED_NORMALIZERS.includes(step.name));
    if (unsupported) {
      ctx.log(`${entry.name}'s declared input normalization is not used: ${unsupported.name} is not available in the browser`);
      continue;
    }
    await setChain(ctx, { input_norm: steps, postprocess: chain.postprocess });
    ctx.log(`Input normalization set from ${entry.name}'s declaration: ${stepNames(steps).join(" → ")}`);
    return entry.name;
  }
  return null;
}

/** The name of the first of `entries` whose declared input normalization is
 * `chain`'s input part, step for step; null when none is, or the chain has
 * no input normalizers (pipeline.py's input_norm_declared_by). GET
 * /api/models/loaded answers it beside the chain, so the page can say where
 * an input normalization a model's start set came from. */
export function inputNormDeclaredBy(chain: Chain, entries: readonly ModelEntry[]): string | null {
  if (!chain.input_norm.length) return null;
  const live = JSON.stringify(chain.input_norm);
  return entries.find((entry) => JSON.stringify(declaredInputNorm(entry)) === live)?.name ?? null;
}

export interface ServeOptions {
  /** The chain's blob, when the caller has it already. */
  blob?: string;
  /** Whether the postprocessing changed: a layer's contrast range is a
   * property of the postprocessing (a SigmoidPostprocessor moves logits to
   * [0, 1]), so the user's shader is dropped then and the layer gets a fresh
   * one over the new range, as pipeline.py does. */
  postprocessChanged?: boolean;
  /** For a model of `entries`, the served model it takes over from (by
   * name): its layer's place and the user's settings on it go to the new
   * one's (a finetune's next iteration, `x-ft2` for `x-ft1`). */
  replacing?: Record<string, string>;
}

/** Serve `entries` through `chain`, in their order, and nothing else: a
 * model already served through this chain is kept as it is; one served
 * through another chain is served anew and the old view disposed of (after
 * the new one exists, so the two share the onnxruntime session rather than
 * rebuilding it); one no longer in `entries` is disposed of and its layer
 * removed. Then the viewer shows one layer per entry, coloured by position.
 * Returns the served models in `entries` order.
 *
 * A model is served in the variant for this machine's GPU: the catalog's
 * entries carry a default variant (fp32, the smallest block), and
 * Catalog.chooseVariant picks the precision and block size the GPU's limits
 * allow, so the entry is tuned here, just before it is served. The session
 * keeps the entries as the catalog gave them, and tunes again on each load. */
export async function serveModels(ctx: AppContext, entries: ModelEntry[], chain: Chain, options: ServeOptions = {}): Promise<ServedModel[]> {
  const blob = options.blob ?? (await blobFor(chain)).blob;
  const before = ctx.infer.served();
  const kept = new Set<ServedModel>();
  const result: ServedModel[] = [];
  let limits: Awaited<ReturnType<AppContext["infer"]["gpuLimits"]>> | null = null;
  for (const entry of entries) {
    const current = before.find((s) => s.entry.id === entry.id && blobOf(s) === blob);
    if (current) {
      kept.add(current);
      result.push(current);
      continue;
    }
    // Served already through another chain: the same variant again, so the
    // onnxruntime session is shared; else the variant for this GPU.
    const previous = before.find((s) => s.entry.id === entry.id);
    let tuned: ModelEntry;
    if (previous) tuned = previous.entry;
    else {
      limits ??= await ctx.infer.gpuLimits();
      tuned = withVariant(entry, ctx.catalog.chooseVariant(entry.manifest, limits));
    }
    ctx.log(`Serving ${tuned.name} (${tuned.variant.variant}, ${sizeLabel(tuned.variant.input_size)} voxels, ${tuned.variant.file_mb} MB)`);
    result.push(await ctx.infer.serve(tuned, chain, blob, () => ctx.catalog.modelBytes(tuned)));
  }
  const names = new Set(entries.map((e) => e.name));
  const replacing = options.replacing ?? {};
  const handedOver = new Set(Object.values(replacing));
  for (const old of before) {
    if (kept.has(old)) continue;
    // A layer handed over goes when its successor's is set, below.
    if (!names.has(old.entry.name)) await stopServing(ctx, old.entry.name, [old], { keepLayer: handedOver.has(old.entry.name) });
    else await old.dispose();
  }
  showLayers(ctx, result, chain, options.postprocessChanged ?? false, replacing);
  return result;
}

/** Stop serving the one model named `name`, the Loaded models list's ✕:
 * its views are disposed of, its layer removed and it leaves the session's
 * models, and nothing else changes. The other models stay served and their
 * layers are not redrawn (serveModels would draw them all again, coloured
 * by their new positions). A finetuned model's bytes and record stay in the
 * catalog, so it is listed and can be served again. In server mode this is
 * services/launch.py's remove_model. */
export async function unserveModel(ctx: AppContext, name: string): Promise<void> {
  await ctx.session.update({ models: ctx.session.state.models.filter((m) => m.name !== name) });
  await stopServing(ctx, name, ctx.infer.served().filter((s) => s.entry.name === name));
}

/** Stop serving the model named `name`, whose `views` are none while it is still starting. */
export async function stopServing(ctx: AppContext, name: string, views: readonly ServedModel[], options: { keepLayer?: boolean } = {}): Promise<void> {
  for (const view of views) await view.dispose();
  if (!options.keepLayer) ctx.viewer.removeLayer(name);
  ctx.log(`Stopped serving ${name}`);
}

/** Put one prediction layer per served model in the viewer, coloured by its
 * position as the Python dashboard colours them, keeping the shader the user
 * set unless the postprocessing changed. Then the shader is dropped (its
 * contrast range belongs to the old chain) but its colour is kept, so the
 * layer the user knows by its colour stays that colour (pipeline.py's
 * previous_shader). */
export function showLayers(ctx: AppContext, served: ServedModel[], chain: Chain, postprocessChanged: boolean, replacing: Record<string, string> = {}): void {
  if (!served.length) return;
  const layers: Record<string, ReturnType<typeof predictionLayer>> = {};
  served.forEach((s, i) => {
    const name = s.entry.name;
    const current = ctx.viewer.layerShader(name);
    const color = (postprocessChanged ? shaderColor(current?.shader) : null) ?? PREDICTION_COLORS[i % PREDICTION_COLORS.length];
    layers[name] = predictionLayer(s, chain, color, postprocessChanged ? undefined : (current ?? undefined));
  });
  ctx.viewer.setLayers(layers, replacing);
}

/** "148" for a cubic block, "148x148x64" otherwise. */
function sizeLabel(size: number[]): string {
  return size.every((n) => n === size[0]) ? String(size[0] ?? "") : size.join("x");
}

/** The names of a chain's steps, for the advice and the export. */
export function stepNames(steps: Chain["input_norm"]): string[] {
  return steps.map((s) => s.name);
}
