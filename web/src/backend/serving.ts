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
// one is Inference.serve and stopping one is ServedModel.dispose.
import { ARGS_KEY } from "../contracts";
import type { AppContext, Chain, ModelEntry, ServedModel } from "../contracts";
import { entryFor } from "../models";
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
      tuned = entryFor(entry.folder, entry.manifest, ctx.catalog.chooseVariant(entry.manifest, limits), entry.repo);
    }
    ctx.log(`Serving ${tuned.name} (${tuned.variant.variant}, ${sizeLabel(tuned.variant.input_size)} voxels, ${tuned.variant.file_mb} MB)`);
    result.push(await ctx.infer.serve(tuned, chain, blob, () => ctx.catalog.modelBytes(tuned)));
  }
  const names = new Set(entries.map((e) => e.name));
  const replacing = options.replacing ?? {};
  const handedOver = new Set(Object.values(replacing));
  for (const old of before) {
    if (kept.has(old)) continue;
    await old.dispose();
    if (!names.has(old.entry.name)) {
      ctx.log(`Stopped serving ${old.entry.name}`);
      // A layer handed over goes when its successor's is set, below.
      if (!handedOver.has(old.entry.name)) ctx.viewer.removeLayer(old.entry.name);
    }
  }
  showLayers(ctx, result, chain, options.postprocessChanged ?? false, replacing);
  return result;
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
