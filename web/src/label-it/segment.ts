// Seed from Prediction's segmentation: what the Finetune tab's "Label it"
// tools make of a model's prediction over the box on screen. The port of
// cellmap_flow/dashboard/routes/finetune/view_labels.py's probability,
// foreground_mask, _seed_plan, methods_for and segment_prediction for the
// browser. The segmenters themselves are already ported and parity-tested:
// connected components with scipy.ndimage.label's numbering
// (infer/label.ts, LabelPostprocessor's) and the mutex watershed with
// mwatershed.agglom's (infer/affinity.ts, AffinityPostprocessor's), so a
// seed's objects are the ones the server's seed makes of the same output.
//
// The threshold is a probability whatever the model's activation (the
// server's rule): the raw output is turned into one by its output class
// (model_info's output_class, serving/probe.py's vocabulary), so 0.5 is the
// model's own decision boundary for every model and the affinity channels
// can be averaged. The arithmetic follows numpy's float32 where the server's
// runs in float32 (scipy's expit in float32 steps, the channel mean summed
// channel by channel and divided once, the threshold compared as a float32),
// so a voxel is foreground here when it is there; expit can still differ by
// one float32 ulp from glibc's expf on about 3 in 10,000 values, which moves
// a voxel only when its probability is within that ulp of the threshold.
//
// What the browser leaves out: "distance_watershed" (segment.py's watershed
// of a distance map grown from its h-maxima) and "instances" (a model that
// serves integer ids, Cellpose's masks). The browser's models are those of
// Hugging Face's cellmap/onnx-web-models: affinity models (channels
// mito_aff_z/y/x) and binary ones (nuc, er), all raw logits. None outputs a
// distance or instance ids, so methodsFor never offers either, and a request
// for one is refused as the server refuses a method the model does not
// offer (the page lists them greyed out, saying what they need).
import type { ModelEntry } from "../contracts";
import { affinityOffsets } from "../finetune/targets";
import { hasAffinityChannels } from "../finetune/volume";
import { mutexWatershed } from "../infer/affinity";
import { connectedComponents } from "../infer/label";

/** How a seed makes objects of the prediction (view_labels.SEED_METHODS), the
 * server's four, so a request names any of them as it would there; which a
 * model offers is methodsFor's. */
export const SEED_METHODS = ["components", "mutex_watershed", "distance_watershed", "instances"] as const;
export type SeedMethod = (typeof SEED_METHODS)[number];

/** What a model is trained on, as the browser tells it (the server's
 * autodetect_output_type, by its channel-name test): "affinities" for a
 * model with a channel named `*_aff*`, else "binary". */
export type OutputType = "affinities" | "binary";

/** serving/probe.py's activation classes. */
export const UNIT = "unit";
export const SIGNED_UNIT = "signed_unit";
export const UNBOUNDED = "unbounded";

export function outputTypeOf(entry: ModelEntry): OutputType {
  return hasAffinityChannels(entry) ? "affinities" : "binary";
}

/** The seed methods that fit a model's output, the best fit first
 * (view_labels.methods_for): the mutex watershed for affinities, then
 * components, which fits every output and so is always offered, last. The
 * server's distance_watershed and instances fit outputs no browser model has
 * (see the top of this file). */
export function methodsFor(outputType: OutputType): SeedMethod[] {
  return outputType === "affinities" ? ["mutex_watershed", "components"] : ["components"];
}

/** What a seed reads of a model (view_labels._seed_plan): `channels`, the
 * output channels it thresholds (null for every one), and, for an affinity
 * model, the `offsets` of its affinity channels. As the trainer targets
 * them: the selected channel when the request names one; else an affinity
 * model's offset channels (the rest, LSDs say, are masked out of its loss);
 * else every channel, which a binary target is broadcast over. An affinity
 * model's offsets are the ones its channel names say (finetune/targets.ts
 * affinityOffsets: mito_aff_z, _y, _x are (1,0,0), (0,1,0), (0,0,1), the
 * unit offsets the server gives the first three `_aff` channels in order). */
export interface SeedPlan {
  outputType: OutputType;
  channels: number[] | null;
  offsets: number[][] | null;
}

export function seedPlan(entry: ModelEntry, selectChannel: number | null): SeedPlan {
  const outputType = outputTypeOf(entry);
  const named = outputType === "affinities" ? affinityOffsets(entry.manifest.channels) : [];
  const offsetChannels = named.flatMap((offset, c) => (offset ? [c] : []));
  const offsets = offsetChannels.length ? offsetChannels.map((c) => named[c]!) : null;
  if (selectChannel !== null) {
    if (!Number.isInteger(selectChannel) || selectChannel < 0) throw new Error(`select_channel must be 0 or more, got ${selectChannel}`);
    return { outputType, channels: [selectChannel], offsets };
  }
  if (outputType === "affinities" && offsets) return { outputType, channels: offsetChannels, offsets };
  return { outputType, channels: null, offsets };
}

// ---------------------------------------------------------------------------
// The prediction as a probability
// ---------------------------------------------------------------------------

/** A block of values, channels first: `shape` is [C, z, y, x]. */
export interface Block {
  data: Float32Array;
  shape: number[];
}

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** serving/probe.py's classify_output_range: the activation class of an
 * observed (min, max), with its small tolerance for a saturating activation
 * landing a hair outside its asymptote in float32. */
export function classifyOutputRange(lo: number, hi: number): string {
  const eps = 1e-4;
  if (lo >= -eps && hi <= 1 + eps) return UNIT;
  if (lo >= -1 - eps && hi <= 1 + eps) return SIGNED_UNIT;
  return UNBOUNDED;
}

/** scipy.special.expit of a float32 as scipy computes it for float32 input,
 * `1 / (1 + exp(-x))` with every step rounded to float32 (expf, then the
 * sum, then the quotient). Each step is done in float64 and rounded once,
 * which gives the float32 result for the sum and the quotient exactly and
 * for the exponential except where glibc's expf is not correctly rounded. */
export function expit32(x: number): number {
  const e = Math.fround(Math.exp(-x));
  return Math.fround(1 / Math.fround(1 + e));
}

/** The prediction as a probability, [channels, z, y, x] float32, 0.5 at the
 * model's decision boundary (view_labels.probability).
 *
 * The boundary depends on the output's activation: 0.5 on [0, 1] output (a
 * sigmoid already applied), 0 on tanh's [-1, 1], and 0 on unbounded output,
 * logits or a signed distance alike; each is turned into a probability.
 * `channels` picks the channels (null: every one); `outputClass` is the
 * served model's (ModelInfo.output_class), or read off the values of the
 * picked channels when it has none. */
export function probability(prediction: Block, outputClass: string | null | undefined, channels: number[] | null): Block {
  const [C, ...spatial] = prediction.shape;
  const n = prod(spatial);
  if (prediction.data.length !== C * n) throw new Error(`a prediction of shape ${prediction.shape.join("×")} has ${prediction.data.length} values`);
  const picked = channels ?? Array.from({ length: C }, (_, c) => c);
  const missing = picked.find((c) => c >= C);
  if (missing !== undefined) throw new Error(`The model has no channel ${missing}`);
  if (!picked.length) throw new Error("The model has no channel to read");
  const out = new Float32Array(picked.length * n);
  picked.forEach((c, k) => out.set(prediction.data.subarray(c * n, (c + 1) * n), k * n));
  let cls = outputClass || null;
  if (!cls) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < out.length; i++) {
      if (out[i] < lo) lo = out[i];
      if (out[i] > hi) hi = out[i];
    }
    cls = classifyOutputRange(lo, hi);
  }
  if (cls === UNIT) return { data: out, shape: [picked.length, ...spatial] };
  if (cls === SIGNED_UNIT) {
    // (prediction + 1) / 2 in float32: the sum rounded, the halving exact.
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(out[i] + 1) / 2;
  } else {
    for (let i = 0; i < out.length; i++) out[i] = expit32(out[i]);
  }
  return { data: out, shape: [picked.length, ...spatial] };
}

/** The channels' mean, as numpy's float32 mean over axis 0 computes it: the
 * channels added one after another in float32, then divided by their count
 * in float32. */
export function channelMean(block: Block): Float32Array {
  const [C, ...spatial] = block.shape;
  const n = prod(spatial);
  const mean = block.data.slice(0, n);
  for (let c = 1; c < C; c++) {
    const plane = block.data.subarray(c * n, (c + 1) * n);
    for (let i = 0; i < n; i++) mean[i] = Math.fround(mean[i] + plane[i]);
  }
  if (C > 1) for (let i = 0; i < n; i++) mean[i] = Math.fround(mean[i] / C);
  return mean;
}

/** Where the model says foreground (view_labels.foreground_mask): its
 * probability, averaged over `channels`, over `threshold`, a probability (0.5
 * is the model's own boundary). numpy compares a float32 array with a
 * Python float as float32, so the threshold is rounded to one first. 1 for
 * foreground, over the block's z, y, x. */
export function foregroundMask(prediction: Block, outputClass: string | null | undefined, channels: number[] | null, threshold = 0.5): Uint8Array {
  const mean = channelMean(probability(prediction, outputClass, channels));
  const t = Math.fround(threshold);
  const mask = new Uint8Array(mean.length);
  for (let i = 0; i < mean.length; i++) mask[i] = mean[i] > t ? 1 : 0;
  return mask;
}

// ---------------------------------------------------------------------------
// The objects
// ---------------------------------------------------------------------------

export interface SegmentOptions {
  /** A probability, 0 to 1: the foreground boundary for components, the mutex watershed's bias. */
  threshold: number;
  /** Objects of fewer voxels become background. */
  minSize: number;
  /** 1, 2 or 3 (infer/label.ts asConnectivity): the components'. */
  connectivity: number;
  /** Components found in each z slice on its own. */
  perSlice: boolean;
}

/** The objects `method` makes of the prediction (view_labels.segment_prediction):
 * 0 background, 1..n, over the block's z, y, x.
 *
 * "components": the foreground mask (foregroundMask over `channels`), an id
 * per connected object (connectivity, perSlice, minSize). "mutex_watershed":
 * the probability of the first offsets.length channels (the affinity
 * channels: seedPlan puts them first, as the server's slice(0, len(offsets))
 * takes them), joined where the affinity is over the threshold
 * (segment.mutex_watershed with the threshold as its bias; it ignores
 * connectivity and per-slice). Any other method is refused; the route
 * offers only methodsFor's. */
export function segmentPrediction(
  method: string,
  prediction: Block,
  outputClass: string | null | undefined,
  channels: number[] | null,
  offsets: number[][] | null,
  options: SegmentOptions,
): Uint32Array {
  const spatial = prediction.shape.slice(1);
  if (spatial.length !== 3) throw new Error(`a seed segments (c, z, y, x) blocks, got shape ${prediction.shape.join("×")}`);
  if (method === "components") {
    const mask = foregroundMask(prediction, outputClass, channels, options.threshold);
    return connectedComponents(mask, spatial, { connectivity: options.connectivity, minSize: options.minSize, perSlice: options.perSlice });
  }
  if (method === "mutex_watershed") {
    if (!offsets || !offsets.length) throw new Error("The model's affinity offsets are unknown, so it cannot seed by mutex_watershed");
    const affinityChannels = channels && channels.length === offsets.length ? channels : offsets.map((_, c) => c);
    const affinities = probability(prediction, outputClass, affinityChannels);
    const labels = mutexWatershed(affinities, offsets, options.threshold, options.minSize);
    const out = new Uint32Array(labels.length);
    for (let i = 0; i < labels.length; i++) out[i] = Number(labels[i]);
    return out;
  }
  throw new Error(`method must be one of ${SEED_METHODS.join(", ")}, got ${JSON.stringify(method)}`);
}
