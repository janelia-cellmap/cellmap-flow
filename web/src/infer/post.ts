// Postprocessors, applied to a model's output block before it is served: the
// browser side of cellmap_flow/post/postprocessors.py, ported from Marwan
// Zouinkhi's web/legacy/pipeline.js (his serverless prototype), with
// SigmoidPostprocessor added: the models on Hugging Face output logits.
//
// A block is a channel-first ndarray {data, shape: [C, z, y, x]}. Each step
// returns a new ndarray in the dtype the Python class declares (uint8 for
// Default and Threshold, float32 for Sigmoid and Lambda; ChannelSelection
// keeps what it is given; Argmax's one channel of ids is uint8, or uint16
// past 255 channels, or with its `ids` the smallest of uint8, uint16 and
// uint32 that holds them), and the chain's output dtype, channel count and
// whether it is labels follow pipeline_spec.py's rule: the last step that
// declares wins.
//
// These are op bodies, as the Python classes are op bodies for chunkmirage's
// pipeline on the server: nothing here reads data, caches or knows about
// views; the engine does that. A step is built once per served layer
// (buildPostprocessors), as the server builds its PostprocessOp once per
// layer, so a stateful step (the blockwise merger's equivalences) keeps its
// state across chunks; each chunk then runs the built chain with its context
// (runPostprocessors): the chunk's index on the served grid and the voxels of
// one output block, which Morton and the affinity segmenter key their ids on,
// as the server's PostprocessOp.apply_at passes them. The segmentation steps
// live in affinity.ts, morton.ts, label.ts and merger.ts.
import type { Step, StepValue } from "../contracts";
import { affinityPostprocessor } from "./affinity";
import { compileExpression } from "./expression";
import { labelPostprocessor } from "./label";
import { blockwiseMerger } from "./merger";
import { mortonRelabeling } from "./morton";
import { asNumber } from "./norms";

export type NdData = Float32Array | Uint8Array | Uint16Array | Uint32Array | Int32Array | Float64Array | BigUint64Array;
/** A block, C order, channel first: shape [C, z, y, x]. */
export type NdArray = { data: NdData; shape: number[] };

/** Where a chunk sits, for the steps that key their ids on it: its index on
 * the served grid (the server's chunk_corner) and the voxels of one output
 * block, spatial axes only (chunk_num_voxels). */
export interface PostContext {
  chunkIndex: number[];
  chunkNumVoxels: number;
}

/** What a served layer gives its stateful steps: the blockwise merger reports
 * the ids it has joined so far as sets of decimal uint64 strings, each set
 * the ids that are one object (neuroglancer's EquivalenceMap.to_json(), and
 * the shape a Neuroglancer layer's `equivalences` take), for the viewer's
 * segmentation layer to show. */
export interface LayerHooks {
  onEquivalences?: (sets: string[][]) => void;
}

/** A built step: a block in, a block out, with the chunk's context. */
export type Postprocessor = (nd: NdArray, context: PostContext) => NdArray;

/** What a step says about its output (SerializableInterface.output_info):
 * a dtype if it declares one, a channel count if it changes it, and whether
 * the output is labels (undefined: it does not say). A dtype that depends on
 * how many channels the step is given (ArgmaxPostprocessor's) is a function
 * of that count, undefined when the caller does not know it; so is the
 * count, though every step that changes it sets its own, as the Python's
 * num_channels does. */
export interface Declares {
  dtype?: string | ((given: number | undefined) => string);
  channels?: (given: number | undefined) => number;
  isSegmentation?: boolean;
}

/** A postprocessor class, as the registry below holds it. */
export interface PostprocessorDef {
  /** Build the step from its parameters, parsed as the Python constructor
   * parses them (throw on a bad one), once per served layer. */
  make(step: Step, hooks: LayerHooks): Postprocessor;
  /** What the step says about its output, from its parameters. */
  declares(step: Step): Declares;
}

/** ChannelSelection's channels: "0,2" from the dashboard's form, or a number,
 * or (from a YAML) a list. */
export function parseChannels(value: StepValue | unknown[] | undefined): number[] {
  if (value === undefined || value === null || value === "") return [0];
  const parts = Array.isArray(value) ? value : typeof value === "string" ? value.split(",") : [value];
  return parts.map((c) => {
    const n = Number.parseInt(String(c).trim(), 10);
    if (Number.isNaN(n) || n < 0) throw new Error(`ChannelSelection.channels: ${JSON.stringify(value)} is not a list of channel indices`);
    return n;
  });
}

// What the dashboard's forms send for "none" (a None default is shown as
// "None"), and the largest id ArgmaxPostprocessor's ids may give.
const NO_VALUE = ["", "none", "null"];
const LARGEST_UINT32 = 2 ** 32 - 1;

/** ArgmaxPostprocessor's threshold: none when absent, empty or "None" (the
 * dashboard's form shows no threshold as "None"), else a number as Python's
 * float() reads it. */
function parseArgmaxThreshold(value: StepValue | undefined): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "string" && NO_VALUE.includes(value.trim().toLowerCase())) return null;
  return asNumber(value, 0, "ArgmaxPostprocessor.threshold");
}

/** ArgmaxPostprocessor's ids (postprocessors._argmax_ids): null for none
 * (absent, empty, "None", or a list of none), else whole numbers from 0 to
 * uint32's largest, from a list, one number, or the form's "50,51,37" or
 * "50 51 37". */
export function parseArgmaxIds(value: StepValue | unknown[] | undefined): number[] | null {
  if (value === undefined || value === null) return null;
  let parts: unknown[];
  if (typeof value === "string") {
    if (NO_VALUE.includes(value.trim().toLowerCase())) return null;
    parts = value.trim().split(/[\s,]+/).filter((part) => part !== "");
  } else {
    parts = Array.isArray(value) ? value : [value];
  }
  const ids = parts.map((part) => {
    const label = typeof part === "string" ? (/^\d+$/.test(part) ? Number(part) : NaN) : typeof part === "number" ? part : NaN;
    if (!Number.isInteger(label) || label < 0 || label > LARGEST_UINT32) {
      throw new Error(`ArgmaxPostprocessor.ids: ${JSON.stringify(value)} is not a list of label ids, whole numbers from 0 to ${LARGEST_UINT32}`);
    }
    return label;
  });
  return ids.length ? ids : null;
}

/** postprocessors.argmax_dtype: the dtype of ArgmaxPostprocessor's ids. With
 * `ids`, the smallest of uint8, uint16 and uint32 that holds the largest of
 * them; without, uint8 while every id (0 to `channels`) fits in it, else
 * uint16, and a count not known is taken to fit uint8. */
export function argmaxDtype(channels: number | undefined, ids: number[] | null = null): "uint8" | "uint16" | "uint32" {
  if (ids) {
    const largest = Math.max(...ids);
    return largest <= 255 ? "uint8" : largest <= 65535 ? "uint16" : "uint32";
  }
  return channels !== undefined && channels + 1 > 256 ? "uint16" : "uint8";
}

/** The typed array of each dtype argmaxDtype answers. */
const ID_ARRAYS = { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array };

/** The steps the browser runs, by the Python class name the dashboard sends. */
export const POSTPROCESSORS: Record<string, PostprocessorDef> = {
  SigmoidPostprocessor: {
    declares: () => ({ dtype: "float32" }),
    make: () => {
      // logits to probabilities, in float32
      return ({ data, shape }) => {
        const out = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = 1 / (1 + Math.exp(-Number(data[i])));
        return { data: out, shape };
      };
    },
  },
  DefaultPostprocessor: {
    declares: () => ({ dtype: "uint8", isSegmentation: false }),
    make: (step) => {
      // clip, shift and scale into uint8: (x + bias) * multiplier, with the
      // defaults mapping [-1, 1] onto [0, 255]
      const clipMin = asNumber(step.clip_min, -1.0, "DefaultPostprocessor.clip_min");
      const clipMax = asNumber(step.clip_max, 1.0, "DefaultPostprocessor.clip_max");
      const bias = asNumber(step.bias, 1.0, "DefaultPostprocessor.bias");
      const multiplier = asNumber(step.multiplier, 127.5, "DefaultPostprocessor.multiplier");
      return ({ data, shape }) => {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) {
          // Storing into a Uint8Array truncates toward zero and wraps modulo
          // 256, which is what numpy's astype(uint8) does on x86.
          out[i] = (Math.min(Math.max(Number(data[i]), clipMin), clipMax) + bias) * multiplier;
        }
        return { data: out, shape };
      };
    },
  },
  ThresholdPostprocessor: {
    declares: () => ({ dtype: "uint8", isSegmentation: true }),
    make: (step) => {
      const threshold = asNumber(step.threshold, 0.5, "ThresholdPostprocessor.threshold");
      return ({ data, shape }) => {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = Number(data[i]) > threshold ? 1 : 0;  // strictly greater, as in Python
        return { data: out, shape };
      };
    },
  },
  ChannelSelection: {
    declares: (step) => { const n = parseChannels(step.channels).length; return { channels: () => n }; },
    make: (step) => {
      // data[channels, :, :, :]: the listed channels, in the listed order; the
      // dtype is kept
      const channels = parseChannels(step.channels);
      return ({ data, shape }) => {
        const [c, z, y, x] = shape;
        const plane = z * y * x;  // voxels per channel
        for (const ch of channels) {
          if (ch >= c) throw new Error(`ChannelSelection: channel ${ch} of an output with ${c} channel(s)`);
        }
        const out = new (data.constructor as { new (n: number): NdData })(channels.length * plane);
        channels.forEach((ch, j) => out.set(data.subarray(ch * plane, (ch + 1) * plane) as never, j * plane));
        return { data: out, shape: [channels.length, z, y, x] };
      };
    },
  },
  LambdaPostprocessor: {
    declares: () => ({ dtype: "float32" }),
    make: (step) => {
      // scalar arithmetic on x (expression.ts); float32 in, float32 out
      const fn = compileExpression(step.expression);
      return ({ data, shape }) => {
        const out = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = fn(Math.fround(Number(data[i])));
        return { data: out, shape };
      };
    },
  },
  ArgmaxPostprocessor: {
    declares: (step) => {
      const ids = parseArgmaxIds(step.ids);
      return { dtype: (given) => argmaxDtype(given, ids), channels: () => 1, isSegmentation: true };
    },
    make: (step) => {
      // One channel of ids: each voxel's highest-scoring channel plus 1, or
      // that channel's id in `ids`, the lowest on a tie (numpy's argmax), a
      // NaN never the one; 0 where the best score is below the threshold,
      // or every channel is NaN. The threshold is compared in float64, as
      // the Python compares it.
      const threshold = parseArgmaxThreshold(step.threshold);
      const ids = parseArgmaxIds(step.ids);
      return ({ data, shape }) => {
        if (shape.length !== 4) throw new Error(`ArgmaxPostprocessor picks between a model's channels: it needs a (c, z, y, x) block, got shape ${shape.join("×")}`);
        const [c, z, y, x] = shape;
        if (ids && ids.length !== c) {
          throw new Error(`ArgmaxPostprocessor has ${ids.length} ids for ${c} channels: give one id per channel it is given (after any ChannelSelection)`);
        }
        const plane = z * y * x;  // voxels per channel
        const out = new ID_ARRAYS[argmaxDtype(c, ids)](plane);
        for (let i = 0; i < plane; i++) {
          let winner = -1;
          let best = -Infinity;
          for (let ch = 0; ch < c; ch++) {
            const v = Number(data[ch * plane + i]);
            if (!Number.isNaN(v) && (winner < 0 || v > best)) {
              winner = ch;
              best = v;
            }
          }
          out[i] = winner < 0 || (threshold !== null && best < threshold) ? 0 : ids ? ids[winner] : winner + 1;
        }
        return { data: out, shape: [1, z, y, x] };
      };
    },
  },
  AffinityPostprocessor: affinityPostprocessor,
  MortonSegmentationRelabeling: mortonRelabeling,
  LabelPostprocessor: labelPostprocessor,
  SimpleBlockwiseMerger: blockwiseMerger,
};

/** The Python classes of these say `browser = True`, which is what the
 * browser page offers (web/test/unit/python-data.test.ts holds the two equal). */
export const SUPPORTED_POSTPROCESSORS: readonly string[] = Object.keys(POSTPROCESSORS);

function definition(step: Step): PostprocessorDef {
  const def = POSTPROCESSORS[step.name];
  if (!def) {
    throw new Error(
      `postprocessor ${JSON.stringify(step.name)} is not available in the browser (supported: ${SUPPORTED_POSTPROCESSORS.join(", ")})`,
    );
  }
  return def;
}

function declares(step: Step): Declares | null {
  return POSTPROCESSORS[step.name]?.declares(step) ?? null;  // not a browser step: declares nothing here
}

/** The chain's steps built once, for a served layer: stateful steps keep
 * their state in the closures; `hooks` is what they report through. */
export function buildPostprocessors(steps: Step[], hooks: LayerHooks = {}): Postprocessor[] {
  return steps.map((step) => definition(step).make(step, hooks));
}

/** Every step's constructor run, to reject a chain before anything is served. */
export function validatePostprocessors(steps: Step[]): void {
  buildPostprocessors(steps);
}

/** `nd` through the built chain, for the chunk `context` describes. */
export function runPostprocessors(processors: Postprocessor[], nd: NdArray, context: PostContext): NdArray {
  let current = nd;
  for (const processor of processors) current = processor(current, context);
  return current;
}

/** `nd` through `steps` in order, built afresh: for stateless chains and
 * tests. Without a context the block is taken as chunk 0 of its own grid. */
export function applyPostprocessors(nd: NdArray, steps: Step[], context?: PostContext): NdArray {
  const ctx = context ?? { chunkIndex: nd.shape.slice(1).map(() => 0), chunkNumVoxels: nd.shape.slice(1).reduce((a, b) => a * b, 1) };
  return runPostprocessors(buildPostprocessors(steps), nd, ctx);
}

/** The dtype the viewer receives (chain_output_dtype): the last step's that
 * declares one, else the model's own. `modelChannels`, when the caller knows
 * it, is for a step whose dtype depends on the channels it is given
 * (ArgmaxPostprocessor's ids: uint16 past 255 channels); the count follows
 * the steps that change it, as `outputChannels` does, and as
 * pipeline_spec's does without the model's: a ChannelSelection's count is
 * its own, so after one it is known either way. */
export function outputDtype(steps: Step[], modelDtype = "float32", modelChannels?: number): string {
  let dtype = modelDtype;
  let channels = modelChannels;
  for (const step of steps) {
    const says = declares(step);
    const own = typeof says?.dtype === "function" ? says.dtype(channels) : says?.dtype;
    dtype = own ?? dtype;
    if (says?.channels) channels = says.channels(channels);
  }
  return dtype;
}

/** How many channels the viewer receives (chain_num_channels), after e.g.
 * ChannelSelection. */
export function outputChannels(steps: Step[], modelChannels: number): number {
  let channels = modelChannels;
  for (const step of steps) {
    const change = declares(step)?.channels;
    if (change) channels = change(channels);
  }
  return channels;
}

/** Whether the output is labels (chain_is_segmentation): false for an empty
 * chain (raw model output is never labels), the last step's answer that
 * says, null when steps are present but none of them says. */
export function isSegmentation(steps: Step[]): boolean | null {
  if (!steps.length) return false;
  let answer: boolean | null = null;
  for (const step of steps) {
    const says = declares(step)?.isSegmentation;
    if (says !== undefined) answer = says;
  }
  return answer;
}
