// Postprocessors, applied to a model's output block before it is served: the
// browser side of cellmap_flow/post/postprocessors.py, ported from Marwan
// Zouinkhi's web/legacy/pipeline.js (his serverless prototype), with
// SigmoidPostprocessor added: the models on Hugging Face output logits.
//
// A block is a channel-first ndarray {data, shape: [C, z, y, x]}. Each step
// returns a new ndarray in the dtype the Python class declares (uint8 for
// Default and Threshold, float32 for Sigmoid and Lambda; ChannelSelection
// keeps what it is given), and the chain's output dtype, channel count and
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
 * the output is labels (undefined: it does not say). */
export interface Declares {
  dtype?: string;
  channels?: (given: number) => number;
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
 * declares one, else the model's own. */
export function outputDtype(steps: Step[], modelDtype = "float32"): string {
  let dtype = modelDtype;
  for (const step of steps) dtype = declares(step)?.dtype ?? dtype;
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
