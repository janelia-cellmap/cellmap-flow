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
// Only the elementwise steps and ChannelSelection are here. The segmentation
// steps (Label, Affinity, Morton relabeling, the blockwise merger, Cellpose
// masks, FillHoles) need libraries or state across chunks; a chain that names
// one is refused before serving, with the error below naming it.
import type { Step, StepValue } from "../contracts";
import { compileExpression } from "./expression";
import { asNumber } from "./norms";

export const SUPPORTED_POSTPROCESSORS: readonly string[] = [
  "SigmoidPostprocessor", "DefaultPostprocessor", "ThresholdPostprocessor", "ChannelSelection", "LambdaPostprocessor",
];

export type NdData = Float32Array | Uint8Array | Uint16Array | Uint32Array | Int32Array | Float64Array;
/** A block, C order, channel first: shape [C, z, y, x]. */
export type NdArray = { data: NdData; shape: number[] };

type Postprocessor = (nd: NdArray) => NdArray;

/** What a step says about its output (SerializableInterface.output_info):
 * a dtype if it declares one, a channel count if it changes it, and whether
 * the output is labels (null: it does not say). */
interface Declares {
  dtype?: string;
  channels?: (given: number) => number;
  isSegmentation?: boolean;
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

function declares(step: Step): Declares | null {
  switch (step.name) {
    case "SigmoidPostprocessor": return { dtype: "float32" };
    case "DefaultPostprocessor": return { dtype: "uint8", isSegmentation: false };
    case "ThresholdPostprocessor": return { dtype: "uint8", isSegmentation: true };
    case "ChannelSelection": { const n = parseChannels(step.channels).length; return { channels: () => n }; }
    case "LambdaPostprocessor": return { dtype: "float32" };
    default: return null;  // not a browser step: declares nothing here
  }
}

function makePostprocessor(step: Step): Postprocessor {
  switch (step.name) {
    case "SigmoidPostprocessor":
      // logits to probabilities, in float32
      return ({ data, shape }) => {
        const out = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = 1 / (1 + Math.exp(-data[i]));
        return { data: out, shape };
      };
    case "DefaultPostprocessor": {
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
          out[i] = (Math.min(Math.max(data[i], clipMin), clipMax) + bias) * multiplier;
        }
        return { data: out, shape };
      };
    }
    case "ThresholdPostprocessor": {
      const threshold = asNumber(step.threshold, 0.5, "ThresholdPostprocessor.threshold");
      return ({ data, shape }) => {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = data[i] > threshold ? 1 : 0;  // strictly greater, as in Python
        return { data: out, shape };
      };
    }
    case "ChannelSelection": {
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
    }
    case "LambdaPostprocessor": {
      // scalar arithmetic on x (expression.ts); float32 in, float32 out
      const fn = compileExpression(step.expression);
      return ({ data, shape }) => {
        const out = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = fn(Math.fround(data[i]));
        return { data: out, shape };
      };
    }
    default:
      throw new Error(
        `postprocessor ${JSON.stringify(step.name)} is not available in the browser (supported: ${SUPPORTED_POSTPROCESSORS.join(", ")})`,
      );
  }
}

/** Every step's constructor run, to reject a chain before anything is served. */
export function validatePostprocessors(steps: Step[]): void {
  for (const step of steps) makePostprocessor(step);
}

/** `nd` through `steps` in order. */
export function applyPostprocessors(nd: NdArray, steps: Step[]): NdArray {
  let current = nd;
  for (const step of steps) current = makePostprocessor(step)(current);
  return current;
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
