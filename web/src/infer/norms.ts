// Input normalizers, applied to a model's input block before the forward pass:
// the browser side of cellmap_flow/norm/input_normalize.py, ported from
// Marwan Zouinkhi's web/legacy/pipeline.js (his serverless prototype).
//
// Each step takes the block as a Float32Array and changes it in place, as the
// Python classes return float32 whatever they are given. Parameters arrive as
// the dashboard's forms send them, every value a string ("0", "255", "true"),
// and are parsed as the Python constructors parse them (float(), and
// "true"/"false" by lower-cased comparison for a bool). Unknown parameters are
// ignored, as a constructor given only its own arguments never sees them.
//
// Only the normalizers that are arithmetic on the voxel values are here. The
// others (EuclideanDistance, Dilate, ChannelSelector) need a label volume or
// a library, and a chain that names one is refused before serving, with the
// error below naming it, rather than silently skipped as the prototype did.
import type { Step, StepValue } from "../contracts";
import { compileExpression } from "./expression";

export const SUPPORTED_NORMALIZERS: readonly string[] = ["MinMaxNormalizer", "ZScoreNormalizer", "LambdaNormalizer"];

type Normalizer = (data: Float32Array) => void;

/** Python's float(value): a number, or a numeric string. */
export function asNumber(value: StepValue | undefined, fallback: number, what: string): number {
  if (value === undefined || value === null || value === "") return fallback;
  const n = typeof value === "boolean" ? Number(value) : Number(value);
  if (Number.isNaN(n)) throw new Error(`${what} must be a number, got ${JSON.stringify(value)}`);
  return n;
}

/** The Python constructors' reading of a bool that may arrive as a string:
 * "true" (any case) is true, every other string false, else bool(value). */
export function asBool(value: StepValue | undefined, fallback: boolean): boolean {
  if (value === undefined || value === null) return fallback;
  if (typeof value === "string") return value.trim().toLowerCase() === "true";
  return Boolean(value);
}

function makeNormalizer(step: Step): Normalizer {
  switch (step.name) {
    case "MinMaxNormalizer": {
      // clip to [min, max], scale to [0, 1], optionally flip
      const min = asNumber(step.min_value, 0.0, "MinMaxNormalizer.min_value");
      const max = asNumber(step.max_value, 255.0, "MinMaxNormalizer.max_value");
      const invert = asBool(step.invert, false);
      const range = max - min;
      return (data) => {
        for (let i = 0; i < data.length; i++) {
          const v = (Math.min(Math.max(data[i], min), max) - min) / range;
          data[i] = invert ? 1 - v : v;
        }
      };
    }
    case "ZScoreNormalizer": {
      const mean = asNumber(step.mean, 0.0, "ZScoreNormalizer.mean");
      const std = asNumber(step.std, 1.0, "ZScoreNormalizer.std");
      return (data) => {
        for (let i = 0; i < data.length; i++) data[i] = (data[i] - mean) / std;
      };
    }
    case "LambdaNormalizer": {
      // Scalar arithmetic on x, parsed against a whitelist (expression.ts);
      // the Python class checks its expression in the constructor too.
      const fn = compileExpression(step.expression);
      return (data) => {
        for (let i = 0; i < data.length; i++) data[i] = fn(data[i]);
      };
    }
    default:
      throw new Error(
        `normalizer ${JSON.stringify(step.name)} is not available in the browser (supported: ${SUPPORTED_NORMALIZERS.join(", ")})`,
      );
  }
}

/** Every step's constructor run, to reject a chain before anything is served. */
export function validateNormalizers(steps: Step[]): void {
  for (const step of steps) makeNormalizer(step);
}

/** `data` through `steps` in order, in place; returns it. */
export function applyNormalizers(data: Float32Array, steps: Step[]): Float32Array {
  for (const step of steps) makeNormalizer(step)(data);
  return data;
}
