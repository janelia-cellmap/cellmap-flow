// The shaders of the viewer's layers, and the value range a prediction is
// drawn over. Ports of cellmap_flow/viewer/raw.py (_invlerp_bounds,
// prediction_shader) and cellmap_flow/serving/probe.py
// (output_display_range), with raw.py's own RAW_SHADER, PREDICTION_SHADER and
// PREDICTION_COLORS and layers.py's _COLOR_RE, read from the export
// (generated/python-data.ts). The strings they produce are the ones the
// Python functions produce, so a layer built here shades as the same layer
// built by the Flask dashboard: test/unit/viewer-display.test.ts holds
// values printed from the Python.
//
// Not ported: raw.py's flow_shader and layers.py's flow_channels, which
// colour a Cellpose model's flow channels by direction. The browser serves
// no Cellpose model (its masks need Cellpose's own dynamics, Python code).
import type { Step, StepValue } from "../contracts";
import { PYTHON, pyFormat, pyRegExp } from "../generated/python-data";

/** The colours prediction layers cycle through, in the order models are
 * submitted: PREDICTION_COLORS[index % PREDICTION_COLORS.length]. */
export const PREDICTION_COLORS: readonly string[] = PYTHON.viewer.prediction_colors;

/** What each of a model's output classes (serving/probe.py's UNIT, SIGNED_UNIT)
 * means as a value range. "unbounded" (logits, distances) and anything else
 * are undetermined. */
const CLASS_RANGE: Record<string, [number, number]> = { unit: [0, 1], signed_unit: [-1, 1] };

/** The values an integer dtype holds, for the dtypes a prediction or raw
 * image layer can have. 64-bit integers are left out: a layer of those is a
 * segmentation, which has no shader, and their limits are not exact doubles. */
const INTEGER_LIMITS: Record<string, [number, number]> = {
  int8: [-128, 127],
  uint8: [0, 255],
  int16: [-32768, 32767],
  uint16: [0, 65535],
  int32: [-2147483648, 2147483647],
  uint32: [0, 4294967295],
};

/** The pattern a shader's colour control is read back through, so a
 * recomputed shader keeps the colour the user navigates by
 * (layers.py's _COLOR_RE). */
const COLOR_CONTROL = pyRegExp(PYTHON.viewer.color_control);

/** Python's f"{value:.6g}": six significant digits, trailing zeros dropped,
 * exponent form (1e-07, 2e+06) below 1e-4 and from 1e6. Used so the shader
 * text matches the Python dashboard's character for character. */
export function formatG6(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  const rounded = Number(value.toPrecision(6));
  const exponent = Math.floor(Math.log10(Math.abs(rounded)));
  if (exponent < -4 || exponent >= 6) {
    const [mantissa, exp] = value.toExponential(5).split("e"); // "1.00000e-7"
    const digits = mantissa.includes(".") ? mantissa.replace(/\.?0+$/, "") : mantissa;
    const n = Number(exp);
    return `${digits}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  return String(rounded);
}

/** An invlerp control's `range=[lo, hi], window=[wlo, whi]`, as Neuroglancer
 * accepts them for data of `dtype` (raw.py's _invlerp_bounds).
 *
 * Neuroglancer reads the four numbers as values of the layer's data type and
 * refuses the whole shader when one is not: over uint8 data, a window padded
 * below 0 or a fractional bound. So for an integer dtype the range is widened
 * to whole numbers and everything is clipped to what the dtype holds; a float
 * dtype, or an unknown one, takes them as they are. */
export function invlerpBounds(lo: number, hi: number, wlo: number, whi: number, dtype?: string | null): string {
  const limits = dtype ? INTEGER_LIMITS[dtype] : undefined;
  if (!limits) {
    return `range=[${formatG6(lo)}, ${formatG6(hi)}], window=[${formatG6(wlo)}, ${formatG6(whi)}]`;
  }
  const [min, max] = limits;
  const whole = (value: number, round: (x: number) => number) => Math.min(Math.max(round(value), min), max);
  const l = whole(lo, Math.floor);
  const h = whole(hi, Math.ceil);
  const wl = Math.min(whole(wlo, Math.floor), l);
  const wh = Math.max(whole(whi, Math.ceil), h);
  return `range=[${l}, ${h}], window=[${wl}, ${wh}]`;
}

/** The shader of a prediction layer, in `color` over `range` (raw.py's
 * prediction_shader). Without a range, [0, 1]: a guess, but a ramp, where the
 * old default range=[0.5, 0.5] was a step that painted everything above 0.5
 * solid. The slider's window is padded by half the range each way so it can
 * still be moved beyond the computed range. `dtype` is the output's when a
 * step declares one (uint8 after a DefaultPostprocessor): see invlerpBounds. */
export function predictionShader(color: string, range?: readonly [number, number] | null, dtype?: string | null): string {
  const [lo, hi] = range ?? [0, 1];
  const pad = (hi - lo) * 0.5 || 1.0;
  return pyFormat(PYTHON.viewer.prediction_shader, { bounds: invlerpBounds(lo, hi, lo - pad, hi + pad, dtype), color });
}

/** The range of what a raw layer of `dtype` holds, as the contrast window
 * when nothing has been sampled: the integer dtype's whole range, [0, 1] for
 * a float. The Python dashboard samples 1-99% percentiles of the data
 * (raw.py's _auto_contrast_range, through the input normalizers); in the
 * browser Neuroglancer reads the raw itself, unnormalized, and the dtype's
 * range is its fallback too (_dtype_default_range). Neuroglancer's own
 * range button in the layer's panel does the sampling when the user wants
 * it. */
export function dtypeRange(dtype: string): [number, number] {
  return INTEGER_LIMITS[dtype] ?? [0, 1];
}

/** The raw layer's shader (raw.py's RAW_SHADER), white over dtypeRange. The
 * window equals the range, as Python's fallback has it. */
export function rawShader(dtype: string): string {
  const [lo, hi] = dtypeRange(dtype);
  return pyFormat(PYTHON.viewer.raw_shader, { bounds: invlerpBounds(lo, hi, lo, hi, dtype) });
}

/** The colour a shader's colour control defaults to, or null when it has
 * none. The pipeline route gives a layer whose shader it drops (the chain
 * changed, so its range is stale) a new shader in this colour, so the colour
 * the user knows the layer by stays (layers.py's prediction_shader_for). */
export function shaderColor(shader: string | null | undefined): string | null {
  const match = COLOR_CONTROL.exec(shader ?? "");
  return match ? match[1] : null;
}

/** Python's float(): a number, or a string that is one; anything else (an
 * empty string, "x", null) is not a number. */
function toFloat(value: StepValue | undefined, fallback: number): number {
  if (value === undefined || value === null) return fallback;
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (value.trim() === "") return NaN;
  return Number(value);
}

/** The value range a chain produces, for its layer's shader (probe.py's
 * output_display_range), or null when the chain leaves it undetermined.
 *
 * A prediction's range need not be sampled, unlike the raw's: the last
 * step's arithmetic fixes it. The model's `outputClass` ("unit",
 * "signed_unit", or "unbounded"/anything else for logits) starts the range;
 * a sigmoid or threshold makes it [0, 1]; a DefaultPostprocessor clips,
 * shifts and scales it into uint8 (clipped against what actually arrives:
 * after a sigmoid, clip_min=-1 is never reached, so the output starts at
 * 127.5, not 0); ChannelSelection leaves it; any other step (a lambda, the
 * segmentation ones) makes it unknown. The steps' values are as the
 * dashboard's forms sent them, strings included. */
export function outputDisplayRange(steps: readonly Step[] | null | undefined, outputClass?: string | null): [number, number] | null {
  let range: [number, number] | null = (outputClass && CLASS_RANGE[outputClass]) || null;
  for (const step of steps ?? []) {
    switch (step.name) {
      case "SigmoidPostprocessor":
      case "ThresholdPostprocessor":
        range = [0, 1];
        break;
      case "DefaultPostprocessor": {
        // Mirrors DefaultPostprocessor._process: clip, shift, scale, land in uint8.
        let clipMin = toFloat(step.clip_min, -1.0);
        let clipMax = toFloat(step.clip_max, 1.0);
        const bias = toFloat(step.bias, 1.0);
        const multiplier = toFloat(step.multiplier, 127.5);
        if ([clipMin, clipMax, bias, multiplier].some((v) => Number.isNaN(v))) {
          range = [0, 255]; // a value that is not a number: Python's except branch
          break;
        }
        if (range) {
          clipMin = Math.max(clipMin, range[0]);
          clipMax = Math.min(clipMax, range[1]);
        }
        const lo = (clipMin + bias) * multiplier;
        const hi = (clipMax + bias) * multiplier;
        range = [Math.max(0, Math.min(lo, hi)), Math.min(255, Math.max(lo, hi))];
        break;
      }
      case "ChannelSelection":
        break; // picks channels, does not rescale
      default:
        range = null;
    }
  }
  if (range === null || range[1] <= range[0]) return null;
  return range;
}
