// A 2D model's training graph (finetune/train_graph.json, format
// cellmap-train-graph/1, written by Marwan Zouinkhi's exports beside the
// .onnx files): what it is, read and checked (parseTrainGraph), and the tile
// sides it can be built at (outputSide, tileSides).
//
// The graph is fully convolutional, so TF.js builds it at any input side the
// arithmetic allows, not only at the sides the .onnx files were exported
// at: valid convolutions shrink a tile by a fixed margin, and a pool or a
// strided convolution must divide it exactly, or the decoder's upsampled
// maps come out a row short of the encoder's cropped skips and no longer
// concatenate (or add). Which sides pass depends on the graph, so they are
// found by walking its shapes at each candidate side (outputSide) rather
// than by a rule written here. On 2026-10-16 both graphs give the same
// answer: setup_53_2d's UNet (three 2×2 pools, skips cropped by 4, 16 and
// 40) and setup_54_2d's ResUNet (strided 2×2 convolutions, nearest
// upsampling, additions) take every side n ≡ 4 (mod 8) from 92 on, 92, 100,
// 108, ..., with an output of n − 88; the exported 300, 556, 1060 and 1068
// are among them.

export interface TrainParam {
  shape: number[];
  trainable: boolean;
}

export type TrainOp =
  | { op: "conv"; x: string; y: string; w: string; b?: string; stride: [number, number] }
  | { op: "conv_t"; x: string; y: string; w: string; b?: string; stride: [number, number] }
  | { op: "maxpool"; x: string; y: string; k: [number, number]; stride: [number, number] }
  | { op: "bn"; x: string; y: string; scale: string; bias: string; mean: string; var: string; eps: number }
  | { op: "inorm"; x: string; y: string; scale: string; bias: string; eps: number }
  | { op: "relu"; x: string; y: string }
  | { op: "add"; x: string; x2: string; y: string }
  | { op: "concat"; xs: string[]; y: string }
  | { op: "crop"; x: string; y: string; axis: "h" | "w"; start: number; end_margin: number }
  | { op: "upsample"; x: string; y: string; factor: number };

export interface TrainGraph {
  format: string;
  layout: "NHWC";
  input: string;
  output: string;
  /** Input side minus output side: what the valid convolutions use up. */
  context: number;
  params: Record<string, TrainParam>;
  ops: TrainOp[];
}

export const TRAIN_GRAPH_FORMAT = "cellmap-train-graph/1";

const OPS = new Set(["conv", "conv_t", "maxpool", "bn", "inorm", "relu", "add", "concat", "crop", "upsample"]);

/** `raw` (train_graph.json parsed) checked: the format and layout this
 * runtime builds, every op one it knows with the tensors it reads made by an
 * earlier op, every param it names declared with a shape. */
export function parseTrainGraph(raw: unknown, where = "train_graph.json"): TrainGraph {
  const fail = (problem: string): never => {
    throw new Error(`${where}: ${problem}`);
  };
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) fail("not a JSON object");
  const g = raw as Record<string, unknown>;
  if (g.format !== TRAIN_GRAPH_FORMAT) fail(`format ${JSON.stringify(g.format)}, where this browser builds ${TRAIN_GRAPH_FORMAT}`);
  if (g.layout !== "NHWC") fail(`layout ${JSON.stringify(g.layout)}, not NHWC`);
  if (typeof g.input !== "string" || typeof g.output !== "string") fail("no input or output tensor named");
  if (typeof g.context !== "number" || !Number.isInteger(g.context) || g.context < 0) fail("context is not a whole number");
  if (!g.params || typeof g.params !== "object") fail("no params");
  const params = g.params as Record<string, TrainParam>;
  for (const [name, p] of Object.entries(params)) {
    if (!p || !Array.isArray(p.shape) || !p.shape.every((n) => Number.isInteger(n) && n > 0) || typeof p.trainable !== "boolean") fail(`param ${name} has no shape or trainable flag`);
  }
  if (!Array.isArray(g.ops) || !g.ops.length) fail("no ops");
  const made = new Set([g.input as string]);
  for (const [i, op] of (g.ops as TrainOp[]).entries()) {
    if (!op || typeof op !== "object" || !OPS.has(op.op)) fail(`op ${i} is ${JSON.stringify((op as { op?: unknown })?.op)}, not one this browser builds`);
    const reads = op.op === "concat" ? op.xs : op.op === "add" ? [op.x, op.x2] : [op.x];
    for (const x of reads) if (!made.has(x)) fail(`op ${i} (${op.op}) reads ${x}, which no earlier op makes`);
    const names = op.op === "conv" || op.op === "conv_t" ? [op.w, op.b] : op.op === "bn" ? [op.scale, op.bias, op.mean, op.var] : op.op === "inorm" ? [op.scale, op.bias] : [];
    for (const name of names) if (name && !params[name]) fail(`op ${i} (${op.op}) names param ${name}, which params does not declare`);
    made.add(op.y);
  }
  if (!made.has(g.output as string)) fail(`no op makes the output ${g.output}`);
  return raw as TrainGraph;
}

/** The output side for an n × n input tile, or null when the graph cannot
 * take n: a pool or strided convolution that would drop a row, a crop or a
 * kernel larger than its input, or two maps that no longer line up where
 * they are concatenated or added. The output must then be n − context. */
export function outputSide(graph: TrainGraph, n: number): number | null {
  const hw = new Map<string, [number, number]>([[graph.input, [n, n]]]);
  const window = (size: number, k: number, s: number) => (size >= k && (size - k) % s === 0 ? (size - k) / s + 1 : 0);
  for (const op of graph.ops) {
    let out: [number, number];
    switch (op.op) {
      case "conv": {
        const [h, w] = hw.get(op.x)!, [, , kh, kw] = graph.params[op.w].shape;
        out = [window(h, kh, op.stride[0]), window(w, kw, op.stride[1])];
        break;
      }
      case "conv_t": {
        const [h, w] = hw.get(op.x)!, [, , kh, kw] = graph.params[op.w].shape;
        out = [(h - 1) * op.stride[0] + kh, (w - 1) * op.stride[1] + kw];
        break;
      }
      case "maxpool": {
        const [h, w] = hw.get(op.x)!;
        out = [window(h, op.k[0], op.stride[0]), window(w, op.k[1], op.stride[1])];
        break;
      }
      case "crop": {
        const [h, w] = hw.get(op.x)!;
        out = op.axis === "h" ? [h - op.start - op.end_margin, w] : [h, w - op.start - op.end_margin];
        break;
      }
      case "upsample": {
        const [h, w] = hw.get(op.x)!;
        out = [h * op.factor, w * op.factor];
        break;
      }
      case "add":
      case "concat": {
        const shapes = (op.op === "add" ? [op.x, op.x2] : op.xs).map((x) => hw.get(x)!);
        if (shapes.some(([h, w]) => h !== shapes[0][0] || w !== shapes[0][1])) return null;
        out = shapes[0];
        break;
      }
      default:  // bn, inorm, relu: the same shape
        out = hw.get(op.x)!;
    }
    if (!(out[0] > 0 && out[1] > 0)) return null;
    hw.set(op.y, out);
  }
  const [h, w] = hw.get(graph.output)!;
  return h === w && h === n - graph.context ? h : null;
}

/** Every side from `min` to `max` the graph takes (outputSide), ascending. */
export function tileSides(graph: TrainGraph, max: number, min = graph.context + 1): number[] {
  const sides: number[] = [];
  for (let n = Math.max(min, 1); n <= max; n++) if (outputSide(graph, n) !== null) sides.push(n);
  return sides;
}

/** `sides` (ascending) in words, for a refusal: "92, 100, 108, … (92 + 8k)"
 * when they step evenly, else the first few. */
export function describeSides(sides: number[]): string {
  if (!sides.length) return "none";
  const step = sides.length > 1 ? sides[1] - sides[0] : 0;
  const even = sides.length > 2 && sides.every((s, i) => s === sides[0] + i * step);
  const head = sides.slice(0, 4).join(", ");
  return even ? `${head}, … (${sides[0]} + ${step}k)` : sides.length > 6 ? `${head}, …` : sides.join(", ");
}

/** The tile sides the Finetune tab offers: the side the graph takes nearest
 * each of TILE_TARGETS, each once. */
export const TILE_TARGETS: readonly number[] = [172, 300, 428, 556];

/** The default tile: the offered side nearest this. 300² trains a step in
 * about 0.3 s on a laptop's GPU in 320 MB (setup_53_2d; setup_54_2d 0.46 s
 * in 908 MB), where 556² takes four times both (1.2 s, 1.3 GB; 1.9 s, 3.7
 * GB, more than many laptops' GPUs give a tab), and half of a 300² tile is
 * output, the paint the loss reaches; at 172² only a quarter is. */
export const DEFAULT_TILE = 300;

/** The largest side looked at for the choices. */
const MAX_SIDE = 1100;

/** One tile side the tab offers, with what it is expected to cost: per
 * step of one tile (batch 1), from the manifest's spike where it measured
 * that side, else scaled by area from the nearest side it measured. */
export interface TileChoice {
  side: number;
  output: number;
  /** (side − context)² / side²: the share of the tile the loss reaches. */
  useful: number;
  step_ms: number | null;
  peak_mb: number | null;
  /** Whether the numbers were measured at this side (else scaled). */
  measured: boolean;
}

/** The offered sides of `graph` with their costs from `spike` (side → numbers). */
export function tileChoices(graph: TrainGraph, spike: Record<string, { step_ms: number; peak_mb: number }> = {}): TileChoice[] {
  const sides = tileSides(graph, MAX_SIDE);
  if (!sides.length) return [];
  const offered = [...new Set(TILE_TARGETS.map((t) => nearest(sides, t)))].sort((a, b) => a - b);
  const measuredSides = Object.keys(spike).map(Number).filter((n) => Number.isInteger(n) && n > 0 && spike[String(n)]);
  return offered.map((side) => {
    const output = side - graph.context;
    const from = measuredSides.length ? nearest(measuredSides, side) : null;
    const scale = from === null ? 0 : (side * side) / (from * from);
    const numbers = from === null ? null : spike[String(from)];
    return {
      side,
      output,
      useful: (output * output) / (side * side),
      step_ms: numbers ? Math.round(numbers.step_ms * scale) : null,
      peak_mb: numbers ? Math.round(numbers.peak_mb * scale) : null,
      measured: from === side,
    };
  });
}

/** The default side among `choices`: the one nearest DEFAULT_TILE. */
export function defaultTile(choices: TileChoice[]): number | null {
  return choices.length ? nearest(choices.map((c) => c.side), DEFAULT_TILE) : null;
}

/** The value of `values` nearest `target`; a tie goes to the larger, whose cost is the one not to underestimate. */
function nearest(values: number[], target: number): number {
  return values.reduce((best, v) => {
    const d = Math.abs(v - target), bd = Math.abs(best - target);
    return d < bd || (d === bd && v > best) ? v : best;
  });
}
