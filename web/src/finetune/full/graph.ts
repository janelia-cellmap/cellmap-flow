// A model's training graph (finetune/train_graph.json, format
// cellmap-train-graph/1, written by Marwan Zouinkhi's exports beside the
// .onnx files): what it is, read and checked (parseTrainGraph), and the tile
// sides it can be built at (outputSide, tileSides). A 2D model's graph is
// NHWC, its tiles squares; a 3D model's is NDHWC, its blocks cubes, with
// the same ops given a third stride, kernel side and crop axis (d).
//
// The graph is fully convolutional, so TF.js builds it at any input side the
// arithmetic allows, not only at the sides the .onnx files were exported
// at: valid convolutions shrink a tile by a fixed margin, and a pool or a
// strided convolution must divide it exactly, or the decoder's upsampled
// maps come out a row short of the encoder's cropped skips and no longer
// concatenate (or add). Which sides pass depends on the graph, so they are
// found by walking its shapes at each candidate side (outputSide) rather
// than by a rule written here. On 2026-10-18 every graph gives the same
// answer: setup_53_2d's and setup_53's UNets (three 2× pools, skips cropped
// by 4, 16 and 40) and setup_54_2d's and setup_54's ResUNets (strided 2×
// convolutions, nearest upsampling, additions) take every side n ≡ 4
// (mod 8) from 92 on, 92, 100, 108, ..., with an output of n − 88; the
// exported 300, 556, 1060 and 1068 (2D), and 116 to 172 (3D), are among
// them.

export interface TrainParam {
  shape: number[];
  trainable: boolean;
}

/** NHWC: a 2D model's tiles; NDHWC: a 3D model's blocks. */
export type Layout = "NHWC" | "NDHWC";

/** A spatial axis: depth (3D only), height, width. */
export type Axis = "d" | "h" | "w";

// Strides and kernels have one entry per spatial axis (two in 2D, three in
// 3D: parseTrainGraph checks), d before h before w.
export type TrainOp =
  | { op: "conv"; x: string; y: string; w: string; b?: string; stride: number[] }
  | { op: "conv_t"; x: string; y: string; w: string; b?: string; stride: number[] }
  | { op: "maxpool"; x: string; y: string; k: number[]; stride: number[] }
  | { op: "bn"; x: string; y: string; scale: string; bias: string; mean: string; var: string; eps: number }
  | { op: "inorm"; x: string; y: string; scale: string; bias: string; eps: number }
  | { op: "relu"; x: string; y: string }
  | { op: "add"; x: string; x2: string; y: string }
  | { op: "concat"; xs: string[]; y: string }
  | { op: "crop"; x: string; y: string; axis: Axis; start: number; end_margin: number }
  | { op: "upsample"; x: string; y: string; factor: number };

export interface TrainGraph {
  format: string;
  layout: Layout;
  input: string;
  output: string;
  /** Input side minus output side: what the valid convolutions use up. */
  context: number;
  params: Record<string, TrainParam>;
  ops: TrainOp[];
}

/** A 3D model's graph (is3d). */
export type TrainGraph3d = TrainGraph & { layout: "NDHWC" };

export const TRAIN_GRAPH_FORMAT = "cellmap-train-graph/1";

const OPS = new Set(["conv", "conv_t", "maxpool", "bn", "inorm", "relu", "add", "concat", "crop", "upsample"]);

/** Whether `graph` is a 3D model's (NDHWC). */
export const is3d = (graph: TrainGraph): graph is TrainGraph3d => graph.layout === "NDHWC";

/** The spatial axes of `layout`, in the order its tensors have them. */
export const spatialAxes = (layout: Layout): Axis[] => (layout === "NDHWC" ? ["d", "h", "w"] : ["h", "w"]);

/** The tensors `op` reads. */
export const opInputs = (op: TrainOp): string[] => (op.op === "concat" ? op.xs : op.op === "add" ? [op.x, op.x2] : [op.x]);

/** `raw` (train_graph.json parsed) checked: the format and a layout this
 * runtime builds, every op one it knows with the tensors it reads made by an
 * earlier op, a stride, kernel or crop axis for each of the layout's
 * spatial axes, every param it names declared with a shape. */
export function parseTrainGraph(raw: unknown, where = "train_graph.json"): TrainGraph {
  const fail = (problem: string): never => {
    throw new Error(`${where}: ${problem}`);
  };
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) fail("not a JSON object");
  const g = raw as Record<string, unknown>;
  if (g.format !== TRAIN_GRAPH_FORMAT) fail(`format ${JSON.stringify(g.format)}, where this browser builds ${TRAIN_GRAPH_FORMAT}`);
  if (g.layout !== "NHWC" && g.layout !== "NDHWC") fail(`layout ${JSON.stringify(g.layout)}, not NHWC or NDHWC`);
  const axes = spatialAxes(g.layout as Layout), rank = axes.length;
  if (typeof g.input !== "string" || typeof g.output !== "string") fail("no input or output tensor named");
  if (typeof g.context !== "number" || !Number.isInteger(g.context) || g.context < 0) fail("context is not a whole number");
  if (!g.params || typeof g.params !== "object") fail("no params");
  const params = g.params as Record<string, TrainParam>;
  for (const [name, p] of Object.entries(params)) {
    if (!p || !Array.isArray(p.shape) || !p.shape.every((n) => Number.isInteger(n) && n > 0) || typeof p.trainable !== "boolean") fail(`param ${name} has no shape or trainable flag`);
  }
  if (!Array.isArray(g.ops) || !g.ops.length) fail("no ops");
  const perAxis = (v: unknown) => Array.isArray(v) && v.length === rank && v.every((n) => Number.isInteger(n) && n > 0);
  const made = new Set([g.input as string]);
  for (const [i, op] of (g.ops as TrainOp[]).entries()) {
    if (!op || typeof op !== "object" || !OPS.has(op.op)) fail(`op ${i} is ${JSON.stringify((op as { op?: unknown })?.op)}, not one this browser builds`);
    for (const x of opInputs(op)) if (!made.has(x)) fail(`op ${i} (${op.op}) reads ${x}, which no earlier op makes`);
    const names = op.op === "conv" || op.op === "conv_t" ? [op.w, op.b] : op.op === "bn" ? [op.scale, op.bias, op.mean, op.var] : op.op === "inorm" ? [op.scale, op.bias] : [];
    for (const name of names) if (name && !params[name]) fail(`op ${i} (${op.op}) names param ${name}, which params does not declare`);
    if ((op.op === "conv" || op.op === "conv_t") && (!perAxis(op.stride) || params[op.w].shape.length !== rank + 2)) fail(`op ${i} (${op.op}) has no stride or kernel for each of ${axes.join(", ")}`);
    if (op.op === "maxpool" && !(perAxis(op.k) && perAxis(op.stride))) fail(`op ${i} (maxpool) has no window or stride for each of ${axes.join(", ")}`);
    if (op.op === "crop" && (!axes.includes(op.axis) || !Number.isInteger(op.start) || !Number.isInteger(op.end_margin) || op.start < 0 || op.end_margin < 0)) fail(`op ${i} (crop) crops ${JSON.stringify(op.axis)} by ${op.start} and ${op.end_margin}, not an axis of ${axes.join(", ")} by whole numbers`);
    if (op.op === "upsample" && !(Number.isInteger(op.factor) && op.factor > 0)) fail(`op ${i} (upsample) has no whole factor`);
    made.add(op.y);
  }
  if (!made.has(g.output as string)) fail(`no op makes the output ${g.output}`);
  return raw as TrainGraph;
}

/** The output side for an input tile of side n (a square in 2D, a cube in
 * 3D), or null when the graph cannot take n: a pool or strided convolution
 * that would drop a row, a crop or a kernel larger than its input, or two
 * maps that no longer line up where they are concatenated or added. The
 * output must then be n − context along every axis. */
export function outputSide(graph: TrainGraph, n: number): number | null {
  const axes = spatialAxes(graph.layout);
  const sizes = new Map<string, number[]>([[graph.input, axes.map(() => n)]]);
  const window = (size: number, k: number, s: number) => (size >= k && (size - k) % s === 0 ? (size - k) / s + 1 : 0);
  for (const op of graph.ops) {
    let out: number[];
    switch (op.op) {
      case "conv": {
        const k = graph.params[op.w].shape.slice(2);
        out = sizes.get(op.x)!.map((size, a) => window(size, k[a], op.stride[a]));
        break;
      }
      case "conv_t": {
        const k = graph.params[op.w].shape.slice(2);
        out = sizes.get(op.x)!.map((size, a) => (size - 1) * op.stride[a] + k[a]);
        break;
      }
      case "maxpool":
        out = sizes.get(op.x)!.map((size, a) => window(size, op.k[a], op.stride[a]));
        break;
      case "crop": {
        const a = axes.indexOf(op.axis);
        out = sizes.get(op.x)!.map((size, b) => (b === a ? size - op.start - op.end_margin : size));
        break;
      }
      case "upsample":
        out = sizes.get(op.x)!.map((size) => size * op.factor);
        break;
      case "add":
      case "concat": {
        const shapes = opInputs(op).map((x) => sizes.get(x)!);
        if (shapes.some((s) => s.some((size, a) => size !== shapes[0][a]))) return null;
        out = shapes[0];
        break;
      }
      default:  // bn, inorm, relu: the same shape
        out = sizes.get(op.x)!;
    }
    if (!out.every((size) => size > 0)) return null;
    sizes.set(op.y, out);
  }
  const out = sizes.get(graph.output)!;
  return out.every((size) => size === n - graph.context) ? out[0] : null;
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

/** The tile sides the Finetune tab offers a 2D model: the side the graph
 * takes nearest each of TILE_TARGETS, each once. */
export const TILE_TARGETS: readonly number[] = [172, 300, 428, 556];

/** The default tile of a 2D model: the offered side nearest this. 300²
 * trains a step in about 0.3 s on a laptop's GPU in 320 MB (setup_53_2d;
 * setup_54_2d 0.46 s in 908 MB), where 556² takes four times both (1.2 s,
 * 1.3 GB; 1.9 s, 3.7 GB, more than many laptops' GPUs give a tab), and half
 * of a 300² tile is output, the paint the loss reaches; at 172² only a
 * quarter is. */
export const DEFAULT_TILE = 300;

/** The block sides the tab offers a 3D model whose files' sides are not
 * known (tileChoices offers those when they are), as TILE_TARGETS. A block
 * costs about its volume, so the range is narrow: a step at 172³ takes 2.4
 * to 2.9 times the memory and time of one at 124³ (net3d.ts), and little
 * of a block is output (36³ of 124³ is 2.4%, 84³ of 172³ 11.6%). */
export const TILE_TARGETS_3D: readonly number[] = [124, 140, 156, 172];

/** The default block of a 3D model, as DEFAULT_TILE: the smallest offered,
 * whose step takes 2.2 GB of GPU memory (setup_53) to 4.0 GB (setup_54). */
export const DEFAULT_TILE_3D = 124;

/** The largest side looked at for the choices, per rank. */
const MAX_SIDE = { 2: 1100, 3: 256 } as const;

/** One tile side the tab offers, with what it is expected to cost: per
 * step of one tile (batch 1), from the manifest's spike where it measured
 * that side, else scaled by area (volume, in 3D) from the nearest side it
 * measured. */
export interface TileChoice {
  side: number;
  output: number;
  /** (side − context)² / side² (cubed in 3D): the share of the tile the loss reaches. */
  useful: number;
  step_ms: number | null;
  peak_mb: number | null;
  /** Whether the numbers were measured at this side (else scaled). */
  measured: boolean;
  /** 3: a cubic block (a 3D model's); 2 or absent: a square tile. */
  dims?: 2 | 3;
}

/** The offered sides of `graph` with their costs from `spike` (side → numbers).
 * A 3D model's are `files`, the block sides of the files a finetune can be
 * written into (run-full.ts writableBlocks), where the graph takes them: an
 * InstanceNorm network normalizes each block by its own statistics, so a
 * finetune serves well only at the block it trained on, and it is served
 * from the file it was written into. Without them, the sides nearest
 * TILE_TARGETS (TILE_TARGETS_3D). */
export function tileChoices(graph: TrainGraph, spike: Record<string, { step_ms: number; peak_mb: number }> = {}, files: readonly number[] = []): TileChoice[] {
  const dims = is3d(graph) ? 3 : 2;
  const sides = tileSides(graph, MAX_SIDE[dims]);
  if (!sides.length) return [];
  const fitting = dims === 3 ? [...new Set(files.filter((f) => sides.includes(f)))].sort((a, b) => a - b) : [];
  const offered = fitting.length ? fitting : [...new Set((dims === 3 ? TILE_TARGETS_3D : TILE_TARGETS).map((t) => nearest(sides, t)))].sort((a, b) => a - b);
  const measuredSides = Object.keys(spike).map(Number).filter((n) => Number.isInteger(n) && n > 0 && spike[String(n)]);
  return offered.map((side) => {
    const output = side - graph.context;
    const from = measuredSides.length ? nearest(measuredSides, side) : null;
    const scale = from === null ? 0 : side ** dims / from ** dims;
    const numbers = from === null ? null : spike[String(from)];
    return {
      side,
      output,
      useful: output ** dims / side ** dims,
      step_ms: numbers ? Math.round(numbers.step_ms * scale) : null,
      peak_mb: numbers ? Math.round(numbers.peak_mb * scale) : null,
      measured: from === side,
      dims,
    };
  });
}

/** The default side among `choices`: `served` when offered (a 3D model's
 * block as it is served, which a finetune is then served at too), else the
 * one nearest DEFAULT_TILE (DEFAULT_TILE_3D for blocks). */
export function defaultTile(choices: TileChoice[], served: number | null = null): number | null {
  if (served !== null && choices.some((c) => c.side === served)) return served;
  return choices.length ? nearest(choices.map((c) => c.side), choices[0].dims === 3 ? DEFAULT_TILE_3D : DEFAULT_TILE) : null;
}

/** The value of `values` nearest `target`; a tie goes to the larger, whose cost is the one not to underestimate. */
function nearest(values: number[], target: number): number {
  return values.reduce((best, v) => {
    const d = Math.abs(v - target), bd = Math.abs(best - target);
    return d < bd || (d === bd && v > best) ? v : best;
  });
}
