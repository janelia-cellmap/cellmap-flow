// The whole network of a model as a trainable TensorFlow.js graph, and its
// weights read from and written back into the served .onnx.
//
// Vendored from Marwan Zouinkhi's reference runtime for cellmap-flow browser
// finetuning ("full" mode, library tfjs): finetune/cellmap_tfjs_train.js,
// the same file in every 2D and 3D folder of the Hugging Face repo
// cellmap/onnx-web-models (commit 3331842ec2, 2026-10-17; its 2D part as at
// 1a480eb78c, 2026-10-16), with its train_graph.json, initializers.json,
// torch_params.json and weight_map.json beside it, exported by him from the
// PyTorch models. Ported to TypeScript with the same functions and the same
// arithmetic; what differs:
//   - TF.js is a parameter typed as @tensorflow/tfjs-core, imported for its
//     types only, so this module costs nothing until a whole-network run
//     loads TF.js (tfjs.ts);
//   - half precision is converted by onnx-head.ts's float16 functions, which
//     the head's write-back already uses (ties to even, where his rounds
//     half up: the same to within one unit in the last place);
//   - an initializer's bytes are checked against the shape it is read with,
//     each weight transform against the tensor it applies to (an axis or
//     index out of range, a permutation that is not one, a reshape or
//     stack of the wrong size), and the graph's ops are typed (graph.ts
//     parses train_graph.json);
//   - buildNet's 3D network (his buildNet3d, here buildPlainNet3d) is kept as
//     he wrote it, the reference the tests hold net3d.ts to on TF.js's CPU
//     backend. On WebGPU it trains blocks of 100³ at most (measured on
//     2026-10-18; 124³ stops): plain autodiff keeps every intermediate, and
//     once a map has more than 16 × 65535 voxels (from 108³ on) its
//     backward pass stops at WebGPU's dispatch limit, as TF.js sums a
//     bias's gradient over a rank-2 view by a transpose that dispatches a
//     workgroup per 16 voxels on one axis. A 3D run trains through
//     net3d.ts's buildNet3d, the same network with a leaner backward pass.
//     Its nets also have dispose(), as the 2D ones do.
// Hugging Face serves the .js without a JavaScript MIME type, so the file is
// carried here rather than imported from the Hub.
//
// 2D (layout NHWC; the params are the ONNX initializers, by name):
//   const net = buildNet(tf, graph, readOnnxWeights(onnxBytes, offsets, Object.keys(graph.params)));
//   const opt = tf.train.adam(1e-4);
//   const loss = trainStep(tf, net, opt, x, target, mask);  // x [B,H,W,1]; target/mask [B,H-ctx,W-ctx,C]
//   const newBytes = writeOnnxWeights(onnxBytes, offsets, net.exportWeights());
// Tensors are NHWC. ONNX conv weights [O,I,kh,kw] -> tf [kh,kw,I,O];
// ConvTranspose [I,O,kh,kw] -> tf [kh,kw,O,I]; the same permutation either way.
//
// 3D (layout NDHWC): the params are the PyTorch state_dict's tensors, rebuilt
// from the served .onnx (whose Conv3d is split into one convolution per
// depth tap, and whose ConvTranspose3d is a 1×1 convolution to 8× the
// channels) by torch_params.json's recipes, and written back by
// weight_map.json's:
//   const net = buildNet(tf, graph3d, readTorchParams(onnxBytes, offsets, torchParams));  // x [1,D,H,W,1]
//   const newBytes = writeOnnxWeights(onnxBytes, offsets, toOnnxWeights(weightMap, net.exportWeights(), offsets));
// WebGPU workarounds in the 3D path (his): bias and norm affine on rank-2
// views (a rank-5 add with a 1-D operand of under 128 values fails to
// compile), crops as rank-4 and rank-3 slices (the rank-5 slice's gradient,
// a pad, fails to compile), ConvTranspose3d with kernel = stride = 2 as a
// matMul and a depth-to-space (conv3dTranspose has no gradient), nearest
// upsampling as rank-3 tiles.
import type * as TF from "@tensorflow/tfjs-core";

import { float16ToFloat32, float32ToFloat16 } from "../onnx-head";
import type { TrainGraph, TrainGraph3d } from "./graph";

export type Tf = typeof TF;

/** Where an initializer's values lie in one .onnx file (finetune/initializers.json, per file). */
export interface InitializerOffset {
  offset: number;
  nbytes: number;
  dtype: string;
  shape: number[];
}

export type InitializerOffsets = Record<string, InitializerOffset>;

const bytesPer = (dtype: string, name: string): 2 | 4 => {
  if (dtype === "float32") return 4;
  if (dtype === "float16") return 2;
  throw new Error(`${name}: dtype ${dtype} is not writable (only float32 and float16 weights are)`);
};

/** The initializers `names` of the .onnx `bytes` as float32, by their offsets. */
export function readOnnxWeights(bytes: Uint8Array, offsets: InitializerOffsets, names: string[]): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const name of names) {
    const o = offsets[name];
    if (!o) throw new Error(`initializer ${name} not in file`);
    const size = bytesPer(o.dtype, name);
    if (o.offset + o.nbytes > bytes.byteLength) throw new Error(`${name} lies past the end of the file (byte ${o.offset + o.nbytes} of ${bytes.byteLength})`);
    const dv = new DataView(bytes.buffer, bytes.byteOffset + o.offset, o.nbytes);
    const a = new Float32Array(o.nbytes / size);
    for (let i = 0; i < a.length; i++) a[i] = size === 4 ? dv.getFloat32(4 * i, true) : float16ToFloat32(dv.getUint16(2 * i, true));
    out[name] = a;
  }
  return out;
}

/** A copy of `bytes` with `weights` written over their initializers, each in the file's own dtype. */
export function writeOnnxWeights(bytes: Uint8Array, offsets: InitializerOffsets, weights: Record<string, Float32Array>): Uint8Array<ArrayBuffer> {
  const b = new Uint8Array(bytes);  // copy
  for (const [name, a] of Object.entries(weights)) {
    const o = offsets[name];
    if (!o) throw new Error(`initializer ${name} not in file`);
    const size = bytesPer(o.dtype, name);
    if (a.length * size !== o.nbytes) throw new Error(`${name}: ${a.length} values for ${o.nbytes} bytes of ${o.dtype}`);
    const dv = new DataView(b.buffer, b.byteOffset + o.offset, o.nbytes);
    for (let i = 0; i < a.length; i++) {
      if (size === 4) dv.setFloat32(4 * i, a[i], true);
      else dv.setUint16(2 * i, float32ToFloat16(a[i]), true);
    }
  }
  return b;
}

/** A network built from a 2D graph: its forward pass, the trainable
 * variables, and their values back in ONNX layout. */
export interface TrainNet {
  forward(x: TF.Tensor4D): TF.Tensor4D;
  /** The trainable params' values, in ONNX layout, by initializer name. */
  exportWeights(): Record<string, Float32Array>;
  vars: TF.Variable[];
  params: Record<string, TF.Tensor>;
  context: number;
  /** Free every tensor the net holds. */
  dispose(): void;
}

/** A tensor's values with its shape, outside TF.js: a param in PyTorch layout. */
export interface TorchWeight {
  data: Float32Array;
  shape: number[];
}

/** A network built from a 3D graph (buildNet on NDHWC): as TrainNet, with
 * its weights by PyTorch state_dict name, in PyTorch layout. */
export interface TrainNet3d {
  forward(x: TF.Tensor5D): TF.Tensor5D;
  /** The trainable params' values, in PyTorch layout, for toOnnxWeights. */
  exportWeights(): Record<string, TorchWeight>;
  vars: TF.Variable[];
  params: Record<string, TF.Tensor>;
  context: number;
  dispose(): void;
}

export interface BuildOptions {
  /** The params to train, by name (default: the graph's trainable flags). */
  trainable?: ReadonlySet<string> | null;
}

/** The net of `graph` with `weights` (by name: ONNX layout in 2D, PyTorch
 * layout in 3D, readTorchParams). `trainable`: the params to train, by name
 * (default: the graph's trainable flags); every other param is a constant
 * (BatchNorm's running statistics, say). A graph typed as 3D (graph.ts is3d)
 * gives a TrainNet3d; any other, a TrainNet, which an NDHWC graph does not
 * make: narrow it with is3d first. */
export function buildNet(tf: Tf, graph: TrainGraph3d, weights: Record<string, Float32Array>, options?: BuildOptions): TrainNet3d;
export function buildNet(tf: Tf, graph: TrainGraph, weights: Record<string, Float32Array>, options?: BuildOptions): TrainNet;
export function buildNet(tf: Tf, graph: TrainGraph, weights: Record<string, Float32Array>, { trainable = null }: BuildOptions = {}): TrainNet | TrainNet3d {
  if (graph.layout === "NDHWC") return buildPlainNet3d(tf, graph as TrainGraph3d, weights, trainable);
  const P: Record<string, TF.Tensor> = {};
  const kind: Record<string, "conv" | "conv_t"> = {};
  for (const op of graph.ops) {
    if (op.op === "conv") kind[op.w] = "conv";
    if (op.op === "conv_t") kind[op.w] = "conv_t";
  }
  for (const [name, meta] of Object.entries(graph.params)) {
    const values = weights[name];
    if (!values) throw new Error(`no weights for ${name}`);
    let t: TF.Tensor = tf.tensor(values, meta.shape, "float32");
    if (kind[name]) {
      const turned = tf.transpose(t, [2, 3, 1, 0]);
      t.dispose();
      t = turned;
    }
    const train = trainable ? trainable.has(name) : meta.trainable;
    // Unnamed: TF.js variable names are global, and several nets may coexist.
    P[name] = train ? tf.variable(t, true) : t;
    if (train) t.dispose();
  }
  const param = <R extends TF.Rank>(name: string) => P[name] as TF.Tensor<R>;

  function forward(x: TF.Tensor4D): TF.Tensor4D {
    const T: Record<string, TF.Tensor4D> = { [graph.input]: x };
    for (const op of graph.ops) {
      let y: TF.Tensor4D;
      switch (op.op) {
        case "conv": {
          y = tf.conv2d(T[op.x], param<TF.Rank.R4>(op.w), op.stride as [number, number], "valid");
          if (op.b) y = tf.add(y, param<TF.Rank.R1>(op.b));
          break;
        }
        case "conv_t": {
          const a = T[op.x];
          const [kh, kw, o] = P[op.w].shape, [n, h, w] = a.shape;
          y = tf.conv2dTranspose(a, param<TF.Rank.R4>(op.w), [n, (h - 1) * op.stride[0] + kh, (w - 1) * op.stride[1] + kw, o], op.stride as [number, number], "valid");
          if (op.b) y = tf.add(y, param<TF.Rank.R1>(op.b));
          break;
        }
        case "maxpool":
          y = tf.maxPool(T[op.x], op.k as [number, number], op.stride as [number, number], "valid");
          break;
        case "bn":  // frozen statistics: the running mean and variance
          y = tf.batchNorm<TF.Rank.R4>(T[op.x], param<TF.Rank.R1>(op.mean), param<TF.Rank.R1>(op.var), param<TF.Rank.R1>(op.bias), param<TF.Rank.R1>(op.scale), op.eps);
          break;
        case "inorm": {
          const a = T[op.x];
          const { mean, variance } = tf.moments(a, [1, 2], true);
          y = tf.add(tf.mul(tf.div(tf.sub(a, mean), tf.sqrt(tf.add(variance, op.eps))), param(op.scale)), param(op.bias));
          break;
        }
        case "relu":
          y = tf.relu(T[op.x]);
          break;
        case "add":
          y = tf.add(T[op.x], T[op.x2]);
          break;
        case "concat":
          y = tf.concat(op.xs.map((k) => T[k]), 3);
          break;
        case "crop": {
          const s = T[op.x].shape, ax = op.axis === "h" ? 1 : 2;
          const begin: [number, number, number, number] = [0, 0, 0, 0], size: [number, number, number, number] = [-1, -1, -1, -1];
          begin[ax] = op.start;
          size[ax] = s[ax] - op.start - op.end_margin;
          y = tf.slice(T[op.x], begin, size);
          break;
        }
        case "upsample": {
          // Nearest neighbour as repeats; rank-3 tiles, since TF.js has no
          // gradient for a rank-6 tile.
          const [n, h, w, c] = T[op.x].shape, f = op.factor;
          const yw = tf.reshape(tf.tile(tf.reshape(T[op.x], [n * h * w, 1, c]), [1, f, 1]), [n * h, 1, w * f * c]);
          y = tf.reshape(tf.tile(yw, [1, f, 1]), [n, h * f, w * f, c]);
          break;
        }
      }
      T[op.y] = y;
    }
    return T[graph.output];
  }

  function exportWeights(): Record<string, Float32Array> {  // back to ONNX layout, trainable params only
    const out: Record<string, Float32Array> = {};
    for (const [name, t] of Object.entries(P)) {
      if (!(t instanceof tf.Variable)) continue;
      const v = kind[name] ? tf.transpose(t, [3, 2, 0, 1]) : t;
      out[name] = Float32Array.from(v.dataSync() as Float32Array);
      if (v !== t) v.dispose();
    }
    return out;
  }

  const vars = Object.values(P).filter((t): t is TF.Variable => t instanceof tf.Variable);
  return {
    forward,
    exportWeights,
    vars,
    params: P,
    context: graph.context,
    dispose: () => Object.values(P).forEach((t) => t.dispose()),
  };
}

/** Masked BCE on logits: the mean over mask > 0 of max(z, 0) − z·t + log(1 + exp(−|z|)). */
export function maskedBce(tf: Tf, z: TF.Tensor, t: TF.Tensor, m: TF.Tensor): TF.Scalar {
  return tf.tidy(() => {
    const l = tf.add(tf.sub(tf.relu(z), tf.mul(z, t)), tf.log1p(tf.exp(tf.neg(tf.abs(z)))));
    return tf.div(tf.sum(tf.mul(l, m)), tf.maximum(tf.sum(m), 1)) as TF.Scalar;
  });
}

/** One optimizer step on the batch; the loss before it. */
export function trainStep(tf: Tf, net: TrainNet, opt: TF.Optimizer, x: TF.Tensor4D, target: TF.Tensor4D, mask: TF.Tensor4D): TF.Scalar;
export function trainStep(tf: Tf, net: TrainNet3d, opt: TF.Optimizer, x: TF.Tensor5D, target: TF.Tensor5D, mask: TF.Tensor5D): TF.Scalar;
export function trainStep(tf: Tf, net: TrainNet | TrainNet3d, opt: TF.Optimizer, x: TF.Tensor, target: TF.Tensor, mask: TF.Tensor): TF.Scalar {
  const forward = net.forward as (x: TF.Tensor) => TF.Tensor;
  const loss = opt.minimize(() => maskedBce(tf, forward(x), target, mask), true, net.vars);
  if (!loss) throw new Error("the optimizer returned no loss");
  return loss;
}

// ---------------------------------------------------------------- structured weight transforms (3D folders)

/** One step of a recipe (torch_params.json from_onnx, weight_map.json
 * to_onnx), on a tensor's values in row-major order. A recipe may start
 * from an ONNX initializer (`from`) or several stacked on a new axis
 * (`stack`); then `index` [axis, k] takes slice k of an axis, dropping it;
 * `permute` reorders the axes (as torch.permute); `reshape` reshapes;
 * `repeat_interleave` [axis, n] repeats each entry of an axis n times in
 * place (as torch.repeat_interleave). The semantics of his
 * finetune_spec.py apply_ops. */
export type WeightTransform =
  | { from: string }
  | { stack: string[]; axis: number }
  | { index: [number, number] }
  | { permute: number[] }
  | { reshape: number[] }
  | { repeat_interleave: [number, number] };

/** A param of a 3D graph and how to read it from the .onnx (finetune/torch_params.json, by state_dict name). */
export interface TorchParam {
  shape: number[];
  from_onnx: WeightTransform[];
}

/** How an ONNX initializer is made from a param (finetune/weight_map.json, by initializer name). */
export interface WeightMapEntry {
  torch: string;
  to_onnx: WeightTransform[];
  /** The transform as PyTorch indexing, for people: "[:, :, 0]", "repeat_interleave(8)". */
  transform?: string;
}

const prod = (a: readonly number[]) => a.reduce((x, y) => x * y, 1);
const isAxis = (ax: unknown, rank: number): ax is number => Number.isInteger(ax) && (ax as number) >= 0 && (ax as number) < rank;

/** `start` (or nothing, for a recipe that names its source) through `ops`;
 * `get` reads an initializer by name. */
function applyTransforms(start: TorchWeight | null, ops: WeightTransform[], get: (name: string) => TorchWeight, what: string): TorchWeight {
  let a = start;
  const fail = (problem: string): never => {
    throw new Error(`${what}: ${problem}`);
  };
  for (const o of ops) {
    if ("from" in o) {
      a = get(o.from);
      continue;
    }
    if ("stack" in o) {
      const parts = o.stack.map(get), s = parts[0]?.shape ?? [], ax = o.axis;
      if (!parts.length || parts.some((p) => p.shape.join() !== s.join())) fail(`stacks ${o.stack.join(", ")}, which are not all of one shape`);
      if (!isAxis(ax, s.length + 1)) fail(`stacks on axis ${ax} of a rank-${s.length + 1} result`);
      const outer = prod(s.slice(0, ax)), inner = prod(s.slice(ax));
      const d = new Float32Array(outer * inner * parts.length);
      for (let i = 0; i < outer; i++) parts.forEach((p, k) => d.set(p.data.subarray(i * inner, (i + 1) * inner), (i * parts.length + k) * inner));
      a = { data: d, shape: [...s.slice(0, ax), parts.length, ...s.slice(ax)] };
      continue;
    }
    if (!a) return fail(`${JSON.stringify(o)} has no tensor to apply to (a recipe starts with from or stack)`);
    const s = a.shape, r = s.length;
    if ("index" in o) {
      const [ax, k] = o.index;
      if (!isAxis(ax, r) || !isAxis(k, s[ax])) fail(`takes index ${k} of axis ${ax} of a [${s}] tensor`);
      const outer = prod(s.slice(0, ax)), inner = prod(s.slice(ax + 1)), n = s[ax];
      const d = new Float32Array(outer * inner);
      for (let i = 0; i < outer; i++) d.set(a.data.subarray((i * n + k) * inner, (i * n + k + 1) * inner), i * inner);
      a = { data: d, shape: [...s.slice(0, ax), ...s.slice(ax + 1)] };
    } else if ("reshape" in o) {
      if (prod(o.reshape) !== a.data.length) fail(`reshapes ${a.data.length} values to [${o.reshape}]`);
      a = { data: a.data, shape: o.reshape.slice() };
    } else if ("permute" in o) {
      const p = o.permute;
      if (p.length !== r || [...p].sort((x, y) => x - y).some((v, i) => v !== i)) fail(`permutes a rank-${r} tensor by [${p}]`);
      const os = p.map((i) => s[i]);
      const st = new Array<number>(r);
      st[r - 1] = 1;
      for (let i = r - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1];
      const pst = p.map((i) => st[i]), d = new Float32Array(a.data.length), idx = new Array<number>(r).fill(0);
      for (let j = 0; j < d.length; j++) {
        let src = 0;
        for (let i = 0; i < r; i++) src += idx[i] * pst[i];
        d[j] = a.data[src];
        for (let i = r - 1; i >= 0; i--) {
          if (++idx[i] < os[i]) break;
          idx[i] = 0;
        }
      }
      a = { data: d, shape: os };
    } else if ("repeat_interleave" in o) {
      const [ax, n] = o.repeat_interleave;
      if (!isAxis(ax, r) || !(Number.isInteger(n) && n > 0)) fail(`repeats axis ${ax} of a [${s}] tensor ${n} times`);
      const outer = prod(s.slice(0, ax + 1)), inner = prod(s.slice(ax + 1));
      const d = new Float32Array(a.data.length * n);
      for (let i = 0; i < outer; i++) for (let k = 0; k < n; k++) d.set(a.data.subarray(i * inner, (i + 1) * inner), (i * n + k) * inner);
      a = { data: d, shape: s.map((v, i) => (i === ax ? v * n : v)) };
    } else {
      fail(`transform ${JSON.stringify(o)} is not one this browser knows`);
    }
  }
  if (!a) return fail("the recipe is empty");
  return a;
}

/** The params of a 3D graph, by state_dict name, in PyTorch layout, read
 * from one served .onnx by torch_params.json's from_onnx recipes. */
export function readTorchParams(bytes: Uint8Array, offsets: InitializerOffsets, torchParams: Record<string, TorchParam>): Record<string, Float32Array> {
  const names = new Set<string>();
  for (const p of Object.values(torchParams)) for (const o of p.from_onnx) ("from" in o ? [o.from] : "stack" in o ? o.stack : []).forEach((n) => names.add(n));
  const raw = readOnnxWeights(bytes, offsets, [...names]);
  const get = (name: string): TorchWeight => {
    const shape = offsets[name].shape;
    if (prod(shape) !== raw[name].length) throw new Error(`initializer ${name}: ${raw[name].length} values for the shape [${shape}]`);
    return { data: raw[name], shape };
  };
  const out: Record<string, Float32Array> = {};
  for (const [name, p] of Object.entries(torchParams)) {
    const a = applyTransforms(null, p.from_onnx, get, `torch_params.json ${name}`);
    if (a.data.length !== prod(p.shape)) throw new Error(`torch_params.json ${name}: ${a.data.length} values for the shape [${p.shape}]`);
    out[name] = a.data;
  }
  return out;
}

/** Trained params in PyTorch layout (a 3D net's exportWeights) as ONNX
 * initializers by weight_map.json's to_onnx recipes, for writeOnnxWeights:
 * those of the params given, float in this file (`offsets`), each at its
 * initializer's shape. */
export function toOnnxWeights(weightMap: Record<string, WeightMapEntry>, params: Record<string, TorchWeight>, offsets: InitializerOffsets): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const [name, e] of Object.entries(weightMap)) {
    const o = offsets[name];
    if (!(e.torch in params) || !o || (o.dtype !== "float32" && o.dtype !== "float16")) continue;
    const a = applyTransforms(params[e.torch], e.to_onnx, () => {
      throw new Error(`weight_map.json ${name}: to_onnx reads an initializer, where it starts from ${e.torch}`);
    }, `weight_map.json ${name}`);
    if (a.data.length !== prod(o.shape)) throw new Error(`weight_map.json ${name}: ${a.data.length} values from ${e.torch} for the initializer's shape [${o.shape}]`);
    out[name] = a.data;
  }
  return out;
}

/** His buildNet3d: the 3D network for plain autodiff (see the header). */
function buildPlainNet3d(tf: Tf, graph: TrainGraph3d, weights: Record<string, Float32Array>, trainable: ReadonlySet<string> | null): TrainNet3d {
  const P: Record<string, TF.Tensor> = {}, shape: Record<string, number[]> = {};
  const kind: Record<string, "conv" | "conv_t"> = {};
  for (const op of graph.ops) {
    if (op.op === "conv") kind[op.w] = "conv";
    if (op.op === "conv_t") kind[op.w] = "conv_t";
  }
  for (const [name, meta] of Object.entries(graph.params)) {
    const values = weights[name];
    if (!values) throw new Error(`no weights for ${name}`);
    shape[name] = meta.shape;
    let t: TF.Tensor = tf.tensor(values, meta.shape, "float32");
    const turned = kind[name] === "conv" ? tf.transpose(t, [2, 3, 4, 1, 0])  // [O,I,kd,kh,kw] -> [kd,kh,kw,I,O]
      : kind[name] === "conv_t" ? tf.reshape(t, [meta.shape[0], prod(meta.shape.slice(1))])  // [I,O,2,2,2] -> [I, O*8]
      : t;
    if (turned !== t) {
      t.dispose();
      t = turned;
    }
    const train = trainable ? trainable.has(name) : meta.trainable;
    P[name] = train ? tf.variable(t, true) : t;
    if (train) t.dispose();
  }
  const flat = (a: TF.Tensor5D, f: (v: TF.Tensor2D) => TF.Tensor2D): TF.Tensor5D => tf.reshape(f(tf.reshape(a, [-1, a.shape[4]])), a.shape);
  const addBias = (a: TF.Tensor5D, b: TF.Tensor) => flat(a, (v) => tf.add(v, b));
  // A crop of axis 1, 2 or 3 (d, h, w) of [N,D,H,W,C] as a slice of a rank-4 or rank-3 view.
  function crop(a: TF.Tensor5D, ax: number, start: number, endMargin: number): TF.Tensor5D {
    const [n, d, h, w, c] = a.shape, len = a.shape[ax] - start - endMargin;
    const v = ax === 1 ? [n, d, h * w, c] : ax === 2 ? [n * d, h, w, c] : [n * d * h, w, c];
    const begin = v.map(() => 0), size = v.map(() => -1);
    begin[1] = start;
    size[1] = len;
    const out = a.shape.slice() as [number, number, number, number, number];
    out[ax] = len;
    return tf.reshape(tf.slice(tf.reshape(a, v), begin, size), out);
  }
  // Nearest upsampling along one axis: a rank-3 tile.
  function repeatAxis(a: TF.Tensor5D, ax: number, f: number): TF.Tensor5D {
    const s = a.shape, outer = prod(s.slice(0, ax + 1)), inner = prod(s.slice(ax + 1));
    const out = s.slice() as [number, number, number, number, number];
    out[ax] *= f;
    return tf.reshape(tf.tile(tf.reshape(a, [outer, 1, inner]), [1, f, 1]), out);
  }

  function forward(x: TF.Tensor5D): TF.Tensor5D {
    const T: Record<string, TF.Tensor5D> = { [graph.input]: x };
    for (const op of graph.ops) {
      let y: TF.Tensor5D;
      switch (op.op) {
        case "conv":
          y = tf.conv3d(T[op.x], P[op.w] as TF.Tensor5D, op.stride as [number, number, number], "valid");
          if (op.b) y = addBias(y, P[op.b]);
          break;
        case "conv_t": {  // k = s = 2: [P,I] x [I, O*8] (+ the bias repeated 8×), then depth-to-space (rank ≤ 6)
          const a = T[op.x], [n, d, h, w, ci] = a.shape, o = (P[op.w] as TF.Tensor2D).shape[1] / 8;
          if (op.stride.some((s) => s !== 2) || shape[op.w].slice(2).some((k) => k !== 2)) throw new Error(`${op.w}: a transposed convolution of kernel and stride 2 only`);
          let z: TF.Tensor = tf.matMul(tf.reshape(a, [-1, ci]), P[op.w] as TF.Tensor2D);
          if (op.b) z = tf.add(z, tf.reshape(tf.tile(tf.reshape(P[op.b], [o, 1]), [1, 8]), [o * 8]));
          z = tf.transpose(tf.reshape(z, [n * d, h, w, o, 2, 4]), [0, 4, 1, 2, 5, 3]);  // [nd, a, h, w, (b e), o]
          z = tf.transpose(tf.reshape(z, [n * d * 2 * h, w, 2, 2, o]), [0, 2, 1, 3, 4]);  // [nd a h, b, w, e, o]
          y = tf.reshape(z, [n, 2 * d, 2 * h, 2 * w, o]);
          break;
        }
        case "maxpool":
          y = tf.maxPool3d(T[op.x], op.k as [number, number, number], op.stride as [number, number, number], "valid");
          break;
        case "bn":  // frozen statistics
          y = flat(T[op.x], (v) => tf.batchNorm(v, P[op.mean] as TF.Tensor1D, P[op.var] as TF.Tensor1D, P[op.bias] as TF.Tensor1D, P[op.scale] as TF.Tensor1D, op.eps));
          break;
        case "inorm": {
          const a = T[op.x];
          const { mean, variance } = tf.moments(a, [1, 2, 3], true);
          y = flat(tf.div(tf.sub(a, mean), tf.sqrt(tf.add(variance, op.eps))), (v) => tf.add(tf.mul(v, P[op.scale]), P[op.bias]));
          break;
        }
        case "relu":
          y = tf.relu(T[op.x]);
          break;
        case "add":
          y = tf.add(T[op.x], T[op.x2]);
          break;
        case "concat":
          y = tf.concat(op.xs.map((k) => T[k]), 4);
          break;
        case "crop":
          y = crop(T[op.x], 1 + "dhw".indexOf(op.axis), op.start, op.end_margin);
          break;
        case "upsample":
          y = [1, 2, 3].reduce((v, ax) => repeatAxis(v, ax, op.factor), T[op.x]);
          break;
      }
      T[op.y] = y;
    }
    return T[graph.output];
  }

  function exportWeights(): Record<string, TorchWeight> {  // trainable params, PyTorch layout, by state_dict name
    const out: Record<string, TorchWeight> = {};
    for (const [name, t] of Object.entries(P)) {
      if (!(t instanceof tf.Variable)) continue;
      // conv_t: [I, O*8] is [I,O,2,2,2] flattened already.
      const v = kind[name] === "conv" ? tf.transpose(t, [4, 3, 0, 1, 2]) : t;
      out[name] = { data: Float32Array.from(v.dataSync() as Float32Array), shape: shape[name] };
      if (v !== t) v.dispose();
    }
    return out;
  }

  const vars = Object.values(P).filter((t): t is TF.Variable => t instanceof tf.Variable);
  return {
    forward,
    exportWeights,
    vars,
    params: P,
    context: graph.context,
    dispose: () => Object.values(P).forEach((t) => t.dispose()),
  };
}
