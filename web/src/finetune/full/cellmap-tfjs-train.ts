// The whole network of a 2D model as a trainable TensorFlow.js graph, and
// its weights read from and written back into the served .onnx.
//
// Vendored from Marwan Zouinkhi's reference runtime for cellmap-flow browser
// finetuning ("full" mode, library tfjs): finetune/cellmap_tfjs_train.js in
// each 2D folder of the Hugging Face repo cellmap/onnx-web-models (commit
// 1a480eb78c, 2026-10-16), with its train_graph.json and initializers.json
// beside it, exported by him from the PyTorch models. Ported to TypeScript
// with the same functions and the same arithmetic; what differs:
//   - TF.js is a parameter typed as @tensorflow/tfjs-core, imported for its
//     types only, so this module costs nothing until a whole-network run
//     loads TF.js (tfjs.ts);
//   - half precision is converted by onnx-head.ts's float16 functions, which
//     the head's write-back already uses (ties to even, where his rounds
//     half up: the same to within one unit in the last place);
//   - an initializer's bytes are checked against the shape it is read with,
//     and the graph's ops are typed (graph.ts parses train_graph.json).
// Hugging Face serves the .js without a JavaScript MIME type, so the file is
// carried here rather than imported from the Hub.
//
//   const net = buildNet(tf, graph, readOnnxWeights(onnxBytes, offsets, Object.keys(graph.params)));
//   const opt = tf.train.adam(1e-4);
//   const loss = trainStep(tf, net, opt, x, target, mask);  // x [B,H,W,1]; target/mask [B,H-ctx,W-ctx,C]
//   const newBytes = writeOnnxWeights(onnxBytes, offsets, net.exportWeights());
//
// Tensors are NHWC. ONNX conv weights [O,I,kh,kw] -> tf [kh,kw,I,O];
// ConvTranspose [I,O,kh,kw] -> tf [kh,kw,O,I]; the same permutation either way.
import type * as TF from "@tensorflow/tfjs-core";

import { float16ToFloat32, float32ToFloat16 } from "../onnx-head";
import type { TrainGraph } from "./graph";

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

/** A network built from the graph: its forward pass, the trainable
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

/** The net of `graph` with `weights` (ONNX layout, by name). `trainable`:
 * the params to train, by name (default: the graph's trainable flags); every
 * other param is a constant (BatchNorm's running statistics, say). */
export function buildNet(tf: Tf, graph: TrainGraph, weights: Record<string, Float32Array>, { trainable = null }: { trainable?: ReadonlySet<string> | null } = {}): TrainNet {
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
          y = tf.conv2d(T[op.x], param<TF.Rank.R4>(op.w), op.stride, "valid");
          if (op.b) y = tf.add(y, param<TF.Rank.R1>(op.b));
          break;
        }
        case "conv_t": {
          const a = T[op.x];
          const [kh, kw, o] = P[op.w].shape, [n, h, w] = a.shape;
          y = tf.conv2dTranspose(a, param<TF.Rank.R4>(op.w), [n, (h - 1) * op.stride[0] + kh, (w - 1) * op.stride[1] + kw, o], op.stride, "valid");
          if (op.b) y = tf.add(y, param<TF.Rank.R1>(op.b));
          break;
        }
        case "maxpool":
          y = tf.maxPool(T[op.x], op.k, op.stride, "valid");
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
export function trainStep(tf: Tf, net: TrainNet, opt: TF.Optimizer, x: TF.Tensor4D, target: TF.Tensor4D, mask: TF.Tensor4D): TF.Scalar {
  const loss = opt.minimize(() => maskedBce(tf, net.forward(x), target, mask), true, net.vars);
  if (!loss) throw new Error("the optimizer returned no loss");
  return loss;
}
