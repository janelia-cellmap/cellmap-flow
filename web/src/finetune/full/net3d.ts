// The whole network of a 3D model, trainable in TF.js within a browser's GPU
// memory: the interface run-full.ts trains a 3D model through.
//
// Marwan Zouinkhi's reference runtime (cellmap-tfjs-train.ts, buildNet on an
// NDHWC graph) builds a 3D network for plain autodiff, which keeps every
// intermediate until the backward pass ends: about 8 GB of tensors (10 GB
// held by TF.js's buffer pool) for one 124³ block of setup_53, and no larger
// block fits an 11 GB GPU; as he wrote it, its backward pass does not even
// run there (its bias gradients are summed by a transpose past WebGPU's
// dispatch limit, see channelSum). This builds the same network from the
// same graph and weights, with his WebGPU workarounds (per-channel
// arithmetic on rank-2 views, the transposed convolution as a matMul and a
// depth-to-space, upsampling as rank-3 tiles), and trains it with
// - every op's gradient but max pooling's and a lone ReLU's written here
//   (tf.customGrad), keeping only what it needs: TF.js's own gradients keep
//   every input they might want for a shape (a crop or a concatenation
//   keeps its whole input, a bias add the convolution's output before the
//   bias);
// - convolution, BatchNorm and ReLU fused into one op that keeps only the
//   convolution's input and the ReLU's output (the normalized values are
//   recovered from the output where the ReLU passed them); InstanceNorm,
//   with the residual addition and the ReLU after it, into one that keeps
//   the norm's input and output; a chain of crops into one slice;
// - a convolution's gradients as matrix products and a forward convolution
//   (convGrads), where TF.js's own kernels for them took 10 of a step's 15
//   s at 172³;
// - the backward pass recomputing the forward segment by segment from
//   checkpoints (every ReLU's output, and whatever else a later segment
//   reads), rather than keeping every activation;
// - every op freeing what it made once the next has read it, TF.js
//   submitting its commands after every op so that a freed buffer can be
//   reused at once, and between segments the GPU synced and TF.js's
//   free-buffer pool emptied: the pool reuses a buffer only for a tensor of
//   the same byte size, in 3D nearly every size is new, and the GPU frees a
//   destroyed buffer only once the work queued before it is done;
// - Adam of our own, as TF.js's (adam.ts: same arithmetic, same epsilon),
//   so that a step's update is computed beside the weights and kept only if
//   the GPU reported no error, at the step's own learning rate (the
//   schedule's, ../schedule.ts).
// A step of one block, after the starting network's forward pass for
// distillation (forwardAsync), on an RTX 2080 Ti (11 GB), TF.js 4.22.0
// WebGPU in Chromium, 2026-10-18 (step + forward pass, GPU memory at the peak):
//   setup_53 (UNet, BatchNorm)    124³ 2.5 + 0.3 s, 2.2 GB; 140³ 3.6 + 0.5 s, 3.0 GB; 172³ 7.2 + 1.0 s, 5.3 GB
//   setup_54 (ResUNet, InstNorm)  124³ 3.4 + 0.7 s, 4.0 GB; 140³ 4.9 + 1.0 s, 5.5 GB; 172³ 9.8 + 2.1 s, 9.9 GB
// where the spike of 2026-10-17 measured setup_53 at 124³ in 4.5 s and 3.0
// GB, and setup_54 at 124³ in 10.9 s and 9.7 GB, and nothing larger of it.
// Its logits match PyTorch's to 4·10⁻⁵ (setup_53 at 124³, setup_54 at 116³).
//
// Running out of GPU memory throws nothing in TF.js: a failed allocation
// leaves a garbage loss and a corrupted update. Each step runs inside WebGPU
// error scopes, and a step that hit one throws GpuMemoryError, the weights
// as they were before it; the network trains on after it (a smaller block).
import type * as TF from "@tensorflow/tfjs-core";

import { adamUpdate, BETA1, BETA2 } from "./adam";
import type { Tf } from "./cellmap-tfjs-train";
import { describeSides, opInputs, outputSide, tileSides, type TrainGraph, type TrainOp } from "./graph";
import type { LossTerms } from "./loss";

/** A step's loss on the network's logits: the total it minimizes, and its terms for the log (loss.ts lossTerms). */
export type Loss3d = (z: TF.Tensor5D) => LossTerms;

/** A step's terms as numbers, as they were before the step: Loss3d's, null where it had none. */
export interface StepTerms { total: number; supervised: number; distillation: number | null; otherChannels: number | null }

export interface Net3d {
  /** One Adam step on the block `x` ([1, D, H, W, 1], NDHWC) with `loss` on
   * the logits ([1, D − context, H − context, W − context, C]), at
   * `learningRate` (the schedule's for the step); the terms as they were
   * before the step. Throws GpuMemoryError when the GPU ran out of memory,
   * the weights left as they were. */
  step(x: TF.Tensor5D, loss: Loss3d, learningRate: number): Promise<StepTerms>;
  /** The logits on `x` without a gradient, freeing as it goes (the starting network's, for distillation). */
  forward(x: TF.Tensor5D): TF.Tensor5D;
  /** forward, waiting for the GPU between segments so that it gives back
   * what each freed: half forward's GPU memory at its peak, a fifth slower
   * (forward cannot wait, and the GPU frees a buffer only once the work
   * queued before it is done). Throws GpuMemoryError as step does. */
  forwardAsync(x: TF.Tensor5D): Promise<TF.Tensor5D>;
  /** The trainable params in PyTorch layout, by state_dict name, for toOnnxWeights. */
  exportWeights(): Record<string, { data: Float32Array; shape: number[] }>;
  /** Free every tensor and the optimizer's state. */
  dispose(): void;
}

/** The GPU ran out of memory during a step: the block is too large for this machine. */
export class GpuMemoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GpuMemoryError";
  }
}

export interface Net3dOptions {
  /** The params to train, by name; the graph's trainable flags when absent.
   * None builds a network that only runs forward (a teacher). */
  trainable?: ReadonlySet<string> | null;
}

type Triple = [number, number, number];
type Saver = (tensors: TF.Tensor[]) => void;

/** One piece of the network as it runs: an op of the graph, or a few fused. */
interface Unit {
  /** The graph's ops it computes, by index; the last makes its output. */
  ops: number[];
  /** The tensors it reads, by name. */
  inputs: string[];
  output: string;
  /** The params it uses, by name. */
  params: string[];
  /** It ends in a ReLU: where the step keeps a checkpoint. */
  relu: boolean;
  /** Its output from its inputs (in `inputs` order). */
  run(xs: TF.Tensor[]): TF.Tensor;
}

/** The ops run between two checkpoints, units [from, to). */
interface Segment {
  from: number;
  to: number;
  /** The tensors it reads from before it: checkpoints, or the block. */
  inputs: string[];
  /** Its last unit's output, the only tensor of it read after it. */
  output: string;
  params: string[];
}

/** A conv's BatchNorm scale this close to 0 is not fused: the fused
 * gradient divides by it (convBnReluOp). The exported models' are 0.73
 * and up. */
const MIN_FUSED_SCALE = 1e-3;

/** The trainable 3D network of `graph` (layout NDHWC) with `weights` in
 * PyTorch layout (readTorchParams). */
export function buildNet3d(tf: Tf, graph: TrainGraph, weights: Record<string, Float32Array>, options: Net3dOptions): Net3d {
  if (graph.layout !== "NDHWC") throw new Error(`buildNet3d builds a 3D model's graph (NDHWC), not ${graph.layout}`);
  const trains = new Set(Object.keys(graph.params).filter((n) => (options.trainable ? options.trainable.has(n) : graph.params[n].trainable)));

  // The params, in TF.js's layouts: a conv's [O,I,kd,kh,kw] as
  // [kd,kh,kw,I,O]; a transposed conv's [I,O,2,2,2] as [I, O·8] (the same
  // values in the same order). Plain tensors, kept out of any caller's
  // tidy: the step replaces them, so they need not be variables.
  const kind: Record<string, "conv" | "conv_t"> = {};
  for (const op of graph.ops) if (op.op === "conv" || op.op === "conv_t") kind[op.w] = op.op;
  const P: Record<string, TF.Tensor> = {};
  const constants: TF.Tensor[] = [];
  const keep = <T extends TF.Tensor>(t: T, constant = false): T => {
    tf.keep(t);
    if (constant) constants.push(t);
    return t;
  };
  try {
    for (const [name, meta] of Object.entries(graph.params)) {
      const values = weights[name];
      if (!values) throw new Error(`no weights for ${name}`);
      if (values.length !== meta.shape.reduce((a, b) => a * b, 1)) throw new Error(`${name}: ${values.length} values for the shape [${meta.shape}]`);
      P[name] = keep(tf.tidy(() => {
        const t = tf.tensor(values, meta.shape, "float32");
        return kind[name] === "conv" ? tf.transpose(t, [2, 3, 4, 1, 0]) : kind[name] === "conv_t" ? tf.reshape(t, [meta.shape[0], -1]) : t;
      }));
    }
  } catch (e) {
    Object.values(P).forEach((t) => t.dispose());
    throw e;
  }

  /** TF.js's WebGPU backend, null on its CPU backend (the tests'). */
  const gpu = webgpuOf(tf);

  // ---- the ops, each with its gradient
  //
  // Every op frees what it makes as soon as the next has read it: a tidy
  // frees only when it ends, and in 3D one map is hundreds of MB. Blocks
  // are one at a time (N = 1), so a map's per-channel arithmetic and
  // statistics run on its rank-2 view [voxels, C].
  const C = (a: TF.Tensor) => a.shape[a.rank - 1];
  const SPATIAL = [1, 2, 3];
  const flat = (a: TF.Tensor) => tf.reshape(a, [-1, C(a)]) as TF.Tensor2D;
  /** `fs` applied in turn to `t`, which is consumed: it and each result are
   * freed once the next op has read them. */
  const then = (t: TF.Tensor, ...fs: ((t: TF.Tensor) => TF.Tensor)[]): TF.Tensor => fs.reduce((cur, f) => {
    const next = f(cur);
    cur.dispose();
    return next;
  }, t);
  /** f on `a` viewed as [-1, C], in a's shape again: per-channel arithmetic
   * on a rank-2 view, since TF.js 4.22.0's WebGPU binary op takes a
   * shared-memory path for a 1-D operand of under 128 values that does not
   * compile for rank 5 ("cannot index type 'vec5'"), and the op then
   * writes nothing. */
  const perChannel = (a: TF.Tensor, f: (v: TF.Tensor) => TF.Tensor) => then(flat(a), f, (v) => tf.reshape(v, a.shape));
  /** v·scale + shift per channel, in one op (a batch norm of mean 0 and variance 1). */
  const affine = (v: TF.Tensor, scale: TF.Tensor, shift: TF.Tensor) =>
    tf.tidy(() => tf.batchNorm(v as TF.Tensor2D, tf.zerosLike(scale), tf.onesLike(scale), shift as TF.Tensor1D, scale as TF.Tensor1D, 0));
  /** Each channel's sum over the voxels, as the product [1, N]·[N, C]: TF.js
   * reduces a leading axis by transposing it last, a full-size copy, and
   * its transpose of a rank-2 [N, C] dispatches N/16 workgroups on one
   * axis, past WebGPU's limit of 65535 from N ≈ 101³ on. */
  const channelSum = (a: TF.Tensor): TF.Tensor1D => tf.tidy(() => tf.reshape(tf.matMul(tf.ones([1, a.size / C(a)]), flat(a)), [C(a)]));

  /** An op of `f`, whose forward pass returns its value and its gradient
   * (one tensor per input, in order). The gradient runs in a tidy: TF.js
   * calls it outside one, and its temporaries would otherwise live until
   * the segment's backward pass ends. */
  function custom(f: (inputs: TF.Tensor[], save: Saver) => { value: TF.Tensor; grad: (dy: TF.Tensor, saved: TF.Tensor[]) => TF.Tensor[] }): (...inputs: TF.Tensor[]) => TF.Tensor {
    return tf.customGrad((...args: (TF.Tensor | Saver)[]) => {
      const save = args.pop() as Saver;
      const { value, grad } = f(args as TF.Tensor[], save);
      return { value, gradFunc: (dy: TF.Tensor, saved: TF.Tensor[]) => tf.tidy(() => grad(dy, saved)) };
    });
  }

  /** [P, k·k·k·c] (P = d·h·w voxels, each column a tap of its k³ window,
   * then a channel) laid out as [1, d·kd, h·kh, w·kw, c]: each voxel's taps
   * in its window, by two rank-4 transposes. */
  const depthToSpace = (m: TF.Tensor, [d, h, w]: number[], [kd, kh, kw]: number[], c: number) => then(m,
    (t) => tf.reshape(t, [d, h * w, kd, kh * kw * c]),
    (t) => tf.transpose(t, [0, 2, 1, 3]),  // [d, a, (h w), (b e c)]
    (t) => tf.reshape(t, [d * kd * h, w, kh, kw * c]),
    (t) => tf.transpose(t, [0, 2, 1, 3]),  // [(d a h), b, w, (e c)]
    (t) => tf.reshape(t, [1, d * kd, h * kh, w * kw, c]));

  /** The gradients of a valid conv3d of `x` by `w` ([kd,kh,kw,I,O]) at
   * `stride`, from dz, its output's (freed here). TF.js's own kernels for
   * them loop over the whole block per weight or per voxel: at 172³ they
   * took 4 and 5 s for one 32-channel layer whose forward takes 0.7 s.
   * Here, for a stride of 1 or of the kernel's size (the exported graphs'):
   * the filter's, one product per tap of the input's window-shifted voxels
   * with dz ([I, P]·[P, O], P the output's voxels); the input's, the
   * forward conv of dz padded by k − 1 with the filter flipped and
   * transposed (stride 1), or dz·wᵀ laid back out into the windows (stride
   * = kernel, which do not overlap). Other strides use TF.js's kernels. */
  function convGrads(x: TF.Tensor, w: TF.Tensor, dz: TF.Tensor, stride: Triple, needDx: boolean, needDw: boolean): [TF.Tensor, TF.Tensor] {
    const k = w.shape.slice(0, 3), [ci, co] = w.shape.slice(3), [, D, H, W] = x.shape, out = dz.shape.slice(1, 4);
    const unit = stride.every((s) => s === 1), tiled = stride.every((s, a) => s === k[a]) && out.every((n, a) => n * k[a] === x.shape[a + 1]);
    let dw: TF.Tensor;
    if (!needDw) dw = tf.zerosLike(w);
    else if (unit || tiled) {
      const x4 = tf.reshape(x, [D, H, W, ci]), dz2 = flat(dz), taps: TF.Tensor[] = [];
      for (let a = 0; a < k[0]; a++) for (let b = 0; b < k[1]; b++) for (let c = 0; c < k[2]; c++) {
        taps.push(tf.tidy(() => {
          const shifted = unit ? tf.slice(x4, [a, b, c, 0], [...out, ci])
            : tf.stridedSlice(x4, [a, b, c, 0], [a + k[0] * (out[0] - 1) + 1, b + k[1] * (out[1] - 1) + 1, c + k[2] * (out[2] - 1) + 1, ci], [...stride, 1]);
          return tf.matMul(tf.reshape(shifted, [-1, ci]), dz2, true, false);
        }));
      }
      dw = tf.tidy(() => tf.reshape(tf.stack(taps), w.shape));
      [x4, dz2, ...taps].forEach((t) => t.dispose());
    } else dw = tf.engine().runKernel("Conv3DBackpropFilterV2", { x, dy: dz }, { strides: stride, pad: "valid", filterShape: w.shape });
    let dx: TF.Tensor;
    if (!needDx) {
      dx = tf.zerosLike(x);
      dz.dispose();
    } else if (unit && k.every((n) => n === 1)) {
      dx = tf.tidy(() => tf.reshape(tf.matMul(flat(dz), tf.reshape(w, [ci, co]), false, true), x.shape));
      dz.dispose();
    } else if (unit) {
      // Flipped on a rank-4 view: TF.js 4.22.0's WebGPU reverse takes rank 4 at most.
      const flipped = tf.tidy(() => tf.transpose(tf.reshape(tf.reverse(tf.reshape(w, [...k, ci * co]), [0, 1, 2]), w.shape), [0, 1, 2, 4, 3]));
      const padded = then(tf.reshape(dz, [...out, co]), (t) => tf.pad(t, [...k.map((n) => [n - 1, n - 1] as [number, number]), [0, 0]]), (t) => tf.expandDims(t, 0));
      dz.dispose();
      dx = tf.conv3d(padded as TF.Tensor5D, flipped as TF.Tensor5D, 1, "valid");
      padded.dispose();
      flipped.dispose();
    } else if (tiled) {
      dx = depthToSpace(tf.tidy(() => tf.matMul(flat(dz), tf.reshape(w, [-1, co]), false, true)), out, k, ci);
      dz.dispose();
    } else {
      dx = tf.conv3dTranspose(dz as TF.Tensor5D, w as TF.Tensor5D, x.shape as [number, number, number, number, number], stride, "valid");
      dz.dispose();
    }
    return [dx, dw];
  }

  /** A valid conv3d plus a bias, keeping only its input and filter: TF.js's
   * bias add would keep the convolution's output too. `needDx`: false for
   * the block itself, whose gradient nothing reads. */
  function convOp(stride: Triple, needDx: boolean, needDw: boolean) {
    return custom(([x, w, b], save) => {
      save([x, w]);
      const z = tf.conv3d(x as TF.Tensor5D, w as TF.Tensor5D, stride, "valid");
      return {
        value: b ? then(z, (z) => perChannel(z, (v) => tf.add(v, b))) : z,
        grad: (dy, [x, w]) => [...convGrads(x, w, tf.clone(dy), stride, needDx, needDw), ...(b ? [channelSum(dy)] : [])],
      };
    });
  }

  /** conv → BatchNorm (frozen statistics) → ReLU as one op, keeping the
   * conv's input and the ReLU's output: y = relu(bn(conv(x) + b)), the
   * bias folded into the mean. Its gradient, with u = dy where y > 0 and s
   * = γ/√(var + ε): dβ = Σu; dγ = Σu·x̂, where x̂ = (y − β)/γ wherever u is
   * not 0; db = s·dβ; then the conv's, from dz = s·u. */
  function convBnReluOp(stride: Triple, bias: boolean, mean: TF.Tensor, variance: TF.Tensor, rstd: TF.Tensor, eps: number, needDx: boolean, needDw: boolean) {
    return custom((inputs, save) => {
      const [x, w] = inputs, b = bias ? inputs[2] : null, [gamma, beta] = inputs.slice(bias ? 3 : 2);
      const shift = b ? tf.sub(mean, b) : mean;
      const value = then(tf.conv3d(x as TF.Tensor5D, w as TF.Tensor5D, stride, "valid"),
        (z) => perChannel(z, (v) => then(tf.batchNorm(v as TF.Tensor2D, shift as TF.Tensor1D, variance as TF.Tensor1D, beta as TF.Tensor1D, gamma as TF.Tensor1D, eps), tf.relu)));
      if (b) shift.dispose();
      save([x, w, value, gamma, beta]);
      return {
        value,
        grad: (dy, [x, w, y, gamma, beta]) => {
          const u = tf.tidy(() => tf.mul(dy, tf.step(y)));
          const dbeta = channelSum(u);
          const dgamma = tf.tidy(() => tf.div(channelSum(tf.mul(tf.sub(flat(y), beta), flat(u))), gamma));
          const s = tf.mul(gamma, rstd);
          const dz = then(u, (u) => perChannel(u, (v) => tf.mul(v, s)));
          return [...convGrads(x, w, dz, stride, needDx, needDw), ...(bias ? [tf.mul(dbeta, s)] : []), dgamma, dbeta];
        },
      };
    });
  }

  /** InstanceNorm, then (`skip`) the residual added and (`relu`) a ReLU, as
   * one op keeping the norm's input z, its statistics and the output y:
   * x̂ = (z − μ)/σ over each channel's N voxels (σ² the biased variance
   * plus ε, as PyTorch's), y = relu(γ·x̂ + β + skip). Its gradient, with u
   * = dy where y > 0: dskip = u; dβ = Σu; dγ = Σu·x̂; dz = (γ/σ)·(u − dβ/N
   * − x̂·dγ/N). */
  function normActOp(eps: number, skip: boolean, relu: boolean) {
    return custom(([z, gamma, beta, s], save) => {
      const n = z.size / C(z);
      const mean = tf.tidy(() => tf.div(channelSum(z), n));
      const variance = tf.tidy(() => tf.div(channelSum(tf.squaredDifference(flat(z), mean)), n));
      const value = then(flat(z),
        (v) => tf.batchNorm(v as TF.Tensor2D, mean as TF.Tensor1D, variance as TF.Tensor1D, beta as TF.Tensor1D, gamma as TF.Tensor1D, eps),
        ...(s ? [(v: TF.Tensor) => tf.add(v, flat(s))] : []),
        ...(relu ? [(v: TF.Tensor) => tf.relu(v)] : []),
        (v) => tf.reshape(v, z.shape));
      save([z, mean, variance, gamma, value]);
      return {
        value,
        grad: (dy, [z, mean, variance, gamma, y]) => {
          const u = relu ? tf.tidy(() => tf.mul(dy, tf.step(y))) : tf.clone(dy);
          const xhat = tf.batchNorm(flat(z), mean as TF.Tensor1D, variance as TF.Tensor1D, undefined, undefined, eps);
          const dbeta = channelSum(u);
          const dgamma = tf.tidy(() => channelSum(tf.mul(xhat, flat(u))));
          const rs = tf.tidy(() => tf.div(gamma, tf.sqrt(tf.add(variance, eps))));
          const part = then(xhat, (v) => tf.tidy(() => affine(v, tf.mul(tf.mul(rs, -1 / n), dgamma), tf.mul(tf.mul(rs, -1 / n), dbeta))));
          const dz = tf.tidy(() => tf.reshape(tf.add(part, affine(flat(u), rs, tf.zerosLike(rs))), z.shape));
          part.dispose();
          if (skip) return [dz, dgamma, dbeta, u];
          u.dispose();
          return [dz, dgamma, dbeta];
        },
      };
    });
  }

  /** BatchNorm alone (after something it does not fuse with), keeping its
   * input: y = (x − mean)·s + β, s = γ/√(var + ε). Its gradient: dx = s·dy;
   * dβ = Σdy; dγ = Σdy·x̂; and the statistics' (frozen in the exports, but
   * a caller may train them): dmean = −s·Σdy, dvar = −½·s·rstd·Σdy·x̂,
   * rstd = 1/√(var + ε). TF.js's own reduces its sums over a rank-2 view
   * by the transpose that fails from ~101³ voxels (channelSum). */
  function bnOp(eps: number) {
    return custom(([x, mean, variance, gamma, beta], save) => {
      save([x, mean, variance, gamma]);
      return {
        value: perChannel(x, (v) => tf.batchNorm(v as TF.Tensor2D, mean as TF.Tensor1D, variance as TF.Tensor1D, beta as TF.Tensor1D, gamma as TF.Tensor1D, eps)),
        grad: (dy, [x, mean, variance, gamma]) => {
          const rstd = tf.rsqrt(tf.add(variance, eps)), s = tf.mul(gamma, rstd);
          const xhat = tf.batchNorm(flat(x), mean as TF.Tensor1D, variance as TF.Tensor1D, undefined, undefined, eps);
          const dbeta = channelSum(dy), dgamma = tf.tidy(() => channelSum(tf.mul(flat(dy), xhat)));
          xhat.dispose();
          const dx = perChannel(dy, (v) => tf.mul(v, s));
          return [dx, tf.neg(tf.mul(s, dbeta)), tf.mul(tf.mul(dgamma, s), tf.mul(rstd, -0.5)), dgamma, dbeta];
        },
      };
    });
  }

  /** Crops of d, h and w (`margins`: each axis's [start, end]) as one slice
   * of the rank-4 view [D, H, W, C], keeping nothing (TF.js's slice keeps
   * its input, for its shape): its gradient is the gradient padded back by
   * one pad, on the same view (TF.js 4.22.0's WebGPU pad does not compile
   * for rank 5). One op rather than the graph's three keeps one padded
   * gradient where TF.js's backward pass would keep all three. */
  function cropOp(margins: [number, number][]) {
    return custom(([x]) => {
      const [, ...dhwc] = x.shape, c = dhwc[3];
      const keep = margins.map(([a, b], ax) => dhwc[ax] - a - b);
      return {
        value: then(tf.reshape(x, dhwc), (t) => tf.slice(t, [...margins.map(([a]) => a), 0], [...keep, c]), (t) => tf.expandDims(t, 0)),
        grad: (dy) => [then(tf.reshape(dy, [...keep, c]), (t) => tf.pad(t, [...margins, [0, 0]]), (t) => tf.expandDims(t, 0))],
      };
    });
  }

  /** Concatenation on the channel axis, keeping nothing (TF.js's keeps
   * every input, for their shapes): its gradient is the gradient's slices,
   * on a rank-2 view. */
  const concatOp = custom((xs) => {
    const channels = xs.map(C), shapes = xs.map((x) => x.shape);
    return {
      value: tf.concat(xs, 4),
      grad: (dy) => {
        const d2 = flat(dy);
        let from = 0;
        return channels.map((c, i) => {
          const g = tf.reshape(tf.slice(d2, [0, from], [-1, c]), shapes[i]);
          from += c;
          return g;
        });
      },
    };
  });

  /** An addition of two maps of one shape, keeping nothing. */
  const addOp = custom(([a, b]) => ({ value: tf.add(a, b), grad: (dy) => [tf.clone(dy), tf.clone(dy)] }));

  /** A transposed conv of kernel = stride = 2 (his): [P, I]·[I, O·8], the
   * bias repeated 8×, then depth-to-space by reshapes and transposes of
   * rank 6 at most (conv3dTranspose has no gradient in TF.js). Keeps its
   * input and filter; its gradient runs the depth-to-space backwards. */
  const convTOp = custom(([x, w, b], save) => {
    save([x, w]);
    const [n, d, h, wd] = x.shape, o = (w as TF.Tensor2D).shape[1] / 8;
    const value = then(tf.matMul(flat(x), w as TF.Tensor2D),
      ...(b ? [(z: TF.Tensor) => tf.tidy(() => tf.add(z, tf.reshape(tf.tile(tf.reshape(b, [o, 1]), [1, 8]), [o * 8])))] : []),
      (z) => tf.reshape(z, [n * d, h, wd, o, 2, 4]),
      (z) => tf.transpose(z, [0, 4, 1, 2, 5, 3]),  // [nd, a, h, w, (b e), o]
      (z) => tf.reshape(z, [n * d * 2 * h, wd, 2, 2, o]),
      (z) => tf.transpose(z, [0, 2, 1, 3, 4]),  // [nd a h, b, w, e, o]
      (z) => tf.reshape(z, [n, 2 * d, 2 * h, 2 * wd, o]));
    return {
      value,
      grad: (dy, [x, w]) => {
        const dz = then(tf.reshape(dy, [n * d * 2 * h, 2, wd, 2, o]),  // space-to-depth: back to [P, o·8]
          (t) => tf.transpose(t, [0, 2, 1, 3, 4]),  // [nd a h, w, b, e, o]
          (t) => tf.reshape(t, [n * d, 2, h, wd, 4, o]),
          (t) => tf.transpose(t, [0, 2, 3, 5, 1, 4]),  // [nd, h, w, o, a, (b e)]
          (t) => tf.reshape(t, [-1, o * 8]));
        const dx = tf.tidy(() => tf.reshape(tf.matMul(dz, w as TF.Tensor2D, false, true), x.shape));
        const dw = tf.matMul(flat(x), dz, true, false);
        return b ? [dx, dw, tf.tidy(() => tf.sum(tf.reshape(channelSum(dz), [o, 8]), 1))] : [dx, dw];
      },
    };
  });

  /** Nearest upsampling by `f` on each of d, h, w, as rank-3 tiles (TF.js
   * has no gradient for a higher-rank tile), keeping nothing: its gradient
   * sums each block of f³ back. */
  function upsampleOp(f: number) {
    const view = (s: number[], ax: number) => [s.slice(0, ax + 1).reduce((a, b) => a * b, 1), s.slice(ax + 1).reduce((a, b) => a * b, 1)];
    return custom(([x]) => {
      const shapes: number[][] = [];
      const value = SPATIAL.reduce((y: TF.Tensor, ax) => {
        const s = y.shape, [outer, inner] = view(s, ax), out = s.slice();
        shapes.push(s);
        out[ax] *= f;
        const next = tf.tidy(() => tf.reshape(tf.tile(tf.reshape(y, [outer, 1, inner]), [1, f, 1]), out));
        if (y !== x) y.dispose();
        return next;
      }, x);
      return {
        value,
        grad: (dy) => [[...SPATIAL.entries()].reverse().reduce((g: TF.Tensor, [i, ax]) => {
          const s = shapes[i], [outer, inner] = view(s, ax);
          const next = tf.tidy(() => tf.reshape(tf.sum(tf.reshape(g, [outer, f, inner]), 1), s));
          if (g !== dy) g.dispose();
          return next;
        }, dy)],
      };
    });
  }

  // ---- the units: the graph's ops in order, some fused
  const readers = new Map<string, number[]>();
  graph.ops.forEach((op, i) => opInputs(op).forEach((x) => readers.set(x, [...(readers.get(x) ?? []), i])));
  /** The one op that reads `t`, if one does and nothing else (the graph's output is read by the caller). */
  const onlyReader = (t: string): number | null => {
    const r = readers.get(t) ?? [];
    return r.length === 1 && t !== graph.output ? r[0] : null;
  };
  const scaleSafe = (name: string) => weights[name].every((v) => Math.abs(v) >= MIN_FUSED_SCALE);
  const fusedAt = new Map<number, Unit>(), absorbed = new Set<number>(), chained = new Set<number>();
  for (const [i, op] of graph.ops.entries()) {
    if (op.op === "conv") {
      const j = onlyReader(op.y), bn = j === null ? null : graph.ops[j];
      if (j === null || bn?.op !== "bn") continue;
      const k = onlyReader(bn.y);
      if (k === null || graph.ops[k].op !== "relu" || trains.has(bn.mean) || trains.has(bn.var) || !scaleSafe(bn.scale)) continue;
      const rstd = keep(tf.tensor1d(weights[bn.var].map((v) => 1 / Math.sqrt(v + bn.eps))), true);
      const fn = convBnReluOp(op.stride as Triple, !!op.b, P[bn.mean], P[bn.var], rstd, bn.eps, op.x !== graph.input, trains.has(op.w));
      const params = [op.w, ...(op.b ? [op.b] : []), bn.scale, bn.bias];
      fusedAt.set(k, { ops: [i, j, k], inputs: [op.x], output: graph.ops[k].y, params: [...params, bn.mean, bn.var], relu: true, run: ([x]) => fn(x, ...params.map((n) => P[n])) });
      absorbed.add(i).add(j);
    } else if (op.op === "crop" && !chained.has(i)) {
      // A chain of crops (setup_53's skips are cropped on d, then h, then w), each read only by the next: one op.
      const ops = [i], margins: [number, number][] = [[0, 0], [0, 0], [0, 0]];
      for (let j: number | null = i; j !== null && graph.ops[j].op === "crop"; j = onlyReader(graph.ops[j].y)) {
        const c = graph.ops[j] as Extract<TrainOp, { op: "crop" }>, m = margins["dhw".indexOf(c.axis)];
        m[0] += c.start;
        m[1] += c.end_margin;
        if (j !== i) ops.push(j);
        chained.add(j);
      }
      const fn = cropOp(margins), last = ops[ops.length - 1];
      fusedAt.set(last, { ops, inputs: [op.x], output: graph.ops[last].y, params: [], relu: false, run: ([x]) => fn(x) });
      ops.slice(0, -1).forEach((o) => absorbed.add(o));
    } else if (op.op === "inorm") {
      const ops = [i];
      let skip: string | null = null, relu = false;
      const j = onlyReader(op.y), next = j === null ? null : graph.ops[j];
      if (j !== null && next?.op === "add" && next.x !== next.x2) {
        skip = next.x === op.y ? next.x2 : next.x;
        ops.push(j);
        const k = onlyReader(next.y);
        if (k !== null && graph.ops[k].op === "relu") ops.push(k), relu = true;
      } else if (j !== null && next?.op === "relu") {
        ops.push(j);
        relu = true;
      }
      const fn = normActOp(op.eps, skip !== null, relu), last = ops[ops.length - 1];
      fusedAt.set(last, {
        ops, inputs: skip ? [op.x, skip] : [op.x], output: graph.ops[last].y, params: [op.scale, op.bias], relu,
        run: ([z, s]) => fn(z, P[op.scale], P[op.bias], ...(s ? [s] : [])),
      });
      ops.slice(0, -1).forEach((o) => absorbed.add(o));
    }
  }
  const units: Unit[] = [];
  for (const [i, op] of graph.ops.entries()) {
    if (absorbed.has(i)) continue;
    units.push(fusedAt.get(i) ?? single(i, op));
  }

  /** The unit of one op, unfused. */
  function single(i: number, op: TrainOp): Unit {
    const unit = (params: string[], run: Unit["run"]): Unit => ({ ops: [i], inputs: opInputs(op), output: op.y, params, relu: op.op === "relu", run });
    switch (op.op) {
      case "conv": {
        const fn = convOp(op.stride as Triple, op.x !== graph.input, trains.has(op.w)), params = op.b ? [op.w, op.b] : [op.w];
        return unit(params, ([x]) => fn(x, ...params.map((n) => P[n])));
      }
      case "conv_t": {
        if (op.stride.some((s) => s !== 2) || graph.params[op.w].shape.slice(2).some((k) => k !== 2)) throw new Error(`${op.w}: a transposed convolution of kernel and stride 2 only`);
        const fn = convTOp, params = op.b ? [op.w, op.b] : [op.w];
        return unit(params, ([x]) => fn(x, ...params.map((n) => P[n])));
      }
      case "maxpool":
        return unit([], ([x]) => tf.maxPool3d(x as TF.Tensor5D, op.k as Triple, op.stride as Triple, "valid"));
      case "bn": {  // not after a conv it fuses with
        const fn = bnOp(op.eps), params = [op.mean, op.var, op.scale, op.bias];
        return unit(params, ([x]) => fn(x, ...params.map((n) => P[n])));
      }
      case "relu":
        return unit([], ([x]) => tf.relu(x));
      case "add":
        return unit([], ([a, b]) => addOp(a, b));
      case "concat":
        return unit([], (xs) => concatOp(...xs));
      case "upsample": {
        const fn = upsampleOp(op.factor);
        return unit([], ([x]) => fn(x));
      }
      case "crop":  // fused above, alone if nothing follows it
      case "inorm":
        throw new Error("unreachable");
    }
  }

  // ---- the segments: cut after every unit ending in a ReLU, and after
  // every unit whose output a later segment reads (a skip connection's
  // source, the residual branch of a ResUNet's block), to a fixpoint: then
  // a segment reads only checkpoints, and hands on only its last output.
  const producer = new Map<string, number>(units.map((u, i) => [u.output, i]));
  const cuts = new Set(units.flatMap((u, i) => (u.relu && i < units.length - 1 ? [i] : [])));
  for (let grew = true; grew;) {
    grew = false;
    const sorted = [...cuts].sort((a, b) => a - b);
    for (const [j, u] of units.entries()) {
      for (const x of u.inputs) {
        const p = producer.get(x);
        if (p === undefined || cuts.has(p) || !sorted.some((c) => p <= c && c < j)) continue;
        cuts.add(p);
        grew = true;
      }
    }
  }
  const bounds = [...[...cuts].sort((a, b) => a - b).map((c) => c + 1), units.length];
  const segments: Segment[] = bounds.map((to, k) => {
    const from = k ? bounds[k - 1] : 0, made = new Set<string>(), inputs = new Set<string>(), params = new Set<string>();
    for (const u of units.slice(from, to)) {
      u.inputs.forEach((x) => made.has(x) || inputs.add(x));
      made.add(u.output);
      u.params.forEach((n) => params.add(n));
    }
    return { from, to, inputs: [...inputs], output: units[to - 1].output, params: [...params] };
  });

  /** Units [from, to) on `given` (by name; never freed), each in a tidy, each
   * intermediate freed after its last reader among them; the tensors named
   * in `kept`. Freeing early is safe under a gradient tape: what a gradient
   * needs it has kept (TF.js clones it), and the tape reads only the shapes
   * and ids of the rest. `after` runs after each unit (a trim). */
  function runUnits(from: number, to: number, given: Map<string, TF.Tensor>, kept: ReadonlySet<string>, after?: (i: number) => void): Map<string, TF.Tensor> {
    const T = new Map(given), left = new Map<string, number>();
    for (let i = from; i < to; i++) units[i].inputs.forEach((x) => left.set(x, (left.get(x) ?? 0) + 1));
    const release = (x: string) => {
      if (given.has(x) || kept.has(x) || left.get(x)) return;
      T.get(x)?.dispose();
      T.delete(x);
    };
    try {
      for (let i = from; i < to; i++) {
        const u = units[i];
        const xs = u.inputs.map((x) => T.get(x) ?? fail(`unit ${i} reads ${x}, which is not computed`));
        T.set(u.output, tf.tidy(() => u.run(xs)));
        for (const x of u.inputs) left.set(x, left.get(x)! - 1), release(x);
        release(u.output);  // read by nothing here, nor kept
        after?.(i);
      }
    } catch (e) {
      for (const [x, t] of T) if (!given.has(x)) t.dispose();
      throw e;
    }
    return new Map([...kept].map((x) => [x, T.get(x) ?? fail(`${x} was not computed`)]));
  }

  // ---- the optimizer: TF.js's Adam, its state beside the params
  const moments = new Map<string, { m: TF.Tensor; v: TF.Tensor }>();
  let accBeta1 = Math.fround(BETA1), accBeta2 = Math.fround(BETA2);  // βᵗ, in float32 as TF.js keeps them

  /** The params and moments after one Adam step on `grads` at `learningRate`, as new tensors (the old ones untouched). */
  function adam(grads: Map<string, TF.Tensor>, learningRate: number): Map<string, { w: TF.Tensor; m: TF.Tensor; v: TF.Tensor }> {
    const c1 = Math.fround(1 - accBeta1), c2 = Math.fround(1 - accBeta2);
    const out = new Map<string, { w: TF.Tensor; m: TF.Tensor; v: TF.Tensor }>();
    for (const [name, g] of grads) out.set(name, tf.tidy(() => adamUpdate(tf, P[name], g, moments.get(name), learningRate, c1, c2)));
    return out;
  }

  // ---- the GPU
  let lost: string | null = null;
  void gpu?.device.lost.then((info) => {
    lost = info.message || info.reason || "lost";
  });
  const isCheckpoint = new Set(segments.slice(0, -1).map((s) => s.to - 1));

  /** A block's checks: one single-channel cube, of a side the graph takes, on a GPU still there. */
  const checkBlock = (x: TF.Tensor5D) => {
    if (lost) throw new Error(`the GPU device was lost (${lost}); reload the page`);
    const [n, d, h, w, c] = x.shape;
    if (n !== 1 || c !== 1 || d !== h || d !== w) throw new Error(`buildNet3d takes one single-channel cube at a time, not [${x.shape}]`);
    if (outputSide(graph, d) === null) throw new Error(`the network cannot take a block of ${d}³: its convolutions and pools take sides of ${describeSides(tileSides(graph, d + 64))}`);
  };

  /** `run` with TF.js submitting its commands to the GPU after every op
   * rather than every 15 (WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE): it holds a
   * freed buffer until the commands that use it are submitted, and with
   * maps of hundreds of MB, 15 ops' worth of them was half the peak of a
   * step (2.3 GB rather than 1.5 at setup_53's worst segment at 124³). No
   * slower: the ops are large. */
  function submittingEachOp<T>(run: () => T): T {
    if (!gpu) return run();
    const flag = "WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE", was = tf.env().getNumber(flag);
    tf.env().set(flag, 1);
    const restore = () => tf.env().set(flag, was);
    let result: T;
    try {
      result = run();
    } catch (e) {
      restore();
      throw e;
    }
    if (result instanceof Promise) return result.finally(restore) as T;
    restore();
    return result;
  }

  function forward(x: TF.Tensor5D): TF.Tensor5D {
    checkBlock(x);
    return submittingEachOp(() => runUnits(0, units.length, new Map([[graph.input, x]]), new Set([graph.output]), (i) => {
      // At each checkpoint, the pool emptied, not waited for (forward
      // returns a tensor): the GPU frees the buffers once it gets there.
      if (gpu && isCheckpoint.has(i)) dropPool(gpu);
    }).get(graph.output) as TF.Tensor5D);
  }

  /** The segments' outputs, segment by segment, the GPU settled and the
   * pool emptied after each: the step's forward pass (`upTo` the last
   * segment, every output kept), or forwardAsync's (all of them, each
   * output freed after its last reader). */
  async function runSegments(x: TF.Tensor5D, upTo: number, keepAll: boolean): Promise<Map<string, TF.Tensor>> {
    const C = new Map<string, TF.Tensor>([[graph.input, x]]);
    const lastRead = new Map<string, number>();
    segments.forEach((s, k) => s.inputs.forEach((n) => lastRead.set(n, k)));
    try {
      for (const [k, s] of segments.slice(0, upTo).entries()) {
        C.set(s.output, runUnits(s.from, s.to, new Map(s.inputs.map((n) => [n, C.get(n)!])), new Set([s.output])).get(s.output)!);
        if (!keepAll) for (const n of s.inputs) if (n !== graph.input && lastRead.get(n) === k) C.get(n)!.dispose(), C.delete(n);
        if (gpu) await settle(gpu, true);
      }
    } catch (e) {
      for (const [n, t] of C) if (n !== graph.input) t.dispose();
      throw e;
    }
    return C;
  }

  async function forwardAsync(x: TF.Tensor5D): Promise<TF.Tensor5D> {
    checkBlock(x);
    // At 172³, setup_53's forward pass held 5.9 GB of GPU memory at its
    // peak with forward, 2.4 GB with this, in 0.95 s rather than 0.72;
    // setup_54's 9.9 and 4.6 GB, in 2.1 s rather than 1.7.
    // Unlike forward's, its errors can be waited for: a pass that ran out
    // of memory throws GpuMemoryError rather than hand on garbage.
    const { value: C, error } = await guarded(gpu, () => submittingEachOp(() => runSegments(x, segments.length, false)));
    if (error || !C) {
      for (const [n, t] of C ?? []) if (n !== graph.input) t.dispose();
      if (gpu) await settle(gpu, true);
      throw error ?? new Error("the forward pass computed nothing");
    }
    return C.get(graph.output) as TF.Tensor5D;
  }

  function step(x: TF.Tensor5D, loss: Loss3d, learningRate: number): Promise<StepTerms> {
    if (!trains.size) return Promise.reject(new Error("this network trains no param: it runs forward only"));
    if (!(learningRate > 0)) return Promise.reject(new Error(`the learning rate ${learningRate} is not a number above 0`));
    try {
      checkBlock(x);
    } catch (e) {
      return Promise.reject(e);
    }
    return submittingEachOp(() => runStep(x, loss, learningRate));
  }

  async function runStep(x: TF.Tensor5D, loss: Loss3d, learningRate: number): Promise<StepTerms> {
    if (gpu) await settle(gpu, true);  // the teacher's forward pass, and the last step's leftovers
    const C = new Map<string, TF.Tensor>([[graph.input, x]]);  // checkpoints by name
    const G = new Map<string, TF.Tensor>();  // their gradients
    const grads = new Map<string, TF.Tensor>();  // the params'
    let terms: { total: TF.Tensor; supervised: TF.Tensor; distillation: TF.Tensor | null; otherChannels: TF.Tensor | null } | null = null;
    const add = (into: Map<string, TF.Tensor>, name: string, g: TF.Tensor | undefined) => {
      if (!g) return;
      const had = into.get(name);
      if (!had) return void into.set(name, g);
      into.set(name, tf.add(had, g));
      had.dispose();
      g.dispose();
    };
    const given = (s: Segment) => new Map(s.inputs.map((n) => [n, C.get(n)!]));
    const cleanUp = () => {
      for (const [n, t] of C) if (n !== graph.input) t.dispose();
      [...G.values(), ...grads.values()].forEach((t) => t.dispose());
      if (terms) [terms.total, terms.supervised, terms.distillation, terms.otherChannels].forEach((t) => t?.dispose());
      C.clear(); G.clear(); grads.clear();
    };
    const computed = await guarded(gpu, async () => {
      // Forward: every segment but the last, keeping their outputs.
      for (const [n, t] of await runSegments(x, segments.length - 1, true)) C.set(n, t);
      // Backward, from the loss: each segment run again under a tape, from
      // its checkpoints, with the gradient of its output from the segments
      // after it.
      for (let k = segments.length - 1; k >= 0; k--) {
        const s = segments[k], last = k === segments.length - 1;
        let dy: TF.Tensor | undefined;
        if (!last) {
          // Read by later segments only, all done: the run below makes it again.
          C.get(s.output)!.dispose();
          C.delete(s.output);
          dy = G.get(s.output);
          if (!dy) continue;  // nothing after it reached the loss
        }
        const ins = s.inputs.filter((n) => n !== graph.input), ps = s.params.filter((n) => trains.has(n));
        if (!ins.length && !ps.length) {
          dy?.dispose();
          G.delete(s.output);
          continue;
        }
        const { value, grads: gs } = tf.engine().gradients(() => {
          const out = runUnits(s.from, s.to, given(s), new Set([s.output])).get(s.output)!;
          if (!last) return out;
          const t = loss(out as TF.Tensor5D);
          terms = { total: t.total, supervised: tf.keep(t.supervised), distillation: t.distillation && tf.keep(t.distillation), otherChannels: t.otherChannels && tf.keep(t.otherChannels) };
          return t.total;
        }, [...ins.map((n) => C.get(n)!), ...ps.map((n) => P[n])], dy, true);
        if (last) terms!.total = value;
        else {
          value.dispose();
          dy!.dispose();
          G.delete(s.output);
        }
        ins.forEach((n, j) => add(G, n, gs[j]));
        ps.forEach((n, j) => add(grads, n, gs[ins.length + j]));
        if (gpu) await settle(gpu, true);
      }
      const t = terms!;
      const [total, supervised, distillation, otherChannels] = await Promise.all([t.total.data(), t.supervised.data(), t.distillation?.data(), t.otherChannels?.data()]);
      return { total: total[0], supervised: supervised[0], distillation: distillation ? distillation[0] : null, otherChannels: otherChannels ? otherChannels[0] : null };
    });
    if (computed.error || !computed.value) {
      cleanUp();
      if (gpu) await settle(gpu, true);
      throw computed.error ?? new Error("the step computed no loss");
    }
    // The update, beside the weights: kept only if the GPU made it whole.
    const updated = await guarded(gpu, async () => {
      const next = adam(grads, learningRate);
      if (gpu) await settle(gpu, false);
      return next;
    });
    if (updated.error || !updated.value) {
      for (const n of updated.value?.values() ?? []) [n.w, n.m, n.v].forEach((t) => t.dispose());
      cleanUp();
      if (gpu) await settle(gpu, true);
      throw updated.error ?? new Error("the update computed nothing");
    }
    for (const [name, n] of updated.value) {
      P[name].dispose();
      const old = moments.get(name);
      old?.m.dispose();
      old?.v.dispose();
      P[name] = keep(n.w);
      moments.set(name, { m: keep(n.m), v: keep(n.v) });
    }
    accBeta1 = Math.fround(accBeta1 * Math.fround(BETA1));
    accBeta2 = Math.fround(accBeta2 * Math.fround(BETA2));
    cleanUp();
    if (gpu) await settle(gpu, true);  // the pool's gigabytes back to the GPU between steps
    return computed.value;
  }

  function exportWeights(): Record<string, { data: Float32Array; shape: number[] }> {
    const out: Record<string, { data: Float32Array; shape: number[] }> = {};
    for (const name of trains) {
      const shape = graph.params[name].shape, t = P[name];
      // A transposed conv's [I, O·8] is its [I,O,2,2,2] in order already.
      const v = kind[name] === "conv" ? tf.transpose(t, [4, 3, 0, 1, 2]) : t;
      out[name] = { data: Float32Array.from(v.dataSync() as Float32Array), shape: shape.slice() };
      if (v !== t) v.dispose();
    }
    return out;
  }

  return {
    step,
    forward,
    forwardAsync,
    exportWeights,
    dispose: () => {
      [...Object.values(P), ...constants, ...[...moments.values()].flatMap((s) => [s.m, s.v])].forEach((t) => t.dispose());
      moments.clear();
    },
  };
}

function fail(problem: string): never {
  throw new Error(`buildNet3d: ${problem}`);
}

// ---- WebGPU: TF.js 4.22.0's backend internals (package.json pins the version)

/** What of TF.js's WebGPU backend this file reaches into. */
interface WebGpuInternals {
  device: GPUDevice;
  bufferManager: { freeBuffers: Map<string, GPUBuffer[]>; numFreeBuffers: number; numBytesAllocated: number };
  ensureCommandEncoderReady(): void;
  endComputePassEncoder(): void;
  submitQueue(): void;
}

/** TF.js's WebGPU backend, when it is the one running, else null (the CPU's, in the tests). */
function webgpuOf(tf: Tf): WebGpuInternals | null {
  if (tf.getBackend() !== "webgpu") return null;
  const be = tf.backend() as unknown as Partial<WebGpuInternals>;
  return be.device && be.bufferManager && typeof be.submitQueue === "function" ? (be as WebGpuInternals) : null;
}

/** TF.js's queued commands sent to the GPU. */
function submit(gpu: WebGpuInternals): void {
  gpu.ensureCommandEncoderReady();
  gpu.endComputePassEncoder();
  gpu.submitQueue();
}

/** TF.js's pool of free buffers destroyed. Its map's keys stay: TF.js
 * releases nothing into a pool whose map is empty. */
function dropPool(gpu: WebGpuInternals): void {
  const bm = gpu.bufferManager;
  bm.freeBuffers.forEach((list) => {
    for (const b of list) {
      bm.numBytesAllocated -= b.size;
      b.destroy();
    }
    list.length = 0;
  });
  bm.numFreeBuffers = 0;
}

const ticks = new WeakMap<GPUDevice, GPUBuffer>();

/** The GPU's queue run dry; with `trim`, the pool emptied and its memory
 * given back. Dawn frees a destroyed buffer once a submission made after
 * the destruction completes: hence two more small ones, each waited for. */
async function settle(gpu: WebGpuInternals, trim: boolean): Promise<void> {
  submit(gpu);
  const queue = gpu.device.queue;
  await queue.onSubmittedWorkDone();
  if (!trim) return;
  dropPool(gpu);
  let tick = ticks.get(gpu.device);
  if (!tick) ticks.set(gpu.device, (tick = gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST })));
  for (let k = 0; k < 2; k++) {
    queue.writeBuffer(tick, 0, new Uint8Array(4));
    queue.submit([gpu.device.createCommandEncoder().finish()]);
    await queue.onSubmittedWorkDone();
  }
}

const SCOPES: GPUErrorFilter[] = ["out-of-memory", "validation", "internal"];

/** `run` inside WebGPU error scopes (on the CPU, as it is): its value if it
 * returned one, and what went wrong, or null. A GPU error outranks a
 * thrown one, which it likely caused; out of memory outranks the rest. */
async function guarded<T>(gpu: WebGpuInternals | null, run: () => Promise<T>): Promise<{ value: T | null; error: Error | null }> {
  const asError = (e: unknown) => (e instanceof Error ? e : new Error(String(e)));
  if (!gpu) {
    try {
      return { value: await run(), error: null };
    } catch (e) {
      return { value: null, error: asError(e) };
    }
  }
  for (const filter of SCOPES) gpu.device.pushErrorScope(filter);
  let value: T | null = null, thrown: unknown = null;
  try {
    value = await run();
    submit(gpu);  // so that the scopes see the last commands' errors
  } catch (e) {
    thrown = e;
  }
  const errors = new Map<GPUErrorFilter, GPUError>();
  for (const filter of [...SCOPES].reverse()) {
    const e = await gpu.device.popErrorScope();
    if (e) errors.set(filter, e);
  }
  const oom = errors.get("out-of-memory"), other = errors.get("validation") ?? errors.get("internal");
  const error = oom ? new GpuMemoryError(`the GPU ran out of memory: ${oom.message}`)
    : other ? new Error(`the GPU refused the step: ${other.message}`)
    : thrown === null ? null : asError(thrown);
  return { value, error };
}
