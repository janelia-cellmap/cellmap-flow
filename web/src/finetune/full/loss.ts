// The whole network's loss in TF.js: the same terms the head trains with
// (head-trainer.ts batchLoss and distillationTerm, the port of the server's
// losses.py), so a whole-network run and a head run on the same paint are
// judged alike, as the server's runs are.
//
// - The supervised term on the painted voxels: BCE on the logits against
//   the smoothed target, the squared distance of the probability from it
//   (MSE), or the margin loss; reduced over the mask as one mean, or with
//   class balance as the mean of the foreground's and the background's
//   means, split by the target before smoothing.
// - λ × the distillation term: the mean squared distance of the logits from
//   the starting network's on the same (augmented) input, over the voxels
//   the scope picks: those the mask leaves out (unlabeled), or all.
//
// As on the server (lora_trainer.py), the starting network sees the tile
// the student sees, after augmentation; its logits are not turned.
import type * as TF from "@tensorflow/tfjs-core";

import type { FinetuneOptions } from "../types";
import type { Tf, TrainNet } from "./cellmap-tfjs-train";

export type FullLossOptions = Pick<FinetuneOptions, "lossType" | "margin" | "labelSmoothing" | "balanceClasses" | "distillationLambda" | "distillationScope">;

/** Σ l·w / max(Σ w, 1): masked_mean, whose clamp makes a set that weighs nothing cost nothing. */
function weightedMean(tf: Tf, l: TF.Tensor, w: TF.Tensor): TF.Scalar {
  return tf.div(tf.sum(tf.mul(l, w)), tf.maximum(tf.sum(w), 1)) as TF.Scalar;
}

/** The supervised term of logits `z` against the 0/1 targets `y` over the mask `m` (batchLoss). */
export function supervisedLoss(tf: Tf, z: TF.Tensor, y: TF.Tensor, m: TF.Tensor, o: FullLossOptions): TF.Scalar {
  const s = o.lossType === "margin" ? 0 : o.labelSmoothing;
  const target = s ? tf.add(tf.mul(y, 1 - s), s / 2) : y;
  let l: TF.Tensor;
  if (o.lossType === "mse") {
    l = tf.square(tf.sub(tf.sigmoid(z), target));
  } else if (o.lossType === "margin") {
    const p = tf.sigmoid(z);
    const short = tf.add(tf.mul(y, tf.relu(tf.sub(1 - o.margin, p))), tf.mul(tf.sub(1, y), tf.relu(tf.sub(p, o.margin))));
    l = tf.square(short);
  } else {
    // max(z, 0) − z·y' + log(1 + exp(−|z|)): BCE on logits that overflows for no z.
    l = tf.add(tf.sub(tf.relu(z), tf.mul(z, target)), tf.log1p(tf.exp(tf.neg(tf.abs(z)))));
  }
  if (!o.balanceClasses) return weightedMean(tf, l, m);
  const fg = tf.mul(m, y), bg = tf.mul(m, tf.sub(1, y));
  return tf.mul(tf.add(weightedMean(tf, l, fg), weightedMean(tf, l, bg)), 0.5) as TF.Scalar;
}

/** The distillation term (distillationTerm): the weighted mean of (z − z₀)², weights `w`. */
export function distillationLoss(tf: Tf, z: TF.Tensor, teacher: TF.Tensor, w: TF.Tensor): TF.Scalar {
  return weightedMean(tf, tf.square(tf.sub(z, teacher)), w);
}

/** The distillation weights of mask `m` for `scope`: 1 − m (unlabeled) or 1 everywhere (all). */
export function distillationWeights(tf: Tf, m: TF.Tensor, scope: FullLossOptions["distillationScope"]): TF.Tensor {
  return scope === "all" ? tf.onesLike(m) : tf.sub(1, m);
}

/** One Adam step of `net` on the batch: the supervised term, plus λ × the
 * distillation term towards `teacher`'s logits when there are some. Both
 * terms as they were before the step; the caller disposes them. */
export function fullTrainStep(tf: Tf, net: TrainNet, opt: TF.Optimizer, x: TF.Tensor4D, y: TF.Tensor4D, m: TF.Tensor4D, teacher: TF.Tensor4D | null, o: FullLossOptions): { total: TF.Scalar; supervised: TF.Scalar; distillation: TF.Scalar | null } {
  let supervised: TF.Scalar | null = null, distillation: TF.Scalar | null = null;
  const total = opt.minimize(() => {
    const z = net.forward(x);
    const sup = tf.keep(supervisedLoss(tf, z, y, m, o));
    supervised = sup;
    if (!teacher || !(o.distillationLambda > 0)) return tf.clone(sup);
    const dist = tf.keep(distillationLoss(tf, z, teacher, distillationWeights(tf, m, o.distillationScope)));
    distillation = dist;
    return tf.add(sup, tf.mul(dist, o.distillationLambda)) as TF.Scalar;
  }, true, net.vars);
  if (!total || !supervised) throw new Error("the optimizer returned no loss");
  return { total, supervised, distillation };
}
