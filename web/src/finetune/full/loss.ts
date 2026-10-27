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
// - For a run of one channel of several (selectChannel), both terms on that
//   channel alone, as the server slices the prediction and the teacher to
//   it, plus OTHER_CHANNELS_WEIGHT × the hold on the others: the mean
//   squared distance of their logits from the starting network's over every
//   voxel of the tile. The network is shared, so a step for one channel
//   moves the others too; the hold keeps them where they were, as the
//   server's trainer/distillation.py holds them.
//
// As on the server (lora_trainer.py), the starting network sees the tile
// the student sees, after augmentation; its logits are not turned. A 3D
// model's step is its runtime's (net3d.ts), which takes the same terms
// (lossTerms) on its NDHWC logits.
import type * as TF from "@tensorflow/tfjs-core";

import type { FinetuneOptions } from "../types";
import type { Tf, TrainNet } from "./cellmap-tfjs-train";

export type FullLossOptions = Pick<FinetuneOptions, "lossType" | "margin" | "labelSmoothing" | "balanceClasses" | "distillationLambda" | "distillationScope"> & Partial<Pick<FinetuneOptions, "selectChannel">>;

/** The weight of the hold on the channels a run of one channel does not
 * train (lossTerms): the server's OTHER_CHANNELS_WEIGHT
 * (finetune/trainer/distillation.py), fixed rather than an option, since
 * those channels are already good and are to stay as they were, not be
 * traded against the paint as λ trades the trained channel's drift. */
export const OTHER_CHANNELS_WEIGHT = 1.0;

/** The loss's terms on a batch, each a scalar the caller disposes:
 * `total`, the one minimized, `supervised` and `distillation` (null when λ
 * is 0 or there is no teacher), and `otherChannels`, the hold on the
 * channels a run of one channel does not train (null without one). */
export interface LossTerms {
  total: TF.Scalar;
  supervised: TF.Scalar;
  distillation: TF.Scalar | null;
  otherChannels: TF.Scalar | null;
}

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

/** Channel `c` of `t` (channels last), keeping the axis: [..., 1]. */
function channelOf(tf: Tf, t: TF.Tensor, c: number): TF.Tensor {
  const last = t.rank - 1;
  return tf.slice(t, t.shape.map((_, a) => (a === last ? c : 0)), t.shape.map((_, a) => (a === last ? 1 : -1)));
}

/** Whether a run of a model of `channels` output channels trains one of
 * them and holds the others (lossTerms): a channel picked, and more than one. */
export function holdsOtherChannels(o: Pick<FullLossOptions, "selectChannel">, channels: number): boolean {
  return (o.selectChannel ?? null) !== null && channels > 1;
}

/** The loss's terms on the logits `z` (any rank, channels last): the
 * supervised term against `y` over `m`, and, with `teacher`'s logits and
 * λ > 0, the distillation term, both on every channel or, with a channel
 * picked (`o.selectChannel`), on that one, `y` and `m` then its alone; and
 * with a channel picked of several and a teacher, the hold on the others
 * (`otherChannels`): their distance from the teacher's logits, weight 1
 * on every voxel (distillationLoss over all of them, as the server's
 * distillation_loss with scope "all"). `total` is the supervised term plus
 * λ × the distillation term plus OTHER_CHANNELS_WEIGHT × the hold, a
 * tensor of its own either way (the 3D runtime and the optimizer may
 * dispose it apart from the terms). */
export function lossTerms(tf: Tf, z: TF.Tensor, y: TF.Tensor, m: TF.Tensor, teacher: TF.Tensor | null, o: FullLossOptions): LossTerms {
  const c = o.selectChannel ?? null, channels = z.shape[z.rank - 1];
  const own = c === null ? z : channelOf(tf, z, c);
  const supervised = supervisedLoss(tf, own, y, m, o);
  const distillation = teacher && o.distillationLambda > 0
    ? distillationLoss(tf, own, c === null ? teacher : channelOf(tf, teacher, c), distillationWeights(tf, m, o.distillationScope))
    : null;
  // 1 on every channel but the one picked, on every voxel.
  const others = Array.from({ length: channels }, (_, k) => (k === c ? 0 : 1));
  const otherChannels = teacher && holdsOtherChannels(o, channels)
    ? distillationLoss(tf, z, teacher, tf.mul(tf.onesLike(z), tf.tensor1d(others)))
    : null;
  let total: TF.Tensor = supervised;
  if (distillation) total = tf.add(total, tf.mul(distillation, o.distillationLambda));
  if (otherChannels) total = tf.add(total, tf.mul(otherChannels, OTHER_CHANNELS_WEIGHT));
  return { total: (total === supervised ? tf.clone(supervised) : total) as TF.Scalar, supervised, distillation, otherChannels };
}

/** One Adam step of `net` on the batch: lossTerms on its logits. The
 * terms as they were before the step; the caller disposes them. */
export function fullTrainStep(tf: Tf, net: TrainNet, opt: TF.Optimizer, x: TF.Tensor4D, y: TF.Tensor4D, m: TF.Tensor4D, teacher: TF.Tensor4D | null, o: FullLossOptions): LossTerms {
  let supervised: TF.Scalar | null = null, distillation: TF.Scalar | null = null, otherChannels: TF.Scalar | null = null;
  const total = opt.minimize(() => {
    const terms = lossTerms(tf, net.forward(x), y, m, teacher, o);
    supervised = tf.keep(terms.supervised);
    distillation = terms.distillation && tf.keep(terms.distillation);
    otherChannels = terms.otherChannels && tf.keep(terms.otherChannels);
    return terms.total;
  }, true, net.vars);
  if (!total || !supervised) throw new Error("the optimizer returned no loss");
  return { total, supervised, distillation, otherChannels };
}
