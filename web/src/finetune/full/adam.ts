// Adam as TF.js's tf.train.adam computes it (its betas, its epsilon for
// float32, the same arithmetic, β₁ᵗ and β₂ᵗ kept in float32), with the
// learning rate given at each step. TF.js's own optimizer keeps the rate it
// was made with for its whole life: AdamOptimizer's learningRate is
// protected, nothing in TF.js 4.22 sets it, and it has no schedules, where
// every trainer eases the rate down over the run (../schedule.ts). So both
// runtimes step their own: the 2D one with createAdam (run-full.ts, through
// loss.ts fullTrainStep), on its network's variables; the 3D one in
// net3d.ts, which computes a step beside the weights and keeps it only when
// the GPU made it whole. They share the update (adamUpdate).
import type * as TF from "@tensorflow/tfjs-core";

import type { Tf } from "./cellmap-tfjs-train";

/** Adam's constants: TF.js's defaults (tf.train.adam), its epsilon the
 * backend's for float32. */
export const BETA1 = 0.9, BETA2 = 0.999, EPSILON = 1e-7;

/** One param's Adam step at `learningRate`: its value and moments after it,
 * new tensors, from its value `w`, its gradient `g` and its moments before
 * (none at its first step); `c1` and `c2` are the step's bias corrections,
 * 1 − β₁ᵗ and 1 − β₂ᵗ. Inside a tidy, which keeps what it returns. */
export function adamUpdate(tf: Tf, w: TF.Tensor, g: TF.Tensor, moments: { m: TF.Tensor; v: TF.Tensor } | undefined, learningRate: number, c1: number, c2: number): { w: TF.Tensor; m: TF.Tensor; v: TF.Tensor } {
  const m = tf.add(tf.mul(moments?.m ?? tf.zerosLike(w), BETA1), tf.mul(g, 1 - BETA1));
  const v = tf.add(tf.mul(moments?.v ?? tf.zerosLike(w), BETA2), tf.mul(tf.square(g), 1 - BETA2));
  const update = tf.mul(tf.div(tf.div(m, c1), tf.add(tf.sqrt(tf.div(v, c2)), EPSILON)), -learningRate);
  return { w: tf.add(update, w), m, v };
}

/** Adam over a network's variables, its rate given at each step. */
export interface Adam {
  /** One step of the variables `vars` down the gradient of `f`, at
   * `learningRate`; `f`'s value before it, which the caller disposes (as
   * TF.js's minimize(f, true, vars) returns it). */
  minimize(f: () => TF.Scalar, vars: TF.Variable[], learningRate: number): TF.Scalar;
  /** Free its moments. */
  dispose(): void;
}

/** An Adam with no step taken yet: TF.js's minimize and applyGradients, the
 * rate a step's own. */
export function createAdam(tf: Tf): Adam {
  const moments = new Map<string, { m: TF.Tensor; v: TF.Tensor }>();
  let accBeta1 = Math.fround(BETA1), accBeta2 = Math.fround(BETA2);  // βᵗ, in float32 as TF.js keeps them
  return {
    minimize(f, vars, learningRate) {
      const { value, grads } = tf.variableGrads(f, vars);
      const c1 = Math.fround(1 - accBeta1), c2 = Math.fround(1 - accBeta2);
      for (const variable of vars) {
        const g = grads[variable.name];
        if (!g) continue;  // not on the loss's path, as TF.js skips it
        const before = moments.get(variable.name);
        const next = tf.tidy(() => adamUpdate(tf, variable, g, before, learningRate, c1, c2));
        variable.assign(next.w);
        next.w.dispose();
        before?.m.dispose();
        before?.v.dispose();
        moments.set(variable.name, { m: tf.keep(next.m), v: tf.keep(next.v) });  // kept from a caller's tidy, as TF.js's variables are
      }
      tf.dispose(Object.values(grads));
      accBeta1 = Math.fround(accBeta1 * Math.fround(BETA1));
      accBeta2 = Math.fround(accBeta2 * Math.fround(BETA2));
      return value;
    },
    dispose() {
      for (const { m, v } of moments.values()) tf.dispose([m, v]);
      moments.clear();
    },
  };
}
