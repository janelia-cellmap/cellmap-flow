// Whole-network augmentation and loss (src/finetune/augment.ts, full/loss.ts):
// the raw and the paint stay registered through a flip and a turn, targets
// made after the turn mean what they did, intensity stays in the raw's
// range, and the TF.js loss is the head's (head-trainer.ts) term for term.
import { describe, expect, it } from "vitest";

import { readTraining } from "../../src/backend/routes/finetune";
import { augmentIntensity, intensityRange, transformBlock } from "../../src/finetune/augment";
import { distillationLoss, distillationWeights, supervisedLoss } from "../../src/finetune/full/loss";
import { loadTf } from "../../src/finetune/full/tfjs";
import { tileTargets } from "../../src/finetune/full/tiles";
import { batchLoss, mulberry32 } from "../../src/finetune/head-trainer";
import { affinityOffsets, affinityTargets } from "../../src/finetune/targets";

describe("augmentation of a tile", () => {
  it("turns the raw and the paint alike, about their shared centre", () => {
    // A 6² raw around a 2² output (a halo of 2): a mark on the output's top left voxel.
    const side = 6, output = 2, halo = 2;
    const raw = new Float32Array(side * side), labels = new Uint8Array(output * output);
    raw[halo * side + halo] = 1;
    labels[0] = 1;
    for (const draw of [{ flips: [false, true, false], turns: 0 }, { flips: [false, false, true], turns: 1 }, { flips: [true, true, true], turns: 3 }] as const) {
      const d = { flips: [...draw.flips] as [boolean, boolean, boolean], turns: draw.turns, scale: 1 };
      const r = transformBlock(raw, [1, side, side], d), l = transformBlock(labels, [1, output, output], d);
      const at = r.indexOf(1), lt = l.indexOf(1);
      expect([Math.floor(at / side) - halo, (at % side) - halo]).toEqual([Math.floor(lt / output), lt % output]);
    }
  });

  it("makes affinities from the turned paint, so a boundary across x becomes one across y", () => {
    // Two objects side by side (labels 2 | 3): a boundary between columns 1 and 2.
    const n = 4, labels = new Uint8Array(n * n).map((_, i) => (i % n < 2 ? 2 : 3));
    const transform = affinityTargets(affinityOffsets(["mito_aff_z", "mito_aff_y", "mito_aff_x"]), 3);
    // The painted pairs across a boundary on channel c: masked, target 0.
    const cut = (t: ReturnType<typeof tileTargets>, c: number) => Array.from({ length: n * n }, (_, p) => t.mask[p * 3 + c] && !t.target[p * 3 + c]).filter(Boolean).length;
    const before = tileTargets(labels, n, transform);
    const after = tileTargets(transformBlock(labels, [1, n, n], { flips: [false, false, false], turns: 1, scale: 1 }), n, transform);
    expect(cut(before, 2)).toBeGreaterThan(0);
    expect(cut(before, 1)).toBe(0);
    expect([cut(after, 1), cut(after, 2)]).toEqual([cut(before, 2), 0]);
  });

  it("keeps brightness and noise inside the raw's range, and the padding at zero", () => {
    const raw = new Float32Array(1000).map((_, i) => (i * 7) % 256), inside = new Uint8Array(1000).fill(1);
    inside.fill(0, 900);
    const range = intensityRange([{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }], "uint8");
    expect(range).toEqual([0, 255]);
    expect(intensityRange([], "uint16")).toEqual([0, 65535]);
    expect(intensityRange([], "float32")).toBeNull();
    const out = augmentIntensity(raw, inside, 1.2, range, mulberry32(1));
    expect(Math.min(...out)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...out)).toBeLessThanOrEqual(255);
    expect(out.subarray(900).every((v) => v === 0)).toBe(true);
    // Brighter on average, by about the factor.
    const mean = (a: ArrayLike<number>) => Array.from(a).slice(0, 900).reduce((s, v) => s + v, 0) / 900;
    expect(mean(out) / mean(raw)).toBeGreaterThan(1.1);
  });
});

describe("the whole network's loss", () => {
  it("is the head's supervised term for every loss, balanced or not, smoothed or not", { timeout: 60_000 }, async () => {
    const { tf } = await loadTf();
    const random = mulberry32(3), n = 64;
    const z = Float64Array.from({ length: n }, () => 6 * random() - 3);
    const y = Float32Array.from({ length: n }, () => (random() < 0.3 ? 1 : 0));
    const m = Float32Array.from({ length: n }, () => (random() < 0.7 ? 1 : 0));
    for (const lossType of ["bce", "mse", "margin"] as const) for (const balanceClasses of [false, true]) for (const labelSmoothing of [0, 0.1]) {
      const o = { lossType, margin: 0.3, balanceClasses, labelSmoothing, distillationLambda: 0, distillationScope: "unlabeled" as const };
      const expected = batchLoss(z, y, m, o, new Float64Array(n));
      const got = tf.tidy(() => supervisedLoss(tf, tf.tensor1d(Float32Array.from(z)), tf.tensor1d(y), tf.tensor1d(m), o).dataSync()[0]);
      expect(got).toBeCloseTo(expected, 5);
    }
  });

  it("holds the logits to the starting network's where the scope says", { timeout: 60_000 }, async () => {
    const { tf } = await loadTf();
    const z = [1, 2, 3, 4], z0 = [1, 0, 3, 0], m = [1, 1, 0, 0];
    const term = (scope: "unlabeled" | "all") => tf.tidy(() => distillationLoss(tf, tf.tensor1d(z), tf.tensor1d(z0), distillationWeights(tf, tf.tensor1d(m), scope)).dataSync()[0]);
    expect(term("unlabeled")).toBeCloseTo(16 / 2);  // the two unpainted voxels: 0 and 16
    expect(term("all")).toBeCloseTo((4 + 16) / 4);
  });
});

describe("the whole network's training controls", () => {
  it("reads the head's loss fields and augmentation into the run's options and the job's params", () => {
    const read = readTraining({ finetune_mode: "full", num_epochs: 4, loss_type: "mse", label_smoothing: 0.1, balance_classes: false, distillation_lambda: 0.1, distillation_scope: "all", augment: true, batch_size: 2 });
    if ("error" in read) throw new Error(read.error);
    expect(read.options).toMatchObject({ mode: "full", lossType: "mse", labelSmoothing: 0.1, balanceClasses: false, distillationLambda: 0.1, distillationScope: "all", augment: true, batchPatches: 2 });
    expect(read.params).toMatchObject({ loss_type: "mse", label_smoothing: 0.1, balance_classes: false, distillation_lambda: 0.1, distillation_scope: "all", augment: true, finetune_mode: "full" });
    const head = readTraining({ augment: true });
    if ("error" in head) throw new Error(head.error);
    expect(head.options.augment).toBe(false);  // the head trains on features computed once
  });
});
