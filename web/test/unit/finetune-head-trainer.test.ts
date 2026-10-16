// The head trainer on data made here: gatherPixels follows the feature
// layout rule (channel on axis 1, the other axes the chunk's voxels in
// order) and takes the voxels any channel's mask covers; batchLoss is the
// server's masked, balanced, smoothed BCE-with-logits and its gradient is
// the loss's (finite differences); trainHead on separable data (a planted
// linear map plus noise) brings the loss down and recovers the labels,
// class balance changes what a 95/5 skew teaches, label smoothing moves the
// optimum, a cancel stops it between batches, the same seed trains the same
// head, and the epoch line is the one the dashboard's loss plot parses. The
// server's other per-voxel losses, MSE on probabilities and the margin
// loss, have the server's values and their own gradients; with a
// distillation set and λ the head is held to the teacher (the head it is
// given, whatever it starts from) by λ, the epoch line names the supervised
// and distillation terms whose sum it reports, and without one, or at λ 0,
// it trains exactly as before.
import { describe, expect, it } from "vitest";

import { batchLoss, concatPixels, type DistillationSet, gatherPixels, mulberry32, type PixelSet, trainHead, yieldEvery } from "../../src/finetune/head-trainer";
import { DEFAULT_OPTIONS, type FinetuneOptions, type Head, type ModelFeatures, type RunReporter, type Targets } from "../../src/finetune/types";

// The dashboard's parser (cellmap_flow/dashboard/static/js/dashboard/finetune/loss-plot.js EPOCH_LOSS).
const EPOCH_LOSS = /Epoch\s+(\d+)\/(\d+)\s*-\s*Loss:\s*([0-9]*\.?[0-9]+)/i;

function head(inChannels: number, outChannels: number, weight?: number[], bias?: number[]): Head {
  return {
    featureTensor: "features", weightName: "w", biasName: "b", inChannels, outChannels, dtype: "float32",
    weight: Float32Array.from(weight ?? new Array(inChannels * outChannels).fill(0)),
    bias: Float32Array.from(bias ?? new Array(outChannels).fill(0)),
  };
}

function reporter(cancelAfterLines = Infinity): RunReporter & { lines: string[] } {
  const lines: string[] = [];
  return { lines, log: (l) => lines.push(l), cancelled: () => lines.length >= cancelAfterLines };
}

/** A standard normal from a uniform generator (Box–Muller). */
function normal(random: () => number): () => number {
  return () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

/** `count` rows of `inChannels` normal features labelled by a planted
 * linear map with `noise` added to its logits (so some labels are wrong
 * even for the planted map); `positiveFraction` sets the planted bias so
 * that fraction of each channel's noise-free logits is positive. The
 * targets are the hard labels, masked everywhere. */
function separable(count: number, inChannels: number, outChannels: number, seed: number, noise = 0.5, positiveFraction = 0.5): PixelSet & { planted: { weight: Float64Array; bias: Float64Array } } {
  const random = mulberry32(seed), gauss = normal(random);
  const weight = Float64Array.from({ length: outChannels * inChannels }, gauss), bias = new Float64Array(outChannels);
  const x = Float32Array.from({ length: count * inChannels }, gauss);
  const y = new Float32Array(count * outChannels), m = new Float32Array(count * outChannels).fill(1);
  const z = new Float64Array(count * outChannels);
  for (let p = 0; p < count; p++) for (let c = 0; c < outChannels; c++) {
    let acc = 0;
    for (let i = 0; i < inChannels; i++) acc += weight[c * inChannels + i] * x[p * inChannels + i];
    z[p * outChannels + c] = acc;
  }
  for (let c = 0; c < outChannels; c++) {
    const sorted = Float64Array.from({ length: count }, (_, p) => z[p * outChannels + c]).sort();
    bias[c] = -sorted[Math.floor(count * (1 - positiveFraction))];
    for (let p = 0; p < count; p++) y[p * outChannels + c] = z[p * outChannels + c] + bias[c] + noise * gauss() > 0 ? 1 : 0;
  }
  return { x, y, m, count, inChannels, outChannels, planted: { weight, bias } };
}

/** The fraction of masked elements whose sign agrees with the label. */
function accuracy(pixels: PixelSet, weight: Float32Array, bias: Float32Array): number {
  let right = 0, total = 0;
  const { inChannels: cin, outChannels: cout } = pixels;
  for (let p = 0; p < pixels.count; p++) for (let c = 0; c < cout; c++) {
    if (pixels.m[p * cout + c] <= 0) continue;
    let z = bias[c];
    for (let i = 0; i < cin; i++) z += weight[c * cin + i] * pixels.x[p * cin + i];
    total++;
    if ((z > 0) === (pixels.y[p * cout + c] > 0.5)) right++;
  }
  return right / total;
}

/** The mean loss over every row at once, with batchLoss's reduction. */
function fullLoss(pixels: PixelSet, weight: Float32Array, bias: Float32Array, options: Pick<FinetuneOptions, "balanceClasses" | "labelSmoothing">): number {
  const { inChannels: cin, outChannels: cout } = pixels;
  const z = new Float64Array(pixels.count * cout);
  for (let p = 0; p < pixels.count; p++) for (let c = 0; c < cout; c++) {
    let acc = bias[c];
    for (let i = 0; i < cin; i++) acc += weight[c * cin + i] * pixels.x[p * cin + i];
    z[p * cout + c] = acc;
  }
  return batchLoss(z, pixels.y, pixels.m, options, new Float64Array(z.length));
}

const options = (over: Partial<FinetuneOptions> = {}): FinetuneOptions => ({ ...DEFAULT_OPTIONS, epochs: 30, learningRate: 0.05, batchPixels: 256, balanceClasses: false, ...over });

describe("gatherPixels", () => {
  // A chunk of 2×1×3 voxels whose features come as [2, C, 3]: z as the batch
  // axis, as the 3D exports run their final Conv; logits as [1, 2, 2, 1, 3].
  const chunk = [2, 1, 3], n = 6, cin = 3, cout = 2;
  const features: ModelFeatures = {
    // features[(batch * cin + c) * 3 + s] = 100 * voxel + c, voxel = batch * 3 + s
    features: Float32Array.from({ length: 2 * cin * 3 }, (_, i) => { const batch = Math.floor(i / (cin * 3)), c = Math.floor(i / 3) % cin, s = i % 3; return 100 * (batch * 3 + s) + c; }),
    featuresDims: [2, cin, 3],
    logits: new Float32Array(cout * n),
    logitsDims: [1, cout, 2, 1, 3],
  };
  const targets: Targets = {
    channels: cout,
    target: Float32Array.from([1, 0, 0, 1, 0, 0, /* channel 1 */ 0, 0, 1, 0, 0, 0]),
    mask: Float32Array.from([1, 1, 0, 0, 0, 0, /* channel 1 */ 0, 0, 1, 0, 0, 1]),
  };

  it("takes the voxels any channel masks, with their features by the layout rule and their targets per channel", () => {
    const set = gatherPixels(features, targets, head(cin, cout), chunk);
    expect(set.count).toBe(4);  // voxels 0, 1 (channel 0), 2 and 5 (channel 1)
    expect(Array.from(set.x)).toEqual([0, 1, 2, 100, 101, 102, 200, 201, 202, 500, 501, 502]);
    expect(Array.from(set.y)).toEqual([1, 0, 0, 0, 0, 1, 0, 0]);
    expect(Array.from(set.m)).toEqual([1, 0, 1, 0, 0, 1, 0, 1]);
  });

  it("refuses features or targets that do not fit the chunk", () => {
    expect(() => gatherPixels({ ...features, featuresDims: [1, cin, 5] }, targets, head(cin, cout), chunk)).toThrow(/holds 5 voxels, the chunk 2×1×3 has 6/);
    expect(() => gatherPixels(features, targets, head(cin, cout), [2, 1, 2])).toThrow(/targets hold 12 values for 2 channels × 4 voxels/);
    expect(() => gatherPixels({ ...features, featuresDims: [1, 6, 3] }, targets, head(cin, cout), chunk)).toThrow(/channel axis \(axis 1\) should have the head's 3 channels/);
    expect(() => gatherPixels(features, { ...targets, channels: 3 }, head(cin, cout), chunk)).toThrow(/3 channels where the head writes 2/);
    expect(() => gatherPixels({ ...features, logitsDims: [1, cout, 5] }, targets, head(cin, cout), chunk)).toThrow(/logits 1×2×5 hold 5 voxels/);
  });

  it("concatenates sets in order", () => {
    const set = gatherPixels(features, targets, head(cin, cout), chunk);
    const both = concatPixels([set, set], head(cin, cout));
    expect(both.count).toBe(8);
    expect(Array.from(both.x.subarray(12))).toEqual(Array.from(set.x));
    expect(concatPixels([], head(cin, cout)).count).toBe(0);
  });
});

describe("batchLoss", () => {
  const z = Float64Array.from([2, -1, 0.5, -3, 4, 0]), y = Float32Array.from([1, 0, 0, 1, 1, 0]), m = Float32Array.from([1, 1, 1, 0, 1, 1]);
  const bce = (zi: number, t: number) => Math.max(zi, 0) - zi * t + Math.log1p(Math.exp(-Math.abs(zi)));

  it("is the masked mean of BCE-with-logits, every channel together", () => {
    const expected = [0, 1, 2, 4, 5].reduce((s, i) => s + bce(z[i], y[i]), 0) / 5;
    expect(batchLoss(z, y, m, { balanceClasses: false, labelSmoothing: 0 }, new Float64Array(6))).toBeCloseTo(expected, 12);
    expect(bce(50, 1)).toBeCloseTo(0, 12);  // the stable form: no overflow at a large logit
    expect(batchLoss(Float64Array.from([-800]), Float32Array.from([0]), Float32Array.from([1]), { balanceClasses: false, labelSmoothing: 0 }, new Float64Array(1))).toBeCloseTo(0, 12);
  });

  it("averages the foreground's and the background's means when balanced, split by the hard target", () => {
    const fg = (bce(z[0], 1) + bce(z[4], 1)) / 2, bg = (bce(z[1], 0) + bce(z[2], 0) + bce(z[5], 0)) / 3;
    expect(batchLoss(z, y, m, { balanceClasses: true, labelSmoothing: 0 }, new Float64Array(6))).toBeCloseTo((fg + bg) / 2, 12);
    // With smoothing the split stays by the hard target, the per-element loss takes the smoothed one.
    const s = 0.2, fgS = (bce(z[0], 0.9) + bce(z[4], 0.9)) / 2, bgS = (bce(z[1], 0.1) + bce(z[2], 0.1) + bce(z[5], 0.1)) / 3;
    expect(batchLoss(z, y, m, { balanceClasses: true, labelSmoothing: s }, new Float64Array(6))).toBeCloseTo((fgS + bgS) / 2, 12);
    // A class with no voxel contributes nothing (masked_mean's clamp), not NaN.
    expect(batchLoss(Float64Array.from([1]), Float32Array.from([1]), Float32Array.from([1]), { balanceClasses: true, labelSmoothing: 0 }, new Float64Array(1))).toBeCloseTo(bce(1, 1) / 2, 12);
  });

  it("returns the loss's gradient (finite differences), zero where the mask is", () => {
    for (const opts of [{ balanceClasses: false, labelSmoothing: 0 }, { balanceClasses: true, labelSmoothing: 0.1 }]) {
      const dz = new Float64Array(6);
      batchLoss(z, y, m, opts, dz);
      expect(dz[3]).toBe(0);
      for (let i = 0; i < 6; i++) {
        const h = 1e-6, up = Float64Array.from(z), down = Float64Array.from(z);
        up[i] += h; down[i] -= h;
        const numeric = (batchLoss(up, y, m, opts, new Float64Array(6)) - batchLoss(down, y, m, opts, new Float64Array(6))) / (2 * h);
        expect(dz[i]).toBeCloseTo(numeric, 6);
      }
    }
  });
});

describe("trainHead", () => {
  it("brings the loss down on separable data and recovers the labels, from the head's own weights", async () => {
    const pixels = separable(2000, 4, 2, 11, 0.25);
    const h = head(4, 2);
    const report = reporter();
    const before = fullLoss(pixels, h.weight, h.bias, options());
    const trained = await trainHead(pixels, h, options(), report);
    expect(trained.losses).toHaveLength(30);
    expect(trained.losses[29]).toBeLessThan(trained.losses[0]);
    expect(fullLoss(pixels, trained.weight, trained.bias, options())).toBeLessThan(before / 3);
    // As right as the planted map itself (the noise mislabels a few), near enough.
    const planted = accuracy(pixels, Float32Array.from(pixels.planted.weight), Float32Array.from(pixels.planted.bias));
    expect(accuracy(pixels, trained.weight, trained.bias)).toBeGreaterThan(Math.max(0.93, planted - 0.02));
    // The planted direction, up to scale: the cosine of each output row with its planted row.
    for (let c = 0; c < 2; c++) {
      let dot = 0, a = 0, b = 0;
      for (let i = 0; i < 4; i++) { const w = trained.weight[c * 4 + i], p = pixels.planted.weight[c * 4 + i]; dot += w * p; a += w * w; b += p * p; }
      expect(dot / Math.sqrt(a * b)).toBeGreaterThan(0.97);
    }
    expect(report.lines).toHaveLength(30);
    expect(report.lines[0]).toMatch(/^Epoch 1\/30 - Loss: \d\.\d{6}$/);
    const parsed = EPOCH_LOSS.exec(report.lines[29])!;
    expect([parsed[1], parsed[2], Number(parsed[3])]).toEqual(["30", "30", Number(trained.losses[29].toFixed(6))]);
  });

  it("starts from `initial` when given, and from the head otherwise", async () => {
    const pixels = separable(300, 3, 1, 5);
    const h = head(3, 1, [1, 2, 3], [4]);
    const tiny = options({ epochs: 1, learningRate: 1e-9 });
    const fromHead = await trainHead(pixels, h, tiny, reporter());
    expect(Array.from(fromHead.weight).map((w) => Math.round(w))).toEqual([1, 2, 3]);
    const fromInitial = await trainHead(pixels, h, tiny, reporter(), { weight: Float32Array.from([-1, -2, -3]), bias: Float32Array.from([-4]) });
    expect(Array.from(fromInitial.weight).map((w) => Math.round(w))).toEqual([-1, -2, -3]);
    expect(Math.round(fromInitial.bias[0])).toBe(-4);
  });

  it("with class balance, a 95/5 skew teaches a head that finds the rare class", async () => {
    const pixels = separable(4000, 4, 1, 3, 0.3, 0.05);  // one voxel in twenty positive
    const positives = pixels.y.reduce((s, v) => s + v, 0) / pixels.count;
    expect(positives).toBeGreaterThan(0.03);
    expect(positives).toBeLessThan(0.08);
    const plain = await trainHead(pixels, head(4, 1), options({ balanceClasses: false }), reporter());
    const balanced = await trainHead(pixels, head(4, 1), options({ balanceClasses: true }), reporter());
    expect(Array.from(balanced.weight)).not.toEqual(Array.from(plain.weight));
    // Recall of the positives: balance lifts it (the plain head calls nearly everything background).
    const recall = (w: Float32Array, b: Float32Array) => {
      let found = 0, total = 0;
      for (let p = 0; p < pixels.count; p++) {
        if (pixels.y[p] < 0.5) continue;
        total++;
        let z = b[0];
        for (let i = 0; i < 4; i++) z += w[i] * pixels.x[p * 4 + i];
        if (z > 0) found++;
      }
      return found / total;
    };
    expect(recall(balanced.weight, balanced.bias)).toBeGreaterThan(recall(plain.weight, plain.bias) + 0.05);
  });

  it("label smoothing moves the optimum: the logits stay finite where hard labels drive them outwards", async () => {
    const pixels = separable(1000, 3, 1, 9, 0);  // noise-free: perfectly separable, hard labels have no optimum
    const hard = await trainHead(pixels, head(3, 1), options({ epochs: 60, labelSmoothing: 0 }), reporter());
    const smooth = await trainHead(pixels, head(3, 1), options({ epochs: 60, labelSmoothing: 0.2 }), reporter());
    const norm = (w: Float32Array) => Math.hypot(...w);
    expect(norm(smooth.weight)).toBeLessThan(norm(hard.weight) * 0.8);
    expect(accuracy(pixels, smooth.weight, smooth.bias)).toBeGreaterThan(0.98);
    // The smoothed loss floors at the targets' own entropy (0.325 at s = 0.2), the hard one keeps falling.
    expect(smooth.losses[59]).toBeGreaterThan(0.3);
    expect(hard.losses[59]).toBeLessThan(0.2);
  });

  it("stops between batches when cancelled, returning what it trained so far", async () => {
    const pixels = separable(1000, 3, 1, 2);
    const report = reporter(3);  // cancelled once three epoch lines are logged
    const trained = await trainHead(pixels, head(3, 1), options({ epochs: 30 }), report);
    expect(trained.losses).toHaveLength(3);
    expect(report.lines).toHaveLength(3);
    expect(Array.from(trained.weight)).not.toEqual([0, 0, 0]);
    // Cancelled before the first epoch: nothing trained, the start returned.
    const cancelledAtOnce = await trainHead(pixels, head(3, 1, [1, 2, 3], [4]), options(), { log: () => {}, cancelled: () => true });
    expect(cancelledAtOnce).toEqual({ weight: new Float32Array([1, 2, 3]), bias: new Float32Array([4]), losses: [], supervised: [] });
    // Cancelled mid-epoch: the batches so far count, the epoch does not.
    let batches = 0;
    const midEpoch = await trainHead(pixels, head(3, 1), options({ epochs: 30 }), { log: () => {}, cancelled: () => ++batches >= 2 });
    expect(midEpoch.losses).toEqual([]);
    expect(Array.from(midEpoch.weight)).not.toEqual([0, 0, 0]);
  });

  it("is deterministic for a seed", async () => {
    const pixels = separable(500, 3, 2, 4);
    const a = await trainHead(pixels, head(3, 2), options({ epochs: 3, seed: 42 }), reporter());
    const b = await trainHead(pixels, head(3, 2), options({ epochs: 3, seed: 42 }), reporter());
    const c = await trainHead(pixels, head(3, 2), options({ epochs: 3, seed: 43 }), reporter());
    expect(a).toEqual(b);
    expect(Array.from(a.weight)).not.toEqual(Array.from(c.weight));
    expect(a.losses).not.toEqual(c.losses);
    const random = mulberry32(1);
    expect([random(), random()]).toEqual([mulberry32(1)(), (() => { const r = mulberry32(1); r(); return r(); })()]);
  });

  it("refuses an empty set and bad options", async () => {
    const empty: PixelSet = { x: new Float32Array(0), y: new Float32Array(0), m: new Float32Array(0), count: 0, inChannels: 3, outChannels: 1 };
    await expect(trainHead(empty, head(3, 1), options(), reporter())).rejects.toThrow(/no painted voxels/);
    const pixels = separable(10, 3, 1, 1);
    await expect(trainHead(pixels, head(3, 1), options({ epochs: 0 }), reporter())).rejects.toThrow(/epochs/);
    await expect(trainHead(pixels, head(3, 1), options({ learningRate: 0 }), reporter())).rejects.toThrow(/learning rate/);
    await expect(trainHead(pixels, head(3, 1), options({ labelSmoothing: 1 }), reporter())).rejects.toThrow(/smoothing/);
    await expect(trainHead(pixels, head(2, 1), options(), reporter())).rejects.toThrow(/initial weights of 2 and 1 values for a 1 × 3 head/);
  });
});

describe("batchLoss's MSE and margin losses", () => {
  const z = Float64Array.from([2, -1, 0.5, -3, 4, 0]), y = Float32Array.from([1, 0, 0, 1, 1, 0]), m = Float32Array.from([1, 1, 1, 0, 1, 1]);
  const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
  const masked = [0, 1, 2, 4, 5];

  it("MSE is the squared distance of the probability from the smoothed target, reduced as BCE is", () => {
    const s = 0.1, t = (i: number) => y[i] * (1 - s) + s / 2;
    const expected = masked.reduce((sum, i) => sum + (sigmoid(z[i]) - t(i)) ** 2, 0) / 5;
    expect(batchLoss(z, y, m, { lossType: "mse", balanceClasses: false, labelSmoothing: s }, new Float64Array(6))).toBeCloseTo(expected, 12);
    const fg = ((sigmoid(z[0]) - t(0)) ** 2 + (sigmoid(z[4]) - t(4)) ** 2) / 2, bg = [1, 2, 5].reduce((sum, i) => sum + (sigmoid(z[i]) - t(i)) ** 2, 0) / 3;
    expect(batchLoss(z, y, m, { lossType: "mse", balanceClasses: true, labelSmoothing: s }, new Float64Array(6))).toBeCloseTo((fg + bg) / 2, 12);
  });

  it("margin squares how far the probability falls short of 1 - m (foreground) or past m (background), and takes no smoothing", () => {
    const margin = 0.3;
    const per = (i: number) => (y[i] > 0.5 ? Math.max(1 - margin - sigmoid(z[i]), 0) : Math.max(sigmoid(z[i]) - margin, 0)) ** 2;
    // z 2 and 4 (foreground) and -1 (background) are past the margin and cost nothing; 0.5 and 0 do.
    expect([0, 1, 4].map(per)).toEqual([0, 0, 0]);
    const expected = masked.reduce((sum, i) => sum + per(i), 0) / 5;
    expect(expected).toBeGreaterThan(0);
    for (const labelSmoothing of [0, 0.2]) {
      expect(batchLoss(z, y, m, { lossType: "margin", margin, balanceClasses: false, labelSmoothing }, new Float64Array(6))).toBeCloseTo(expected, 12);
    }
    const fg = (per(0) + per(4)) / 2, bg = (per(1) + per(2) + per(5)) / 3;
    expect(batchLoss(z, y, m, { lossType: "margin", margin, balanceClasses: true, labelSmoothing: 0 }, new Float64Array(6))).toBeCloseTo((fg + bg) / 2, 12);
    // A smaller margin asks for more: at 0.1 the background at z = -1 (p 0.27) costs too.
    expect(batchLoss(z, y, m, { lossType: "margin", margin: 0.1, balanceClasses: false, labelSmoothing: 0 }, new Float64Array(6))).toBeGreaterThan(expected);
  });

  it("returns each loss's gradient (finite differences), zero where the mask is", () => {
    const zs = Float64Array.from([2, -1, 0.5, -3, 0.3, 0]);  // a foreground voxel short of the margin too
    for (const opts of [
      { lossType: "mse" as const, balanceClasses: false, labelSmoothing: 0 },
      { lossType: "mse" as const, balanceClasses: true, labelSmoothing: 0.1 },
      { lossType: "margin" as const, margin: 0.3, balanceClasses: false, labelSmoothing: 0 },
      { lossType: "margin" as const, margin: 0.2, balanceClasses: true, labelSmoothing: 0 },
    ]) {
      const dz = new Float64Array(6);
      batchLoss(zs, y, m, opts, dz);
      expect(dz[3]).toBe(0);
      for (let i = 0; i < 6; i++) {
        const h = 1e-6, up = Float64Array.from(zs), down = Float64Array.from(zs);
        up[i] += h; down[i] -= h;
        const numeric = (batchLoss(up, y, m, opts, new Float64Array(6)) - batchLoss(down, y, m, opts, new Float64Array(6))) / (2 * h);
        expect(dz[i]).toBeCloseTo(numeric, 6);
      }
    }
  });

  it("trains a head with either: the loss comes down and the labels are recovered", async () => {
    const pixels = separable(2000, 4, 1, 21, 0.25);
    for (const lossType of ["mse", "margin"] as const) {
      const trained = await trainHead(pixels, head(4, 1), options({ lossType, epochs: 40, learningRate: 0.1 }), reporter());
      expect(trained.losses[39]).toBeLessThan(trained.losses[0]);
      expect(accuracy(pixels, trained.weight, trained.bias)).toBeGreaterThan(0.9);
    }
  });
});

/** `count` rows of `inChannels` standard normal features, every channel weighed 1. */
function heldRows(count: number, inChannels: number, outChannels: number, seed: number): DistillationSet {
  const gauss = normal(mulberry32(seed));
  return { x: Float32Array.from({ length: count * inChannels }, gauss), w: new Float32Array(count * outChannels).fill(1), count, inChannels, outChannels };
}

/** The mean squared distance of `trained`'s logits from `teacher`'s over the rows of `held`. */
function distance(held: DistillationSet, trained: { weight: Float32Array; bias: Float32Array }, teacher: Head): number {
  let sum = 0;
  for (let p = 0; p < held.count; p++) for (let c = 0; c < held.outChannels; c++) {
    let a = trained.bias[c] - teacher.bias[c];
    for (let i = 0; i < held.inChannels; i++) a += (trained.weight[c * held.inChannels + i] - teacher.weight[c * held.inChannels + i]) * held.x[p * held.inChannels + i];
    sum += a * a;
  }
  return sum / (held.count * held.outChannels);
}

describe("trainHead with distillation", () => {
  const EPOCH_PARTS = /^Epoch (\d+)\/(\d+) - Loss: (\d+\.\d{6}) - Supervised: (\d+\.\d{6}) - Distillation: (\d+\.\d{6})$/;

  it("holds the head to the teacher, the head it is given, wherever it starts from", async () => {
    // Nothing supervised (every mask 0): distillation alone moves the head.
    const pixels = { ...separable(400, 3, 1, 7), m: new Float32Array(400) };
    const teacher = head(3, 1, [1, -2, 0.5], [0.25]);
    const start = { weight: Float32Array.from([0, 0, 0]), bias: Float32Array.from([0]) };
    const held = heldRows(500, 3, 1, 8);
    const trained = await trainHead(pixels, teacher, options({ epochs: 150, learningRate: 0.05, batchPixels: 100, distillationLambda: 1 }), reporter(), start, held);
    Array.from(trained.weight).forEach((w, i) => expect(w).toBeCloseTo(teacher.weight[i], 2));
    expect(trained.bias[0]).toBeCloseTo(0.25, 2);
    expect(trained.supervised.every((loss) => loss === 0)).toBe(true);
    // At λ 0 nothing moves it.
    const still = await trainHead(pixels, teacher, options({ epochs: 5, distillationLambda: 0 }), reporter(), start, held);
    expect(Array.from(still.weight)).toEqual([0, 0, 0]);
  });

  it("trades the paint against the teacher by λ: the heavier, the nearer the teacher and the further from the paint's optimum", async () => {
    const pixels = separable(1000, 4, 1, 11, 0.25);
    const teacher = head(4, 1);  // the zero head: logit 0 everywhere
    const held = heldRows(800, 4, 1, 12);
    const runs = [];
    for (const distillationLambda of [0, 0.01, 1]) {
      runs.push(await trainHead(pixels, teacher, options({ epochs: 30, distillationLambda }), reporter(), undefined, held));
    }
    const distances = runs.map((r) => distance(held, r, teacher)), fits = runs.map((r) => fullLoss(pixels, r.weight, r.bias, options()));
    expect(distances[1]).toBeLessThan(distances[0]);
    expect(distances[2]).toBeLessThan(distances[1] / 2);
    expect(fits[2]).toBeGreaterThan(fits[0]);
    // A light hold still learns the paint.
    expect(accuracy(pixels, runs[1].weight, runs[1].bias)).toBeGreaterThan(0.9);
  });

  it("names the supervised and distillation terms on the epoch line, the loss their sum, the supervised term kept apart", async () => {
    const pixels = separable(600, 3, 2, 13);
    const report = reporter();
    const lambda = 0.5;
    const trained = await trainHead(pixels, head(3, 2), options({ epochs: 6, distillationLambda: lambda }), report, undefined, heldRows(300, 3, 2, 14));
    expect(report.lines).toHaveLength(6);
    report.lines.forEach((line, e) => {
      const parts = EPOCH_PARTS.exec(line)!;
      const [epoch, total, loss, supervised, distillation] = parts.slice(1).map(Number);
      expect([epoch, total]).toEqual([e + 1, 6]);
      expect(loss).toBeCloseTo(supervised + lambda * distillation, 5);
      expect(loss).toBe(Number(trained.losses[e].toFixed(6)));
      expect(supervised).toBe(Number(trained.supervised[e].toFixed(6)));
      // The dashboard's parser reads the total.
      expect(Number(EPOCH_LOSS.exec(line)![3])).toBe(loss);
    });
    // The head starts at the teacher, where the term is 0; it grows as the paint moves the head.
    expect(Number(EPOCH_PARTS.exec(report.lines[5])![5])).toBeGreaterThan(0);
  });

  it("trains exactly as before without a distillation set, with an empty one, or at λ 0", async () => {
    const pixels = separable(500, 3, 2, 4);
    const plain = reporter();
    const before = await trainHead(pixels, head(3, 2), options({ epochs: 4, distillationLambda: 0 }), plain);
    const noSet = await trainHead(pixels, head(3, 2), options({ epochs: 4, distillationLambda: 0.5 }), reporter(), undefined, null);
    const empty = await trainHead(pixels, head(3, 2), options({ epochs: 4, distillationLambda: 0.5 }), reporter(), undefined, heldRows(0, 3, 2, 1));
    const off = await trainHead(pixels, head(3, 2), options({ epochs: 4, distillationLambda: 0 }), reporter(), undefined, heldRows(100, 3, 2, 1));
    for (const other of [noSet, empty, off]) expect(other).toEqual(before);
    expect(before.losses).toEqual(before.supervised);
    expect(plain.lines[0]).toMatch(/^Epoch 1\/4 - Loss: \d\.\d{6}$/);
  });

  it("refuses a distillation set for another head, a loss it does not have, a margin past 0.5 and a negative λ", async () => {
    const pixels = separable(10, 3, 1, 1);
    await expect(trainHead(pixels, head(3, 1), options(), reporter(), undefined, heldRows(5, 4, 1, 1))).rejects.toThrow(/distillation set of 4 → 1 channels for a 3 → 1 head/);
    await expect(trainHead(pixels, head(3, 1), options({ lossType: "dice" as never }), reporter())).rejects.toThrow(/bce, mse, margin losses, not dice/);
    await expect(trainHead(pixels, head(3, 1), options({ margin: 0.6 }), reporter())).rejects.toThrow(/margin is from 0 to 0.5/);
    await expect(trainHead(pixels, head(3, 1), options({ distillationLambda: -1 }), reporter())).rejects.toThrow(/distillation weight/);
  });
});

describe("yieldEvery", () => {
  it("gives the event loop a turn only once the slice has gone by, and lets a timer queued meanwhile run", async () => {
    let clock = 0;
    const breathe = yieldEvery(16, () => clock);
    clock = 10;
    expect(await breathe()).toBe(false);
    let timerRan = false;
    setTimeout(() => { timerRan = true; }, 0);
    clock = 20;
    expect(await breathe()).toBe(true);
    expect(timerRan).toBe(true);
    // the slice starts again from the turn
    clock = 30;
    expect(await breathe()).toBe(false);
  });
});
