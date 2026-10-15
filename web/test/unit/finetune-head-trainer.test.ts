// The head trainer on data made here: gatherPixels follows the feature
// layout rule (channel on axis 1, the other axes the chunk's voxels in
// order) and takes the voxels any channel's mask covers; batchLoss is the
// server's masked, balanced, smoothed BCE-with-logits and its gradient is
// the loss's (finite differences); trainHead on separable data (a planted
// linear map plus noise) brings the loss down and recovers the labels,
// class balance changes what a 95/5 skew teaches, label smoothing moves the
// optimum, a cancel stops it between batches, the same seed trains the same
// head, and the epoch line is the one the dashboard's loss plot parses.
import { describe, expect, it } from "vitest";

import { batchLoss, concatPixels, gatherPixels, mulberry32, type PixelSet, trainHead } from "../../src/finetune/head-trainer";
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
    expect(cancelledAtOnce).toEqual({ weight: new Float32Array([1, 2, 3]), bias: new Float32Array([4]), losses: [] });
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
