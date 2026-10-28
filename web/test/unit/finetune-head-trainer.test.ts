// The head trainer on data made here: batchLoss returns each loss's
// gradient (BCE-with-logits, MSE on probabilities and the margin loss,
// balanced or not, smoothed or not; checked by finite differences);
// trainHead on separable data (a planted linear map plus noise) brings the
// loss down and recovers the labels, its epoch line the one the dashboard's
// loss plot parses, its steps at the schedule's rates, the form's first and
// the shared floor of it last; and with a distillation set and λ the head is
// held to the teacher (the head it is given, whatever it starts from).
import { describe, expect, it } from "vitest";

import { PYTHON } from "../../src/generated/python-data";
import { batchLoss, type DistillationSet, mulberry32, type PixelSet, trainHead } from "../../src/finetune/head-trainer";
import { DEFAULT_OPTIONS, type FinetuneOptions, type Head, type RunReporter } from "../../src/finetune/types";

// The dashboard's parser (cellmap_flow/dashboard/static/js/dashboard/finetune/loss-plot.js EPOCH_LOSS).
const EPOCH_LOSS = /Epoch\s+(\d+)\/(\d+)\s*-\s*Loss:\s*([0-9]*\.?[0-9]+)/i;

function head(inChannels: number, outChannels: number, weight?: number[], bias?: number[]): Head {
  return {
    featureTensor: "features", weightName: "w", biasName: "b", inChannels, outChannels, dtype: "float32",
    weight: Float32Array.from(weight ?? new Array(inChannels * outChannels).fill(0)),
    bias: Float32Array.from(bias ?? new Array(outChannels).fill(0)),
  };
}

function reporter(): RunReporter & { lines: string[] } {
  const lines: string[] = [];
  return { lines, log: (l) => lines.push(l), cancelled: () => false };
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

describe("batchLoss", () => {
  const y = Float32Array.from([1, 0, 0, 1, 1, 0]), m = Float32Array.from([1, 1, 1, 0, 1, 1]);

  it("returns each loss's gradient (finite differences), zero where the mask is", () => {
    const z = Float64Array.from([2, -1, 0.5, -3, 0.3, 0]);  // a foreground voxel short of the margin too
    for (const opts of [
      { balanceClasses: false, labelSmoothing: 0 },
      { balanceClasses: true, labelSmoothing: 0.1 },
      { lossType: "mse" as const, balanceClasses: false, labelSmoothing: 0 },
      { lossType: "mse" as const, balanceClasses: true, labelSmoothing: 0.1 },
      { lossType: "margin" as const, margin: 0.3, balanceClasses: false, labelSmoothing: 0 },
      { lossType: "margin" as const, margin: 0.2, balanceClasses: true, labelSmoothing: 0 },
    ]) {
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

  it("steps at the form's rate first and at the shared floor of it last", async () => {
    // Nine copies of a voxel whose feature is 0: only the bias learns, and
    // its gradient, σ(b) − 1, stays near −0.5 over steps this small, so each
    // of Adam's steps moves it by that step's rate. A run is stopped after
    // `steps` steps: cancelled() is asked before the epoch and after each.
    const steps = 9, rate = 1e-3, floor = PYTHON.finetune.learning_rate_floor;
    const pixels: PixelSet = { x: new Float32Array(steps), y: new Float32Array(steps).fill(1), m: new Float32Array(steps).fill(1), count: steps, inChannels: 1, outChannels: 1 };
    const biasAfter = async (done: number) => {
      let asked = 0;
      const report = { log: () => {}, cancelled: () => asked++ === done };
      const trained = await trainHead(pixels, head(1, 1), options({ epochs: 1, batchPixels: 1, learningRate: rate, lossType: "bce", labelSmoothing: 0 }), report);
      return trained.bias[0];
    };
    expect((await biasAfter(1)) / rate).toBeCloseTo(1, 2);
    expect(((await biasAfter(steps)) - (await biasAfter(steps - 1))) / (rate * floor)).toBeCloseTo(1, 2);
  });
});

/** `count` rows of `inChannels` standard normal features, every channel weighed 1. */
function heldRows(count: number, inChannels: number, outChannels: number, seed: number): DistillationSet {
  const gauss = normal(mulberry32(seed));
  return { x: Float32Array.from({ length: count * inChannels }, gauss), w: new Float32Array(count * outChannels).fill(1), count, inChannels, outChannels };
}

describe("trainHead with distillation", () => {
  it("holds the head to the teacher, the head it starts from, against the paint", async () => {
    // The paint's planted map is another head than the teacher's: free, the
    // head leaves the teacher for it; distilled, it stays near.
    const pixels = separable(400, 3, 1, 7);
    const teacher = head(3, 1, [1, -2, 0.5], [0.25]);
    const held = heldRows(500, 3, 1, 8);
    const away = (t: { weight: Float32Array; bias: Float32Array }) =>
      Math.hypot(...Array.from(t.weight, (w, i) => w - teacher.weight[i]), t.bias[0] - teacher.bias[0]);
    const train = (lambda: number) => trainHead(pixels, teacher, options({ epochs: 60, learningRate: 0.05, batchPixels: 100, distillationLambda: lambda }), reporter(), held);
    const free = await train(0), distilled = await train(10);
    expect(away(free)).toBeGreaterThan(0.5);
    expect(away(distilled)).toBeLessThan(away(free) / 3);
  });
});
