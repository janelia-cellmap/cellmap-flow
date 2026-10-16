// The head trained on the painted voxels: their features gathered from each
// chunk's feature map into one matrix (gatherPixels), then the head's
// weights fitted to their targets with Adam on masked, class-balanced,
// label-smoothed BCE-with-logits (trainHead), the loss of the server's
// trainer (cellmap_flow/finetune/lora_trainer.py with losses.py's
// masked_mean and balanced_mean) so a head trained here and a model
// finetuned on the cluster are judged alike. A 1×1 convolution is a linear
// map per voxel, so the whole network the trainer sees is `logits = W x + b`
// on a row of features, and its gradient is a matrix product: plain
// TypedArray arithmetic on the main thread, milliseconds per epoch for the
// tens of thousands of painted voxels a session has, handing the event loop
// back every SLICE_MS (yieldEvery) so the page keeps drawing, the log
// stream reaches the job card as the epochs end, and a cancel gets through. Everything is deterministic for a seed: the shuffle is
// mulberry32's, the arithmetic doubles in one order.
import type { FinetuneOptions, Head, ModelFeatures, RunReporter, Targets } from "./types";

/** The painted voxels of one or more chunks: `count` rows of the head's
 * input features (`x`, count × inChannels), their targets and masks per
 * output channel (`y`, `m`, count × outChannels), both row-major. */
export interface PixelSet {
  x: Float32Array;
  y: Float32Array;
  m: Float32Array;
  count: number;
  inChannels: number;
  outChannels: number;
}

export interface TrainedHead {
  weight: Float32Array;
  bias: Float32Array;
  /** One mean batch loss per epoch trained. */
  losses: number[];
}

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** The painted voxels of a chunk: those whose mask is set on any channel,
 * with their features from the model's feature map and their targets.
 *
 * The feature map's layout (PLAN_browser_finetune.md, "What trains"): the
 * channel on axis 1 and the other axes, flattened in order, the chunk's
 * voxels in C order. That covers the 2D exports' `[1, C, y, x]` and the 3D
 * exports' `[z, C, y, x]` (their final Conv runs with z as the batch axis)
 * alike: voxel v = n × spatial + s has channel c at (n × C + c) × spatial +
 * s. The logits are checked to the same rule, so the features gathered
 * belong to the voxels the targets describe. */
export function gatherPixels(features: ModelFeatures, targets: Targets, head: Head, chunkShape: number[]): PixelSet {
  const n = prod(chunkShape);
  const { inChannels, outChannels } = head;
  if (targets.channels !== outChannels) throw new Error(`the targets have ${targets.channels} channels where the head writes ${outChannels}`);
  if (targets.target.length !== outChannels * n || targets.mask.length !== outChannels * n) {
    throw new Error(`the targets hold ${targets.target.length} values for ${outChannels} channels × ${n} voxels of a ${chunkShape.join("×")} chunk`);
  }
  const fd = features.featuresDims;
  if (fd.length < 2 || fd[1] !== inChannels) throw new Error(`the feature map is ${fd.join("×")}; its channel axis (axis 1) should have the head's ${inChannels} channels`);
  if (features.features.length !== prod(fd) || prod(fd) !== inChannels * n) {
    throw new Error(`the feature map ${fd.join("×")} holds ${prod(fd) / inChannels} voxels, the chunk ${chunkShape.join("×")} has ${n}`);
  }
  const ld = features.logitsDims;
  if (prod(ld) !== outChannels * n) throw new Error(`the logits ${ld.join("×")} hold ${prod(ld) / outChannels} voxels, the chunk ${chunkShape.join("×")} has ${n}`);
  const spatial = prod(fd.slice(2));

  const { target, mask } = targets;
  let count = 0;
  const painted = new Uint8Array(n);
  for (let c = 0; c < outChannels; c++) {
    const base = c * n;
    for (let v = 0; v < n; v++) if (mask[base + v] > 0 && !painted[v]) { painted[v] = 1; count++; }
  }
  const x = new Float32Array(count * inChannels), y = new Float32Array(count * outChannels), m = new Float32Array(count * outChannels);
  let p = 0;
  for (let v = 0; v < n; v++) {
    if (!painted[v]) continue;
    const batch = Math.floor(v / spatial), s = v - batch * spatial;
    for (let c = 0; c < inChannels; c++) x[p * inChannels + c] = features.features[(batch * inChannels + c) * spatial + s];
    for (let c = 0; c < outChannels; c++) {
      y[p * outChannels + c] = target[c * n + v];
      m[p * outChannels + c] = mask[c * n + v];
    }
    p++;
  }
  return { x, y, m, count, inChannels, outChannels };
}

/** One PixelSet of several, their rows in order. */
export function concatPixels(sets: PixelSet[], head: Head): PixelSet {
  const { inChannels, outChannels } = head;
  const count = sets.reduce((n, s) => n + s.count, 0);
  const x = new Float32Array(count * inChannels), y = new Float32Array(count * outChannels), m = new Float32Array(count * outChannels);
  let at = 0;
  for (const s of sets) {
    if (s.inChannels !== inChannels || s.outChannels !== outChannels) throw new Error(`a pixel set of ${s.inChannels} → ${s.outChannels} channels for a ${inChannels} → ${outChannels} head`);
    x.set(s.x, at * inChannels);
    y.set(s.y, at * outChannels);
    m.set(s.m, at * outChannels);
    at += s.count;
  }
  return { x, y, m, count, inChannels, outChannels };
}

/** mulberry32: a small seeded generator of doubles in [0, 1), the same
 * sequence on every machine for a seed. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The masked BCE-with-logits of a batch and its gradient with respect to the
 * logits, as the server's trainer reduces it.
 *
 * Per element, in the form that overflows for no logit (PyTorch's
 * BCEWithLogitsLoss): max(z, 0) − z·y' + log(1 + exp(−|z|)), against the
 * smoothed target y' = y(1 − s) + s/2. Reduced over the masked elements of
 * the batch, every channel together: their mean (masked_mean), or with
 * class balance the mean over the foreground voxels and the mean over the
 * background ones averaged (balanced_mean), the split by the hard target
 * y, before smoothing, as the server splits it. `dz` gets ∂loss/∂z. */
export function batchLoss(z: Float64Array, y: Float32Array, m: Float32Array, options: Pick<FinetuneOptions, "balanceClasses" | "labelSmoothing">, dz: Float64Array): number {
  const s = options.labelSmoothing;
  let sumFg = 0, sumBg = 0, weightFg = 0, weightBg = 0;
  for (let i = 0; i < z.length; i++) {
    const w = m[i];
    if (w <= 0) { dz[i] = 0; continue; }
    const zi = z[i], target = y[i] * (1 - s) + s / 2;
    const loss = Math.max(zi, 0) - zi * target + Math.log1p(Math.exp(-Math.abs(zi)));
    dz[i] = w * (1 / (1 + Math.exp(-zi)) - target);  // w · (σ(z) − y'), the reduction's weights applied below
    if (y[i] > 0.5) { sumFg += w * loss; weightFg += w; } else { sumBg += w * loss; weightBg += w; }
  }
  // masked_mean's clamp(min=1): a class with no voxel contributes zero, not NaN.
  if (options.balanceClasses) {
    const scaleFg = 1 / (2 * Math.max(weightFg, 1)), scaleBg = 1 / (2 * Math.max(weightBg, 1));
    for (let i = 0; i < z.length; i++) if (m[i] > 0) dz[i] *= y[i] > 0.5 ? scaleFg : scaleBg;
    return sumFg * scaleFg + sumBg * scaleBg;
  }
  const scale = 1 / Math.max(weightFg + weightBg, 1);
  for (let i = 0; i < z.length; i++) dz[i] *= scale;
  return (sumFg + sumBg) * scale;
}

/** The head fitted to `pixels` from `initial`'s weights (the head's own
 * when absent): `options.epochs` passes over the rows in a seeded shuffle,
 * `options.batchPixels` rows per Adam step (lr `options.learningRate`,
 * betas 0.9 and 0.999, eps 1e-8, PyTorch's defaults), the loss of
 * `batchLoss`. After each epoch one line to `report`, in the words the
 * dashboard's job card and loss plot parse: `Epoch i/N - Loss: 0.012345`.
 * `report.cancelled()` is asked before each epoch and after every batch;
 * when it says so the weights as trained so far come back, with the losses
 * of the epochs that completed: a cancel right after an epoch's line (the
 * job's Stop Early) returns that epoch's head exactly. */
export async function trainHead(pixels: PixelSet, head: Head, options: FinetuneOptions, report: RunReporter, initial?: { weight: Float32Array; bias: Float32Array }): Promise<TrainedHead> {
  const { count, inChannels: cin, outChannels: cout } = pixels;
  if (count === 0) throw new Error("no painted voxels to train on");
  if (!(options.epochs >= 1) || !Number.isInteger(options.epochs)) throw new Error(`epochs must be a whole number of 1 or more, not ${options.epochs}`);
  if (!(options.batchPixels >= 1) || !Number.isInteger(options.batchPixels)) throw new Error(`batchPixels must be a whole number of 1 or more, not ${options.batchPixels}`);
  if (!(options.learningRate > 0)) throw new Error(`the learning rate must be above zero, not ${options.learningRate}`);
  if (!(options.labelSmoothing >= 0 && options.labelSmoothing < 1)) throw new Error(`label smoothing is a fraction below 1, not ${options.labelSmoothing}`);
  const start = initial ?? head;
  if (start.weight.length !== cout * cin || start.bias.length !== cout) throw new Error(`initial weights of ${start.weight.length} and ${start.bias.length} values for a ${cout} × ${cin} head`);

  // The parameters and Adam's moments, doubles; the weights go back to float32 at the end.
  const w = Float64Array.from(start.weight), b = Float64Array.from(start.bias);
  const mw = new Float64Array(w.length), vw = new Float64Array(w.length), mb = new Float64Array(b.length), vb = new Float64Array(b.length);
  const gw = new Float64Array(w.length), gb = new Float64Array(b.length);
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lr = options.learningRate;
  let step = 0;

  const batch = Math.min(options.batchPixels, count);
  const z = new Float64Array(batch * cout), dz = new Float64Array(batch * cout);
  const yBatch = new Float32Array(batch * cout), mBatch = new Float32Array(batch * cout);
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  const random = mulberry32(options.seed);
  const losses: number[] = [];
  const result = () => ({ weight: Float32Array.from(w), bias: Float32Array.from(b), losses });
  const breathe = yieldEvery(SLICE_MS);

  for (let epoch = 1; epoch <= options.epochs; epoch++) {
    if (report.cancelled()) return result();
    // Fisher–Yates, from the seeded generator.
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      const t = order[i]; order[i] = order[j]; order[j] = t;
    }
    let epochLoss = 0, batches = 0;
    for (let from = 0; from < count; from += batch) {
      const rows = Math.min(batch, count - from);
      // Forward: z[p, c] = b[c] + Σ_i w[c, i] x[p, i]; the batch's targets and masks beside it.
      for (let p = 0; p < rows; p++) {
        const row = order[from + p], xo = row * cin;
        for (let c = 0; c < cout; c++) {
          let acc = b[c];
          const wo = c * cin;
          for (let i = 0; i < cin; i++) acc += w[wo + i] * pixels.x[xo + i];
          z[p * cout + c] = acc;
          yBatch[p * cout + c] = pixels.y[row * cout + c];
          mBatch[p * cout + c] = pixels.m[row * cout + c];
        }
      }
      const n = rows * cout;
      const loss = batchLoss(z.subarray(0, n), yBatch.subarray(0, n), mBatch.subarray(0, n), options, dz.subarray(0, n));
      // Backward: ∂loss/∂w[c, i] = Σ_p dz[p, c] x[p, i]; ∂loss/∂b[c] = Σ_p dz[p, c].
      gw.fill(0); gb.fill(0);
      for (let p = 0; p < rows; p++) {
        const xo = order[from + p] * cin;
        for (let c = 0; c < cout; c++) {
          const d = dz[p * cout + c];
          if (d === 0) continue;
          gb[c] += d;
          const wo = c * cin;
          for (let i = 0; i < cin; i++) gw[wo + i] += d * pixels.x[xo + i];
        }
      }
      // Adam, as torch.optim.Adam steps it.
      step++;
      const correction1 = 1 - beta1 ** step, correction2 = 1 - beta2 ** step;
      adam(w, gw, mw, vw, lr, beta1, beta2, eps, correction1, correction2);
      adam(b, gb, mb, vb, lr, beta1, beta2, eps, correction1, correction2);
      epochLoss += loss;
      batches++;
      await breathe();
      if (report.cancelled()) return result();
    }
    const mean = epochLoss / batches;
    losses.push(mean);
    report.log(`Epoch ${epoch}/${options.epochs} - Loss: ${mean.toFixed(6)}`);
  }
  return result();
}

function adam(p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array, lr: number, beta1: number, beta2: number, eps: number, correction1: number, correction2: number): void {
  const stepSize = lr / correction1, sqrtCorrection2 = Math.sqrt(correction2);
  for (let i = 0; i < p.length; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * g[i];
    v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
    p[i] -= stepSize * m[i] / (Math.sqrt(v[i]) / sqrtCorrection2 + eps);
  }
}

/** How long the trainer computes before it hands the event loop back. */
export const SLICE_MS = 16;

/** A function to call between steps: it gives the event loop a turn once
 * `ms` have gone by since the last one, and says whether it did.
 *
 * The turn is a timer task, not scheduler.yield: Chrome runs a yield's
 * continuation ahead of every other task waiting, so a loop that yields
 * after each batch keeps the page's other work (the log stream's events, the
 * job card's polls) waiting until it ends, which showed every epoch's line
 * at once, after the last. A timer waits its turn with the rest; browsers
 * clamp it to 4 ms when nested, which a 16 ms slice keeps to a fifth. */
export function yieldEvery(ms: number, now: () => number = () => performance.now()): () => Promise<boolean> {
  let since = now();
  return async () => {
    if (now() - since < ms) return false;
    await new Promise((resolve) => setTimeout(resolve, 0));
    since = now();
    return true;
  };
}
