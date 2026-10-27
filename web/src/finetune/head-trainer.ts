// The head trained on the painted voxels: their features gathered from each
// chunk's feature map into one matrix (gatherPixels), then the head's
// weights fitted to their targets with Adam on a masked, class-balanced
// loss (trainHead): BCE-with-logits by default, label-smoothed, or the
// server's MSE on probabilities or margin loss (batchLoss), each the server
// trainer's (cellmap_flow/finetune/lora_trainer.py with losses.py's
// masked_mean, balanced_mean and MarginLoss) so a head trained here and a
// model finetuned on the cluster are judged alike. With distillation the
// loss adds λ × the mean squared distance of the logits from the starting
// head's on a second set of voxels (a DistillationSet, which
// distillation.ts samples): losses.py's distillation_loss, with the
// teacher's logits W₀x + b₀ computed on the same cached features, so the
// teacher costs no model run at all. A 1×1 convolution is a linear map per
// voxel, so the whole network the trainer sees is `logits = W x + b` on a
// row of features, and its gradient is a matrix product: plain TypedArray
// arithmetic on the main thread, milliseconds per epoch for the tens of
// thousands of painted voxels a session has, handing the event loop back
// every SLICE_MS (yieldEvery) so the page keeps drawing, the log stream
// reaches the job card as the epochs end, and a cancel gets through.
// Everything is deterministic for a seed: the shuffles are mulberry32's,
// the arithmetic doubles in one order.
import type { FinetuneOptions, Head, HeadLoss, ModelFeatures, RunReporter, Targets } from "./types";

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

/** The voxels the distillation term holds to the teacher (the head the job
 * started from): `count` rows of features (`x`, count × inChannels) and,
 * per row and output channel, the weight the term gives that logit (`w`,
 * count × outChannels: 1 where it counts, 0 where it does not, as the
 * server's anchor and unlabeled masks are), both row-major. */
export interface DistillationSet {
  x: Float32Array;
  w: Float32Array;
  count: number;
  inChannels: number;
  outChannels: number;
}

export interface TrainedHead {
  weight: Float32Array;
  bias: Float32Array;
  /** One mean batch loss per epoch trained: the supervised term, plus λ ×
   * the distillation term when there is one (what the epoch line's Loss
   * says). */
  losses: number[];
  /** The supervised term alone, per epoch: the same as `losses` without distillation. */
  supervised: number[];
}

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** The voxel count of a chunk and the spatial size of its feature map's
 * batch slices, once `features` and `targets` are checked to describe the
 * chunk's voxels by the layout rule (see gatherPixels). `head` is the head
 * trained, which for a run of one channel is its row alone (run.ts), so the
 * targets have its channels; the logits, the model's every channel, are
 * checked against the features by the feature runner (features.ts). */
export function chunkLayout(features: ModelFeatures, targets: Targets, head: Head, chunkShape: number[]): { n: number; spatial: number } {
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
  return { n, spatial: prod(fd.slice(2)) };
}

/** Voxel `v`'s features copied into row `row` of `x` (rows of `inChannels`),
 * by the layout rule: voxel v = batch × spatial + s has channel c at
 * (batch × C + c) × spatial + s. */
export function copyFeatures(features: ModelFeatures, inChannels: number, spatial: number, v: number, x: Float32Array, row: number): void {
  const batch = Math.floor(v / spatial), s = v - batch * spatial;
  for (let c = 0; c < inChannels; c++) x[row * inChannels + c] = features.features[(batch * inChannels + c) * spatial + s];
}

/** The painted voxels of a chunk: those whose mask is set on any channel,
 * with their features from the model's feature map and their targets.
 *
 * The feature map's layout (PLAN_browser_finetune.md, "What trains"): the
 * channel on axis 1 and the other axes, flattened in order, the chunk's
 * voxels in C order. That covers the 2D exports' `[1, C, y, x]` and the 3D
 * exports' `[z, C, y, x]` (their final Conv runs with z as the batch axis)
 * alike: voxel v = n × spatial + s has channel c at (n × C + c) × spatial +
 * s. The feature map is checked to hold the chunk's voxels, so the features
 * gathered belong to the voxels the targets describe. */
export function gatherPixels(features: ModelFeatures, targets: Targets, head: Head, chunkShape: number[]): PixelSet {
  const { inChannels, outChannels } = head;
  const { n, spatial } = chunkLayout(features, targets, head, chunkShape);
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
    copyFeatures(features, inChannels, spatial, v, x, p);
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

/** The losses trainHead and batchLoss take. */
export const HEAD_LOSSES: readonly HeadLoss[] = ["bce", "mse", "margin"];

/** What batchLoss reads of the options: the loss (BCE when absent) and its margin, the reduction, the smoothing. */
export type LossOptions = Pick<FinetuneOptions, "balanceClasses" | "labelSmoothing"> & Partial<Pick<FinetuneOptions, "lossType" | "margin">>;

/** The masked loss of a batch and its gradient with respect to the logits,
 * as the server's trainer computes and reduces it.
 *
 * Per element, by the loss:
 * - "bce", BCE-with-logits in the form that overflows for no logit
 *   (PyTorch's BCEWithLogitsLoss): max(z, 0) − z·y' + log(1 + exp(−|z|)),
 *   against the smoothed target y' = y(1 − s) + s/2;
 * - "mse", the squared distance of the probability from that target,
 *   (σ(z) − y')²: the server compares probabilities, not logits, for MSE
 *   (comparing logits with 0/1 trains them toward 0 and 1, which the
 *   sigmoid serves as 0.5 and 0.73);
 * - "margin", MarginLoss: relu((1 − m) − σ(z))² for a foreground voxel and
 *   relu(σ(z) − m)² for a background one, nothing once the probability is
 *   past the margin; no smoothing (the server sets it to 0 for this loss).
 * Reduced over the masked elements of the batch, every channel together:
 * their mean (masked_mean), or with class balance the mean over the
 * foreground voxels and the mean over the background ones averaged
 * (balanced_mean), the split by the hard target y, before smoothing, as the
 * server splits it. `dz` gets ∂loss/∂z. */
export function batchLoss(z: Float64Array, y: Float32Array, m: Float32Array, options: LossOptions, dz: Float64Array): number {
  const lossType = options.lossType ?? "bce";
  const margin = options.margin ?? 0.3;
  const s = lossType === "margin" ? 0 : options.labelSmoothing;
  let sumFg = 0, sumBg = 0, weightFg = 0, weightBg = 0;
  for (let i = 0; i < z.length; i++) {
    const w = m[i];
    if (w <= 0) { dz[i] = 0; continue; }
    const zi = z[i], foreground = y[i] > 0.5, target = y[i] * (1 - s) + s / 2;
    const p = 1 / (1 + Math.exp(-zi));
    let loss: number, slope: number;
    if (lossType === "bce") {
      loss = Math.max(zi, 0) - zi * target + Math.log1p(Math.exp(-Math.abs(zi)));
      slope = p - target;
    } else if (lossType === "mse") {
      loss = (p - target) ** 2;
      slope = 2 * (p - target) * p * (1 - p);  // through the sigmoid: dσ/dz = σ(1 − σ)
    } else {
      const short = foreground ? Math.max(1 - margin - p, 0) : Math.max(p - margin, 0);
      loss = short * short;
      slope = (foreground ? -2 : 2) * short * p * (1 - p);
    }
    dz[i] = w * slope;  // the reduction's weights applied below
    if (foreground) { sumFg += w * loss; weightFg += w; } else { sumBg += w * loss; weightBg += w; }
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

/** The distillation term of rows `rows` of `held` (losses.py's
 * distillation_loss): Σ w (z − z₀)² / max(Σ w, 1) over the rows and output
 * channels, z the head's logits (`w`, `b`), z₀ the teacher's (`teacher`,
 * one row per row of `held`), w the set's weights; masked_mean's clamp, so
 * a slice that weighs nothing costs nothing. `scaledDiff` gets each
 * element's w (z − z₀) and the scale 1 / max(Σ w, 1) comes back with the
 * term, for the gradient: ∂term/∂z = 2 w (z − z₀) / max(Σ w, 1). */
export function distillationTerm(held: DistillationSet, rows: Uint32Array, w: Float64Array, b: Float64Array, teacher: Float64Array, scaledDiff: Float64Array): { term: number; scale: number } {
  const { inChannels: cin, outChannels: cout } = held;
  let sumW = 0, sumSq = 0;
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r], xo = row * cin;
    for (let c = 0; c < cout; c++) {
      let acc = b[c];
      const wo = c * cin;
      for (let i = 0; i < cin; i++) acc += w[wo + i] * held.x[xo + i];
      const weight = held.w[row * cout + c], diff = acc - teacher[row * cout + c];
      scaledDiff[r * cout + c] = weight * diff;
      sumW += weight;
      sumSq += weight * diff * diff;
    }
  }
  const scale = 1 / Math.max(sumW, 1);
  return { term: sumSq * scale, scale };
}

/** The teacher's logits on every row of `held`: b₀ + W₀ x, in doubles. */
function teacherLogits(held: DistillationSet, teacher: { weight: Float32Array; bias: Float32Array }): Float64Array {
  const { count, inChannels: cin, outChannels: cout } = held;
  const out = new Float64Array(count * cout);
  for (let p = 0; p < count; p++) for (let c = 0; c < cout; c++) {
    let acc = teacher.bias[c];
    for (let i = 0; i < cin; i++) acc += teacher.weight[c * cin + i] * held.x[p * cin + i];
    out[p * cout + c] = acc;
  }
  return out;
}

/** The seed of the distillation rows' shuffle: a stream apart from the
 * painted voxels', so that a run without distillation shuffles exactly as
 * it did before there was any. */
const DISTILLATION_STREAM = 0x9e3779b9;

/** The head fitted to `pixels` from `initial`'s weights (the head's own
 * when absent): `options.epochs` passes over the rows in a seeded shuffle,
 * `options.batchPixels` rows per Adam step (lr `options.learningRate`,
 * betas 0.9 and 0.999, eps 1e-8, PyTorch's defaults), the loss of
 * `batchLoss`.
 *
 * With `distillation` and λ (`options.distillationLambda`) above 0, each
 * step's loss adds λ × distillationTerm on a share of the distillation
 * rows: they are shuffled once an epoch (their own seeded stream) and dealt
 * out over the epoch's steps in equal slices, so every row is seen once an
 * epoch whatever the batch, as the server's patches bring their unlabeled
 * voxels along with their painted ones. The teacher is `head`, the weights
 * the job started from (FinetuneRun), not `initial`: the server's teacher
 * is the model before any iteration of the job changed it
 * (adaptation.py's teacher), so a restart does not distil toward what an
 * earlier run already moved.
 *
 * After each epoch one line to `report`, in the words the dashboard's job
 * card and loss plot parse: `Epoch i/N - Loss: 0.012345`, with distillation
 * `Epoch i/N - Loss: total - Supervised: s - Distillation: d` (the
 * server's line names the supervised term too: it is the one that says how
 * well the paint is fitted, since the distillation term is smallest for a
 * head that has not moved). The head returned is the last epoch's: the
 * problem is convex in the head's weights, and no epoch is kept aside as
 * the server's best checkpoint is. `report.cancelled()` is asked before
 * each epoch and after every batch; when it says so the weights as trained
 * so far come back, with the losses of the epochs that completed: a cancel
 * right after an epoch's line (the job's Stop Early) returns that epoch's
 * head exactly. */
export async function trainHead(pixels: PixelSet, head: Head, options: FinetuneOptions, report: RunReporter, initial?: { weight: Float32Array; bias: Float32Array }, distillation?: DistillationSet | null): Promise<TrainedHead> {
  const { count, inChannels: cin, outChannels: cout } = pixels;
  if (count === 0) throw new Error("no painted voxels to train on");
  if (!(options.epochs >= 1) || !Number.isInteger(options.epochs)) throw new Error(`epochs must be a whole number of 1 or more, not ${options.epochs}`);
  if (!(options.batchPixels >= 1) || !Number.isInteger(options.batchPixels)) throw new Error(`batchPixels must be a whole number of 1 or more, not ${options.batchPixels}`);
  if (!(options.learningRate > 0)) throw new Error(`the learning rate must be above zero, not ${options.learningRate}`);
  if (!(options.labelSmoothing >= 0 && options.labelSmoothing < 1)) throw new Error(`label smoothing is a fraction below 1, not ${options.labelSmoothing}`);
  if (!HEAD_LOSSES.includes(options.lossType)) throw new Error(`the head trains with the ${HEAD_LOSSES.join(", ")} losses, not ${options.lossType}`);
  if (!(options.margin >= 0 && options.margin <= 0.5)) throw new Error(`the margin is from 0 to 0.5, not ${options.margin}`);
  if (!(options.distillationLambda >= 0 && Number.isFinite(options.distillationLambda))) throw new Error(`the distillation weight is a number of 0 or more, not ${options.distillationLambda}`);
  const start = initial ?? head;
  if (start.weight.length !== cout * cin || start.bias.length !== cout) throw new Error(`initial weights of ${start.weight.length} and ${start.bias.length} values for a ${cout} × ${cin} head`);
  const lambda = options.distillationLambda;
  const held = distillation && distillation.count > 0 && lambda > 0 ? distillation : null;
  if (held && (held.inChannels !== cin || held.outChannels !== cout)) throw new Error(`a distillation set of ${held.inChannels} → ${held.outChannels} channels for a ${cin} → ${cout} head`);

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

  // The distillation rows: the teacher's logits once, their order, and each
  // step's slice of them (steps × slice ≥ rows).
  const steps = Math.ceil(count / batch);
  const teacher = held ? teacherLogits(held, head) : new Float64Array(0);
  const heldOrder = new Uint32Array(held?.count ?? 0);
  for (let i = 0; i < heldOrder.length; i++) heldOrder[i] = i;
  const heldRandom = mulberry32((options.seed ^ DISTILLATION_STREAM) >>> 0);
  const scaledDiff = new Float64Array((held ? Math.ceil(held.count / steps) : 0) * cout);

  const losses: number[] = [], supervised: number[] = [];
  const result = () => ({ weight: Float32Array.from(w), bias: Float32Array.from(b), losses, supervised });
  const breathe = yieldEvery(SLICE_MS);

  for (let epoch = 1; epoch <= options.epochs; epoch++) {
    if (report.cancelled()) return result();
    shuffle(order, random);
    if (held) shuffle(heldOrder, heldRandom);
    let epochLoss = 0, epochSupervised = 0, epochDistillation = 0, batches = 0;
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
      // The distillation term on this step's slice, its gradient added:
      // λ · 2 w (z − z₀) / max(Σ w, 1) per element, times the row's features.
      let term = 0;
      if (held) {
        const slice = heldOrder.subarray(Math.floor((batches * held.count) / steps), Math.floor(((batches + 1) * held.count) / steps));
        const t = distillationTerm(held, slice, w, b, teacher, scaledDiff);
        term = t.term;
        const factor = 2 * lambda * t.scale;
        for (let r = 0; r < slice.length; r++) {
          const xo = slice[r] * cin;
          for (let c = 0; c < cout; c++) {
            const d = factor * scaledDiff[r * cout + c];
            if (d === 0) continue;
            gb[c] += d;
            const wo = c * cin;
            for (let i = 0; i < cin; i++) gw[wo + i] += d * held.x[xo + i];
          }
        }
      }
      // Adam, as torch.optim.Adam steps it.
      step++;
      const correction1 = 1 - beta1 ** step, correction2 = 1 - beta2 ** step;
      adam(w, gw, mw, vw, lr, beta1, beta2, eps, correction1, correction2);
      adam(b, gb, mb, vb, lr, beta1, beta2, eps, correction1, correction2);
      epochLoss += loss + lambda * term;
      epochSupervised += loss;
      epochDistillation += term;
      batches++;
      await breathe();
      if (report.cancelled()) return result();
    }
    const mean = epochLoss / batches, meanSupervised = epochSupervised / batches;
    losses.push(held ? mean : meanSupervised);
    supervised.push(meanSupervised);
    report.log(
      held
        ? `Epoch ${epoch}/${options.epochs} - Loss: ${mean.toFixed(6)} - Supervised: ${meanSupervised.toFixed(6)} - Distillation: ${(epochDistillation / batches).toFixed(6)}`
        : `Epoch ${epoch}/${options.epochs} - Loss: ${meanSupervised.toFixed(6)}`,
    );
  }
  return result();
}

/** Fisher–Yates over `order`, from the seeded generator. */
export function shuffle(order: Uint32Array, random: () => number): void {
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
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
