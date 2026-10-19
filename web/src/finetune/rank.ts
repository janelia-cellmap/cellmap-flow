// How well a model's prediction already matches the paint: the score the
// Finetune tab ranks models by (backend/routes/rank.ts), to pick the one to
// start finetuning from.
//
// A model is scored on the targets its own finetune would train on
// (targets.ts targetTransformFor: an object or not on every channel of a
// binary model, the painted pairs of an affinity model), over the painted
// voxels of the volume, its prediction read on the volume's grid (each
// painted voxel takes the model's voxel its centre lies in, as Seed from
// Prediction reads one: label-it/prediction.ts servedIndices), so a model
// served at another voxel size is scored on the same paint. Its decision is
// at 0, as the trainer's: the models' outputs are logits. A model finer
// than the paint (16 nm under 128 nm strokes) needs a block of its own for
// every few painted voxels, so the route reads at most RANK_BLOCKS of its
// blocks, spread over the paint, and scores the painted voxels under them.
//
// The score is the balanced accuracy, the mean of the share of object
// targets it calls object and of background targets it calls background, so
// that a painted volume mostly of one class does not reward a model that
// calls everything that class. Rough strokes are forgiven along the edge
// between painted object and painted background: for a binary model a
// painted voxel next to one of the other class (its six neighbours in the
// chunk) is not counted. An affinity model's painted pairs across an edge
// are what its boundary channels learn from, so its targets are counted as
// they are. The finetune's default loss on the same targets (class-balanced
// BCE, label smoothing 0.1) is given beside the score: about 0.69 is
// guessing, about 0.20 the paint fitted exactly.
//
// A binary model is scored channel by channel and answers with its best:
// painted nuclei rank a model by its nucleus channel, whatever else it
// predicts. An affinity model is scored on its channels together.
import type { PaintedChunk, Targets } from "./types";

/** The label smoothing of the finetune's default loss (FinetuneOptions), which the loss beside the score uses. */
export const RANK_SMOOTHING = 0.1;

/** The most blocks of a model's output one score reads: 32 passes of a 3D
 * model take half a minute on a laptop GPU. */
export const RANK_BLOCKS = 32;

/** `count` of `items`, spread evenly over them in their order; all of them when there are no more. */
export function spread<T>(items: readonly T[], count: number): T[] {
  if (items.length <= count) return [...items];
  return Array.from({ length: count }, (_, j) => items[Math.floor(((j + 0.5) * items.length) / count)]);
}

/** One channel's agreement with the paint so far: the targets counted, how
 * many of them the prediction gets right, and the loss summed, apart for
 * object and background targets. */
export interface Tally {
  objects: number;
  objectsRight: number;
  objectLoss: number;
  background: number;
  backgroundRight: number;
  backgroundLoss: number;
}

export function emptyTallies(channels: number): Tally[] {
  return Array.from({ length: channels }, () => ({ objects: 0, objectsRight: 0, objectLoss: 0, background: 0, backgroundRight: 0, backgroundLoss: 0 }));
}

/** The painted voxels of `chunk` that touch a painted voxel of the other
 * class (object, label 2 and up, against background, label 1) along z, y or
 * x within the chunk: 1 there, else 0. The strokes' edges, which a binary
 * model's score leaves out. */
export function paintEdges(chunk: PaintedChunk): Uint8Array {
  const [d, h, w] = chunk.shape, labels = chunk.labels;
  const edges = new Uint8Array(labels.length);
  const steps = [h * w, w, 1], sizes = [d, h, w];
  for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (z * h + y) * w + x, label = labels[i];
    if (!label) continue;
    const at = [z, y, x];
    for (let a = 0; a < 3 && !edges[i]; a++) {
      for (const s of [-1, 1]) {
        const p = at[a] + s;
        if (p < 0 || p >= sizes[a]) continue;
        const other = labels[i + s * steps[a]];
        if (other && (other > 1) !== (label > 1)) { edges[i] = 1; break; }
      }
    }
  }
  return edges;
}

/** The model's logit on channel `c` at voxel `i` of a chunk (C order over
 * its shape), or null where it was not read: past what the model's level
 * covers, or under none of the blocks a fine model's score reads. */
export type LogitAt = (c: number, i: number) => number | null;

/** Add one painted chunk to `tallies`: `targets` the chunk's (channel
 * outermost over the chunk's voxels, targets.ts), `logitAt` the model's
 * prediction on its voxels, and `skip` the voxels not to count (paintEdges,
 * or null). */
export function tallyChunk(tallies: Tally[], chunk: Pick<PaintedChunk, "shape">, targets: Targets, logitAt: LogitAt, skip: Uint8Array | null): void {
  const n = chunk.shape.reduce((a, b) => a * b, 1), C = targets.channels;
  if (C !== tallies.length) throw new Error(`targets of ${C} channels for ${tallies.length} tallies`);
  const smoothHigh = 1 - RANK_SMOOTHING / 2, smoothLow = RANK_SMOOTHING / 2;
  for (let c = 0; c < C; c++) {
    const t = tallies[c];
    for (let i = 0; i < n; i++) {
      if (!targets.mask[c * n + i] || (skip && skip[i])) continue;
      const logit = logitAt(c, i);
      if (logit === null) continue;
      // BCE on the logit against the smoothed target: softplus(z) − y·z.
      const softplus = Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit)));
      if (targets.target[c * n + i] > 0.5) {
        t.objects++;
        if (logit > 0) t.objectsRight++;
        t.objectLoss += softplus - smoothHigh * logit;
      } else {
        t.background++;
        if (logit <= 0) t.backgroundRight++;
        t.backgroundLoss += softplus - smoothLow * logit;
      }
    }
  }
}

/** A model's score on the paint, from its tallies. */
export interface RankScore {
  /** The balanced accuracy: the mean of objectsFound and backgroundRight, or the one of them the paint has. */
  matches: number;
  objectsFound: number | null;
  backgroundRight: number | null;
  /** The finetune's default loss on the same targets, class-balanced. */
  loss: number;
  scored: number;
  objects: number;
  background: number;
  /** A binary model's channel the score is of; null for an affinity model's, scored on all of them. */
  channel: string | null;
}

/** The score of `tallies`: an affinity model's over its channels together,
 * a binary model's best channel's (by the score, then the lower loss).
 * null when nothing was counted. */
export function rankScore(tallies: Tally[], channels: readonly string[], affinity: boolean): RankScore | null {
  if (affinity) {
    const all = tallies.reduce<Tally>((s, t) => ({
      objects: s.objects + t.objects, objectsRight: s.objectsRight + t.objectsRight, objectLoss: s.objectLoss + t.objectLoss,
      background: s.background + t.background, backgroundRight: s.backgroundRight + t.backgroundRight, backgroundLoss: s.backgroundLoss + t.backgroundLoss,
    }), emptyTallies(1)[0]);
    return scoreOf(all, null);
  }
  let best: RankScore | null = null;
  tallies.forEach((t, c) => {
    const s = scoreOf(t, channels[c] ?? `channel ${c}`);
    if (s && (!best || s.matches > best.matches || (s.matches === best.matches && s.loss < best.loss))) best = s;
  });
  return best;
}

function scoreOf(t: Tally, channel: string | null): RankScore | null {
  if (!t.objects && !t.background) return null;
  const objectsFound = t.objects ? t.objectsRight / t.objects : null;
  const backgroundRight = t.background ? t.backgroundRight / t.background : null;
  const rates = [objectsFound, backgroundRight].filter((r): r is number => r !== null);
  const losses = [t.objects ? t.objectLoss / t.objects : null, t.background ? t.backgroundLoss / t.background : null].filter((l): l is number => l !== null);
  return {
    matches: rates.reduce((a, b) => a + b, 0) / rates.length,
    objectsFound, backgroundRight,
    loss: losses.reduce((a, b) => a + b, 0) / losses.length,
    scored: t.objects + t.background, objects: t.objects, background: t.background,
    channel,
  };
}
