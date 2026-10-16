// The voxels the distillation term holds to the starting head, sampled from
// each chunk's feature map as the run computes it: the port of where the
// server's trainer applies losses.py's distillation_loss (lora_trainer.py,
// the scope chosen per batch) for a head trained on cached features.
//
// Where. The server distils over the voxels of each training patch that its
// scope picks: "unlabeled", the voxels the supervised mask leaves out
// (1 - mask, per channel); "all", every voxel; and "anchor" whenever the
// session has good regions, the unpainted voxels of the regions' patches
// (`annotation == 0`, one weight for every channel), which then decides
// alone, whatever the form's scope says. Here the chunks are the patches: a
// painted chunk's voxels for "unlabeled" and "all", the regions' boxes
// (good-regions.ts anchorBoxes) for "anchor".
//
// How many. A chunk is a whole model output block, a hundred thousand
// voxels or more, and every one of them has a row of features (16 to 256
// floats), so the candidates are subsampled uniformly, by a seeded
// reservoir, to DISTILLATION_VOXELS over the whole run. The cap is split
// over the chunks before any is read: equally over the painted chunks
// (each painted place weighs alike, as the server draws a chunk first and
// a voxel in it), and over the regions' chunks in proportion to the region
// voxels each holds (every anchored voxel as likely as any other). A chunk
// with fewer candidates than its share gives all it has.
import { chunkLayout, copyFeatures, type DistillationSet } from "./head-trainer";
import type { Head, ModelFeatures, PaintedChunk, Targets } from "./types";

/** The most voxels a run distils on, over every chunk: 32,768. About the
 * painted voxels of a session (tens of thousands), so the term at most
 * doubles an epoch's arithmetic, and its rows take 32,768 × inChannels
 * floats: 4 MB at 32 channels, 32 MB at 256. The term is a mean, and a mean
 * squared distance over 32,768 voxels has a sampling error far below
 * anything λ trades it against. */
export const DISTILLATION_VOXELS = 32768;

/** Which voxels of a chunk the term counts: the form's two scopes, or a good region's anchor. */
export type DistillationPick = "unlabeled" | "all" | "anchor";

export interface ChunkDistillation {
  pick: DistillationPick;
  /** The most voxels kept from this chunk. */
  quota: number;
  /** "anchor": the chunk's voxels inside a good region's box, 1 each, in the chunk's C order. */
  inRegion?: Uint8Array;
  /** The run's seeded generator (one stream, the chunks in order). */
  random: () => number;
}

/** Each chunk's share of `cap`: in proportion to `weights` (a chunk's
 * candidates as far as they are known before it is read), rounded up. */
export function quotas(weights: readonly number[], cap: number = DISTILLATION_VOXELS): number[] {
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map((w) => (total > 0 ? Math.ceil((cap * w) / total) : 0));
}

/** The voxels of `chunk` the term counts, at most `plan.quota` of them drawn
 * uniformly among the candidates (Algorithm R over the chunk's voxels in
 * order, then put back in that order), with their features and per-channel
 * weights:
 * - "unlabeled": a voxel some channel's mask leaves out, weighted 1 on
 *   those channels and 0 on the masked ones (the server's 1 - mask);
 * - "all": every voxel, 1 on every channel;
 * - "anchor": a voxel in a good region (`plan.inRegion`) that is not
 *   painted (label 0), 1 on every channel: painted, it is the paint's, so
 *   a scribble inside a good region wins over the region. */
export function sampleDistillation(features: ModelFeatures, targets: Targets, chunk: PaintedChunk, head: Head, plan: ChunkDistillation): DistillationSet {
  const { inChannels, outChannels } = head;
  const { n, spatial } = chunkLayout(features, targets, head, chunk.shape);
  const { mask } = targets;
  const { pick, quota, inRegion, random } = plan;
  if (pick === "anchor" && (!inRegion || inRegion.length !== n)) throw new Error(`an anchor needs the chunk's ${n} voxels' region mask`);
  const candidate = (v: number): boolean => {
    if (pick === "all") return true;
    if (pick === "anchor") return inRegion![v] === 1 && chunk.labels[v] === 0;
    for (let c = 0; c < outChannels; c++) if (!(mask[c * n + v] > 0)) return true;
    return false;
  };
  // Algorithm R: the first `quota` candidates, then the i-th replaces a kept
  // one with probability quota / (i + 1); every candidate as likely as any.
  const kept: number[] = [];
  let seen = 0;
  if (quota > 0) {
    for (let v = 0; v < n; v++) {
      if (!candidate(v)) continue;
      if (kept.length < quota) kept.push(v);
      else {
        const j = Math.floor(random() * (seen + 1));
        if (j < quota) kept[j] = v;
      }
      seen++;
    }
  }
  kept.sort((a, b) => a - b);
  const count = kept.length;
  const x = new Float32Array(count * inChannels), w = new Float32Array(count * outChannels);
  kept.forEach((v, row) => {
    copyFeatures(features, inChannels, spatial, v, x, row);
    for (let c = 0; c < outChannels; c++) w[row * outChannels + c] = pick === "unlabeled" ? (mask[c * n + v] > 0 ? 0 : 1) : 1;
  });
  return { x, w, count, inChannels, outChannels };
}

/** One DistillationSet of several, their rows in order. */
export function concatDistillation(sets: readonly DistillationSet[], head: Head): DistillationSet {
  const { inChannels, outChannels } = head;
  const count = sets.reduce((n, s) => n + s.count, 0);
  const x = new Float32Array(count * inChannels), w = new Float32Array(count * outChannels);
  let at = 0;
  for (const s of sets) {
    if (s.inChannels !== inChannels || s.outChannels !== outChannels) throw new Error(`a distillation set of ${s.inChannels} → ${s.outChannels} channels for a ${inChannels} → ${outChannels} head`);
    x.set(s.x, at * inChannels);
    w.set(s.w, at * outChannels);
    at += s.count;
  }
  return { x, w, count, inChannels, outChannels };
}
