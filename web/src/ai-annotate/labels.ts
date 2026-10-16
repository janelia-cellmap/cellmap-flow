// The ids a box of labels gets from a segmentation: the port of
// cellmap_flow/finetune/session/fill.py's _fresh_ids and seed_labels, and of
// the scipy.ndimage.label call pipeline.labels_for_box makes on the mask
// (label2d), for the browser's AI annotation. The connected-component
// labelling itself is infer/label.ts's (labelComponents, which numbers
// objects as scipy does); here it is run on one plane with scipy's default
// structure, 4-connectivity, so two objects the model left touching only at
// a corner stay apart, as it was asked to leave them.
//
// The ids: an object the user has already painted part of keeps the id
// painted there most; the rest get fresh ones, the lowest the box does not
// use from 2 up for a uint8 volume, counting up from the box's largest for an
// instance target on a uint16 one (countUp). Ids running out is a refusal
// (AIAnnotateError "refused", 409: the Python raises ValueError and its
// accept route answers 409 with the message), with fill.py's own words.
import { earlierOffsets, labelComponents } from "../infer/label";
import { AIAnnotateError, type Labels } from "./types";

/** The objects of `mask` (rows × cols, rows first, nonzero = foreground),
 * each pixel's id in an int32 array (scipy.ndimage.label's dtype), 0 for
 * background, numbered 1..count in raster order of their first pixel as
 * scipy numbers them; 4-connected (scipy's default structure in 2D: pixels
 * touching only at a corner are different objects). */
export function label2d(mask: Uint8Array, rows: number, cols: number): { objects: Int32Array; count: number } {
  if (mask.length !== rows * cols) throw new Error(`a ${rows}×${cols} mask has ${rows * cols} pixels, not ${mask.length}`);
  const { labels, count } = labelComponents(mask, [1, rows, cols], earlierOffsets(2, 1));
  return { objects: Int32Array.from(labels), count };
}

/** The largest id the labels' dtype holds (np.iinfo(dtype).max). */
function biggestId(existing: Labels): number {
  return existing instanceof Uint16Array ? 65535 : 255;
}

function dtypeName(existing: Labels): string {
  return existing instanceof Uint16Array ? "uint16" : "uint8";
}

/** `count` ids for new objects in a box that holds `existing` (fill._fresh_ids).
 *
 * `countUp`: from one past the box's largest label, so an affinity target's
 * objects stay apart from every object nearby (ids elsewhere in the volume
 * are not looked at: affinities only compare nearby voxels). Else the lowest
 * ids the box does not use, so a uint8 volume's 254 ids are never run
 * through: a repeat in another box is only a repeated colour, which a binary
 * or distance target does not see. Throws AIAnnotateError("refused", 409)
 * when the dtype has no room. */
export function freshIds(existing: Labels, count: number, countUp: boolean): number[] {
  const biggest = biggestId(existing);
  const ids: number[] = [];
  if (countUp) {
    let max = 0;
    for (let i = 0; i < existing.length; i++) if (existing[i] > max) max = existing[i];
    const first = Math.max(max, 1) + 1;
    for (let k = 0; k < count; k++) ids.push(first + k);
  } else {
    const used = new Set<number>();
    for (let i = 0; i < existing.length; i++) if (existing[i] >= 2) used.add(existing[i]);
    const last = Math.min(biggest, used.size + count + 2);
    for (let id = 2; id <= last && ids.length < count; id++) if (!used.has(id)) ids.push(id);
  }
  if (ids.length < count || (ids.length && ids[ids.length - 1] > biggest)) {
    throw new AIAnnotateError(
      "refused",
      `${count} new objects do not fit in the volume's ${dtypeName(existing)} labels: make a new annotation volume, which for an instance model (Cellpose, affinities) now holds uint16 ids, or raise Min size or the threshold so the seed makes fewer objects`,
      409,
    );
  }
  return ids;
}

/** Labels for a box from a segmentation: an id per object inside, 1 outside
 * (fill.seed_labels).
 *
 * `objects` is a segmenter's output over the box (0 background, objects
 * 1..n; label2d's here), `existing` what the box holds now, the same length.
 * Each object gets an id of its own (freshIds; `countUp` for an instance
 * target), so neighbouring objects are told apart, and a merge the model
 * made shows as one colour to split; but an object the user has already
 * painted part of takes the id painted there most (on a tie, the larger id,
 * as the Python's stable sort of the pairs leaves it), or the unpainted rest
 * of it would be taught as a different object from its painted part. A
 * binary or distance target reads every id as foreground, so the ids cost
 * it nothing. The result is of `existing`'s dtype. */
export function seedLabels(objects: ArrayLike<number>, existing: Labels, countUp: boolean): Labels {
  const n = existing.length;
  if (objects.length !== n) throw new Error(`${objects.length} object labels for a box of ${n} voxels`);
  const labels: Labels = existing instanceof Uint16Array ? new Uint16Array(n) : new Uint8Array(n);
  let maxObject = 0;
  for (let i = 0; i < n; i++) {
    labels[i] = objects[i] > 0 ? 2 : 1;
    if (objects[i] > maxObject) maxObject = objects[i];
  }
  if (maxObject === 0) return labels;

  // For each object, how many of its voxels hold each painted id.
  const painted = new Map<number, Map<number, number>>();
  const present = new Uint8Array(maxObject + 1);
  for (let i = 0; i < n; i++) {
    const o = objects[i];
    if (o <= 0) continue;
    present[o] = 1;
    if (existing[i] < 2) continue;
    let counts = painted.get(o);
    if (!counts) painted.set(o, (counts = new Map()));
    counts.set(existing[i], (counts.get(existing[i]) ?? 0) + 1);
  }
  const ids = new Array<number>(maxObject + 1).fill(0);
  for (const [o, counts] of painted) {
    let best = 0, bestCount = -1;
    for (const [id, count] of counts) {
      if (count > bestCount || (count === bestCount && id > best)) {
        best = id;
        bestCount = count;
      }
    }
    ids[o] = best;
  }
  const fresh: number[] = [];
  for (let o = 1; o <= maxObject; o++) if (present[o] && ids[o] === 0) fresh.push(o);
  const newIds = freshIds(existing, fresh.length, countUp);
  fresh.forEach((o, k) => (ids[o] = newIds[k]));
  for (let i = 0; i < n; i++) if (objects[i] > 0) labels[i] = ids[objects[i]];
  return labels;
}
