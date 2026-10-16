// From an EM plane to the grey image a model is sent, and from its reply to a
// mask: the port of cellmap_flow/ai_annotate/mask_extraction.py (adapted there
// from ask-to-mask) and of staging.py's overlay. Pure functions over typed
// arrays; the canvas is images.ts's business.
import type { RgbaImage } from "./types";

/** `plane` (rows × cols values, rows first) as uint8 grey levels
 * (mask_extraction.normalize_to_uint8). Values already 0..255 integers
 * (`isUint8`) pass through unchanged; anything else is clipped to its 1st to
 * 99th percentile over the plane and stretched to 0-255, as raw EM is not
 * always 8-bit and a few saturated pixels should not flatten the contrast. */
export function normalizeToUint8(plane: ArrayLike<number>, isUint8: boolean): Uint8Array {
  const n = plane.length;
  const out = new Uint8Array(n);
  if (isUint8) {
    for (let i = 0; i < n; i++) out[i] = plane[i];
    return out;
  }
  const sorted = Float64Array.from(plane as ArrayLike<number>).sort();
  let lo = percentile(sorted, 1);
  let hi = percentile(sorted, 99);
  if (hi <= lo) hi = lo + 1;
  const scale = 255 / (hi - lo);
  for (let i = 0; i < n; i++) {
    const v = Math.min(Math.max(plane[i], lo), hi);
    out[i] = Math.trunc((v - lo) * scale);  // astype(np.uint8) truncates
  }
  return out;
}

/** numpy's default (linear) percentile of sorted values. */
function percentile(sorted: Float64Array, q: number): number {
  if (sorted.length === 0) return 0;
  const rank = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(rank), hi = Math.ceil(rank);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

/** A grey plane (rows × cols uint8, rows first) as the RGBA image models
 * take (mask_extraction.slice_to_rgb): the grey in every channel, alpha 255. */
export function greyToRgba(grey: Uint8Array, rows: number, cols: number): RgbaImage {
  if (grey.length !== rows * cols) throw new Error(`a ${rows}×${cols} plane has ${rows * cols} pixels, not ${grey.length}`);
  const data = new Uint8ClampedArray(rows * cols * 4);
  for (let i = 0, j = 0; i < grey.length; i++, j += 4) {
    data[j] = data[j + 1] = data[j + 2] = grey[i];
    data[j + 3] = 255;
  }
  return { width: cols, height: rows, data };
}

/** The pixels of the model's reply painted `target` (mask_extraction.extract_mask),
 * one byte per pixel, rows first, 1 where it is.
 *
 * Each pixel scores how strongly it shows the target colour: its weakest
 * channel among those the target has at full strength (255), minus its
 * strongest deviation elsewhere: a channel the target has off (0) being on,
 * or a channel the target has at part strength being away from that
 * strength. For red (255, 0, 0) that is R − max(G, B): red that dominates.
 * A pixel counts when its score reaches `threshold`. The part-strength term
 * is what lets orange (255, 128, 0) or purple (128, 0, 255) be found at all. */
export function extractMask(image: RgbaImage, target: readonly [number, number, number], threshold = 200): Uint8Array {
  const full: number[] = [], off: number[] = [], part: number[] = [];
  target.forEach((t, c) => (t >= 255 ? full : t <= 0 ? off : part).push(c));
  const { width, height, data } = image;
  const mask = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    let strength = 255;
    for (const c of full) strength = Math.min(strength, data[j + c]);
    let deviation = -Infinity;
    for (const c of off) deviation = Math.max(deviation, data[j + c]);
    for (const c of part) deviation = Math.max(deviation, Math.abs(data[j + c] - target[c]));
    const score = deviation === -Infinity ? strength : strength - deviation;
    if (score >= threshold) mask[i] = 1;
  }
  return mask;
}

/** How strongly the overlay tints the mask with the target colour (staging.py OVERLAY_ALPHA). */
export const OVERLAY_ALPHA = 0.5;

/** `image` with the pixels `mask` marks (one byte per pixel of `image`, rows
 * first, as the mask of the write box laid out as the image is and scaled to
 * it) tinted `rgb` (staging.overlay): the review's third picture, what
 * accepting writes, not the model's own image. */
export function overlay(image: RgbaImage, mask: Uint8Array, rgb: readonly [number, number, number], alpha = OVERLAY_ALPHA): RgbaImage {
  if (mask.length !== image.width * image.height) throw new Error(`the mask has ${mask.length} pixels, the image ${image.width * image.height}`);
  const data = new Uint8ClampedArray(image.data);
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    if (!mask[i]) continue;
    for (let c = 0; c < 3; c++) data[j + c] = Math.round((1 - alpha) * data[j + c] + alpha * rgb[c]);
  }
  return { width: image.width, height: image.height, data };
}

/** The share of `mask`'s pixels set (staging.py's mask_fraction). */
export function maskFraction(mask: Uint8Array): number {
  if (mask.length === 0) return 0;
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
  return n / mask.length;
}
