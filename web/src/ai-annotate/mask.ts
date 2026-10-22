// The mask in a model's reply and what the review shows of it, in the
// browser: the Python both modes run decides it all
// (cellmap_flow/portable/ai_annotate.py, in the engine's Pyodide worker
// through portable/ai_annotate_calls.py), as the server's backend and
// staging do: the reply resized back to the size sent (Pillow's bilinear
// filter; the worker loads Pillow the first time), the padding cut off and
// the target colour's pixels taken (model_mask); the mask's objects put on
// the write box and the image sent tinted where they are (stage). Here are
// the calls, with the images as the canvas holds them.
import { AI_PYTHON } from "./plane";
import type { PlanePlan, PythonCall, RgbaImage } from "./types";

/** The mask in `reply`, the image a model sent back (decoded), and the reply
 * lined up with the plane (portable/ai_annotate.py model_mask): resized back
 * to `sent`, the image it was sent (the plane padded to a square), when it
 * came at another size, the padding cut off the right and bottom to the
 * plane `image`'s size, and the pixels painted `target` taken. The mask has
 * a byte a pixel of the plane, rows first, 1 where painted. */
export async function modelMask(
  python: PythonCall,
  reply: RgbaImage,
  sent: { width: number; height: number },
  image: { width: number; height: number },
  target: readonly [number, number, number],
): Promise<{ mask: Uint8Array; model_image: RgbaImage }> {
  const pixels = new Uint8Array(reply.data).buffer;
  const { arrays } = await python<{ value: Record<string, never>; arrays: ArrayBuffer[] }>(`${AI_PYTHON}:model_mask`, {
    shape: [reply.height, reply.width], sent: [sent.height, sent.width], image: [image.height, image.width], target_rgb: target,
  }, [pixels]);
  return { mask: new Uint8Array(arrays[0]), model_image: { width: image.width, height: image.height, data: new Uint8ClampedArray(arrays[1]) } };
}

/** Have the Python import what `stageMask` labels the mask's objects with
 * (scipy, which the worker loads the first time a call needs it: seconds,
 * once a page), while the model works on the plane, so the answer is not
 * kept waiting for it. Never fails: a failure here is the stage's to report. */
export function prepareMask(python: PythonCall): void {
  python(`${AI_PYTHON}:prepare`, {}).catch(() => {});
}

/** What the review shows of `mask` (a byte a pixel of the plan's image, rows
 * first, nonzero where painted), as the server stages it
 * (ai_annotate_calls.py stage): its objects on the write box's in-plane
 * voxels, z, y, x order, 0 background and 1..n the objects
 * (mask_to_write_shape), the share of the box they mark, and `image`, the
 * image sent, tinted `target` where they are (overlay). */
export async function stageMask(
  python: PythonCall,
  plan: PlanePlan,
  mask: Uint8Array,
  image: RgbaImage,
  target: readonly [number, number, number],
): Promise<{ objects: Int32Array; maskFraction: number; overlay: RgbaImage }> {
  const { value, arrays } = await python<{ value: { mask_fraction: number }; arrays: ArrayBuffer[] }>(
    `${AI_PYTHON}:stage`, { plan, target_rgb: target }, [new Uint8Array(mask).buffer, new Uint8Array(image.data).buffer]);
  return {
    objects: new Int32Array(arrays[0]),
    maskFraction: value.mask_fraction,
    overlay: { width: image.width, height: image.height, data: new Uint8ClampedArray(arrays[1]) },
  };
}

/** numpy's default (linear) percentile of sorted values (the raw's window
 * the ranking stretches a model's input to, backend/routes/rank.ts). */
export function percentile(sorted: Float64Array, q: number): number {
  if (sorted.length === 0) return 0;
  const rank = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(rank), hi = Math.ceil(rank);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}
