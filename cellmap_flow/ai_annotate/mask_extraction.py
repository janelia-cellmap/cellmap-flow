"""From an EM plane to the RGB image a model is sent, and from its reply to a mask.

Adapted from ask-to-mask (``_normalize_to_uint8``/``_slice_to_rgb`` and
``extract_mask``'s colour-difference mode): the model is asked to paint the
structure one colour and everything else black, and the mask is the pixels
that came back that colour.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """A 2D array as uint8 grey levels.

    uint8 passes through unchanged. Anything else is clipped to its 1st-99th
    percentile over the plane and stretched to 0-255, as raw EM is not always
    8-bit and a few saturated pixels should not flatten the contrast.
    """
    if data.dtype == np.uint8:
        return data
    p_lo, p_hi = np.percentile(data, (1, 99))
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    clipped = np.clip(data.astype(np.float32), p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)


def slice_to_rgb(data_2d: np.ndarray) -> Image.Image:
    """A 2D plane as a grey RGB image, the form image models take."""
    normed = normalize_to_uint8(data_2d)
    return Image.fromarray(np.stack([normed] * 3, axis=-1), mode="RGB")


def extract_mask(
    output_image: Image.Image,
    target_rgb: tuple[int, int, int],
    threshold: float = 200.0,
) -> np.ndarray:
    """The pixels of the model's reply painted ``target_rgb``, as a bool (H, W) mask.

    Each pixel scores how strongly it shows the target colour: its weakest
    channel among those the target has at full strength, minus its strongest
    deviation elsewhere -- a channel the target has off (0) being on, or a
    channel the target has at part strength being away from that strength.
    For red (255, 0, 0) that is ``R - max(G, B)``: red that dominates. A
    pixel counts when its score reaches ``threshold``.

    The part-strength term is what lets orange (255, 128, 0) or purple
    (128, 0, 255) be found at all: counting their half channel as "on" would
    cap the score at 128, under the threshold, and counting it as "off"
    would reject the colour itself.
    """
    out = np.asarray(output_image.convert("RGB") if isinstance(output_image, Image.Image) else output_image)
    out = out.astype(np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)

    full = target >= 255
    off = target <= 0
    part = ~full & ~off

    strength = np.min(out[:, :, full], axis=-1) if full.any() else np.full(out.shape[:2], 255.0, np.float32)
    deviations = []
    if off.any():
        deviations.append(np.max(out[:, :, off], axis=-1))
    if part.any():
        deviations.append(np.max(np.abs(out[:, :, part] - target[part]), axis=-1))
    score = strength - np.max(deviations, axis=0) if deviations else strength
    return score >= threshold
