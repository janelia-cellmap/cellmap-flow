"""EM-crop <-> RGB conversion and color-diff mask extraction.

Ported from ask-to-mask's zarr_io.py (_normalize_to_uint8/_slice_to_rgb) and
postprocess.py (extract_mask) - the default color-diff mode only, since v1
needs a single deterministic binary mask, not the instance/direct/invert
variants ask-to-mask also supports.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to uint8 [0, 255].

    uint8 data is returned as-is; other dtypes are percentile-clipped
    (1st-99th) and rescaled, since raw EM data isn't guaranteed 8-bit.
    """
    if data.dtype == np.uint8:
        return data
    p_lo, p_hi = np.percentile(data, (1, 99))
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    clipped = np.clip(data.astype(np.float32), p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)


def slice_to_rgb(data_2d: np.ndarray) -> Image.Image:
    """Convert a 2D array to an RGB PIL Image."""
    normed = normalize_to_uint8(data_2d)
    return Image.fromarray(np.stack([normed] * 3, axis=-1), mode="RGB")


def extract_mask(
    input_image: Image.Image,
    output_image: Image.Image,
    target_rgb: tuple[int, int, int],
    threshold: float = 200.0,
) -> np.ndarray:
    """Extract a binary mask by finding pixels with high saturation in the target color.

    Uses the max channel value in the target color direction: for a red target
    (255,0,0), finds pixels where red is high and dominates over green and blue.

    Args:
        input_image: Original EM crop (RGB), unused but kept for parity with
            ask-to-mask's signature (some mask modes there need it).
        output_image: Gemini-recolored image (RGB).
        target_rgb: The color used to highlight the label, e.g. (255, 0, 0).
        threshold: Minimum score in the target channel(s) to count as colored.

    Returns:
        Binary mask as uint8 array (0 or 255), same spatial dims as input.
    """
    out = np.array(output_image).astype(np.float32)
    target = np.array(target_rgb, dtype=np.float32)

    on_channels = np.where(target > 0)[0]
    off_channels = np.where(target == 0)[0]

    on_min = np.min(out[:, :, on_channels], axis=-1)
    if len(off_channels) > 0:
        off_max = np.max(out[:, :, off_channels], axis=-1)
        score = on_min - off_max
    else:
        score = on_min

    return np.where(score >= threshold, 255, 0).astype(np.uint8)
