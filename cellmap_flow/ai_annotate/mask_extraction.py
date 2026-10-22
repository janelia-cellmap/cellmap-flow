"""From an EM plane to the RGB image a model is sent, and from its reply to a mask.

Adapted from ask-to-mask (``_normalize_to_uint8``/``_slice_to_rgb`` and
``extract_mask``'s colour-difference mode): the model is asked to paint the
structure one colour and everything else black, and the mask is the pixels
that came back that colour. The arithmetic is
``cellmap_flow.portable.ai_annotate``'s (``normalize_to_uint8``,
``extract_mask``; ``model_mask`` lines a reply up with the plane first),
which the browser runs too; here are the Pillow images the server's
backends and staging pass around.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cellmap_flow.portable import ai_annotate as portable
from cellmap_flow.portable.ai_annotate import normalize_to_uint8  # noqa: F401  (re-exported)


def slice_to_rgb(data_2d: np.ndarray) -> Image.Image:
    """A 2D plane as a grey RGB image, the form image models take."""
    normed = normalize_to_uint8(data_2d)
    return Image.fromarray(np.stack([normed] * 3, axis=-1), mode="RGB")


def extract_mask(
    output_image,
    target_rgb: tuple[int, int, int],
    threshold: float = 200.0,
) -> np.ndarray:
    """The pixels of the model's reply (a Pillow image or an RGB array)
    painted ``target_rgb``, as a bool (H, W) mask
    (``portable.ai_annotate.extract_mask`` says how a pixel is scored)."""
    out = np.asarray(output_image.convert("RGB") if isinstance(output_image, Image.Image) else output_image)
    return portable.extract_mask(out, target_rgb, threshold)
