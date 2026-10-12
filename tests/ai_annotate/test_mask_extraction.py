"""EM plane to RGB, and the model's painted reply to a bool mask."""

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate.mask_extraction import extract_mask, normalize_to_uint8, slice_to_rgb
from cellmap_flow.ai_annotate.organelles import ORGANELLES


def test_uint8_passes_through_and_other_dtypes_are_stretched():
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert normalize_to_uint8(data) is data
    wide = np.linspace(1000, 5000, 10_000, dtype=np.float32).reshape(100, 100)
    out = normalize_to_uint8(wide)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_a_flat_plane_does_not_divide_by_zero():
    out = normalize_to_uint8(np.full((4, 4), 7, dtype=np.uint16))
    assert out.dtype == np.uint8 and (out == 0).all()


def test_slice_to_rgb_is_grey_rgb_of_the_plane_shape():
    image = slice_to_rgb(np.arange(6, dtype=np.uint8).reshape(2, 3))
    assert image.mode == "RGB" and image.size == (3, 2)
    pixels = np.asarray(image)
    assert (pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all()


def _reply(colours):
    """A one-row image of the given RGB colours."""
    return Image.fromarray(np.array([colours], dtype=np.uint8))


def test_red_dominant_pixels_are_the_mask():
    reply = _reply([(255, 0, 0), (230, 25, 20), (255, 255, 255), (0, 0, 0), (128, 128, 128), (255, 100, 0)])
    mask = extract_mask(reply, (255, 0, 0))
    assert mask.dtype == bool and mask.shape == (1, 6)
    assert mask.tolist() == [[True, True, False, False, False, False]]


@pytest.mark.parametrize("key", list(ORGANELLES))
def test_every_catalog_colour_can_be_found_and_neighbours_are_not(key):
    target = ORGANELLES[key].rgb
    near = tuple(int(np.clip(c + (-20 if c > 128 else 15), 0, 255)) for c in target)
    others = [p.rgb for p in ORGANELLES.values() if p.rgb != target]
    mask = extract_mask(_reply([target, near, (0, 0, 0), (255, 255, 255), *others]), target)
    assert mask[0, :2].all(), f"{key}: its own colour was not found"
    assert not mask[0, 2:].any(), f"{key}: another colour counted as {target}"


def test_a_numpy_reply_works_too():
    assert extract_mask(np.array([[[255, 0, 0]]], dtype=np.uint8), (255, 0, 0)).tolist() == [[True]]
