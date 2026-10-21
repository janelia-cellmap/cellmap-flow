"""The model's painted reply to a bool mask."""

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate.mask_extraction import extract_mask
from cellmap_flow.ai_annotate.organelles import ORGANELLES


def _reply(colours):
    """A one-row image of the given RGB colours."""
    return Image.fromarray(np.array([colours], dtype=np.uint8))


@pytest.mark.parametrize("key", list(ORGANELLES))
def test_every_catalog_colour_can_be_found_and_neighbours_are_not(key):
    target = ORGANELLES[key].rgb
    near = tuple(int(np.clip(c + (-20 if c > 128 else 15), 0, 255)) for c in target)
    others = [p.rgb for p in ORGANELLES.values() if p.rgb != target]
    mask = extract_mask(_reply([target, near, (0, 0, 0), (255, 255, 255), *others]), target)
    assert mask[0, :2].all(), f"{key}: its own colour was not found"
    assert not mask[0, 2:].any(), f"{key}: another colour counted as {target}"
