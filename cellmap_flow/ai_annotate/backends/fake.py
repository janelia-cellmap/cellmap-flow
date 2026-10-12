"""A provider that runs locally and calls nothing, for tests and demos.

It stands in for a hosted model wherever the dashboard's own behaviour is
what is being checked: the review panel, Accept and Reject, the undo. Its
mask is deterministic and plausible on EM: the pixels darker than the
plane's median within a disk around the click, which on most EM picks out
membranes and dense organelles near where the user pointed.
"""

import numpy as np
from PIL import Image

from cellmap_flow.ai_annotate.backends.base import SegmentRequest, SegmentResult


class FakeBackend:
    def segment(self, request: SegmentRequest, model: str) -> SegmentResult:
        """Dark pixels within ``min(H, W) / 4`` of the click (or the centre),
        and the input with them painted ``target_rgb`` as the "model's" image."""
        grey = np.asarray(request.image.convert("L"), dtype=np.float32)
        height, width = grey.shape
        row, col = request.click_px if request.click_px is not None else (height / 2, width / 2)
        rows, cols = np.ogrid[:height, :width]
        disk = (rows - row) ** 2 + (cols - col) ** 2 <= (min(height, width) / 4) ** 2
        mask = disk & (grey < np.median(grey))

        painted = np.array(request.image.convert("RGB"))
        painted[mask] = request.target_rgb
        return SegmentResult(mask=mask, model_image=Image.fromarray(painted), model=model, usage={})
