"""What every provider takes and gives back, so the pipeline does not know which one ran.

A provider turns one RGB EM plane and a prompt into a mask of the same size.
How it gets there is its own business: an image model asked to paint the
structure (Vertex Gemini), a local stand-in (fake), or later a model that
takes the click itself as a point prompt (``click_px``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentRequest:
    """One plane to segment.

    ``image`` is exactly what is sent: the RGB EM plane at the resolution the
    prompt states. ``target_rgb`` is the colour the prompt asks for, and
    ``click_px`` the (row, col) the user pointed at in ``image``, or None.
    """

    image: Image.Image
    prompt: str
    target_rgb: tuple
    click_px: tuple | None = None


@dataclass
class SegmentResult:
    """A provider's answer.

    ``mask`` is bool with ``image``'s (height, width). ``model_image`` is
    what the model drew, resized to ``image``'s size, for the review preview
    (None for a provider that draws nothing). ``usage`` holds whatever token
    counts the provider reports, possibly none.
    """

    mask: np.ndarray
    model_image: Image.Image | None
    model: str
    usage: dict = field(default_factory=dict)


class Backend(Protocol):
    def segment(self, request: SegmentRequest, model: str) -> SegmentResult:
        """Segment ``request`` with ``model``, raising ``AIAnnotateError`` on failure."""
        ...
