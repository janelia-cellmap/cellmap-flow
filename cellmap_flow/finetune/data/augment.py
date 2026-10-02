"""Training-time augmentation of a patch pair: flips, XY rotations, brightness and noise.

Brightness and noise act on the raw before it is normalized, spatial
transforms on the pair after: see Augmentation.intensity and .spatial.
Targets (affinities, SKOOTS vectors) are derived from the annotation by the
trainer, after this, so they are computed from the transformed labels and
stay consistent. Move target computation into the dataset and that stops
being true: a flip would then need the affinity channels permuted and
negated.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _intensity_range(arr: np.ndarray, normalizers=()) -> Tuple[Optional[float], Optional[float]]:
    """The value range intensity augmentation keeps raw inside, as (low, high).

    The first normalizer's own window when it is a MinMaxNormalizer -- that
    is the range the data is known to use, and the one it is clipped to next
    anyway. Otherwise the dtype's range for integers, and no bound at all for
    floats: signed and float raw keep their negative half, and uint16 data
    using 0-4000 gets noise of 1% of 4000, not of 65535.
    """
    first = normalizers[0] if normalizers else None
    if type(first).__name__ == "MinMaxNormalizer":
        low, high = float(first.min_value), float(first.max_value)
        if high > low:
            return low, high
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    return None, None


class Augmentation:
    """Draws each patch's augmentation, applies it, and says what it applied.

    ``intensity`` then ``spatial``, then ``report``, once per patch. Each
    DataLoader worker is its own (spawned) process, so the tallies never
    reach the parent: each worker reports its own, tagged, straight to the
    training log. Without that, the only evidence augmentation ran would be
    a flag echoed at startup.
    """

    def __init__(self, normalizers=()):
        # The raw's normalizers, whose first one may say its value range.
        self.normalizers = normalizers
        self._n = 0
        self._flips = np.zeros(3, dtype=np.int64)
        self._rots = np.zeros(4, dtype=np.int64)
        self._scales: list = []
        self._pending: dict = {}

    def intensity(self, patch: np.ndarray, rng) -> np.ndarray:
        """Random brightness scale (x0.8-x1.2) plus Gaussian noise (1% of range), as float32.

        On the raw as stored, before normalization: the scale and noise are
        in the raw dtype's own units (0-255 for uint8 EM), which is what they
        physically describe; on the normalized [-1, 1] signal a
        multiplicative factor would pull toward mid-grey instead of changing
        brightness. The range is the data's own (see _intensity_range), and
        the result is clipped to it; floats with no known range are not
        clipped, and their noise is scaled by the patch's own magnitude.
        """
        low, high = _intensity_range(patch, self.normalizers)
        if low is None:
            span = max(1.0, float(np.nanmax(np.abs(patch)))) if patch.size else 1.0
        else:
            span = high - low
        scale = rng.uniform(0.8, 1.2)
        noise = rng.normal(0.0, 0.01 * span, patch.shape)
        out = patch.astype(np.float32) * scale + noise
        if low is not None:
            out = np.clip(out, low, high)
        self._pending["scale"] = float(scale)
        self._pending["value_range"] = (low, high)
        return out.astype(np.float32)

    def spatial(self, raw: np.ndarray, ann: np.ndarray, rng):
        """Random flips, and XY rotations when the YX plane is square, of both patches alike.

        Raw and annotation are different sizes but share a centre, and for
        even-sized patches reflection about index ``(n-1)/2`` lands on the
        same physical plane for both -- so applying the identical transform
        to each keeps them registered.

        Rotation is skipped unless Y and X are equal in both patches, since
        the model's input shape is fixed and a non-square rot90 would change
        it. Z is never rotated into -- EM is routinely anisotropic there.
        """
        flips = [False, False, False]
        for axis in (0, 1, 2):
            if rng.random() < 0.5:
                raw = np.flip(raw, axis=axis)
                ann = np.flip(ann, axis=axis)
                flips[axis] = True

        k = 0
        rotatable = raw.shape[1] == raw.shape[2] and ann.shape[1] == ann.shape[2]
        if rotatable:
            k = int(rng.integers(0, 4))
            if k:
                raw = np.rot90(raw, k=k, axes=(1, 2))
                ann = np.rot90(ann, k=k, axes=(1, 2))

        self._pending["flips"] = flips
        self._pending["k"] = k
        self._pending["rotatable"] = rotatable

        # np.flip/np.rot90 return views; torch.from_numpy needs real strides.
        return np.ascontiguousarray(raw), np.ascontiguousarray(ann)

    def report(self) -> None:
        """Log what was applied: the first few patches in full, then rolling summaries.

        Runs inside whichever DataLoader worker produced the patch, so the
        line is tagged with the worker id -- several workers interleave in the
        log and otherwise the counts look contradictory.
        """
        p = self._pending
        if not p:
            return
        flips = p.get("flips", [False, False, False])
        k = p.get("k", 0)
        scale = p.get("scale")

        self._n += 1
        self._flips += np.array(flips, dtype=np.int64)
        self._rots[k] += 1
        if scale is not None:
            self._scales.append(scale)

        worker_info = torch.utils.data.get_worker_info()
        tag = f"aug w{0 if worker_info is None else worker_info.id}"

        # First few in full, so concrete values are visible immediately
        # rather than only after a summary interval.
        if self._n <= 3:
            applied = [f"flip{ax}" for ax, on in zip("ZYX", flips) if on]
            if k:
                applied.append(f"rot90_xy x{k}")
            if scale is not None:
                applied.append(f"brightness x{scale:.3f}")
            applied.append("noise sigma=1% of range")
            if not p.get("rotatable", True):
                applied.append("(rotation skipped: YX not square)")
            logger.info(f"[{tag}] patch {self._n}: " + ", ".join(applied))

        # Rolling summary: proves augmentation is still running deep into a
        # long job, and that the draws are distributed as intended.
        if self._n % 100 == 0:
            n = self._n
            fz, fy, fx = (100.0 * self._flips / n)
            rot = ", ".join(f"{i}:{100.0 * c / n:.0f}%" for i, c in enumerate(self._rots))
            if self._scales:
                s = np.array(self._scales)
                brightness = (
                    f"brightness mean {s.mean():.3f} "
                    f"range [{s.min():.3f}, {s.max():.3f}]"
                )
            else:
                brightness = "brightness n/a"
            logger.info(
                f"[{tag}] {n} patches augmented: flips Z {fz:.0f}% / "
                f"Y {fy:.0f}% / X {fx:.0f}% (expect ~50%), rot90_xy {rot} "
                f"(expect ~25% each), {brightness}"
            )

        self._pending = {}
