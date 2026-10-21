"""Reading an array resampled to another voxel size: worldarray's resampling.

When a model asks for a voxel size that a dataset has no level at,
``ImageDataInterface(..., on_voxel_size_mismatch="resample")`` reads the
level ``io.multiscale.select_level(..., mode="resample")`` picks and
worldarray resamples it (``WorldArray.read(box, voxel_size=...)``, its spec's
Behaviour 4): the target grid's voxel 0 starts at the source's corner, there
are ``ceil(n·s/t)`` target voxels on an axis, a whole factor of 2 or more is a
block mean and any other a linear interpolation ("linear"), and label data
takes one source voxel ("nearest"; on a tie, the lower index). A target
voxel's value depends only on its index and the source array, so two reads
that meet agree with one read across both.

What stays here: which data is labels (``is_label_dtype``, which picks the
method; worldarray has no dtype heuristic), and the resampled shape and the
per-axis method names ImageDataInterface reports.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cellmap_flow.io.metadata import snap_integral


def is_label_dtype(dtype) -> bool:
    """Whether data of ``dtype`` is resampled as labels (nearest voxel)
    rather than as intensities (mean and linear): bool, or an integer type
    of 32 bits or more."""
    dtype = np.dtype(dtype)
    return dtype == np.bool_ or (dtype.kind in "iu" and dtype.itemsize >= 4)


def _whole_factor(factor: float):
    """``factor`` as an int when it is a whole number (float noise allowed), else None."""
    snapped = float(snap_integral([factor])[0])
    return int(snapped) if snapped.is_integer() else None


@dataclass(frozen=True)
class Resampling:
    """An array of ``source_shape`` voxels of ``source_voxel_size`` (nm,
    spatial axes), seen at ``target_voxel_size`` from the same corner."""

    source_voxel_size: Tuple[float, ...]
    target_voxel_size: Tuple[float, ...]
    source_shape: Tuple[int, ...]

    def factor(self, axis) -> float:
        """Source voxels per target voxel along ``axis``: above 1 is a downsampling."""
        return float(self.target_voxel_size[axis]) / float(self.source_voxel_size[axis])

    @property
    def shape(self) -> Tuple[int, ...]:
        """Target voxels per axis: enough to cover every source voxel."""
        extent = np.asarray(self.source_shape, dtype=float) / [self.factor(a) for a in range(len(self.source_shape))]
        return tuple(int(n) for n in np.ceil(snap_integral(extent)))

    def methods(self, labels=False) -> Tuple[str, ...]:
        """How each axis is resampled: "same", "mean", "linear" or "nearest"."""
        methods = []
        for axis in range(len(self.source_shape)):
            whole = _whole_factor(self.factor(axis))
            if whole == 1:
                methods.append("same")
            elif labels:
                methods.append("nearest")
            elif whole is not None and whole >= 2:
                methods.append("mean")
            else:
                methods.append("linear")
        return tuple(methods)
