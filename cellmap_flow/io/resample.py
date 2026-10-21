"""Reading an array resampled to another voxel size: worldarray's resampling.

When a model asks for a voxel size that a dataset has no level at,
``ImageDataInterface(..., on_voxel_size_mismatch="resample")`` opens it with
``worldarray.open(..., voxel_size=..., on_mismatch="resample")`` (its spec's
Behaviour 4): the target grid's voxel 0 starts at the source's corner, a whole
factor of 2 or more is a block mean (a block cut short by the data's end is
the mean of what it has) and any other a linear interpolation ("linear"), and
label data takes the source voxel the target centre is in ("nearest"). A
target voxel's value depends only on its index and the source array, so two
reads that meet agree with one read across both.

What stays here: which data is labels, which picks the method; worldarray has
no dtype heuristic.
"""

import numpy as np


def is_label_dtype(dtype) -> bool:
    """Whether data of ``dtype`` is resampled as labels (nearest voxel)
    rather than as intensities (mean and linear): bool, or an integer type
    of 32 bits or more."""
    dtype = np.dtype(dtype)
    return dtype == np.bool_ or (dtype.kind in "iu" and dtype.itemsize >= 4)
