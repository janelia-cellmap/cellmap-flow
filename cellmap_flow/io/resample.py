"""Reading an array resampled to another voxel size, axis by axis.

When a model asks for a voxel size that a dataset has no level at,
``ImageDataInterface(..., on_voxel_size_mismatch="resample")`` reads the
level ``io.multiscale.select_level(..., mode="resample")`` picks and
resamples it here, so the model sees data at its own voxel size. The
default, "relabel", instead reads that level as if it were at the model's
voxel size, voxel for voxel.

**Geometry.** The resampled grid has voxels of the target size, and its
voxel 0 starts at the source's voxel 0's corner (the OME translation minus
half a source voxel; see ``io.geometry``), so both grids cover the data from
the same corner. On an axis with source voxels of ``s`` and target voxels of
``t``, target voxel ``j`` is centred at ``corner + (j + 1/2)·t``, which is
the source index ``p = (j + 1/2)·t/s − 1/2`` (source voxel ``i`` is centred
at index ``i``). There are ``ceil(n·s/t)`` target voxels, so they cover all
``n`` source voxels; the last may run past the data by less than a voxel.

**Values**, per axis, by the factor ``f = t/s``:

- ``f`` is 1: the source voxels as they are.
- ``f`` is a whole number of 2 or more: each target voxel is the mean of the
  ``f`` source voxels it covers exactly (a block or area mean). The last
  one, cut short by the end of the data, is the mean of those it has.
- otherwise (upsampling, or a factor that is not whole): linear
  interpolation at ``p`` between the two source voxels around it. Past the
  first or last voxel's centre, ``p`` is clamped to it, so the data's own
  border voxel is repeated rather than blended with padding.
- label data, whatever ``f``: the source voxel the target voxel's centre is
  in, so no label is invented by averaging two others. Data is taken as
  labels by its dtype (``is_label_dtype``): bool, or integers of 32 bits or
  more. EM intensities are uint8 or uint16, and segmentations uint32 or
  uint64.

The axes are resampled one after another. Each step works along its own
axis only, so the order does not matter. Integer intensity data is rounded
back to its dtype, as a stored pyramid level is, so the input chain sees
the dtype it would see at a real level.

**Chunks.** A target voxel's value depends only on its index and on the
source array, never on where a read starts or ends. A read takes, from the
source, the voxels its target voxels need (the read's margin), clamped to
the array. So two reads that meet give, at their shared border, exactly the
voxels one read across both gives.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cellmap_flow.io.geometry import Box
from cellmap_flow.io.metadata import snap_integral
from cellmap_flow.io.source import read_padded


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

    def read(self, store, box: Box, fill=0, transform=None) -> np.ndarray:
        """The target voxels of ``box`` (``Box`` of target indices, one entry per
        spatial axis), resampled from ``store`` (its spatial axes, voxel 0 at
        index 0, as ``io.source`` opens arrays), as a numpy array.

        Padding and ``transform`` work as in ``io.source.read_padded``: the
        target voxels outside the resampled array are ``fill``, or with
        "edge" its nearest voxel; ``transform`` (the input chain) is applied
        to the resampled voxels before the padding is added.
        """
        dtype = np.dtype(store.dtype.numpy_dtype)
        methods = self.methods(labels=is_label_dtype(dtype))
        edge = fill == "edge"
        wanted, pad_width = [], []
        for begin, end, n in zip(box.begin, box.end, self.shape):
            j = np.arange(begin, end)
            if edge:
                # The array's nearest voxel, for each one outside it.
                wanted.append(np.clip(j, 0, n - 1))
                pad_width.append((0, 0))
            else:
                inside = j[(j >= 0) & (j < n)]
                wanted.append(inside)
                before = int(np.count_nonzero(j < 0))
                pad_width.append((before, len(j) - len(inside) - before))

        # Resample the contiguous run of target voxels the box needs, then
        # pick the box's voxels from it (a voxel "edge" repeats appears twice).
        runs = [np.arange(w.min(), w.max() + 1) if len(w) else w for w in wanted]
        steps = [self._axis_step(axis, run, method) for axis, (run, method) in enumerate(zip(runs, methods))]
        source_box = Box(tuple(s[0] for s in steps), tuple(s[1] - s[0] for s in steps))
        data = read_padded(store, source_box)
        # float32 for uint8/uint16/float32 data, float64 for float64: exact
        # enough for both, at a quarter of float64's memory for uint8.
        work = np.result_type(dtype, np.float32)
        # Block means first: they shrink what the other steps work on.
        for axis in sorted(range(len(steps)), key=lambda axis: methods[axis] != "mean"):
            data = steps[axis][2](data, axis, work)
        for axis, (w, run) in enumerate(zip(wanted, runs)):
            if not np.array_equal(w, run):
                data = np.take(data, w - run[0], axis=axis)
        data = _as_dtype(data, dtype)
        if transform is not None:
            data = transform(data)
        if any(any(p) for p in pad_width):
            data = np.pad(data, pad_width=pad_width, mode="constant", constant_values=0 if edge else fill)
        return data

    def _axis_step(self, axis, run, method):
        """``(first, end, apply)``: the source voxels ``[first, end)`` that
        target voxels ``run`` (a contiguous range) need along ``axis``, and
        ``apply(data, axis, work)``, which turns those source voxels into
        them, computing means and interpolations in the float dtype ``work``."""
        n = self.source_shape[axis]
        if len(run) == 0:
            return 0, 0, lambda data, axis, work: data
        f = self.factor(axis)
        if method == "same":
            first, end = int(run[0]), int(run[-1]) + 1
            return first, end, lambda data, axis, work: data
        if method == "nearest":
            # The source voxel the target voxel's centre, (j + 1/2)·f in
            # source voxels from the corner, lies in.
            index = np.clip(np.floor(snap_integral((run + 0.5) * f)).astype(int), 0, n - 1)
            first = int(index.min())
            return first, int(index.max()) + 1, lambda data, axis, work: np.take(data, index - first, axis=axis)
        if method == "mean":
            whole = _whole_factor(f)
            starts = run * whole
            first, end = int(starts[0]), int(min(starts[-1] + whole, n))
            counts = np.minimum(starts + whole, end) - starts
            shape = [1] * len(self.source_shape)
            shape[axis] = len(run)

            def mean(data, axis, work):
                # Block i is data[i*whole : (i+1)*whole] along axis, so the
                # k-th voxel of every block is data[k::whole]: f strided adds,
                # about ten times faster than np.add.reduceat. A last block cut
                # short is missing its last few k.
                size = list(data.shape)
                size[axis] = len(run)
                sums = np.zeros(size, dtype=work)
                for k in range(whole):
                    part = data[_along(axis, slice(k, None, whole))]
                    sums[_along(axis, slice(0, part.shape[axis]))] += part
                return sums / counts.reshape(shape).astype(work)

            return first, end, mean
        # linear
        position = np.clip((run + 0.5) * f - 0.5, 0, n - 1)
        lower = np.floor(position).astype(int)
        upper = np.minimum(lower + 1, n - 1)
        weight = position - lower
        first = int(lower.min())
        shape = [1] * len(self.source_shape)
        shape[axis] = len(run)

        def linear(data, axis, work):
            below = np.take(data, lower - first, axis=axis).astype(work, copy=False)
            above = np.take(data, upper - first, axis=axis).astype(work, copy=False)
            w = weight.reshape(shape).astype(work)
            return below * (1 - w) + above * w

        return first, int(upper.max()) + 1, linear


def _along(axis, index):
    """An index that is ``index`` along ``axis`` and everything along the axes before it."""
    return (slice(None),) * axis + (index,)


def _as_dtype(data, dtype):
    """``data`` in ``dtype``: integers rounded to the nearest and kept in range."""
    if data.dtype == dtype:
        return data
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        data = np.clip(np.rint(data), info.min, info.max)
    return data.astype(dtype)
