"""The paint of one annotation volume carried onto another model's grid: the
maths of ``carry_paint`` (``finetune/session/volume.py`` says what it is for),
on arrays alone, so the server runs it on its zarrs and the browser on its
paint store's chunks (``portable/calls.py``).

Each new voxel takes the paint of the old voxels whose centres lie inside it
(one for a grid as fine as the old, several for a coarser one), or, where
none does (a finer grid), the old voxel its own centre lies in. A new voxel
whose old voxels were painted different labels is left unpainted, on a
boundary the coarser grid cannot draw; unpainted old voxels do not count
against it, so a stroke thinner than the new voxel still lands, a voxel
wide. Only the boxes of the old volume's painted chunks are visited.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class Grid:
    """An annotation volume's grid, z y x: voxel 0's lower corner in nm (not
    the OME translation, which is its centre: ``volume_corner_nm``), its voxel
    size in nm, and its shape and chunks in voxels."""

    corner_nm: tuple
    voxel_nm: tuple
    shape: tuple
    chunks: tuple


def axis_sources(old: Grid, new: Grid, axis: int):
    """Per voxel of ``new`` along ``axis``, the voxels of ``old`` it takes, as
    ``(first, end)`` arrays (end <= first where none lies in the old volume):
    those whose centres lie inside it, or, when none does (a finer grid), the
    one its own centre lies in."""
    fv, tv = float(old.voxel_nm[axis]), float(new.voxel_nm[axis])
    fc, tc = old.corner_nm[axis], new.corner_nm[axis]
    lo = tc + np.arange(new.shape[axis]) * tv
    eps = 1e-6
    first = np.ceil((lo - fc) / fv - 0.5 - eps).astype(np.int64)
    end = np.ceil((lo + tv - fc) / fv - 0.5 - eps).astype(np.int64)
    none = end <= first
    first[none] = np.floor((lo[none] + tv / 2 - fc) / fv).astype(np.int64)
    end[none] = first[none] + 1
    return np.maximum(first, 0), np.minimum(end, old.shape[axis])


def parts(chunks, lo, shape) -> Iterator[Tuple[tuple, tuple, tuple]]:
    """The chunks (of ``chunks`` voxels) the box ``lo`` + ``shape`` touches:
    each one's index, the box's part of it and the chunk's part of the box,
    both as slices."""
    first = [l // c for l, c in zip(lo, chunks)]
    last = [(l + n - 1) // c for l, n, c in zip(lo, shape, chunks)]
    for index in itertools.product(*(range(f, e + 1) for f, e in zip(first, last))):
        corner = [i * c for i, c in zip(index, chunks)]
        a = [max(l, k) for l, k in zip(lo, corner)]
        b = [min(l + n, k + c) for l, n, k, c in zip(lo, shape, corner, chunks)]
        in_chunk = tuple(slice(x - k, y - k) for x, y, k in zip(a, b, corner))
        in_box = tuple(slice(x - l, y - l) for x, y, l in zip(a, b, lo))
        yield index, in_chunk, in_box


def first_visit(visited: dict, chunks, lo, shape) -> np.ndarray:
    """The voxels of the box ``lo`` + ``shape`` not visited before, marking
    them visited in ``visited`` (a mask per chunk index of ``chunks``)."""
    fresh = np.zeros(shape, bool)
    for index, in_chunk, in_box in parts(chunks, lo, shape):
        seen = visited.setdefault(index, np.zeros(chunks, bool))
        fresh[in_box] = ~seen[in_chunk]
        seen[in_chunk] = True
    return fresh


def carry(old: Grid, new: Grid, painted: Iterable[Tuple[tuple, np.ndarray]], read: Callable[[list, list], np.ndarray],
          dtype) -> Tuple[Dict[tuple, np.ndarray], dict]:
    """Carry the paint of the old volume's ``painted`` chunks (each its index
    and its labels, a whole chunk) onto the new grid. ``read(lo, hi)`` gives
    the old labels over a box in old voxels (several chunks' worth, for a
    coarser new grid); ``dtype`` is the new volume's (its labels capped at
    its maximum).

    Returns the new chunks that got paint, by index, each whole (0 where the
    old paint says nothing: the caller writes them over what is there where
    they are not 0), and the counts ``{"painted", "source", "conflicts",
    "chunks"}``: the new voxels painted, the old painted voxels they came
    from, the new voxels left unpainted where their old voxels disagreed,
    and the new chunks returned.
    """
    maps = [axis_sources(old, new, a) for a in range(3)]
    cap = np.iinfo(dtype).max
    painted_voxels = source = conflicts = 0
    out: Dict[tuple, np.ndarray] = {}
    visited: dict = {}
    for index, chunk in painted:
        lo = [i * c for i, c in zip(index, old.chunks)]
        where = np.nonzero(chunk)
        if not where[0].size:
            continue
        source += where[0].size
        box_lo = [l + int(w.min()) for l, w in zip(lo, where)]
        box_hi = [l + int(w.max()) + 1 for l, w in zip(lo, where)]
        # The new voxels whose sources meet the painted box, along each axis.
        spans = []
        for (first, end), a, b in zip(maps, box_lo, box_hi):
            meets = np.flatnonzero((end > first) & (end > a) & (first < b))
            spans.append((int(meets[0]), int(meets[-1]) + 1) if meets.size else None)
        if None in spans:
            continue
        firsts = [m[0][s:e] for m, (s, e) in zip(maps, spans)]
        ends = [m[1][s:e] for m, (s, e) in zip(maps, spans)]
        read_lo = [int(f.min()) for f in firsts]
        read_hi = [int(e.max()) for e in ends]
        src = read(read_lo, read_hi)
        found = np.zeros([e - s for s, e in spans], dtype=src.dtype)
        clash = np.zeros(found.shape, bool)
        # Each new voxel's sources, an offset at a time along each axis.
        reach = [int((e - f).max()) for f, e in zip(firsts, ends)]
        for dz in range(reach[0]):
            for dy in range(reach[1]):
                for dx in range(reach[2]):
                    inside = [(f + d < e) for f, e, d in zip(firsts, ends, (dz, dy, dx))]
                    at = [np.minimum(f + d, e - 1) - r for f, e, d, r in zip(firsts, ends, (dz, dy, dx), read_lo)]
                    v = src[np.ix_(*at)]
                    valid = inside[0][:, None, None] & inside[1][None, :, None] & inside[2][None, None, :] & (v > 0)
                    clash |= valid & (found > 0) & (v != found)
                    found = np.where(valid & (found == 0), v, found)
        found[clash] = 0
        target_lo = [s for s, _ in spans]
        # Into the new chunks the span reaches, where the old paint says something,
        # the labels capped at the new dtype's maximum (a cap the old dtype cannot
        # reach is left out: numpy refuses an integer its dtype cannot hold).
        capped = found if cap >= np.iinfo(found.dtype).max else np.minimum(found, cap)
        for chunk_index, in_chunk, in_box in parts(new.chunks, target_lo, found.shape):
            part = capped[in_box]
            if not part.any():
                continue
            piece = out.setdefault(chunk_index, np.zeros(new.chunks, dtype))
            piece[in_chunk] = np.where(part > 0, part, piece[in_chunk]).astype(dtype)
        # A new voxel two old chunks' boxes reach (a coarser grid) is the
        # same either time, and counted once.
        fresh = first_visit(visited, new.chunks, target_lo, found.shape)
        painted_voxels += int(((found > 0) & fresh).sum())
        conflicts += int((clash & fresh).sum())
    return out, {"painted": painted_voxels, "source": int(source), "conflicts": conflicts, "chunks": len(out)}
