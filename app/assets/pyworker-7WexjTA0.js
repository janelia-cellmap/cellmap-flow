var e=`"""Byte-bounded LRU cache for numpy chunks (and encoded bytes)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


def _nbytes(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 1


class LRUCache(Generic[T]):
    """Thread-safe LRU keyed by any hashable, bounded by total bytes.

    A single shared instance is normally used for every pipeline stage; keys embed the
    stage hash so that changing a parameter downstream never evicts upstream entries
    except through normal LRU pressure.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict[Hashable, T] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> T | None:
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                self.misses += 1
                return None
            self._data[key] = value
            self.hits += 1
            return value

    def __contains__(self, key: Hashable) -> bool:
        """Whether \`\`key\`\` is held, without counting a hit or making it recently used."""
        with self._lock:
            return key in self._data

    def put(self, key: Hashable, value: T) -> None:
        size = _nbytes(value)
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._data:
                self._bytes -= _nbytes(self._data.pop(key))
            self._data[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes and self._data:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= _nbytes(evicted)

    def invalidate(self, prefix: Hashable | None = None) -> int:
        """Drop entries; if \`\`prefix\`\` is given, only tuple keys starting with it."""
        with self._lock:
            if prefix is None:
                n = len(self._data)
                self._data.clear()
                self._bytes = 0
                return n
            doomed = [k for k in self._data if isinstance(k, tuple) and k[:1] == (prefix,)]
            for k in doomed:
                self._bytes -= _nbytes(self._data.pop(k))
            return len(doomed)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "entries": len(self._data),
            "bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
`,t=`"""Core value types shared by sources, ops, frontends and the server.

Conventions
-----------
* Arrays are handled in numpy C order. For a 3-D volume that means axes \`\`(z, y, x)\`\`
  with \`\`x\`\` varying fastest in memory. Frontends that use the opposite convention
  (N5, precomputed) reverse axis lists when emitting metadata / parsing chunk keys.
* \`\`Box\`\` is half-open \`\`[start, stop)\`\` in voxel coordinates of a given scale level.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Half-open voxel box \`\`[start, stop)\`\` in C-order axes."""

    start: tuple[int, ...]
    stop: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.start) != len(self.stop):
            raise ValueError("start and stop must have same length")

    @property
    def ndim(self) -> int:
        return len(self.start)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.start, self.stop))

    @property
    def empty(self) -> bool:
        return any(s <= 0 for s in self.shape)

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop))

    def pad(self, halo: Sequence[int]) -> Box:
        return Box(
            tuple(a - h for a, h in zip(self.start, halo)),
            tuple(b + h for b, h in zip(self.stop, halo)),
        )

    def clip(self, shape: Sequence[int]) -> Box:
        return Box(
            tuple(min(max(a, 0), s) for a, s in zip(self.start, shape)),
            tuple(min(max(b, 0), s) for b, s in zip(self.stop, shape)),
        )

    def relative_to(self, other: Box) -> Box:
        """Express this box in coordinates whose origin is \`\`other.start\`\`."""
        return Box(
            tuple(a - o for a, o in zip(self.start, other.start)),
            tuple(b - o for b, o in zip(self.stop, other.start)),
        )

    @staticmethod
    def from_chunk(index: Sequence[int], chunk_shape: Sequence[int]) -> Box:
        start = tuple(i * c for i, c in zip(index, chunk_shape))
        stop = tuple(s + c for s, c in zip(start, chunk_shape))
        return Box(start, stop)


KINDS = (None, "image", "label", "mask")


def kind_for_dtype(dtype) -> str | None:
    """What values of \`\`dtype\`\` most likely are when nothing says (a stored array): \`\`mask\`\`
    for booleans, \`\`label\`\` for integers of 32 bits or more (segment ids; intensities are
    rarely stored that wide), else unknown. Labels and masks are resampled by nearest voxel
    and downsampled by their most common value, never averaged."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return "mask"
    if dtype.kind in "iu" and dtype.itemsize >= 4:
        return "label"
    return None


@dataclass(frozen=True)
class ArrayInfo:
    """Static description of one scale level of a chunked array (C-order axes). \`\`kind\`\` is
    what its values are, for viewers choosing how to show it: \`\`image\`\` (intensities),
    \`\`label\`\` (segment ids) or \`\`mask\`\` (inside or not); \`\`None\`\` if unknown."""

    shape: tuple[int, ...]
    dtype: np.dtype
    chunk_shape: tuple[int, ...]
    voxel_size: tuple[float, ...]
    units: tuple[str, ...]
    axes: tuple[str, ...]
    translation: tuple[float, ...] = field(default=None)  # type: ignore[assignment]
    kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "chunk_shape", tuple(int(s) for s in self.chunk_shape))
        n = len(self.shape)
        if self.translation is None:
            object.__setattr__(self, "translation", (0.0,) * n)
        for name in ("chunk_shape", "voxel_size", "units", "axes", "translation"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have length {n}, got {getattr(self, name)}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunk_grid(self) -> tuple[int, ...]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunk_shape))

    def chunk_box(self, index: Sequence[int], clip: bool = True) -> Box:
        box = Box.from_chunk(index, self.chunk_shape)
        return box.clip(self.shape) if clip else box

    def chunks_covering(self, box: Box) -> Iterator[tuple[int, ...]]:
        """Yield chunk indices whose extent intersects \`\`box\`\` (assumed already clipped)."""
        if box.empty:
            return
        lo = [a // c for a, c in zip(box.start, self.chunk_shape)]
        hi = [(b - 1) // c for b, c in zip(box.stop, self.chunk_shape)]
        for idx in np.ndindex(*[h - lo_i + 1 for lo_i, h in zip(lo, hi)]):
            yield tuple(int(lo_i + i) for lo_i, i in zip(lo, idx))

    def with_(self, **changes) -> ArrayInfo:
        from dataclasses import replace

        return replace(self, **changes)

    def rescaled(self, voxel_size: Sequence[float]) -> ArrayInfo:
        """This array's extent on voxels of \`\`voxel_size\`\` along its last axes (as many as
        given): the shape rounded up, chunks as they were (in voxels), and the translation
        moved so each voxel's position is its centre, as OME-Zarr's is (a voxel twice as big
        covers two, its centre half an old voxel on)."""
        k = len(voxel_size)
        old, new = self.voxel_size[-k:], tuple(float(v) for v in voxel_size)
        shape = tuple(math.ceil(n * o / v - 1e-9) for n, o, v in zip(self.shape[-k:], old, new))
        shift = tuple(t + (v - o) / 2 for t, o, v in zip(self.translation[-k:], old, new))
        return self.with_(
            shape=self.shape[:-k] + shape,
            voxel_size=self.voxel_size[:-k] + new,
            translation=self.translation[:-k] + shift,
        )

    @staticmethod
    def default_axes(ndim: int) -> tuple[str, ...]:
        names = ("t", "c", "z", "y", "x")
        return names[-ndim:] if ndim <= 5 else tuple(f"d{i}" for i in range(ndim))
`,n=`"""Ops run back to back on one block: what a pipeline stage computes for a chunk.

Imports nothing but numpy and the ops, so the browser engine runs this same code in
Pyodide: the page reads the padded input, this computes the chunk.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops.base import Op


def _ratio(a: ArrayInfo, b: ArrayInfo, k: int) -> tuple[float, ...]:
    """Input voxels per output voxel along the last \`\`k\`\` axes, from \`\`a\`\`'s grid to \`\`b\`\`'s."""
    return tuple(float(w) / float(v) for v, w in zip(a.voxel_size[-k:], b.voxel_size[-k:]))


CHANNEL_AXES = ("c", "channel", "channels")


def _gridded(info: ArrayInfo) -> int:
    """How many trailing axes are chunked and padded: those after the last channel axis.
    A channel axis, and any axis before it, is read whole, so an op may change its length
    (select channels, combine them) or drop it."""
    last = max((a for a, name in enumerate(info.axes) if name in CHANNEL_AXES), default=-1)
    return info.ndim - last - 1


def _infos(info: ArrayInfo, ops: Sequence[Op]) -> tuple[list[ArrayInfo], int]:
    """Each op's output, and how many trailing axes every op keeps: those no op drops or
    adds, and none of them a channel axis."""
    infos, kept = [info], _gridded(info)
    for op in ops:
        infos.append(op.output_info(infos[-1]))
        kept = min(kept, _gridded(infos[-1]))
    return infos, kept


def scale(info: ArrayInfo, ops: Sequence[Op]) -> tuple[float, ...]:
    """Input voxels each output voxel spans along the axes \`\`ops\`\` keep: 1 for ops that keep
    the grid. Only a stage's first op may change it (\`\`downsample\`\`, or a model that reads 8
    nm voxels and writes 16 nm ones)."""
    infos, kept = _infos(info, ops)
    for i, (a, b) in enumerate(zip(infos[1:], infos[2:]), 1):
        if any(abs(r - 1) > 1e-9 for r in _ratio(a, b, kept)):
            raise ValueError(f"op {ops[i].name!r} changes the voxel size: only a stage's first op may")
    return _ratio(infos[0], infos[-1], kept) if ops else (1.0,) * kept


def plan(info: ArrayInfo, ops: Sequence[Op]) -> tuple[ArrayInfo, int, tuple[int, ...]]:
    """What \`\`ops\`\` make of input \`\`info\`\`: the output's info (its chunks still the input's),
    how many leading axes they consume (the channels of a \`\`stack://\`\` source, which are
    read whole), and the halo the axes they keep need, in input voxels: the sum of the ops',
    those after a first op that changes the grid counted in its output's voxels (\`\`scale\`\`).
    An op may also add leading axes (a model's channels; read whole too): the output's axes
    are those it added, then those kept. Planned on the finest level, the halo is the widest
    any level needs."""
    infos, kept = _infos(info, ops)
    out, r = infos[-1], scale(info, ops)
    lead = info.ndim - kept
    # each op as it runs on this level (an op in physical units counts its halo in its voxels)
    halos = [op.for_level(out).halo_for(kept) for op in ops]
    halo = tuple(
        int(np.ceil((halos[0][a] if halos else 0) + r[a] * sum(h[a] for h in halos[1:]) - 1e-9))
        for a in range(kept)
    )
    return out, lead, halo


def _whole(values, what: str) -> tuple[int, ...]:
    out = tuple(int(round(v)) for v in values)
    if any(abs(v - o) > 1e-6 for v, o in zip(values, out)):
        raise ValueError(f"{what} {tuple(values)} is not whole voxels of the input: choose chunks the voxel ratio divides")
    return out


def input_box(info: ArrayInfo, out_box: Box, lead: int, halo: Sequence[int],
              scale: Sequence[float] | None = None) -> Box:
    """The input to read for output box \`\`out_box\`\`: every leading axis, the kept ones
    (the last \`\`len(halo)\`\` of the output's), on the input's grid (\`\`scale\`\` input voxels
    to each output voxel), padded."""
    k = len(halo)
    r = scale or (1.0,) * k
    start = _whole([o * f for o, f in zip(out_box.start[-k:], r)], "an output chunk's start")
    stop = _whole([o * f for o, f in zip(out_box.stop[-k:], r)], "an output chunk's end")
    padded = Box(start, stop).pad(halo)
    return Box((0,) * lead + padded.start, info.shape[:lead] + padded.stop)


def run(ops: Sequence[Op], block: np.ndarray, in_box: Box, out_box: Box, out: ArrayInfo,
        halo: Sequence[int] | None = None, scale: Sequence[float] | None = None):
    """\`\`ops\`\` on \`\`block\`\` (the input over \`\`in_box\`\`), cropped to \`\`out_box\`\`. \`\`halo\`\` and
    \`\`scale\`\` are \`\`plan\`\`'s and \`\`scale\`\`'s: the halo's length is how many trailing axes the
    ops keep (by default all the output's). Each op returns its block as it came, or shaved
    by its own halo on every side (as a valid convolution does), and may drop leading axes
    or add some; a first op that changes the grid returns its output's voxels over either."""
    k = len(halo) if halo is not None else min(in_box.ndim, out.ndim)
    r = tuple(scale) if scale is not None else (1.0,) * k
    space = Box(in_box.start[-k:], in_box.stop[-k:])  # where the block sits in the kept axes
    box = in_box  # and in all the axes it has
    for i, op in enumerate(ops):
        level_op = op.for_level(out)
        result = level_op.apply_at(block, box)
        h = level_op.halo_for(k)
        got = result.shape[-k:] if result.ndim >= k else None
        if i == 0 and any(abs(f - 1) > 1e-9 for f in r):  # onto the output's grid
            for cut in (h, (0,) * k):  # shaved by its halo first, as a model's output is
                lo = [(a + c) / f for a, c, f in zip(space.start, cut, r)]
                hi = [(b - c) / f for b, c, f in zip(space.stop, cut, r)]
                if all(abs(v - round(v)) < 1e-6 for v in lo + hi) and got == tuple(round(b - a) for a, b in zip(lo, hi)):
                    space = Box(tuple(round(v) for v in lo), tuple(round(v) for v in hi))
                    break
            else:
                raise ValueError(
                    f"op {op.name!r} returned {result.shape} from {block.shape}: on a grid {r} times "
                    f"coarser, the block's voxels or those within its halo {h}"
                )
        elif got != block.shape[-k:]:
            if got != tuple(s - 2 * a for s, a in zip(block.shape[-k:], h)):
                raise ValueError(
                    f"op {op.name!r} changed block shape {block.shape} -> {result.shape}: it may "
                    f"drop or add leading axes, and shave its halo {h} off the last {k}"
                )
            space = space.pad([-a for a in h])  # a valid convolution: it returned the interior
        lead = result.shape[: result.ndim - k]  # leading axes are read whole
        same = lead == block.shape[: block.ndim - k]  # as they came, or made anew
        box = Box(box.start[: len(lead)] if same else (0,) * len(lead),
                  box.stop[: len(lead)] if same else tuple(lead))
        box = Box(box.start + space.start, box.stop + space.stop)
        block = result
    if block.ndim != out.ndim:
        raise ValueError(f"ops left a block of {block.ndim} axes for an output of {out.ndim}")
    crop = Box(out_box.start[-k:], out_box.stop[-k:]).relative_to(space)
    lead = Box(out_box.start[: out.ndim - k], out_box.stop[: out.ndim - k])
    return np.asarray(block[lead.slices() + crop.slices()], dtype=out.dtype)


def tent(i0: Sequence[int], i1: Sequence[int], c0: Sequence[int], c1: Sequence[int],
         overlap: int | Sequence[int]) -> np.ndarray:
    """A block's weight at the voxels \`\`i0:i1\`\` where blocks overlap: 1 on its own voxels
    \`\`c0:c1\`\`, falling linearly to \`\`1 / (overlap + 1)\`\` at the edge of its overlap and 0
    beyond, the product over the axes (\`\`overlap\`\` one for all, or per axis). Two
    neighbours' weights cross in the middle of their overlap, so their mean turns smoothly
    from one block's values to the other's instead of stepping where they meet."""
    k = len(i0)
    o = tuple(overlap) if isinstance(overlap, (tuple, list)) else (int(overlap),) * k
    weight = np.ones(())
    for a, b, lo, hi, ov in zip(i0, i1, c0, c1, o):
        idx = np.arange(a, b)
        d = np.maximum(np.maximum(lo - idx, idx - (hi - 1)), 0)
        weight = np.multiply.outer(weight, np.clip(1 - d / (ov + 1), 0, 1))
    return weight


def pad_edge(data: np.ndarray, box: Box, shape: Sequence[int]) -> np.ndarray:
    """\`\`data\`\`, read over \`\`box\`\` clipped to an array of \`\`shape\`\`, extended to all of
    \`\`box\`\` by repeating the nearest voxel inside (\`\`Source.read_padded(edge=True)\`\`)."""
    clipped = box.clip(shape)
    if clipped == box:
        return data
    inner = clipped.relative_to(box)
    return np.pad(data, [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)], "edge")
`,r=`"""Meshes computed when a viewer asks for them, in Neuroglancer's precomputed (legacy) mesh
format: one segment, \`\`1\`\`, whose fragments are the chunks of one level of a pipeline, each
meshed when it is fetched (the format names fragments in a manifest and leaves them to be
fetched one by one, so nothing is computed up front; its multi-resolution sibling needs
every fragment's byte offsets first).

    <dataset>/mesh/info        {"@type": "neuroglancer_legacy_mesh"}
    <dataset>/mesh/1:0         {"fragments": ["1:0:0_0_0", ...]}, every chunk of the level
    <dataset>/mesh/1:0:i_j_k   chunk (i, j, k) meshed: uint32 vertex count, float32 x, y, z
                               per vertex (nanometres), uint32 triangle corners

With \`\`lods\`\` above 1, a surface is served in Neuroglancer's multi-resolution format instead,
whose meshes get finer where the viewer zooms in:

    <dataset>/mesh/info        {"@type": "neuroglancer_multilod_draco", ...}
    <dataset>/mesh/1.index     the octree: per level of detail, the nodes and their sizes
    <dataset>/mesh/1           every node's fragment, one after another (HTTP Range requests)

Level of detail \`\`i\`\` is pyramid level \`\`level - lods + 1 + i\`\`, its nodes the chunks of that
level, so the coarsest is \`\`level\`\`'s chunks and each finer one splits them in eight. The
format lists every fragment's byte size before any is fetched, so each is padded to
\`\`FRAGMENT_BYTES\`\` (Draco decoders read what they need and ignore the rest) and its mesh
kept under \`\`MAX_TRIANGLES\`\`; nodes are listed only near the surface of the coarsest level
(\`\`multires_nodes\`\`), so the index stays small while the finer levels exist only where they
are fetched.

Two kinds. \`\`surface\`\`: the boundary of the voxels at or above \`\`threshold\`\`, by marching
cubes (scikit-image), from the chunk and one voxel more on its high sides so neighbouring
fragments meet, closed at the array's edges. \`\`terrain\`\`: an elevation model (\`\`y, x\`\`, or
\`\`z, y, x\`\` with one z) as a surface of two triangles per cell, the elevation times
\`\`exaggeration\`\` as height; cells with a NaN corner are left out. Imports nothing but
numpy, pydantic and scikit-image, so the browser engine's workers run it too.
"""

from __future__ import annotations

import struct
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from chunkmirage.core import ArrayInfo, Box

_TO_NM = {"": 1.0, "nm": 1.0, "nanometer": 1.0, "um": 1e3, "micrometer": 1e3, "µm": 1e3,
          "mm": 1e6, "millimeter": 1e6, "m": 1e9, "meter": 1e9, "km": 1e12, "kilometer": 1e12}  # fmt: skip
MAX_SIDE = 512  # the default level: the finest whose longest side is at most this
FRAGMENT_BYTES = 1 << 17  # every multi-resolution fragment, padded: sizes are listed up front
MAX_TRIANGLES = 100_000  # a fragment's mesh, coarsened above it: Draco takes under a byte each
BITS = 16  # multi-resolution vertex positions: integers in [0, 2**BITS) across a node
BAND = 2  # nodes are listed within this many coarsest-level voxels of its surface
MAX_NODES = 2_000_000  # an index's nodes (16 bytes each): fewer levels of detail past it


class MeshSpec(BaseModel):
    """What a dataset's \`\`mesh\`\` frontend meshes."""

    kind: Literal["surface", "terrain"] = Field(
        "surface", description="surface: an isosurface of the volume; terrain: an elevation model"
    )
    level: int | None = Field(
        None,
        ge=0,
        description=f"The level meshed; default the finest whose longest side is at most {MAX_SIDE}",
    )
    threshold: float = Field(128.0, description="surface: values at or above this are inside")
    exaggeration: float = Field(1.0, gt=0, description="terrain: the elevation's scale")
    lods: int = Field(
        1,
        ge=1,
        le=8,
        description="surface: levels of detail. 1 is one mesh of \`level\`; more serve "
        "Neuroglancer's multi-resolution format, \`level\` the coarsest and each further one a "
        "pyramid level finer, meshed where the viewer zooms in",
    )


def mesh_level(infos: list[ArrayInfo], spec: MeshSpec) -> int:
    if spec.level is not None:
        if spec.level >= len(infos):
            raise ValueError(f"mesh level {spec.level}: the dataset has {len(infos)} levels")
        return spec.level
    small = [i for i, info in enumerate(infos) if max(info.shape[-3:]) <= MAX_SIDE]
    return small[0] if small else len(infos) - 1


def fragment_names(info: ArrayInfo) -> list[str]:
    return ["1:0:" + "_".join(map(str, idx)) for idx in np.ndindex(*info.chunk_grid)]


def fragment_box(info: ArrayInfo, index) -> Box:
    """A fragment's voxels: its chunk and one more on each high side (clipped), so its
    surface meets its neighbours'."""
    box = info.chunk_box(index)
    return Box(box.start, tuple(min(b + 1, s) for b, s in zip(box.stop, info.shape))).clip(
        info.shape
    )


def _nm(info: ArrayInfo) -> np.ndarray:
    return np.array([v * _TO_NM.get(u, 1.0) for v, u in zip(info.voxel_size, info.units)])


def fragment(spec: MeshSpec, block: np.ndarray, box: Box, info: ArrayInfo) -> bytes:
    """Mesh \`\`block\`\` (the level's voxels over \`\`box\`\`), vertices in nanometres."""
    scale = _nm(info)
    origin = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(info.translation, info.units)])
    if spec.kind == "terrain":
        return _terrain(spec, block, box, info, scale, origin)
    mask = np.asarray(block) >= spec.threshold
    if mask.ndim != 3:
        raise ValueError(f"a surface mesh needs a z, y, x volume, not {mask.ndim} axes")
    # closed at the array's edges: one voxel of outside beyond them
    lo = [int(a == 0) for a in box.start]
    hi = [int(b == s) for b, s in zip(box.stop, info.shape)]
    padded = np.pad(mask, list(zip(lo, hi)))
    if not padded.any() or padded.all():
        return encode(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), 0.5)
    zyx = (verts - lo + np.array(box.start)) * scale[-3:] + origin[-3:]
    return encode(zyx[:, ::-1].astype(np.float32), faces[:, ::-1].astype(np.uint32))


def _terrain(spec, block, box: Box, info: ArrayInfo, scale, origin) -> bytes:
    z = np.asarray(block, dtype=np.float64)
    z = z.reshape(z.shape[-2:])  # y, x (a z of one dropped)
    h, w = z.shape
    y0, x0 = box.start[-2:]
    ys = (np.arange(y0, y0 + h) * scale[-2] + origin[-2])[:, None]
    xs = (np.arange(x0, x0 + w) * scale[-1] + origin[-1])[None, :]
    up = _TO_NM.get(info.units[-1], 1.0) * spec.exaggeration  # elevation in the grid's unit
    verts = np.stack(np.broadcast_arrays(xs, ys, np.nan_to_num(z) * up), -1).reshape(-1, 3)
    i = np.arange(h * w).reshape(h, w)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    ok = ~(
        np.isnan(z[:-1, :-1]) | np.isnan(z[:-1, 1:]) | np.isnan(z[1:, :-1]) | np.isnan(z[1:, 1:])
    )
    faces = np.concatenate([np.stack([a, c, b], -1)[ok], np.stack([b, c, d], -1)[ok]])
    return encode(verts.astype(np.float32), faces.astype(np.uint32))


def encode(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """A legacy mesh fragment: vertex count, the vertices, the triangles' corners (LE)."""
    return (
        struct.pack("<I", len(verts))
        + np.ascontiguousarray(verts, "<f4").tobytes()
        + np.ascontiguousarray(faces, "<u4").tobytes()
    )


# ------------------------------------------------------------------ multi-resolution
def lod_levels(infos: list[ArrayInfo], spec: MeshSpec) -> list[int]:
    """The pyramid level of each level of detail, finest first."""
    if spec.kind != "surface" and spec.lods > 1:
        raise ValueError("levels of detail are for surface meshes")
    top = mesh_level(infos, spec)
    if top - spec.lods + 1 < 0:
        raise ValueError(f"lods={spec.lods} from level {top}: the dataset has levels 0..{top} below it")
    return list(range(top - spec.lods + 1, top + 1))


def _morton(xyz: np.ndarray) -> np.ndarray:
    """Z-curve keys of (n, 3) x, y, z positions (x the lowest bit)."""
    key = np.zeros(len(xyz), np.uint64)
    v = xyz.astype(np.uint64)
    for b in range(21):
        for a in range(3):
            key |= ((v[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return key


def surface_band(mask: np.ndarray, core: Box | None = None) -> np.ndarray:
    """The voxels within \`\`BAND\`\` of the surface of \`\`mask\`\` (inside or not; beyond its edges
    is outside), cropped to \`\`core\`\` (where \`\`mask\`\` is a block with a \`\`BAND\`\` border, so the
    parts of a level can be done apart and joined)."""
    from scipy.ndimage import maximum_filter, minimum_filter

    m = np.asarray(mask).astype(np.uint8)
    size = 2 * BAND + 1  # separable passes: a cube's dilation and erosion
    band = maximum_filter(m, size, mode="constant", cval=0) > minimum_filter(m, size, mode="constant", cval=0)
    return band if core is None else band[core.slices()]


def multires_nodes(spec: MeshSpec, band: np.ndarray, infos: list[ArrayInfo], chunk) -> list[np.ndarray]:
    """Per level of detail (finest first), its nodes, (n, 3) chunk positions \`\`z, y, x\`\` of its
    level in Z-curve order: those that meet \`\`band\`\`, the voxels of the coarsest level near its
    surface (\`\`surface_band\`\`). A finer level's surface lies near the coarser one's, and a node
    listed but empty only costs its fetch."""
    levels = lod_levels(infos, spec)
    band = np.asarray(band, bool)
    chunk = np.asarray(chunk)
    out = []
    for i, level in enumerate(levels):
        scale = 2 ** (len(levels) - 1 - i)  # this level's voxels per coarsest voxel
        grid = -(-np.asarray(infos[level].shape[-3:]) // chunk)
        if (chunk % scale == 0).all():  # a node is a block of whole coarsest voxels
            b = chunk // scale
            pad = np.zeros(tuple(grid * b), bool)
            pad[tuple(slice(0, min(n, g)) for n, g in zip(band.shape, grid * b))] = band[tuple(slice(0, g) for g in grid * b)]
            nodes = np.argwhere(pad.reshape(grid[0], b[0], grid[1], b[1], grid[2], b[2]).any((1, 3, 5)))
        else:  # nodes smaller than a coarsest voxel: every node each band voxel covers
            f = scale // chunk
            at = np.argwhere(band)
            offsets = np.array(np.meshgrid(*[np.arange(n) for n in f], indexing="ij")).reshape(3, -1).T
            nodes = (at[:, None, :] * f + offsets[None]).reshape(-1, 3)
            nodes = nodes[(nodes < grid).all(1)]
        out.append(nodes[np.argsort(_morton(nodes[:, ::-1]), kind="stable")])
    if (n := sum(len(o) for o in out)) > MAX_NODES:
        raise ValueError(f"lods={spec.lods} lists {n} mesh nodes, over {MAX_NODES}: ask for fewer levels of detail")
    return out


def multires_index(nodes: list[np.ndarray], infos: list[ArrayInfo], levels: list[int], chunk) -> bytes:
    """The \`\`1.index\`\` manifest: nodes sized \`\`chunk\`\` voxels of each level (nanometres, x y z),
    each level's origin offset from the finest's, every fragment \`\`FRAGMENT_BYTES\`\`."""
    first = infos[levels[0]]
    voxel = _nm(first)[-3:]
    origin = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(first.translation, first.units)])[-3:]
    shape = np.asarray(chunk, float) * voxel
    lod_scales = [float(_nm(infos[lv])[-3:].min()) for lv in levels]
    offsets = []
    for lv in levels:
        i = infos[lv]
        o = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(i.translation, i.units)])[-3:]
        offsets.extend((o - origin)[::-1])
    out = struct.pack("<3f", *shape[::-1]) + struct.pack("<3f", *origin[::-1]) + struct.pack("<I", len(levels))
    out += struct.pack(f"<{len(levels)}f", *lod_scales) + struct.pack(f"<{3 * len(levels)}f", *offsets)
    out += struct.pack(f"<{len(levels)}I", *[len(n) for n in nodes])
    for n in nodes:
        out += np.ascontiguousarray(n[:, ::-1].T, "<u4").tobytes() + np.full(len(n), FRAGMENT_BYTES, "<u4").tobytes()
    return out


def multires_info() -> dict:
    return {"@type": "neuroglancer_multilod_draco", "vertex_quantization_bits": BITS,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "lod_scale_multiplier": 1}


def node_box(info: ArrayInfo, node, chunk) -> Box:
    """A node's voxels: its chunk of the level and one more on its high sides (clipped)."""
    start = tuple(int(n) * c for n, c in zip(node, chunk))
    stop = tuple(min(s + c + 1, n) for s, c, n in zip(start, chunk, info.shape))
    return Box(start, stop)


def multires_fragment(spec: MeshSpec, block: np.ndarray, box: Box, info: ArrayInfo, chunk) -> tuple[np.ndarray, np.ndarray]:
    """A node's mesh as the format stores it: vertices as integers across the node (x, y, z,
    \`\`[0, 2**BITS)\`\`) and triangles, marching cubes run on each octant apart so no triangle
    crosses one (as coarser levels must be), coarsened past \`\`MAX_TRIANGLES\`\`."""
    from skimage.measure import marching_cubes

    mask = np.asarray(block) >= spec.threshold
    lo = [int(a == 0) for a in box.start]
    hi = [int(b == s) for b, s in zip(box.stop, info.shape)]
    padded = np.pad(mask, list(zip(lo, hi)))  # closed at the array's edges
    chunk = np.asarray(chunk)
    half = chunk // 2
    for step in (1, 2, 4, 8):
        verts, faces, n = [], [], 0
        for octant in np.ndindex(2, 2, 2):
            o = np.array(octant) * half
            stop = np.where(octant, chunk, half) + 1
            # in the padded block, node voxel j is at j + lo
            sub = padded[tuple(slice(a + e, b + e) for a, b, e in zip(o, stop, lo))]
            if min(sub.shape) < 2 or sub.all() or not sub.any():
                continue
            v, f, _, _ = marching_cubes(sub.astype(np.float32), 0.5, step_size=step)
            verts.append(v + o)
            faces.append(f + n)
            n += len(v)
        if not verts:
            return np.zeros((0, 3), np.uint32), np.zeros((0, 3), np.uint32)
        f = np.concatenate(faces)
        if len(f) <= MAX_TRIANGLES or step == 8:
            break
    v = np.concatenate(verts)  # node coordinates: the node's voxel 0 at 0
    q = np.clip(np.rint(v / chunk * (2**BITS - 1)), 0, 2**BITS - 1).astype(np.uint32)
    return np.ascontiguousarray(q[:, ::-1]), np.ascontiguousarray(f[:, ::-1].astype(np.uint32))


def encode_draco(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """A fragment as Neuroglancer reads it: Draco, its integer positions kept as they are
    (quantized to \`\`BITS\`\` over exactly \`\`[0, 2**BITS - 1]\`\`), padded to \`\`FRAGMENT_BYTES\`\`.
    Needs DracoPy (the \`\`mesh\`\` extra); the browser engine encodes with Draco's own wasm."""
    import DracoPy

    if not len(faces):
        return b"\\0" * FRAGMENT_BYTES
    data = DracoPy.encode(verts, faces, quantization_bits=BITS, quantization_range=2**BITS - 1,
                          quantization_origin=[0, 0, 0], compression_level=7)
    if len(data) > FRAGMENT_BYTES:
        raise ValueError(f"a mesh fragment took {len(data)} bytes, over {FRAGMENT_BYTES}")
    return data + b"\\0" * (FRAGMENT_BYTES - len(data))
`,i=`"""Ops: pure functions on blocks with declared halo, output dtype and cacheability.

The core has two: \`\`downsample\`\`, which the pipeline itself runs to make the coarser levels
of an op bound to one resolution, and \`\`python\`\`, code carried in the spec. Every other op
(filters, thresholds, detectors, terrain) is in \`\`chunkmirage_contrib.ops\`\`, found through
the \`\`chunkmirage.ops\`\` entry point as any other package's ops are.
"""

from chunkmirage.ops.base import Op, get_op, list_ops, op_from_spec, register
from chunkmirage.ops.code import PythonOp, python_op
from chunkmirage.ops.downsample import Downsample

__all__ = [
    "Downsample",
    "Op",
    "PythonOp",
    "get_op",
    "list_ops",
    "op_from_spec",
    "python_op",
    "register",
]
`,a=`from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from chunkmirage.core import ArrayInfo

log = logging.getLogger("chunkmirage")


class Op(BaseModel):
    """A block-wise operation.

    Subclass, set \`\`name\`\`, declare parameters as pydantic fields, and implement \`\`apply\`\`.

    * \`\`halo\`\`: voxels of upstream context needed on every side (int or per-axis tuple).
      The framework reads the padded region, calls \`\`apply\`\` on it, and crops the result.
    * \`\`cache\`\`: whether this stage's output chunks should be memoized. Turn on for
      expensive stages (inference) so cheap downstream tweaks (threshold) are free. A
      pipeline can override it per op: \`\`{"op": "gaussian", "sigma": 4, "cache": true}\`\`.
    * \`\`output_dtype\`\` / \`\`output_info\`\`: describe the result; default is unchanged.
    * \`\`output_kind\`\`: what the result's values are (\`\`ArrayInfo.kind\`\`): \`\`image\`\`,
      \`\`label\`\` or \`\`mask\`\`; \`\`None\`\` (the default) if the op does not say.
    * \`\`cache_token()\`\`: what the result depends on beyond the parameters (a file's
      modification time, weights replaced in place); part of the op's identity.
    * \`\`slots\`\`: at most this many \`\`apply\`\` calls of this op at once, process-wide (a GPU,
      a memory-hungry step), queued finest level first; \`\`None\`\` (the default): no limit.
    * \`\`overlap\`\`: output voxels beyond its chunk that each block of this op also computes
      (int or per-axis tuple), for ops whose blocks computed on their own disagree where
      they meet (a fit per block, a model's tiles). The blocks, chunk plus overlap, are
      cached as a stage of their own, and a served chunk is the weighted mean of every block
      reaching it (\`\`fused.tent\`\`): smooth across chunk edges, and the same whatever order
      the blocks are computed in. The op runs as a stage of its own.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ClassVar[str] = ""
    halo: ClassVar[int | tuple[int, ...]] = 0
    overlap: ClassVar[int | tuple[int, ...]] = 0
    cache: ClassVar[bool] = False
    # packages its apply imports beyond numpy (the browser engine loads them up front)
    packages: ClassVar[tuple[str, ...]] = ()
    output_kind: ClassVar[str | None] = None
    slots: ClassVar[int | None] = None
    _cache: bool | None = PrivateAttr(None)  # this op's own setting, over the class's

    @property
    def cached(self) -> bool:
        """Whether this op's output is memoized: its own setting, else its class's."""
        return self.cache if self._cache is None else self._cache

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        h = self.halo
        return tuple(h) if isinstance(h, (tuple, list)) else (int(h),) * ndim

    def overlap_for(self, ndim: int) -> tuple[int, ...]:
        """The overlap per axis, from \`\`overlap\`\`, along the last \`\`ndim\`\` axes."""
        o = self.overlap
        return tuple(o) if isinstance(o, (tuple, list)) else (int(o),) * ndim

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype(in_dtype)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        return info.with_(dtype=self.output_dtype(info.dtype), kind=self.output_kind)

    def apply(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def input_voxel_size(self) -> tuple[float, ...] | None:
        """The voxel size this op must read, along the data's last axes and in the source's
        units: a model trained at one resolution. The pipeline then runs it on one level (the
        source's at that size, else one resampled to it), caches what it makes, and makes the
        coarser levels by downsampling that. \`\`None\`\` (the default): it runs on every level."""
        return None

    def for_level(self, info: ArrayInfo) -> Op:
        """This op as it runs on a scale level described by \`\`info\`\` (the stage's output, on
        the input's grid). Ops that work in physical units override it to take the level's
        voxel size: a slope in degrees needs the pixel spacing, which doubles from level to
        level. The default is the op itself."""
        return self

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        """Like \`\`apply\`\` but told where \`\`block\`\` sits (\`\`box\`\` = its halo-padded extent in
        voxels of this scale level). Override when the result must depend on position, e.g. to
        make per-chunk labels globally unique. Default delegates to \`\`apply\`\`."""
        return self.apply(block)

    # --- identity / serialization -------------------------------------------------
    def spec(self) -> dict[str, Any]:
        spec = {"op": self.name, **self.model_dump(mode="json")}
        if self._cache is not None:
            spec["cache"] = self._cache
        return spec

    def cache_token(self) -> str | None:
        """What this op's output depends on outside its parameters, as a string that changes
        when that does: a weights file's modification time, a model version. \`\`None\`\` (the
        default): nothing. It is part of \`\`digest\`\`, so a pipeline built after it changes has
        new stage keys and new links, and viewers refetch; \`\`DatasetRegistry.refresh\`\`
        rebuilds a served dataset to read it again."""
        return None

    def digest(self) -> str:
        """Identity of what the op computes: whether it is cached does not change that."""
        identity = {"op": self.name, **self.model_dump(mode="json")}
        token = self.cache_token()
        if token is not None:
            identity["@cache_token"] = str(token)
        payload = json.dumps(identity, sort_keys=True, default=str).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


_REGISTRY: dict[str, type[Op]] = {}
_ENTRYPOINTS_LOADED = False


def register(cls: type[Op]) -> type[Op]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a class-level \`name\`")
    _REGISTRY[cls.name] = cls
    return cls


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    for ep in entry_points(group="chunkmirage.ops"):
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, Op):
                _REGISTRY.setdefault(cls.name or ep.name, cls)
        except Exception:  # noqa: BLE001 - a broken plugin must not take the server down
            log.warning("op plugin %r (%s) failed to load; skipped", ep.name, ep.value, exc_info=True)


def get_op(name: str) -> type[Op]:
    _load_entrypoints()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown op {name!r}; known: {sorted(_REGISTRY)}") from None


def list_ops() -> dict[str, dict]:
    _load_entrypoints()
    return {
        name: {
            "halo": "dynamic" if isinstance(cls.halo, property) else cls.halo,
            "overlap": "dynamic" if isinstance(cls.overlap, property) else cls.overlap,
            "cache": cls.cache,
            "slots": cls.slots,
            "doc": (cls.__doc__ or "").strip(),
            "schema": cls.model_json_schema(),
        }
        for name, cls in sorted(_REGISTRY.items())
    }


def op_from_spec(spec: dict[str, Any] | Op) -> Op:
    if isinstance(spec, Op):
        return spec
    spec = dict(spec)
    name = spec.pop("op")
    cache = spec.pop("cache", None)
    op = get_op(name)(**spec)
    if cache is not None:
        op._cache = bool(cache)
    return op


def ops_from_specs(specs: Sequence[dict[str, Any] | Op]) -> list[Op]:
    return [op_from_spec(s) for s in specs]
`,o=`"""\`\`python\`\`: an op whose code is part of the pipeline spec, edited live as a Neuroglancer
shader is. Lines starting \`\`#halo\`\`, \`\`#output\`\` and \`\`#uicontrol\`\` say what it needs and
makes; they are read as text, so a pipeline is planned without running the code."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register

KINDS = ("image", "label", "mask")

TEMPLATE = '''\\
# Python run on each block of this view as it is fetched; edit it and the view updates.
# Lines starting with these are read before it runs:
#   #halo 2                 voxels of context apply needs on each side
#   #output uint32 label    the dtype it returns, then image, label or mask (default: as given)
#   #uicontrol float low slider(min=0, max=255, default=128)
#                           a slider, as in Neuroglancer's shaders; apply gets its value
# numpy is np; scipy and scikit-image can be imported.
def apply(block):
    return block
'''


class Control(BaseModel):
    """A \`\`#uicontrol\`\` line: a slider (\`\`float\`\`, \`\`int\`\`) or a checkbox (\`\`bool\`\`)."""

    name: str
    type: Literal["float", "int", "bool"]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: float | int | bool

    def cast(self, value: Any) -> float | int | bool:
        return {"float": float, "int": int, "bool": bool}[self.type](value)


class Directives(BaseModel):
    halo: tuple[int, ...] = (0,)
    dtype: str | None = None
    kind: str | None = None
    controls: list[Control] = []


_CONTROL = re.compile(r"(\\w+)\\s+([A-Za-z_]\\w*)\\s+(\\w+)\\s*\\((.*)\\)\\s*$")


def _number(text: str, line: int) -> float | bool:
    if text in ("true", "false"):
        return text == "true"
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"line {line}: {text!r} is not a number") from None


def _control(rest: str, line: int) -> Control:
    m = _CONTROL.match(rest)
    if not m:
        raise ValueError(f"line {line}: #uicontrol takes \`type name slider(min=..., max=..., default=...)\`")
    kind, name, widget, args = m.groups()
    values = {}
    for part in filter(None, (p.strip() for p in args.split(","))):
        key, eq, value = part.partition("=")
        if not eq:
            raise ValueError(f"line {line}: #uicontrol arguments are key=value, not {part!r}")
        values[key.strip()] = _number(value.strip(), line)
    if name in ("block", "box"):
        raise ValueError(f"line {line}: #uicontrol {name!r}: apply gets that name itself")
    if name in _reserved():
        raise ValueError(f"line {line}: #uicontrol {name!r}: the op has that name itself; pick another")
    if kind in ("float", "int") and widget == "slider":
        if "min" not in values or "max" not in values:
            raise ValueError(f"line {line}: a slider needs min and max")
        values.setdefault("default", values["min"])
        if kind == "int":
            values.setdefault("step", 1)
        if values["min"] > values["max"]:
            raise ValueError(f"line {line}: a slider's min is above its max")
        if not values["min"] <= values["default"] <= values["max"]:
            raise ValueError(f"line {line}: a slider's default is outside min..max")
        if "step" in values and values["step"] <= 0:
            raise ValueError(f"line {line}: a slider's step must be above 0")
    elif kind == "bool" and widget == "checkbox":
        values.setdefault("default", False)
    else:
        raise ValueError(f"line {line}: #uicontrol {kind} {widget}: use float or int slider, or bool checkbox")
    unknown = set(values) - {"min", "max", "step", "default"}
    if unknown:
        raise ValueError(f"line {line}: #uicontrol takes min, max, step and default, not {sorted(unknown)}")
    c = Control(name=name, type=kind, **values)
    return c.model_copy(update={"default": c.cast(c.default)})


@lru_cache(maxsize=1)
def _reserved() -> frozenset[str]:
    """Names a \`\`#uicontrol\`\` may not take: the op's own attributes and fields, which a
    plugin made of the code (\`\`python_op\`\`) would otherwise shadow."""
    return frozenset(dir(_Coded)) | frozenset(PythonOp.model_fields)


MAX_HALO = 1024


@lru_cache(maxsize=256)
def directives(code: str) -> Directives:
    """What the \`\`#\`\` lines of \`\`code\`\` say (lines starting at the first column, so an
    indented comment in \`\`apply\`\` is not one); \`\`ValueError\`\` naming the line if one is
    wrong."""
    found: dict[str, Any] = {"controls": []}
    for n, raw in enumerate(code.splitlines(), 1):
        line = raw.strip()
        word = line.split(maxsplit=1)[0] if raw.startswith("#") else ""
        rest = line[len(word):].strip()
        if word in ("#halo", "#output") and word[1:] in found:
            raise ValueError(f"line {n}: a second {word}")
        if word == "#halo":
            try:
                found["halo"] = tuple(int(v) for v in rest.split())
            except ValueError:
                raise ValueError(f"line {n}: #halo takes whole voxels, one or one per axis") from None
            if not found["halo"] or min(found["halo"]) < 0:
                raise ValueError(f"line {n}: #halo takes whole voxels, one or one per axis")
            if max(found["halo"]) > MAX_HALO:
                raise ValueError(f"line {n}: #halo above {MAX_HALO} voxels: every chunk would read that much more")
        elif word == "#output":
            found["output"] = True
            for token in rest.split():
                if token in KINDS:
                    found["kind"] = token
                    continue
                try:
                    dt = np.dtype(token)
                except TypeError:
                    raise ValueError(f"line {n}: #output {token!r}: a numpy dtype, or one of {KINDS}") from None
                if dt.kind not in "uif":
                    raise ValueError(f"line {n}: #output {token!r}: an integer or float dtype")
                found["dtype"] = dt.name
        elif word == "#uicontrol":
            c = _control(rest, n)
            if any(o.name == c.name for o in found["controls"]):
                raise ValueError(f"line {n}: a second #uicontrol {c.name!r}")
            found["controls"].append(c)
    found.pop("output", None)
    return Directives(**found)


def check(code: str) -> None:
    """\`\`code\`\` parses and defines \`\`apply(block, ...)\`\` taking its controls; checked without
    running it."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"line {e.lineno}: {e.msg}") from None
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply"]
    if not defs:
        raise ValueError("the code defines no \`def apply(block, ...)\`")
    fn = defs[-1]  # the last def is what runs
    for later in tree.body[tree.body.index(fn) + 1:]:  # ... unless the name is rebound after it
        targets = getattr(later, "targets", [getattr(later, "target", None)])
        if isinstance(later, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)) and any(
            isinstance(t, ast.Name) and t.id == "apply" for t in targets
        ):
            raise ValueError(f"line {later.lineno}: apply is no longer the function after this")
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if not names and a.vararg is None:
        raise ValueError(f"line {fn.lineno}: apply takes the block first")
    controls = {c.name for c in directives(code).controls}
    for c in controls:
        if c not in names and a.kwarg is None:
            raise ValueError(f"line {fn.lineno}: apply has no parameter {c!r} for its #uicontrol")
        if names and c == names[0]:
            raise ValueError(f"line {fn.lineno}: {c!r} is apply's first parameter, the block, and a #uicontrol")
    # every other parameter is a slider's, box, or has a default: nothing else is passed
    positional = a.posonlyargs + a.args
    defaulted = {x.arg for x in positional[len(positional) - len(a.defaults):]} | {
        x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None
    }
    for x in names[1:]:
        if x not in controls and x != "box" and x not in defaulted:
            raise ValueError(f"line {fn.lineno}: apply's parameter {x!r} has no #uicontrol and no default")


@lru_cache(maxsize=4)  # a few versions, while an edit is in flight; each holds its module's namespace
def _compiled(code: str):
    """\`\`apply\`\` from \`\`code\`\`, run once per process, and whether it takes \`\`box\`\`."""
    ns: dict[str, Any] = {"np": np, "__name__": "chunkmirage_python_op"}
    exec(compile(code, "<python op>", "exec"), ns)  # noqa: S102 - the point of this op; gated by the server
    fn = ns["apply"]
    params = fn.__code__.co_varnames[: fn.__code__.co_argcount + fn.__code__.co_kwonlyargcount]
    return fn, "box" in params


class _Coded(Op):
    """What an op made from code has: halo, output and \`\`apply\`\` from its code's text."""

    def _code(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _values(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def directives(self) -> Directives:
        return directives(self._code())

    @property
    def halo(self):  # type: ignore[override]
        return self.directives.halo

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        h = self.directives.halo
        if len(h) == 1:
            return h * ndim
        if len(h) > ndim:
            raise ValueError(f"#halo gives {len(h)} axes; the data has {ndim}")
        return (0,) * (ndim - len(h)) + h  # the last axes: z y x of c z y x

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        d = self.directives.dtype
        return np.dtype(d) if d else np.dtype(in_dtype)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        return info.with_(dtype=self.output_dtype(info.dtype), kind=self.directives.kind or info.kind)

    def apply(self, block: np.ndarray) -> np.ndarray:
        return self.apply_at(block, None)

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        fn, wants_box = _compiled(self._code())
        out = fn(block, **self._values(), **({"box": box} if wants_box else {}))
        return np.asarray(out).astype(self.output_dtype(block.dtype), copy=False)


@register
class PythonOp(_Coded):
    """Python of your own, carried in the spec: \`\`apply(block, ...)\`\` returns the block
    computed (or its interior, shaved by the halo). \`\`#halo\`\`, \`\`#output\`\` and
    \`\`#uicontrol\`\` lines say what it reads and makes; a server runs it only when started
    with \`\`--allow-code\`\`."""

    name = "python"
    code: str = Field(
        TEMPLATE,
        description="Python defining apply(block, ...), with #halo, #output and #uicontrol lines.",
        json_schema_extra={"format": "python"},
    )
    params: dict[str, float | int | bool] = Field(
        default_factory=dict, description="Values of the code's #uicontrol sliders, over their defaults."
    )

    @field_validator("code")
    @classmethod
    def _valid(cls, v: str) -> str:
        directives(v)
        check(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def _current(cls, data: Any) -> Any:
        """Values of sliders the code no longer has dropped: edited live, the code renames
        and removes them, and the value of one gone is no part of what the op computes."""
        code = data.get("code", TEMPLATE) if isinstance(data, dict) else None
        if isinstance(code, str) and isinstance(data.get("params"), dict):
            try:
                names = {c.name for c in directives(code).controls}
            except ValueError:
                return data  # the code's own check says what is wrong
            data = {**data, "params": {k: v for k, v in data["params"].items() if k in names}}
        return data

    def _code(self) -> str:
        return self.code

    def _values(self) -> dict[str, Any]:
        return {c.name: c.cast(self.params.get(c.name, c.default)) for c in self.directives.controls}


def python_op(name: str, code: str, doc: str | None = None) -> type[Op]:
    """An op class from \`\`code\`\` written for the \`\`python\`\` op: its \`\`#uicontrol\`\` lines
    become its parameters, so it is a plugin like any other (register it through the
    \`\`chunkmirage.ops\`\` entry point, or \`\`register\`\` it)."""
    check(code)
    controls = directives(code).controls
    body: dict[str, Any] = {
        "__doc__": doc or f"\`\`{name}\`\`, from code written for the \`\`python\`\` op.",
        "__module__": __name__,
        "__annotations__": {c.name: {"float": float, "int": int, "bool": bool}[c.type] for c in controls},
        "name": name,
        "_code": lambda self: code,
        "_values": lambda self: {c.name: getattr(self, c.name) for c in controls},
    }
    for c in controls:
        body[c.name] = Field(c.default, **({} if c.type == "bool" else {"ge": c.min, "le": c.max}))
    return type("".join(w.title() for w in name.split("_")), (_Coded,), body)
`,s=`"""Coarser voxels, as a stored pyramid level is made: the one op the engine itself runs, to
make the coarser levels of an op bound to one resolution (\`\`chunkmirage.pipeline\`\`). It
imports nothing beyond numpy, so the browser engine runs it too."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Downsample(Op):
    """Coarser voxels: each block of \`\`factor\`\` voxels becomes one, their mean, or for labels
    and masks their most common value. The grid changes with it: voxels \`\`factor\`\` times
    bigger, the shape divided (rounded up), each voxel's position the centre of its block.
    A pipeline makes the coarser levels of an op with an input voxel size this way."""

    name = "downsample"
    factor: list[int] = Field(
        [2, 2, 2],
        description="Voxels per output voxel along each of the data's last axes (z, y, x): "
        "2, 2, 2 halves each; 1 keeps an axis as it is.",
    )
    mode: str = Field(
        "auto",
        pattern="^(auto|mean|mode)$",
        description="mean of each block; mode, its most common value (labels, masks); auto: "
        "mode for labels and masks, mean for anything else.",
    )
    _mode: str = PrivateAttr("mean")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        f = self.factor
        if not f or len(f) > info.ndim or any(int(v) < 1 for v in f):
            raise ValueError(f"downsample factor={f}: one whole number of 1 or more per axis, of {info.ndim}")
        return info.rescaled([v * k for v, k in zip(info.voxel_size[-len(f) :], f)])

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._mode = self.mode if self.mode != "auto" else ("mode" if info.kind in ("label", "mask") else "mean")
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        f, k = [int(v) for v in self.factor], len(self.factor)
        lead, space = block.shape[:-k], block.shape[-k:]
        whole = [-(-n // v) * v for n, v in zip(space, f)]
        if list(space) != whole:  # the array's last block along an axis: its last voxel repeated
            block = np.pad(block, [(0, 0)] * len(lead) + [(0, w - n) for w, n in zip(whole, space)], "edge")
        shape = list(lead) + [x for w, v in zip(whole, f) for x in (w // v, v)]
        blocks = block.reshape(shape)
        inner = tuple(len(lead) + 2 * i + 1 for i in range(k))  # the axes within each block
        if self._mode == "mean":  # in the data's own type, integers rounded
            mean = blocks.mean(axis=inner, dtype=np.float64)
            return (np.rint(mean) if np.issubdtype(block.dtype, np.integer) else mean).astype(block.dtype)
        order = [a for a in range(blocks.ndim) if a not in inner] + list(inner)
        g = blocks.transpose(order).reshape(*lead, *(w // v for w, v in zip(whole, f)), -1)
        counts = (g[..., :, None] == g[..., None, :]).sum(-1)  # how often each value of a block occurs in it
        return np.take_along_axis(g, counts.argmax(-1)[..., None], -1)[..., 0]
`,c=`from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import KINDS, ArrayInfo, Box
from chunkmirage.fused import tent


class Source(ABC):
    """Read-only, random-access array. Everything upstream of an op is a \`\`Source\`\`."""

    @property
    @abstractmethod
    def info(self) -> ArrayInfo: ...

    @abstractmethod
    def read(self, box: Box) -> np.ndarray:
        """Return data for \`\`box\`\` (must lie within \`\`info.shape\`\`), shape == box.shape."""

    def read_padded(self, box: Box, fill=0, *, edge: bool = False) -> np.ndarray:
        """Read \`\`box\`\` even if it pokes outside the array. Out-of-range voxels are \`\`fill\`\`,
        or with \`\`edge\`\` the nearest voxel inside (so a filter sees no step at the border)."""
        clipped = box.clip(self.info.shape)
        if clipped == box:
            return self.read(box)
        if edge and not clipped.empty:
            inner = clipped.relative_to(box)
            pad = [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)]
            return np.pad(self.read(clipped), pad, mode="edge")
        out = np.full(box.shape, fill, dtype=self.info.dtype)
        if not clipped.empty:
            out[clipped.relative_to(box).slices()] = self.read(clipped)
        return out

    def cache_key(self) -> str:
        """Stable identity used to build cache keys for downstream stages."""
        return f"{type(self).__name__}:{id(self)}"


class KindSource(Source):
    """\`\`inner\`\` with its values said to be \`\`kind\`\` (\`\`ArrayInfo.kind\`\`), whatever it
    guessed: a uint32 image, or labels stored as uint16."""

    def __init__(self, inner: Source, kind: str | None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.inner = inner
        self._info = inner.info.with_(kind=kind)
        self._key = f"kind={kind}:{inner.cache_key()}"  # resampled and downsampled differently

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        return self.inner.read(box)


class ChunkedSource(Source):
    """A source materialized chunk-by-chunk via \`\`compute_chunk\`\` and memoized in an LRU.

    \`\`read(box)\`\` gathers the covering chunks (from cache or freshly computed) and slices.
    This is the single mechanism behind *both* raw-source caching and per-stage caching:
    every pipeline stage is exposed to the next one as a \`\`ChunkedSource\`\`.
    """

    def __init__(
        self,
        info: ArrayInfo,
        compute_chunk: Callable[[tuple[int, ...]], np.ndarray],
        cache: LRUCache | None,
        key: str,
    ):
        self._info = info
        self._compute = compute_chunk
        self._cache = cache
        self._key = key
        #: the overlapping blocks this stage is blended from, if any (an op with \`\`overlap\`\`)
        self.blocks: BlockSource | None = None
        # In-flight computations keyed by chunk index: concurrent requests for the same chunk
        # (a viewer retrying, or two clients) wait for one computation instead of duplicating it.
        self._inflight: dict[tuple[int, ...], threading.Event] = {}
        self._inflight_lock = threading.Lock()

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    @property
    def cached(self) -> bool:
        """Whether this stage memoizes its chunks."""
        return self._cache is not None

    def in_cache(self, index: Sequence[int]) -> bool:
        """Whether chunk \`\`index\`\` is held in the cache now (computed, and not evicted)."""
        return self._cache is not None and (self._key, tuple(int(i) for i in index)) in self._cache

    def chunk_box(self, index: Sequence[int]) -> Box:
        """What chunk \`\`index\`\` covers: \`\`compute_chunk\`\` returns an array of its shape."""
        return self._info.chunk_box(index)

    def chunk(self, index: Sequence[int]) -> np.ndarray:
        """Chunk \`\`index\`\` as an array of shape \`\`chunk_box(index).shape\`\` (edge-clipped)."""
        index = tuple(int(i) for i in index)
        if self._cache is None:
            return self._compute_checked(index)
        while True:
            hit = self._cache.get((self._key, index))
            if hit is not None:
                return hit
            with self._inflight_lock:
                event = self._inflight.get(index)
                if event is None:
                    event = self._inflight[index] = threading.Event()
                    owner = True
                else:
                    owner = False
            if not owner:
                event.wait()
                continue  # the owner has cached it (or failed); re-check the cache
            try:
                data = self._compute_checked(index)
                self._cache.put((self._key, index), data)
                return data
            finally:
                with self._inflight_lock:
                    self._inflight.pop(index, None)
                event.set()

    def _compute_checked(self, index: tuple[int, ...]) -> np.ndarray:
        data = np.ascontiguousarray(self._compute(index))
        expected = self.chunk_box(index).shape
        if data.shape != expected:
            raise ValueError(
                f"{self._key}: compute_chunk{index} returned shape {data.shape}, expected {expected}"
            )
        return data

    def read(self, box: Box) -> np.ndarray:
        info = self._info
        # Fast path: the box is exactly one chunk.
        idx = tuple(a // c for a, c in zip(box.start, info.chunk_shape))
        if info.chunk_box(idx) == box:
            return self.chunk(idx)
        out = np.empty(box.shape, dtype=info.dtype)
        for cidx in info.chunks_covering(box):
            cbox = info.chunk_box(cidx)
            inter = Box(
                tuple(max(a, b) for a, b in zip(box.start, cbox.start)),
                tuple(min(a, b) for a, b in zip(box.stop, cbox.stop)),
            )
            if inter.empty:
                continue
            data = self.chunk(cidx)
            out[inter.relative_to(box).slices()] = data[inter.relative_to(cbox).slices()]
        return out


class BlockSource(ChunkedSource):
    """Chunks that overlap: block \`\`index\`\` is chunk \`\`index\`\` of \`\`info\`\` and \`\`overlap\`\`
    voxels around it along the last axes (past the array's edge too, so every block has the
    one shape), each computed from its inputs alone and memoized as a chunk is. \`\`read\`\`
    blends them: a voxel is the mean of every block reaching it, each weighted 1 over its
    own chunk and less the further into its overlap (\`\`fused.tent\`\`), so the result turns
    smoothly from one block's values to the next's instead of stepping at a chunk's edge,
    and is the same whatever order the blocks are computed in. An integer output is
    rounded after the mean. The blocks of an op with \`\`overlap\`\`; the leading axes (before
    the last \`\`len(overlap)\`\`) are read whole."""

    def __init__(
        self,
        info: ArrayInfo,
        overlap: Sequence[int],
        compute_block: Callable[[tuple[int, ...]], np.ndarray],
        cache: LRUCache | None,
        key: str,
    ):
        super().__init__(info, compute_block, cache, key)
        self.overlap = tuple(int(o) for o in overlap)

    def chunk_box(self, index: Sequence[int]) -> Box:
        """Block \`\`index\`\`'s extent: chunk \`\`index\`\`, not clipped to the array along the
        last axes, padded by the overlap."""
        info, k = self._info, len(self.overlap)
        lead = info.ndim - k
        whole = info.chunk_box(index)
        core = Box.from_chunk(index, info.chunk_shape)
        return Box(
            whole.start[:lead] + tuple(a - o for a, o in zip(core.start[lead:], self.overlap)),
            whole.stop[:lead] + tuple(b + o for b, o in zip(core.stop[lead:], self.overlap)),
        )

    def read(self, box: Box) -> np.ndarray:
        info, k = self._info, len(self.overlap)
        lead = info.ndim - k
        chunks, grid = info.chunk_shape[lead:], info.chunk_grid[lead:]
        acc = np.zeros(box.shape, np.float64)
        total = np.zeros(box.shape[lead:], np.float64)
        # the blocks whose extent reaches the box
        first = [max((a - o) // c, 0) for a, o, c in zip(box.start[lead:], self.overlap, chunks)]
        last = [min((b - 1 + o) // c, g - 1) for b, o, c, g in zip(box.stop[lead:], self.overlap, chunks, grid)]
        ahead = tuple(a // c for a, c in zip(box.start[:lead], info.chunk_shape[:lead]))
        for rel in np.ndindex(*[hi - lo + 1 for lo, hi in zip(first, last)]):
            index = ahead + tuple(int(lo + r) for lo, r in zip(first, rel))
            extent = self.chunk_box(index)
            inter = Box(
                tuple(max(a, b) for a, b in zip(box.start, extent.start)),
                tuple(min(a, b) for a, b in zip(box.stop, extent.stop)),
            )
            if inter.empty:
                continue
            core = Box.from_chunk(index[lead:], chunks)
            weight = tent(inter.start[lead:], inter.stop[lead:], core.start, core.stop, self.overlap)
            data = self.chunk(index)[inter.relative_to(extent).slices()]
            dst = inter.relative_to(box).slices()
            acc[dst] += weight * data
            total[dst[lead:]] += weight
        out = acc / total
        if info.dtype.kind in "iub":
            out = np.rint(out)
        return out.astype(info.dtype)


class MultiscaleSource:
    """An ordered list of \`\`Source\`\` levels, s0 = full resolution. \`\`shader\`\` is a
    Neuroglancer shader the viewer should show the data with (its channel axis then being
    a shader channel), for sources whose channels mean something particular. \`\`cached\`\`
    says a pipeline should keep no copy of the levels' chunks in a stage of its own: they
    are cached stages already (another dataset's, \`\`dataset://\`\`), or must be read afresh
    each time (\`\`coverage://\`\`)."""

    def __init__(
        self, levels: Sequence[Source], name: str = "", shader: str | None = None, cached: bool = False
    ):
        if not levels:
            raise ValueError("need at least one level")
        self.levels = list(levels)
        self.name = name
        self.shader = shader
        self.cached = cached

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, i: int) -> Source:
        return self.levels[i]

    def __iter__(self):
        return iter(self.levels)

    def cache_key(self) -> str:
        return "|".join(lvl.cache_key() for lvl in self.levels)

    def level_for(self, voxel_size: Sequence[float], rtol: float = 0.01) -> tuple[int, bool]:
        """The level to read data at \`\`voxel_size\`\` from (its last axes, in the source's
        units), and whether it is at that size: one that is (within \`\`rtol\`\`), else the
        coarsest finer on every axis (resampled down, never up), else level 0."""
        want = np.asarray(voxel_size, dtype=float)
        best = 0
        for i, lvl in enumerate(self.levels):
            v = np.asarray(lvl.info.voxel_size[-len(want) :], dtype=float)
            if np.allclose(v, want, rtol=rtol, atol=0):
                return i, True
            if np.all(v <= want * (1 + rtol)):
                best = i
        return best, False

    def nearest_level(self, voxel_size: Sequence[float]) -> int:
        """The level whose voxel size (its last axes) is nearest \`\`voxel_size\`\`, by ratio
        (the sum over axes of the size's log ratio); the finer of two as near."""
        want = np.log(np.asarray(voxel_size, dtype=float))
        dist = [
            float(np.abs(np.log(np.asarray(lvl.info.voxel_size[-len(want) :], dtype=float)) - want).sum())
            for lvl in self.levels
        ]
        return min(range(len(dist)), key=lambda i: (round(dist[i], 9), i))
`,l='"""chunkmirage_contrib: what computes, shipped beside the engine.\n\nThe ``chunkmirage`` core composes data and serves it; it computes nothing but the\ndownsampling a pyramid needs. Everything with an algorithm in it lives here: the ops\n(``chunkmirage_contrib.ops``), the computed sources (``synthetic://``, ``register://``,\n``stitch://``, GeoTIFF reading), and the registration, stitching and tracking code behind\nthem. The core finds all of it through the same entry points any other package would use\n(``chunkmirage.ops``, ``chunkmirage.sources``, ``chunkmirage.source_matchers``,\n``chunkmirage.models``) and imports none of it; ``tests/test_core_boundary.py`` keeps it so.\n"""\n',u=`"""The ops shipped with chunkmirage: pointwise, filters, segmentation, combining channels
and terrain. Importing this registers them (\`\`@register\`\`); an installed package finds them
through the \`\`chunkmirage.ops\`\` entry point without importing anything."""

from chunkmirage_contrib.ops.combine import Contacts, NormalizedDifference
from chunkmirage_contrib.ops.filters import Diff, Gaussian, Gradient, Uniform
from chunkmirage_contrib.ops.pointwise import Cast, Scale, Threshold
from chunkmirage_contrib.ops.segment import DoG, Label, Morphology, Spots
from chunkmirage_contrib.ops.terrain import Hillshade, Slope

__all__ = [
    "Cast",
    "Contacts",
    "Diff",
    "DoG",
    "Gaussian",
    "Gradient",
    "Hillshade",
    "Label",
    "Morphology",
    "NormalizedDifference",
    "Scale",
    "Slope",
    "Spots",
    "Threshold",
    "Uniform",
]
`,d=`"""Ops over the channels of a \`\`stack://\`\` source: results that need two images at once.

Such an op takes a block whose first axis is the stack's channel axis and returns one
without it; the pipeline reads every channel and pads only the spatial axes by the halo.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Contacts(Op):
    """Contact sites between two structures, the first two channels of a \`\`stack://\`\` source:
    the voxels within \`\`radius\`\` of both. A mask; follow with \`\`label\`\` to colour and
    size-filter the sites."""

    name = "contacts"
    packages = ("scipy",)
    radius: float = Field(
        3.0,
        gt=0,
        le=32,
        description="Reach, in voxels: a voxel is a contact site when both structures lie within "
        "this distance of it (Euclidean). The halo is radius + 1.",
    )
    distance: float | None = Field(
        None,
        gt=0,
        description="Reach in the data's units (nm, say), instead of radius: each level counts "
        "it in its own voxels, so a contact means the same at every zoom (a level whose voxels "
        "are bigger than it keeps the voxels in both structures).",
    )
    a_low: float = Field(
        128.0,
        description="Values at or above this in the first channel are the first structure: 128 "
        "for a uint8 probability map, 1 for a segmentation.",
    )
    b_low: float = Field(
        128.0,
        description="The same threshold for the second channel, the second structure.",
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.radius) + 1

    def for_level(self, info: ArrayInfo) -> Op:
        """With \`\`distance\`\`, the radius in this level's voxels (its finest spatial axis)."""
        if self.distance is None:
            return self
        op = self.model_copy(update={"radius": self.distance / min(info.voxel_size[-3:])})
        op._cache = self._cache
        return op

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        if info.ndim < 4 or info.axes[0] != "c" or info.shape[0] < 2:
            raise ValueError(
                "contacts needs a source whose first axis is a channel axis holding two "
                f"structures, such as stack://<a>|<b>; got axes {info.axes}, shape {info.shape}"
            )
        return ArrayInfo(
            shape=info.shape[1:],
            dtype=self.output_dtype(info.dtype),
            chunk_shape=info.chunk_shape[1:],
            voxel_size=info.voxel_size[1:],
            units=info.units[1:],
            axes=info.axes[1:],
            translation=info.translation[1:],
            kind="mask",
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        a = block[0] >= self.a_low
        b = block[1] >= self.b_low
        return (self._within_reach(a) & self._within_reach(b)).astype(np.uint8)

    def _within_reach(self, mask: np.ndarray) -> np.ndarray:
        """The voxels within \`\`radius\`\` of \`\`mask\`\`: the distance to it, thresholded."""
        from scipy.ndimage import distance_transform_edt

        if not mask.any():
            return np.zeros(mask.shape, dtype=bool)
        if mask.all():
            return np.ones(mask.shape, dtype=bool)
        return distance_transform_edt(~mask) <= self.radius


@register
class NormalizedDifference(Op):
    """\`\`(a - b) / (a + b)\`\` of two channels of a \`\`stack://\`\` source (float32): the indices
    remote sensing reads plants, water and burn scars from (NDVI is near infrared and red,
    NBR near and shortwave infrared). With \`\`minus\`\`, a second pair's index is subtracted
    from the first's: a change between two dates. Burn severity (dNBR) is the NBR before a
    fire minus the NBR after, the before and after images' bands stacked as four channels.
    Where a band is zero or less, or the two sum to under \`\`floor\`\` (water), the index is NaN."""

    name = "normalized_difference"
    pair: list[int] = Field(
        [0, 1], min_length=2, max_length=2, description="The channels a and b, counted from the first."
    )
    minus: list[int] | None = Field(
        None,
        min_length=2,
        max_length=2,
        description="Two more channels, whose index is subtracted from the first pair's: [2, 3] "
        "for the after image of a stack of before and after.",
    )
    offset: float = Field(
        0.0,
        description="Added to every channel first, to make reflectances of stored numbers: "
        "Sentinel-2 since 2022 stores them plus 1000 (offset -1000).",
    )
    nodata: float | None = Field(
        None, description="A stored value meaning no data (Sentinel-2: 0): the index there is NaN."
    )
    floor: float = Field(
        0.0,
        ge=0,
        description="Where a pair's two values sum to less than this (after offset), its index is "
        "noise, a ratio of near zeros (water reflects almost no infrared), and is NaN.",
    )
    output_kind = "image"

    def _channels(self) -> list[int]:
        return [*self.pair, *(self.minus or [])]

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        need = max(self._channels()) + 1
        if info.ndim < 3 or info.axes[0] != "c" or info.shape[0] < need:
            raise ValueError(
                f"normalized_difference needs a source whose first axis holds {need} or more "
                f"channels, such as stack://<a>|<b>; got axes {info.axes}, shape {info.shape}"
            )
        return ArrayInfo(
            shape=info.shape[1:], dtype=np.dtype("float32"), chunk_shape=info.chunk_shape[1:],
            voxel_size=info.voxel_size[1:], units=info.units[1:], axes=info.axes[1:],
            translation=info.translation[1:], kind="image",
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        def index(a: int, b: int) -> np.ndarray:
            x, y = (block[c].astype(np.float32) + np.float32(self.offset) for c in (a, b))
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (x - y) / (x + y)
            out[(x <= 0) | (y <= 0) | (x + y < self.floor)] = np.nan  # positive, and not near zero
            if self.nodata is not None:
                out[(block[a] == self.nodata) | (block[b] == self.nodata)] = np.nan
            return out

        out = index(*self.pair)
        return out - index(*self.minus) if self.minus else out
`,f=`"""Neighbourhood filters (most need scipy). These exercise the halo machinery."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Gaussian(Op):
    """Gaussian blur (smoothing). Reduces noise before thresholding; larger sigma = blurrier."""

    name = "gaussian"
    packages = ("scipy",)
    sigma: float = Field(
        1.0,
        gt=0,
        description="Blur width in voxels (standard deviation). 1 removes pixel noise; 3-5 merges small structures.",
    )
    truncate: float = Field(
        3.0,
        ge=1,
        le=6,
        description="Kernel radius in units of sigma. Rarely needs changing; 3 keeps 99.7% of the kernel. "
        "Determines the halo: ceil(sigma × truncate) voxels of neighbouring data are read on each side.",
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.sigma * self.truncate)

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            block.astype(np.float32), self.sigma, truncate=self.truncate, mode="nearest"
        )


@register
class Uniform(Op):
    """Box (mean) filter: each voxel becomes the average of a size³ cube around it."""

    name = "uniform"
    packages = ("scipy",)
    size: int = Field(
        3,
        ge=1,
        le=31,
        description="Edge length of the averaging cube, in voxels (odd values are centred).",
    )

    @property
    def halo(self):  # type: ignore[override]
        return self.size // 2 + 1

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        return uniform_filter(block.astype(np.float32), self.size, mode="nearest")


@register
class Diff(Op):
    """Change along one axis: each voxel minus the one \`\`lag\`\` steps before it on \`\`axis\`\`
    (float32). Along time, what changed since the frame or day before: a hurricane's cold
    wake in sea temperature, a flare brightening the sun. The first \`\`lag\`\` steps of the
    array compare against its first."""

    name = "diff"
    axis: int = Field(
        0,
        ge=0,
        description="The axis to difference along, counted from the first (0: time in a t, y, x series).",
    )
    lag: int = Field(1, ge=1, le=64, description="How many steps back to compare with.")

    @property
    def halo(self):  # type: ignore[override]
        return self.lag

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        if self.axis >= ndim:
            raise ValueError(f"diff axis={self.axis}: the data has {ndim} axes")
        return tuple(self.lag if a == self.axis else 0 for a in range(ndim))

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        a = self.axis
        b = block.astype(np.float32)
        out = np.zeros_like(b)
        now, before = [slice(None)] * b.ndim, [slice(None)] * b.ndim
        now[a], before[a] = slice(self.lag, None), slice(None, -self.lag)
        out[tuple(now)] = b[tuple(now)] - b[tuple(before)]
        return out


@register
class Gradient(Op):
    """Rate of change along each of \`\`axes\`\`, one channel each on a new leading \`\`c\`\` axis
    (float32), per unit of the axes (a level's voxel size): central differences. It returns
    only the interior it can compute, one voxel less on each side of those axes, as a valid
    convolution (or a model) does. Where temperature changes fastest at sea, ocean fronts;
    on terrain, the slope's components; in a volume, edges and their direction. With
    \`\`sigma\`\`, smoothed along the same axes first (a Gaussian derivative), so noise a voxel or
    two across does not point every voxel its own way."""

    name = "gradient"
    packages = ("scipy",)
    sigma: float = Field(
        0.0,
        ge=0,
        le=16,
        description="Smoothing first, in voxels along the axes differentiated (0: none). The halo "
        "grows to 1 + ceil(3 × sigma).",
    )
    axes: list[int] | None = Field(
        None,
        description="The axes to differentiate along, counted from the first (1, 2: latitude and "
        "longitude of a time, lat, lon series); default the last three, or all if fewer.",
    )
    _voxel: tuple[float, ...] | None = PrivateAttr(None)

    def _along(self, ndim: int) -> list[int]:
        axes = list(range(max(0, ndim - 3), ndim)) if self.axes is None else list(self.axes)
        if not axes or any(not 0 <= a < ndim for a in axes) or len(set(axes)) != len(axes):
            raise ValueError(f"gradient axes={self.axes}: the data has {ndim} axes")
        return axes

    @property
    def halo(self):  # type: ignore[override]
        return 1 + math.ceil(3 * self.sigma)

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        along = self._along(ndim)
        return tuple(self.halo if a in along else 0 for a in range(ndim))

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        n = len(self._along(info.ndim))
        return info.with_(
            shape=(n, *info.shape), chunk_shape=(n, *info.chunk_shape), dtype=np.dtype("float32"),
            voxel_size=(1.0, *info.voxel_size), units=("", *info.units), axes=("c", *info.axes),
            translation=(0.0, *info.translation), kind="image",
        )

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._voxel = tuple(float(v) for v in info.voxel_size)
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        b = block.astype(np.float32)
        along = self._along(b.ndim)
        voxel = self._voxel[-b.ndim:] if self._voxel else (1.0,) * b.ndim
        if self.sigma > 0:
            from scipy.ndimage import gaussian_filter

            b = gaussian_filter(b, [self.sigma if a in along else 0 for a in range(b.ndim)], truncate=3, mode="nearest")
        h = self.halo
        inner = [slice(h, -h) if a in along else slice(None) for a in range(b.ndim)]
        out = []
        for a in along:
            hi, lo = list(inner), list(inner)
            hi[a], lo[a] = slice(h + 1, b.shape[a] - h + 1), slice(h - 1, b.shape[a] - h - 1)
            out.append((b[tuple(hi)] - b[tuple(lo)]) / np.float32(2 * voxel[a]))
        return np.stack(out)
`,p=`from __future__ import annotations

import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class Threshold(Op):
    """Binary mask: voxels with \`\`low <= value < high\`\` become \`\`value\`\`, everything else 0."""

    name = "threshold"
    output_kind = "mask"
    low: float = Field(
        0.0,
        description="Lower bound (inclusive), in the source's intensity units. Voxels at or above it pass.",
    )
    high: float | None = Field(
        None,
        description="Upper bound (exclusive). Leave empty for no upper bound. Use it to select an intensity band.",
    )
    value: int = Field(
        1,
        ge=1,
        le=255,
        description="Label written for voxels that pass (1 shows as one segment in Neuroglancer).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        mask = block >= self.low
        if self.high is not None:
            mask &= block < self.high
        return mask.astype(np.uint8) * np.uint8(self.value)


@register
class Cast(Op):
    """Convert to another data type, e.g. float32 → uint8 for viewers that need integers."""

    name = "cast"
    dtype: str = Field(
        "uint8",
        description="Target numpy dtype name: uint8, uint16, uint32, uint64, int16, float32, ...",
    )
    clip: bool = Field(
        True,
        description="Clip values to the target integer range first (avoids wrap-around, e.g. 300 → 255 not 44).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype(self.dtype)

    def output_info(self, info):
        return info.with_(dtype=self.output_dtype(info.dtype))  # still what it was: labels stay labels

    def apply(self, block: np.ndarray) -> np.ndarray:
        out = np.dtype(self.dtype)
        if self.clip and np.issubdtype(out, np.integer):
            ii = np.iinfo(out)
            block = np.clip(block, ii.min, ii.max)
        return block.astype(out)


@register
class Scale(Op):
    """Linear intensity rescale \`\`value * factor + offset\`\` (output is float32)."""

    name = "scale"
    factor: float = Field(1.0, description="Multiply every voxel by this (contrast).")
    offset: float = Field(0.0, description="Then add this (brightness).")

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        return block.astype(np.float32) * np.float32(self.factor) + np.float32(self.offset)
`,m=`"""Showcase ops beyond pointwise: multi-scale filtering, morphology and labelling (need scipy).

These demonstrate what a viewer shader cannot do: neighbourhood context (halos), binary
morphology, and connected-component labelling with size filtering.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class DoG(Op):
    """Difference of Gaussians: enhances blob-like structures of a chosen size, suppresses background."""

    name = "dog"
    output_kind = "image"
    packages = ("scipy",)
    sigma: float = Field(
        2.0, gt=0, description="Size of structures to enhance, in voxels (smaller blur)."
    )
    ratio: float = Field(
        1.6,
        gt=1,
        le=4,
        description="Larger blur = sigma × ratio. 1.6 approximates a Laplacian of Gaussian.",
    )
    gain: float = Field(
        4.0, gt=0, description="Multiply the difference so the result uses the 0..255 range."
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.sigma * self.ratio * 3)

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        b = block.astype(np.float32)
        d = gaussian_filter(b, self.sigma, mode="nearest") - gaussian_filter(
            b, self.sigma * self.ratio, mode="nearest"
        )
        return np.clip(d * self.gain + 128, 0, 255).astype(np.uint8)


@register
class Morphology(Op):
    """Binary morphology on a mask: remove specks (open), fill holes (close), shrink or grow."""

    name = "morphology"
    output_kind = "mask"
    packages = ("scipy",)
    operation: str = Field(
        "open",
        pattern="^(open|close|erode|dilate)$",
        description="open = erode then dilate (removes small objects); close = dilate then erode (fills small gaps); erode; dilate.",
    )
    radius: int = Field(
        2, ge=1, le=16, description="Radius of the spherical structuring element, in voxels."
    )

    @property
    def halo(self):  # type: ignore[override]
        return 2 * self.radius + 1

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

        r = self.radius
        zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        ball = (zz * zz + yy * yy + xx * xx) <= r * r
        mask = block > 0
        fn = {
            "open": binary_opening,
            "close": binary_closing,
            "erode": binary_erosion,
            "dilate": binary_dilation,
        }[self.operation]
        return fn(mask, structure=ball).astype(np.uint8)


@register
class Label(Op):
    """Connected components of a mask, coloured as segments. Labels are unique per chunk, so one
    object spanning several chunks gets several colours; that is the honest per-chunk preview."""

    name = "label"
    output_kind = "label"
    packages = ("scipy",)
    min_size: int = Field(
        0,
        ge=0,
        description="Drop components smaller than this many voxels (counted within the chunk + halo).",
    )
    connectivity: int = Field(
        1,
        ge=1,
        le=3,
        description="1 = faces only (6-connected), 2 = +edges (18), 3 = +corners (26).",
    )

    @property
    def halo(self):  # type: ignore[override]
        # Some context so size filtering near chunk borders sees more of each object.
        return 4 if self.min_size else 0

    def output_dtype(self, in_dtype):
        return np.dtype("uint32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import generate_binary_structure, label

        structure = generate_binary_structure(3, self.connectivity)
        labels, n = label(block > 0, structure=structure)
        labels = labels.astype(np.uint32)
        if self.min_size and n:
            sizes = np.bincount(labels.ravel(), minlength=n + 1)
            keep = sizes >= self.min_size
            keep[0] = False
            labels = np.where(keep[labels], labels, 0).astype(np.uint32)
        return labels

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        labels = self.apply(block)
        # Offset labels by a salt derived from the chunk position so colours differ between
        # chunks (labels stay < 2**16 per chunk; the salt occupies the high 16 bits).
        z, y, x = (int(v) for v in box.start[-3:])
        salt = np.uint32(((z * 73856093) ^ (y * 19349663) ^ (x * 83492791)) & 0x7FFF) << np.uint32(
            16
        )
        return np.where(labels > 0, labels + salt, 0).astype(np.uint32)


@register
class Spots(Op):
    """Bright diffraction-limited spots, such as single mRNA molecules in smFISH or EASI-FISH:
    a difference of Gaussians, its local maxima above \`\`threshold\`\`, each drawn as a small
    ball. A spot's id comes from its position in the volume, so one spot keeps its id and
    colour whichever chunk finds it."""

    name = "spots"
    output_kind = "label"
    packages = ("scipy",)
    sigma: float = Field(
        1.0,
        gt=0,
        le=8,
        description="Spot size, in voxels along y and x: the smaller blur of the difference of "
        "Gaussians (the larger is 1.6 times it). About the spot's standard deviation.",
    )
    sigma_z: float = Field(
        0.6,
        gt=0,
        le=8,
        description="The same along z, in z voxels: smaller than sigma when z voxels are "
        "coarser, as in most light microscopy.",
    )
    threshold: float = Field(
        10.0,
        ge=0,
        description="Least difference-of-Gaussians response a spot needs, in the image's "
        "intensity units: higher finds fewer, brighter spots.",
    )
    separation: int = Field(
        2,
        ge=1,
        le=8,
        description="Spots closer than this, in y-x voxels, are one: the radius of the local "
        "maximum search.",
    )
    radius: int = Field(
        1,
        ge=0,
        le=8,
        description="Radius of the ball drawn for each spot, in y-x voxels (0 marks one voxel).",
    )

    def _z(self, v: float) -> int:
        """\`\`v\`\` y-x voxels in z voxels, as far as sigma_z / sigma says."""
        return int(math.ceil(v * self.sigma_z / self.sigma))

    @property
    def halo(self):  # type: ignore[override]
        xy = math.ceil(3 * 1.6 * self.sigma) + self.separation + self.radius
        z = math.ceil(3 * 1.6 * self.sigma_z) + self._z(self.separation) + self._z(self.radius)
        return (z, xy, xy)

    def output_dtype(self, in_dtype):
        return np.dtype("uint32")

    def output_info(self, info):
        if info.ndim != 3:
            raise ValueError(
                f"spots finds spots in a z, y, x volume, not axes {info.axes}: pin the other "
                'axes with the spec\\'s select, e.g. {"c": 1, "t": 0} (--select c=1,t=0)'
            )
        return super().output_info(info)

    def apply(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - apply_at is used
        return self.apply_at(block, None)

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, grey_dilation, maximum_filter

        b = block.astype(np.float32)
        s = (self.sigma_z, self.sigma, self.sigma)
        dog = gaussian_filter(b, s, mode="nearest") - gaussian_filter(
            b, tuple(1.6 * v for v in s), mode="nearest"
        )
        sz, sxy = self._z(self.separation), self.separation
        peaks = (dog == maximum_filter(dog, size=(2 * sz + 1, 2 * sxy + 1, 2 * sxy + 1))) & (
            dog >= self.threshold
        )
        out = np.zeros(b.shape, dtype=np.uint32)
        where = np.nonzero(peaks)
        if not len(where[0]):
            return out
        origin = np.asarray(box.start[-3:] if box is not None else (0, 0, 0), dtype=np.int64)
        z, y, x = (w.astype(np.int64) + o for w, o in zip(where, origin))
        h = (z * 73856093) ^ (y * 19349663) ^ (x * 83492791)
        out[where] = (h % 0xFFFFFFFE + 1).astype(np.uint32)
        if self.radius:
            rz, r = self._z(self.radius), self.radius
            zz, yy, xx = np.ogrid[-rz : rz + 1, -r : r + 1, -r : r + 1]
            ball = (zz / max(rz, 1)) ** 2 * (rz > 0) + (yy * yy + xx * xx) / (r * r) <= 1
            out = grey_dilation(out, footprint=ball)
        return out
`,h=`"""Terrain from elevation: slope and hillshade, the first things one computes from a digital
elevation model. Both work on the last two axes (\`\`y, x\`\`, rows down the image) in the
level's own pixel spacing, so a coarse level's slope is the slope of its coarser grid."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


class _Terrain(Op):
    halo = 1
    _spacing: tuple[float, float] = PrivateAttr((1.0, 1.0))

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._spacing = (float(info.voxel_size[-2]), float(info.voxel_size[-1]))
        op._cache = self._cache
        return op

    def _gradient(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d(elevation)/dy and /dx, per unit of the axes (y counting down the image)."""
        z = block.astype(np.float64) * self.z_factor
        dy, dx = np.gradient(z, *self._spacing, axis=(-2, -1))
        return dy, dx


@register
class Slope(_Terrain):
    """Slope of an elevation model in degrees (0 flat, 90 a cliff), from the level's pixel
    spacing; output float32, NaN where the elevation is."""

    name = "slope"
    z_factor: float = Field(
        1.0,
        gt=0,
        description="Elevation units per unit of the pixel spacing (1 when both are metres).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        dy, dx = self._gradient(block)
        return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


@register
class Hillshade(_Terrain):
    """Shaded relief: the brightness of the terrain lit by a distant sun at \`\`azimuth\`\` and
    \`\`altitude\`\` (local illumination only: no shadows cast across the terrain). Output
    uint8, 1 (unlit) to 255 (facing the sun), 0 where the elevation is NaN."""

    name = "hillshade"
    azimuth: float = Field(
        315.0,
        ge=0,
        le=360,
        description="Direction the sun shines from, degrees clockwise from the top of the image.",
    )
    altitude: float = Field(
        45.0, gt=0, le=90, description="Height of the sun above the horizon, in degrees."
    )
    z_factor: float = Field(
        1.0,
        gt=0,
        description="Elevation units per unit of the pixel spacing; above 1 exaggerates relief.",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        dy, dx = self._gradient(block)
        az, alt = np.radians(self.azimuth), np.radians(self.altitude)
        # the surface normal (-dz/dx, dz/dy up the image, 1) against the sun's direction
        sun = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
        lit = (-dx * sun[0] + dy * sun[1] + sun[2]) / np.sqrt(dx * dx + dy * dy + 1.0)
        out = (1 + 254 * np.clip(lit, 0.0, 1.0)).round()
        return np.where(np.isfinite(lit), out, 0).astype(np.uint8)
`,g=`"""Deformable registration on a GPU: the solver behind \`\`register://\`\`.

Finds a smooth displacement field \`\`u\`\` in the fixed image's physical space such that the
moving image sampled at \`\`affine(p + u(p))\`\` looks like the fixed image at \`\`p\`\`. The fit
maximizes local normalized cross-correlation, penalizes the field's gradient, and runs
Adam from coarse levels to fine ones. \`\`u\`\` lives on a regular control grid a few voxels
apart, so it stays small (tens of megabytes for a whole organ), and the \`\`scene://\`\`
resampler serves the registered image through it at any resolution.

Needs PyTorch (\`\`chunkmirage[gpu]\`\`); without a GPU it runs on the CPU, slowly.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("chunkmirage")

MAX_VOXELS = 1 << 24  # fixed voxels warped at once; bounds GPU memory to a few GB
FLAT = 1e-4  # windows with less intensity variance than this (in [0, 1] units) are ignored
MAX_FIT_VOXELS = 300_000  # the affine search's finest copy of the images
MIN_FIT_SIZE = 12  # its coarsest copy: at least this many voxels on every axis
FIT_ITERATIONS = (150, 100)  # per copy, coarse to fine (the first alone when there is one)
LR_MATRIX = 2e-3  # Adam steps of the affine's matrix, and of its centre in voxels
LR_CENTRE = 0.4


@dataclass
class Level:
    """One resolution level of an image, spatial axes only (C order)."""

    data: np.ndarray
    voxel_size: np.ndarray  # physical units per voxel
    translation: np.ndarray  # physical position of voxel 0's centre


@dataclass
class Grid:
    """A displacement field on a regular grid: \`\`values[i, j, k]\`\` (physical units, C order
    components) is the displacement at \`\`origin + (i, j, k) * spacing\`\`."""

    values: np.ndarray  # (z, y, x, 3) float32
    origin: np.ndarray
    spacing: np.ndarray


@dataclass
class Settings:
    iterations: tuple[int, ...]  # per level, coarse to fine
    smooth: float = 1.0  # weight of the penalty on the field's gradient
    grid: float = 4.0  # control-point spacing, in voxels of each level
    window: tuple[int, ...] = (7,)  # correlation window per level, voxels (odd)
    step: float = 0.5  # Adam learning rate, in voxels of each level


def pick_device(device: str = "auto"):
    """\`\`auto\`\`: the GPU with the most free memory, else the CPU."""
    import torch

    if device != "auto":
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def normalize(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Intensities to [0, 1] between \`\`lo\`\` and \`\`hi\`\`, clipped (bright spots saturate)."""
    return np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1e-12), 0, 1)


def solve(
    fixed: list[Level],
    moving: list[Level],
    affine: np.ndarray,
    settings: Settings,
    *,
    box: tuple[np.ndarray, np.ndarray],
    device: str = "auto",
    snapshots: int = 1,
    init: np.ndarray | None = None,
    ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    label: str = "",
) -> list[Grid]:
    """Fit \`\`u\`\` level by level (\`\`fixed[i]\`\` against \`\`moving[i]\`\`, coarse to fine).

    \`\`affine\`\` (4x4 or 3x4) maps fixed physical points to moving physical points; the
    moving image is sampled at \`\`affine(p + u(p))\`\`. The control grid spans \`\`box\`\` (the
    fixed image's physical extent, \`\`(low, high)\`\`); points outside it use the nearest
    edge value. Returns \`\`snapshots\`\` fields evenly spaced over the iterations, from the
    starting field to the final one (just the final one when \`\`snapshots\`\` is 1).

    \`\`init\`\` starts from a field given at the first level's control points, \`\`(z, y, x,
    3)\`\`, rather than from zero; \`\`ranges\`\` normalizes the images with the given
    intensity ranges (fixed, moving) rather than the first level's percentiles, so blocks
    of one image are all scaled alike; \`\`label\`\` names the fit in the log.
    """
    lo, hi = (np.asarray(b, dtype=float) for b in box)
    total = sum(settings.iterations)
    if total == 0:  # nothing to fit: the affine alone, or the starting field
        values = np.zeros((2, 2, 2, 3), np.float32) if init is None else init
        return [_grid(values, lo, hi)] * max(1, snapshots)
    if len(settings.iterations) != len(fixed) or len(settings.window) != len(fixed):
        raise ValueError(
            f"settings need one iteration count and one window per level ({len(fixed)})"
        )
    import torch
    import torch.nn.functional as F

    dev = pick_device(device)
    a = torch.as_tensor(np.asarray(affine, dtype=float)[:3], dtype=torch.float32, device=dev)
    if init is None:
        shape = _control_shape(lo, hi, fixed[0].voxel_size, settings.grid)
        u = torch.zeros((1, 3, *shape), device=dev)
    else:
        u = torch.as_tensor(init, dtype=torch.float32, device=dev).permute(3, 0, 1, 2)[None]
    # the iteration count after which each snapshot is taken (repeated when there are more
    # snapshots than iterations)
    n = max(snapshots, 1)
    wanted = [round(k * total / (n - 1)) for k in range(n)] if n > 1 else [total]
    out: list[Grid] = []

    def record(step: int) -> None:
        while len(out) < n and wanted[len(out)] == step:
            out.append(_grid(u, lo, hi))

    record(0)
    step = 0
    started = time.time()
    fixed_range, moving_range = ranges or (_range(fixed[0].data), _range(moving[0].data))
    per_level = zip(fixed, moving, settings.iterations, settings.window)
    for stage, (fl, ml, iters, window) in enumerate(per_level, 1):
        if iters == 0:
            continue
        shape = _control_shape(lo, hi, fl.voxel_size, settings.grid)
        if tuple(u.shape[2:]) != shape:  # refine: both grids span the box exactly
            u = F.interpolate(u, size=shape, mode="trilinear", align_corners=True)
        u = u.detach().requires_grad_(True)
        spacing = (hi - lo) / (np.asarray(shape) - 1)
        opt = _Adam(u, lr=settings.step * float(np.mean(fl.voxel_size)))
        fix = torch.as_tensor(normalize(fl.data, *fixed_range), device=dev)[None, None]
        mov = torch.as_tensor(normalize(ml.data, *moving_range), device=dev)[None, None]
        warp = _Warp(fl, ml, lo, hi, a, dev)
        slabs = _slabs(fl.data.shape, window // 2)
        # Only voxels whose correlation window the moving image covers (under the starting
        # field) count: elsewhere there is nothing to match, and fitting would drag the
        # moving image's edge over whatever lies beyond it. The field there follows from
        # its smoothness alone.
        with torch.no_grad():
            seen = [warp.seen(u, z0, z1, window)[:, :, c0:c1] for z0, z1, c0, c1 in slabs]
        # the fixed image's side of each window does not change as the field does
        stats = [_window_stats(fix[:, :, z0:z1], window) for z0, z1, _, _ in slabs]
        n_vox = math.prod(fl.data.shape)
        t0 = time.time()
        for it in range(iters):
            u.grad = None
            sim = torch.zeros((), device=dev)  # read back only when logged
            for (z0, z1, c0, c1), inside, st in zip(slabs, seen, stats):
                warped = warp(mov, u, z0, z1)
                cc = _lncc(fix[:, :, z0:z1], st, warped, window)[:, :, c0:c1] * inside
                loss = -cc.sum() / n_vox
                loss.backward()
                sim -= loss.detach()
            reg = settings.smooth * _diffusion(u, spacing)
            reg.backward()
            opt.step()
            step += 1
            if it == 0:
                first = float(sim)
            record(step)
        log.info(
            "register%s: stage %d/%d, %s voxels, grid %s, %d iterations in %.1f s: "
            "similarity %.4f -> %.4f",
            label,
            stage,
            len(fixed),
            "x".join(map(str, fl.data.shape)),
            "x".join(map(str, shape)),
            iters,
            time.time() - t0,
            first,
            float(sim),
        )
        del fix, mov, warp, seen, stats
    if not label:  # a whole image: its levels no longer need the GPU's memory
        log.info("register: solved in %.1f s on %s", time.time() - started, dev)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def find_affine(fixed: Level, moving: Level, device: str = "auto") -> tuple[np.ndarray, dict]:
    """The fixed-to-moving affine (4x4, physical units) to start from when none is given,
    as the browser page's \`\`affine.ts\`\` finds it. The images' intensity moments are
    matched first, centre to centre and principal axis to principal axis, which leaves
    each axis's sign open: of the orientations that do not mirror the image, the best
    correlated is kept. A mirror image (one axis reversed) is not searched for: on a nearly
    symmetric specimen (a brain) it correlates as well as the right rotation, so the images
    cannot tell, and a fit cannot reach one from a rotation; give it as an affine. Then the 12 numbers are fitted by gradient ascent on the
    normalized cross-correlation, on a copy of at most \`\`MAX_FIT_VOXELS\`\` voxels after a
    coarser one. The moments assume both images show the same whole object. Also returns
    the correlation with no affine, after the moments and after the fit."""
    import torch

    dev = pick_device(device)
    started = time.time()
    fs = [_normalized(fixed)]
    ms = [_normalized(moving)]
    # copies: the finest within MAX_FIT_VOXELS, after one coarser if that is big enough
    while math.prod(fs[-1].data.shape) > MAX_FIT_VOXELS:
        fs.append(_halve(fs[-1]))
        ms.append(_halve(ms[-1]))
    coarser = _halve(fs[-1])
    stages = [len(fs) - 1]
    if min(coarser.data.shape) >= MIN_FIT_SIZE:
        fs.append(coarser)
        ms.append(_halve(ms[-1]))
        stages.insert(0, len(fs) - 1)

    f0, m0, ff, mf = fs[stages[0]], ms[stages[0]], fs[stages[-1]], ms[stages[-1]]
    cf, sf = _moments(f0)
    cm, sm = _moments(m0)
    ef, vf = np.linalg.eigh(sf)  # ascending, so both images' axes pair up by extent
    em, vm = np.linalg.eigh(sm)
    best = None
    for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).reshape(3, -1).T:
        # A = Vm sqrt(em) S / sqrt(ef) Vf^T: takes the fixed second moments to the moving ones
        a = vm @ np.diag(np.sqrt(np.maximum(em, 1e-12) / np.maximum(ef, 1e-12)) * s) @ vf.T
        if np.linalg.det(a) < 0:  # a mirror: not searched for (see above)
            continue
        v = _ncc(f0, m0, a, cm, cf, dev)
        if best is None or v > best[0]:
            best = (v, a)
    assert best is not None  # four of the eight orientations have each handedness
    scores = {
        "identity": _ncc(ff, mf, np.eye(3), np.zeros(3), np.zeros(3), dev),
        "moments": _ncc(ff, mf, best[1], cm, cf, dev),
    }

    def t32(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev)

    a, u = t32(best[1]).requires_grad_(True), t32(cm).requires_grad_(True)  # u: the centre's image
    for k, s in enumerate(stages):
        f, m = fs[s], ms[s]
        iters = FIT_ITERATIONS[0 if len(stages) == 1 else k]
        opt_a = _Adam(a, lr=LR_MATRIX, eps=1e-12)
        opt_u = _Adam(u, lr=LR_CENTRE * float(np.mean(f.voxel_size)), eps=1e-12)
        fix, mov, p = _fit_tensors(f, m, cf, dev)
        for _ in range(iters):
            a.grad = u.grad = None
            (-_ncc_t(fix, mov, m, p, a, u)).backward()  # ascent
            opt_a.step()
            opt_u.step()
    a_np, u_np = a.detach().cpu().numpy().astype(float), u.detach().cpu().numpy().astype(float)
    scores["final"] = _ncc(ff, mf, a_np, u_np, cf, dev)
    affine = np.eye(4)
    affine[:3, :3] = a_np
    affine[:3, 3] = u_np - a_np @ cf  # q = A (p - cf) + u
    log.info(
        "register: affine found in %.1f s: correlation %.3f with none, %.3f after the moments,"
        " %.3f after the fit",
        time.time() - started,
        scores["identity"],
        scores["moments"],
        scores["final"],
    )
    return affine, scores


def _normalized(level: Level) -> Level:
    return Level(normalize(level.data, *_range(level.data)), level.voxel_size, level.translation)


def _halve(level: Level) -> Level:
    """2x2x2 means (partial blocks at the far faces averaged over what they hold); the
    grid's origin moves to the first block's centre. In float32 whatever the data's type:
    sums of uint8 voxels would wrap."""
    x = level.data.astype(np.float32)
    for axis in range(3):
        n = x.shape[axis]
        pairs = (x.take(range(0, n - n % 2, 2), axis) + x.take(range(1, n, 2), axis)) / 2
        x = np.concatenate([pairs, x.take([n - 1], axis)], axis) if n % 2 else pairs
    grew = np.asarray(level.data.shape) > 1
    return Level(
        x.astype(np.float32),
        np.where(grew, 2 * level.voxel_size, level.voxel_size),
        np.where(grew, level.translation + level.voxel_size / 2, level.translation),
    )


def _moments(level: Level) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted centre and covariance of the tissue (above the 20th percentile
    of the non-zero voxels), physical units."""
    data = level.data
    nonzero = data[data > 0]
    floor = np.percentile(nonzero, 20) if nonzero.size else 0.0
    idx = np.argwhere(data > floor)
    w = data[data > floor].astype(float)
    p = level.translation + idx * level.voxel_size
    c = (w[:, None] * p).sum(0) / w.sum()
    d = p - c
    return c, (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / w.sum()


def _fit_tensors(f: Level, m: Level, cf, dev):
    """The two images as tensors and the fixed voxel centres relative to \`\`cf\`\`, (N, 3)."""
    import torch

    fix = torch.as_tensor(f.data, device=dev).reshape(-1)
    mov = torch.as_tensor(m.data, device=dev)[None, None]
    idx = np.indices(f.data.shape, dtype=np.float32).reshape(3, -1).T
    p = torch.as_tensor(idx * f.voxel_size + (f.translation - cf), dtype=torch.float32, device=dev)
    return fix, mov, p


def _ncc_t(fix, mov, m: Level, p, a, u):
    """Normalized cross-correlation of the fixed image with the moving one sampled at
    \`\`a p + u\`\` (trilinear, 0 beyond it), differentiable in \`\`a\`\` and \`\`u\`\`."""
    import torch
    import torch.nn.functional as F

    q = (
        p @ a.T + u - torch.as_tensor(m.translation, dtype=p.dtype, device=p.device)
    ) / torch.as_tensor(m.voxel_size, dtype=p.dtype, device=p.device)
    g = (
        2
        * q
        / torch.as_tensor(
            np.maximum(np.asarray(m.data.shape) - 1, 1), dtype=p.dtype, device=p.device
        )
        - 1
    )
    v = F.grid_sample(
        mov,
        g.flip(-1).reshape(1, 1, 1, -1, 3),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1)
    fc, vc = fix - fix.mean(), v - v.mean()
    return (fc * vc).sum() / ((fc * fc).sum() * (vc * vc).sum()).clamp_min(1e-24).sqrt()


def _ncc(f: Level, m: Level, a, u, cf, dev) -> float:
    import torch

    fix, mov, p = _fit_tensors(f, m, cf, dev)
    with torch.no_grad():
        return float(
            _ncc_t(
                fix,
                mov,
                m,
                p,
                torch.as_tensor(a, dtype=p.dtype, device=dev),
                torch.as_tensor(u, dtype=p.dtype, device=dev),
            )
        )


def _range(data: np.ndarray) -> tuple[float, float]:
    step = max(1, data.size // 1_000_000)
    lo, hi = np.percentile(data.reshape(-1)[::step], [0.5, 99.5])
    return float(lo), float(hi)


def _control_shape(lo, hi, voxel_size, grid: float) -> tuple[int, ...]:
    """Control points spanning \`\`lo..hi\`\` at most \`\`grid\`\` voxels apart (exactly that far
    when the extent is a whole number of spacings, rounding error allowed for)."""
    n = np.ceil((hi - lo) / (grid * np.asarray(voxel_size, dtype=float)) - 1e-6).astype(int) + 1
    return tuple(int(v) for v in np.maximum(n, 2))


def _grid(u, lo, hi) -> Grid:
    values = u if isinstance(u, np.ndarray) else u.detach()[0].permute(1, 2, 3, 0).cpu().numpy()
    shape = np.asarray(values.shape[:3])
    return Grid(np.ascontiguousarray(values, np.float32), lo.copy(), (hi - lo) / (shape - 1))


def _slabs(shape, halo: int) -> list[tuple[int, int, int, int]]:
    """Runs of whole z planes of at most \`\`MAX_VOXELS\`\` voxels, each padded by \`\`halo\`\`
    planes so the correlation windows at its core are complete: \`\`(start, stop, core
    start, core stop)\`\`, the core relative to the padded run."""
    plane = math.prod(shape[1:])
    rows = max(1, MAX_VOXELS // plane - 2 * halo)
    out = []
    for z in range(0, shape[0], rows):
        z0, z1 = max(0, z - halo), min(shape[0], z + rows + halo)
        out.append((z0, z1, z - z0, min(z + rows, shape[0]) - z0))
    return out


class _Adam:
    """Adam (Kingma and Ba) on one tensor. \`\`torch.optim\`\` would do, but importing it loads
    \`\`torch._dynamo\`\`: seconds of start-up from a network file system, for nothing here."""

    def __init__(self, p, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        import torch

        self.p, self.lr, self.b1, self.b2, self.eps = p, lr, b1, b2, eps
        self.m, self.v = torch.zeros_like(p), torch.zeros_like(p)
        self.t = 0
        self.torch = torch

    def step(self) -> None:
        g, b1, b2 = self.p.grad, self.b1, self.b2
        self.t += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = self.m / (1 - b1**self.t)
        v_hat = self.v / (1 - b2**self.t)
        with self.torch.no_grad():
            self.p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))


class _Warp:
    """The moving image sampled at \`\`affine(p + u(p))\`\` for fixed voxel centres \`\`p\`\`."""

    def __init__(self, fixed: Level, moving: Level, lo, hi, affine, dev):
        import torch

        def t(v):
            return torch.as_tensor(np.asarray(v, dtype=float), dtype=torch.float32, device=dev)

        self.torch = torch
        self.shape = fixed.data.shape
        self.vf, self.tf = t(fixed.voxel_size), t(fixed.translation)
        self.lo, self.span = t(lo), t(hi - lo)
        self.a = affine
        self.vm, self.tm = t(moving.voxel_size), t(moving.translation)
        self.mshape = t(np.maximum(np.asarray(moving.data.shape) - 1, 1))
        self.dev = dev

    def _points(self, z0: int, z1: int):
        """Physical positions of the fixed voxel centres in planes \`\`z0:z1\`\`, (d, h, w, 3)."""
        torch = self.torch
        axes = [
            torch.arange(z0, z1, device=self.dev, dtype=torch.float32),
            torch.arange(self.shape[1], device=self.dev, dtype=torch.float32),
            torch.arange(self.shape[2], device=self.dev, dtype=torch.float32),
        ]
        p = [ax * self.vf[i] + self.tf[i] for i, ax in enumerate(axes)]
        return torch.stack(torch.meshgrid(*p, indexing="ij"), dim=-1)

    def _index(self, x):
        """Moving-image voxel indices of fixed-space physical points \`\`x\`\` under the affine."""
        return (x @ self.a[:, :3].T + self.a[:, 3] - self.tm) / self.vm

    def _displaced(self, u, z0: int, z1: int):
        """Moving-image voxel indices of the fixed voxel centres in planes \`\`z0:z1\`\` under
        the field \`\`u\`\` and the affine, (d, h, w, 3)."""
        import torch.nn.functional as F

        p = self._points(z0, z1)
        g = 2 * (p - self.lo) / self.span - 1  # into the control grid, grid_sample wants x,y,z
        disp = F.grid_sample(
            u, g.flip(-1)[None], mode="bilinear", padding_mode="border", align_corners=True
        )[0].permute(1, 2, 3, 0)
        return self._index(p + disp)

    def seen(self, u, z0: int, z1: int, window: int):
        """Which fixed voxels in planes \`\`z0:z1\`\` have their whole \`\`window\`\` (as far as the
        fixed image reaches) inside the moving image under the starting field \`\`u\`\` and the
        affine: a (1, 1, d, h, w) float mask."""
        import torch.nn.functional as F

        idx = self._displaced(u, z0, z1)
        ok = ((idx >= 0) & (idx <= self.mshape)).all(dim=-1)[None, None].float()
        h = window // 2
        ok = F.pad(ok, (h,) * 6, value=1.0)  # beyond the fixed image: no data to miss
        for k in ((window, 1, 1), (1, window, 1), (1, 1, window)):  # a box minimum
            ok = -F.max_pool3d(-ok, k, stride=1)
        return ok

    def __call__(self, mov, u, z0: int, z1: int):
        import torch.nn.functional as F

        d, h, w = z1 - z0, self.shape[1], self.shape[2]
        g = 2 * self._displaced(u, z0, z1) / self.mshape - 1
        return F.grid_sample(  # 0 beyond the moving image, as the served image has
            mov, g.flip(-1)[None], mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(1, 1, d, h, w)


def _box_mean(x, window: int):
    """Mean over each voxel's window (clipped at the volume's faces), per channel: a
    separable box filter. A prefix-sum version, whose cost would not grow with the window,
    measured 4x slower at the default window (68 vs 17 ms on a 2^25-voxel level, RTX 2080
    Ti; cuDNN's pooling is that fast) and only wins past windows of about 60, so the
    browser engine's running sum is not mirrored here."""
    import torch.nn.functional as F

    h = window // 2
    for k in ((window, 1, 1), (1, window, 1), (1, 1, window)):  # one axis at a time
        pad = tuple(h if n > 1 else 0 for n in k)
        x = F.avg_pool3d(x, k, stride=1, padding=pad, count_include_pad=False)
    return x


def _window_stats(i, window: int):
    """The fixed image's window means and variances."""
    import torch

    mi, ii = _box_mean(torch.cat([i, i * i], dim=1), window).unbind(1)
    return mi, ii - mi * mi


def _lncc(i, stats, j, window: int):
    """Squared local normalized cross-correlation per voxel, in [0, 1]; \`\`stats\`\` are the
    fixed image \`\`i\`\`'s window means and variances (\`\`_window_stats\`\`)."""
    import torch

    mi, vi = stats
    mj, jj, ij = _box_mean(torch.cat([j, j * j, i * j], dim=1), window).unbind(1)
    cov = ij - mi * mj
    vj = jj - mj * mj
    # Windows without contrast in either image carry no signal and are left out (as in
    # ANTs), not damped with a constant: that would reward warps that add contrast.
    ok = (vi > FLAT) & (vj > FLAT)
    den = torch.where(ok, vi * vj, torch.ones_like(vi))
    return torch.where(ok, cov * cov / den, torch.zeros_like(vi))[:, None]


def _diffusion(u, spacing):
    """Mean squared gradient of the field (dimensionless: displacement per distance)."""
    total = 0.0
    for dim, sp in zip((2, 3, 4), spacing):
        d = u.diff(dim=dim) / float(sp)
        total = total + (d * d).mean()
    return total
`,_='"""The computed sources shipped with chunkmirage: ``synthetic://``, ``register://``,\n``stitch://`` and GeoTIFF files. Each is found through the ``chunkmirage.sources`` (or\n``chunkmirage.source_matchers``) entry point and imported only when a URL asks for it, so\nthis module imports none of them: ``register://`` needs torch, GeoTIFFs tifffile."""\n',v=`"""GeoTIFF sources, cloud-optimized GeoTIFFs (COGs) above all: the tiled TIFFs, with their
lower resolutions inside, that most geospatial and planetary rasters are published as.

    https://.../site.tif   (or s3://, gs://, a local path)

A COG is read where it is looked at: its header once (a few range requests), then each
512 x 512 tile the pipeline asks for, decoded by tifffile and kept in a small cache. Its
pages of decreasing size (the overviews) are the levels, so a zoomed-out view reads the
small ones. One sample per pixel; the axes are \`\`y, x\`\` in the projection's units (metres),
\`\`y\`\` counting down the image as rows do (minus the northing, for a north-up raster), and
voxel \`\`(0, 0)\`\` sits at the tiepoint. Float data's no-data value (\`\`GDAL_NODATA\`\`) is NaN.

Needs the \`\`tiff\`\` extra (tifffile, imagecodecs): tensorstore's \`\`tiff\`\` driver reads only
uint8 images.
"""

from __future__ import annotations

import io
import os
import threading
import urllib.request

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source


def is_geotiff(path: str) -> bool:
    return path.split("?")[0].lower().endswith((".tif", ".tiff"))


def match(path: str):
    """\`\`open_geotiff\`\` for a \`\`.tif\`\`/\`\`.tiff\`\` path, else None (the \`\`chunkmirage.source_matchers\`\` entry point)."""
    return open_geotiff if is_geotiff(path) else None


class _Ranges(io.RawIOBase):
    """A file of \`\`size\`\` bytes read through a tensorstore kvstore in 256 KB blocks, each
    fetched once: the header and page directories tifffile reads are many small reads."""

    BLOCK = 1 << 18

    def __init__(self, kv, key: str, size: int):
        self.kv, self.key, self.size, self.pos = kv, key, size, 0
        self.blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def bytes(self, start: int, stop: int) -> bytes:
        got = self.kv.read(self.key, byte_range=slice(start, stop)).result()
        if got.state != "value" or len(got.value) != stop - start:
            raise OSError(f"{self.key}: bytes {start}-{stop} could not be read ({got.state})")
        return bytes(got.value)

    def readinto(self, b) -> int:
        n = 0
        while n < len(b) and self.pos < self.size:
            i, o = divmod(self.pos, self.BLOCK)
            if i not in self.blocks:
                stop = min((i + 1) * self.BLOCK, self.size)
                self.blocks[i] = self.bytes(i * self.BLOCK, stop)
            piece = self.blocks[i][o : o + len(b) - n]
            b[n : n + len(piece)] = piece
            n += len(piece)
            self.pos += len(piece)
        return n


def _size(kv, key: str, url: str) -> int:
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            return int(r.headers["Content-Length"])
    if "://" not in url or url.startswith("file://"):
        return os.path.getsize(url.removeprefix("file://"))
    return len(kv.read(key).result().value)  # a bucket object: read once, whole


class GeoTIFFLevel(Source):
    """One page of a GeoTIFF: tiles fetched by range and decoded where a read covers them."""

    def __init__(self, page, ranges: _Ranges, info: ArrayInfo, key: str, cache: LRUCache):
        self.page, self.ranges, self._info, self._key, self.cache = page, ranges, info, key, cache
        self.nodata = None
        if "GDAL_NODATA" in page.tags and info.dtype.kind == "f":
            self.nodata = float(page.tags["GDAL_NODATA"].value)

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def _tile(self, ty: int, tx: int) -> np.ndarray:
        th, tw = self._info.chunk_shape
        across = -(-self._info.shape[1] // tw)
        i = ty * across + tx
        hit = self.cache.get((self._key, i))
        if hit is not None:
            return hit
        start, count = self.page.dataoffsets[i], self.page.databytecounts[i]
        if count == 0:  # a tile the writer left out: nothing there
            tile = np.full(
                (th, tw), np.nan if self._info.dtype.kind == "f" else 0, self._info.dtype
            )
        else:
            data, _, _ = self.page.decode(self.ranges.bytes(start, start + count), i)
            tile = np.asarray(data).reshape(th, tw)
            if self.nodata is not None and not np.isnan(self.nodata):
                tile = np.where(tile == self.nodata, np.nan, tile).astype(self._info.dtype)
        self.cache.put((self._key, i), tile)
        return tile

    def read(self, box: Box) -> np.ndarray:
        out = np.empty(box.shape, self._info.dtype)
        th, tw = self._info.chunk_shape
        for ty in range(box.start[0] // th, (box.stop[0] - 1) // th + 1):
            for tx in range(box.start[1] // tw, (box.stop[1] - 1) // tw + 1):
                y0, x0 = ty * th, tx * tw
                ys = slice(max(box.start[0], y0), min(box.stop[0], y0 + th))
                xs = slice(max(box.start[1], x0), min(box.stop[1], x0 + tw))
                tile = self._tile(ty, tx)
                out[ys.start - box.start[0] : ys.stop - box.start[0],
                    xs.start - box.start[1] : xs.stop - box.start[1]] = tile[
                    ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
                ]  # fmt: skip
        return out


def open_geotiff(path: str, *, cache_bytes: int = 0, **_) -> MultiscaleSource:
    import tensorstore as ts
    import tifffile

    head, _, key = path.rstrip("/").rpartition("/")
    root = head + "/"  # a kvstore's keys are appended to its root as they are
    kv = ts.KvStore.open(root if "://" in head else {"driver": "file", "path": root}).result()
    ranges = _Ranges(kv, key, _size(kv, key, path))
    with _lock:
        tif = tifffile.TiffFile(io.BufferedReader(ranges, 1 << 16), name=key)
    pages = [p for p in tif.pages if p.shape[:2] and (p.subfiletype & 4) == 0]  # no masks
    base = pages[0]
    if base.samplesperpixel != 1:
        raise ValueError(f"{path}: {base.samplesperpixel} samples per pixel; one is read")
    if not base.is_tiled:
        raise ValueError(f"{path}: stored in strips, not tiles; only tiled GeoTIFFs are read")
    tags = base.tags
    sx, sy = (
        (float(v) for v in tags["ModelPixelScaleTag"].value[:2])
        if "ModelPixelScaleTag" in tags
        else (1.0, 1.0)
    )
    tie = tags["ModelTiepointTag"].value if "ModelTiepointTag" in tags else (0, 0, 0, 0, 0, 0)
    # raster (i, j) -> model (x, y): the top-left corner of pixel (0, 0)
    x0, y0 = float(tie[3]) - float(tie[0]) * sx, float(tie[4]) + float(tie[1]) * sy
    geo = tif.geotiff_metadata or {}
    projected = int(geo.get("GTModelTypeGeoKey", 1)) == 1 and "ModelPixelScaleTag" in tags
    unit = "m" if projected else ""  # a geographic raster's degrees have no viewer unit
    cache = LRUCache(max(int(cache_bytes), 256 * 2**20))
    levels = []
    for i, page in enumerate(pages):
        h, w = page.shape[:2]
        fy, fx = base.shape[0] / h, base.shape[1] / w  # the overview's pixel, in the base's
        vy, vx = sy * fy, sx * fx
        info = ArrayInfo(
            shape=(h, w),
            dtype=page.dtype,
            chunk_shape=(page.tilelength, page.tilewidth),
            voxel_size=(vy, vx),
            units=(unit, unit),
            axes=("y", "x"),
            translation=(-y0 + vy / 2, x0 + vx / 2),  # pixel centres; y is minus the northing
        )
        levels.append(GeoTIFFLevel(page, ranges, info, f"geotiff:{path}:{i}", cache))
    return MultiscaleSource(levels, name=path)


_lock = threading.Lock()
`,y=`"""\`\`register://\`\` sources: a moving image registered onto a fixed image's grid, the
deformable part solved (on a GPU if there is one) when the source opens.

    register://<moving>?fixed=<fixed>&affine=<matrix.npy>
    register://<moving>?fixed=...&show=pair
    register://<moving>?fixed=...&show=field
    register://<moving>?fixed=...&frames=12
    register://<moving>?fixed=...&refine=3&halo=8

(the parameters follow the last \`\`?\`\`, so \`\`<moving>\`\` may have a query of its own;
percent-encode \`\`fixed\`\` if it has one). Opening the source reads a few coarse levels of
both images and fits a smooth displacement field \`\`u\`\` in the fixed image's space
(\`\`chunkmirage.registration\`\`), which stays in memory: a whole organ's is megabytes.
Every level of the output is the moving image sampled at \`\`affine(p + u(p))\`\` for each
fixed voxel \`\`p\`\`, computed chunk by chunk by the \`\`scene://\`\` resampler, so even full
resolution is served without anything being written. Solved fields are kept per process
(the last few), so reopening the same URL, as editing a downstream op does, reuses them.

With \`\`refine\`\`, the field of the levels below the solved ones is fitted where it is
looked at: each has its own control lattice, in blocks of \`\`block\`\` voxels, and a block is
fitted when a chunk it covers is first requested, starting from the solved field over the
block plus \`\`halo\`\` voxels of context, coarse to fine within the block, then kept in the
chunk cache of whoever opened it (a server's, so \`\`--cache-gb\`\` bounds them and clearing
the cache clears them; \`\`_blocks\`\` when opened from Python without one). So the fit's detail grows with the zoom, its cost with what is viewed rather
than with the volume, and any client asking for chunks drives it.

Query parameters:

* \`\`fixed\`\`: the fixed image (required): anything \`\`open_source\`\` reads. The output has
  its grid and levels.
* \`\`affine\`\`: the fixed-to-moving affine in physical units, C order: a \`\`.npy\`\` or text
  file holding a 4x4 or 3x4 matrix, or its 12 or 16 values inline, row by row (default:
  identity), or \`\`auto\`\` to find one from the images' intensity moments and a correlation
  fit, among orientations that do not mirror the image (a mirror image needs its affine
  given). The moving image is sampled at \`\`affine(p + u(p))\`\`.
* \`\`fixed_channel\`\`, \`\`moving_channel\`\`: the channel each image is matched on, for images
  with a \`\`c\`\` axis (default 0). The output has every channel of the moving image.
* \`\`levels\`\`: fixed-image levels to solve on, coarse to fine, e.g. \`\`6,5,4\`\` (default:
  from the coarsest with at least 16 voxels on every axis to the finest with at most
  2^25). Each is matched against the moving level nearest its voxel size.
* \`\`iterations\`\`: Adam steps per level, one value, one per level or one per level with
  the refined ones (default 100). \`\`0\`\` skips the solve: the affine alone, which needs no
  GPU.
* \`\`refine\`\`: levels below the solved ones fitted block by block on demand (default 0),
  \`\`halo\`\`: voxels of context around a block when it is fitted (default 8), \`\`block\`\`:
  voxels per block, C order (default 64,128,128).
* \`\`smooth\`\`: weight of the penalty on the field's gradient (default 1).
* \`\`grid\`\`: control-point spacing of the field, in voxels of each level (default 4).
* \`\`window\`\`: correlation window, voxels, odd: one value, one per level or one per level
  with the refined ones (default 7). Fine levels gain from a wider one.
* \`\`show\`\`: \`\`image\`\` (default), \`\`pair\`\` (a \`\`c\`\` axis of two: the fixed image's channel,
  then the registered moving image's, to compare in one shader) or \`\`field\`\` (\`\`u\`\`
  itself, components first, in physical units).
* \`\`frames\`\`: a leading \`\`t\`\` axis of that many fields from the solve, evenly spaced from
  none (the affine alone) to the final one, to watch it converge. The moving image's own
  \`\`t\`\` axis must have one time point, which the frames replace.
* \`\`interpolation\`\`: \`\`linear\`\` (default) or \`\`nearest\`\` (default for uint32/uint64).
* \`\`chunk\`\`: output chunk shape of the spatial axes, C order (default the fixed image's).
* \`\`device\`\`: \`\`auto\`\` (default: the GPU with the most free memory, else the CPU),
  \`\`cpu\`\`, \`\`cuda:1\`\`, ...
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, field_validator

from chunkmirage import demand
from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.fused import tent
from chunkmirage.neuroglancer import compare_shader, field_shader
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.scene import SceneLevelSource, _pick_level
from chunkmirage.sources.warp import _SPATIAL, FieldLevelSource, FramesSource
from chunkmirage.transforms import Affine, Displacements, Sequence, VectorField, simplify
from chunkmirage_contrib.registration import (
    Grid,
    Level,
    Settings,
    _control_shape,
    _halve,
    _range,
    find_affine,
    solve,
)

log = logging.getLogger("chunkmirage")


class RegisterParams(BaseModel):
    """The query of a \`\`register://\`\` URL. The one definition of it: the browser engine
    (\`\`web/\`\`) generates its TypeScript types and form defaults from this model's JSON
    Schema (\`\`chunkmirage schema\`\`)."""

    model_config = ConfigDict(extra="forbid")

    fixed: str = Field(
        description="The fixed image: anything open_source reads. The output has its grid and levels."
    )
    affine: str | None = Field(
        None,
        description=(
            "The fixed-to-moving affine in physical units, C order: a .npy or text file holding a 4x4"
            " or 3x4 matrix, or its 12 or 16 values inline, row by row; or auto, to find one from"
            " the images' intensity moments and a correlation fit. Default: identity."
        ),
    )
    fixed_channel: int = Field(
        0, ge=0, description="The fixed image's channel to match on, for images with a c axis."
    )
    moving_channel: int = Field(
        0,
        ge=0,
        description=(
            "The moving image's channel to match on; the output has every channel of the moving image."
        ),
    )
    levels: list[int] | None = Field(
        None,
        description=(
            "Fixed-image levels to solve on, coarse to fine. Default: from the coarsest with at least"
            " 16 voxels on every axis to the finest with at most 2^25."
        ),
    )
    iterations: list[NonNegativeInt] = Field(
        [100],
        description=(
            "Adam steps per level: one value, one per level, or one per level including the"
            " refined ones. 0 skips the solve (the affine alone)."
        ),
    )
    refine: int = Field(
        0,
        ge=0,
        description=(
            "Levels below the solved ones whose field is fitted on demand, block by block, where"
            " it is viewed: refine=2 with levels 6,5,4 fits levels 3 and 2, a block (one output"
            " chunk) when a chunk it covers is first requested, from the solved field, coarse to"
            " fine within the block. The finest fitted level's field serves every finer one."
        ),
    )
    halo: int = Field(
        8,
        ge=0,
        description=(
            "Voxels of context on every side of a block whose field is fitted on demand: how far"
            " beyond the block the fit looks, so neighbouring blocks agree."
        ),
    )
    block: list[PositiveInt] = Field(
        [64, 128, 128],
        min_length=3,
        max_length=3,
        description=(
            "Voxels per block of a field fitted on demand, C order. Independent of the output's"
            " chunks: a chunk's field reaches into the neighbouring blocks, so blocks deeper than a"
            " thin chunk fit fewer voxels in all, even for one slice."
        ),
    )
    smooth: float = Field(1.0, ge=0, description="Weight of the penalty on the field's gradient.")
    grid: float = Field(
        4.0, gt=0, description="Control-point spacing of the field, in voxels of each level."
    )
    window: list[int] = Field(
        [7],
        min_length=1,
        description=(
            "Correlation window, voxels, odd: one value, one per level, or one per level"
            " including the refined ones. Fine levels gain from a wider one."
        ),
    )
    show: Literal["image", "pair", "field"] = Field(
        "image",
        description=(
            "image: the registered moving image; pair: a c axis of two, the fixed image's channel"
            " then the registered one; field: u itself, components first, in physical units."
        ),
    )
    frames: int | None = Field(
        None,
        ge=1,
        description=(
            "A leading t axis of that many fields from the solve, from none (the affine alone) to the"
            " final one, to watch it converge."
        ),
    )
    interpolation: Literal["linear", "nearest"] | None = Field(
        None, description="Default: nearest for uint32/uint64 (labels), else linear."
    )
    chunk: list[PositiveInt] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Output chunk shape of the spatial axes, C order. Default: the fixed image's.",
    )
    device: str = Field(
        "auto",
        description="auto (the GPU with the most free memory, else the CPU), cpu, cuda:1, ...",
    )

    @classmethod
    def from_query(cls, q: dict[str, str]) -> RegisterParams:
        if unknown := set(q) - set(cls.model_fields):
            raise ValueError(
                f"unknown register:// parameters {sorted(unknown)}; allowed: {sorted(cls.model_fields)}"
            )
        return cls(**q)

    @field_validator("levels", "iterations", "window", "chunk", "block", mode="before")
    @classmethod
    def _comma_separated(cls, v):
        return [int(x) for x in v.split(",")] if isinstance(v, str) else v

    @field_validator("window")
    @classmethod
    def _odd(cls, v):
        if any(w < 1 or w % 2 == 0 for w in v):
            raise ValueError(f"window must be positive odd numbers, got {v}")
        return v


MAX_SOLVE_VOXELS = 1 << 25  # finest default level: its whole volume sits on the GPU
MIN_SOLVE_SIZE = 16  # coarsest default level: at least this many voxels on every axis
KEEP = 8  # solved fields remembered per process
BLOCK_CACHE_BYTES = 1 << 30  # blocks of refined fields kept when no cache is given
_solved: OrderedDict[str, list[Grid]] = OrderedDict()
_affines: dict[str, np.ndarray] = {}  # found affines, per image pair
_ranges: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}  # per solve key
_blocks: LRUCache[np.ndarray] = LRUCache(BLOCK_CACHE_BYTES)
# fits of refined blocks: three at once (they read while one holds the GPU), the finest level
# first, dropped if every request for one gives up before it starts
_block_queue = demand.queues["refined blocks"] = demand.Queue(slots=3)
_solve_lock = threading.Lock()  # one solve at a time: they share the GPU


class PairSource(Source):
    """The fixed image's channel and the registered moving image's, as the two channels of
    one \`\`(c, z, y, x)\`\` array (to compare them in a single shader)."""

    def __init__(self, fixed: Source, fixed_lead, registered: Source, moving_lead, info, key):
        self.parts = ((fixed, tuple(fixed_lead)), (registered, tuple(moving_lead)))
        self._info = info
        self._key = key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        space = Box(box.start[1:], box.stop[1:])
        out = np.empty(box.shape, dtype=self._info.dtype)
        for i, c in enumerate(range(box.start[0], box.stop[0])):
            src, lead = self.parts[c]
            b = Box(lead + space.start, tuple(v + 1 for v in lead) + space.stop)
            out[i] = src.read(b).reshape(space.shape)
        return out


def _n_lead(ms: MultiscaleSource, url: str) -> int:
    axes = ms.levels[0].info.axes
    n = 0
    while n < len(axes) and axes[n] not in _SPATIAL:
        n += 1
    if len(axes) - n != 3 or any(a not in _SPATIAL for a in axes[n:]):
        raise ValueError(f"{url}: register:// needs axes ending in z, y, x; got {axes}")
    return n


def _lead(ms: MultiscaleSource, n_lead: int, channel: int, what: str) -> tuple[int, ...]:
    """Index of the matched channel on the leading axes (the first of any other)."""
    info = ms.levels[0].info
    axes, shape = info.axes[:n_lead], info.shape[:n_lead]
    if "c" not in axes and channel:
        raise ValueError(f"{what}_channel={channel}, but the {what} image has no c axis")
    if "c" in axes and not 0 <= channel < shape[axes.index("c")]:
        raise ValueError(f"{what}_channel={channel} is outside its {shape[axes.index('c')]}")
    return tuple(channel if a == "c" else 0 for a in axes)


def _level(src: Source, lead: tuple[int, ...], start=None, stop=None) -> Level:
    """The matched channel of \`\`src\`\` (its voxels \`\`start:stop\`\`, default all) as a \`\`Level\`\`."""
    info = src.info
    n = len(lead)
    start = (0,) * 3 if start is None else tuple(int(v) for v in start)
    stop = info.shape[n:] if stop is None else tuple(int(v) for v in stop)
    box = Box(lead + start, tuple(v + 1 for v in lead) + stop)
    vox = np.asarray(info.voxel_size[n:], dtype=float)
    return Level(
        src.read(box).reshape(box.shape[n:]),
        vox,
        np.asarray(info.translation[n:], dtype=float) + np.asarray(start) * vox,
    )


def _spatial_vox(ms: MultiscaleSource, n_lead: int) -> list[np.ndarray]:
    return [np.asarray(lvl.info.voxel_size[n_lead:], dtype=float) for lvl in ms.levels]


def _default_levels(ms: MultiscaleSource, n_lead: int) -> list[int]:
    shapes = [lvl.info.shape[n_lead:] for lvl in ms.levels]
    fits = [i for i, s in enumerate(shapes) if math.prod(s) <= MAX_SOLVE_VOXELS]
    finest = fits[0] if fits else len(shapes) - 1
    coarse = [i for i, s in enumerate(shapes) if min(s) >= MIN_SOLVE_SIZE]
    coarsest = max(max(coarse, default=finest), finest)
    return list(range(coarsest, finest - 1, -1))


def _display_range(data: np.ndarray) -> tuple[float, float]:
    """Contrast limits from the voxels with data (which skips padding)."""
    v = data[data > 0]
    lo, hi = np.percentile(v, [1, 99.8]) if v.size else (0.0, 1.0)
    return float(lo), float(max(hi, lo + 1))


def _digest(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:n]


def _affine(value: str | None) -> np.ndarray:
    if value is None:
        return np.eye(4)
    try:
        m = np.array([float(v) for v in value.split(",")])
    except ValueError:  # a file
        path = Path(value)
        m = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    m = np.asarray(m, dtype=float)
    if m.size in (12, 16):
        m = m.reshape(-1, 4)
    if m.shape == (3, 4):
        m = np.vstack([m, [0, 0, 0, 1]])
    if m.shape != (4, 4):
        raise ValueError(f"affine must be 4x4 or 3x4 (or 12/16 values), got shape {m.shape}")
    return m


def _refined_field(
    fixed: Source,
    flead: tuple[int, ...],
    moving: Source,
    mlead: tuple[int, ...],
    affine: np.ndarray,
    parent: VectorField,
    lo: np.ndarray,
    hi: np.ndarray,
    block_voxels: tuple[int, ...],
    settings: Settings,
    halo: int,
    stages: int,
    ranges,
    device: str,
    level: int,
    blocks: LRUCache,
) -> VectorField:
    """The field of a fixed level below the solved ones: a control lattice \`\`settings.grid\`\`
    voxels apart, in blocks of \`\`block_voxels\`\`. A block is fitted when first read,
    from \`\`parent\`\`'s values (the solved field) over the block plus \`\`halo\`\` voxels of
    context, against the fixed and moving voxels those reach, coarse to fine over
    \`\`stages\`\` halvings of them (so the block's coarsest copy is about as coarse as the
    solved level), and kept in \`\`blocks\`\`, context included. Neighbouring blocks overlap
    there, and a lattice point's value is the average of the blocks covering it, each
    weighted 1 over its own block and less the further into its context (\`\`fused.tent\`\`), so the
    field turns smoothly from one block's fit to the next's across the overlap instead of
    stepping at their boundary. A block's fit depends on nothing but the solved field and
    the images, so neither does the blend on the order blocks are fitted in."""
    finfo, minfo = fixed.info, moving.info
    nf, nm = len(flead), len(mlead)
    fvox = np.asarray(finfo.voxel_size[nf:], dtype=float)
    ft = np.asarray(finfo.translation[nf:], dtype=float)
    fshape = np.asarray(finfo.shape[nf:])
    mvox = np.asarray(minfo.voxel_size[nm:], dtype=float)
    mt = np.asarray(minfo.translation[nm:], dtype=float)
    mshape = np.asarray(minfo.shape[nm:])
    a = np.asarray(affine, dtype=float)
    spacing = settings.grid * fvox
    n = np.asarray(_control_shape(lo, hi, fvox, settings.grid))
    block = np.asarray([math.ceil(c / settings.grid) for c in block_voxels])
    context = math.ceil(halo / settings.grid)  # lattice points
    window = settings.window[0]
    w2 = window // 2
    margin = halo + w2 + 1  # moving voxels beyond the parent field's reach a block may move
    info = ArrayInfo(
        shape=(*n, 3),
        dtype=np.float32,
        chunk_shape=(*block, 3),
        voxel_size=(*spacing, 1.0),
        units=(*finfo.units[nf:], ""),
        axes=("z", "y", "x", "c"),
        translation=(*lo, 0.0),
    )
    key = "register:" + _digest(
        f"refine|{parent.key}|{fixed.cache_key()}|{moving.cache_key()}|{flead}|{mlead}|"
        f"{np.round(a, 12).tolist()}|{dataclasses.asdict(settings)}|{halo}|{block_voxels}"
    )

    def sampled(blo, bhi, shape) -> np.ndarray:
        """\`\`parent\`\` on a grid of \`\`shape\`\` points spanning the box, (z, y, x, 3)."""
        g = np.mgrid[tuple(slice(0, s) for s in shape)].reshape(3, -1).T
        pts = blo + g * (bhi - blo) / (np.asarray(shape) - 1)
        return parent.sample(pts).reshape(*shape, 3).astype(np.float32)

    def extent(index):
        """Block \`\`index\`\`'s own lattice points \`\`c0:c1\`\` and those it is fitted over, \`\`k0:k1\`\`."""
        c0 = np.asarray(index) * block
        c1 = np.minimum(c0 + block, n)
        k0, k1 = np.maximum(c0 - context, 0), np.minimum(c1 + context, n)
        k1 = np.minimum(np.maximum(k1, k0 + 2), n)  # a fit needs two points per axis
        k0 = k1 - np.maximum(k1 - k0, 2)
        return c0, c1, k0, k1

    def compute(index) -> np.ndarray:
        """Block \`\`index\`\` fitted: its values over \`\`k0:k1\`\`, (z, y, x, 3)."""
        c0, c1, k0, k1 = extent(index)
        blo, bhi = lo + k0 * spacing, lo + (k1 - 1) * spacing
        if settings.iterations[0] == 0:
            return sampled(blo, bhi, k1 - k0)
        # the fixed voxels whose correlation windows the block's points reach
        v0 = np.maximum(np.floor((blo - ft) / fvox).astype(int) - w2, 0)
        v1 = np.minimum(np.ceil((bhi - ft) / fvox).astype(int) + w2 + 1, fshape)
        if (v1 - v0 < window).any():  # beyond the image, or too thin to correlate
            return sampled(blo, bhi, k1 - k0)
        fl = _level(fixed, flead, v0, v1)
        # the moving voxels those reach under the parent field and the affine, with room to move
        q = fl.translation + np.mgrid[0:9, 0:9, 0:9].reshape(3, -1).T / 8 * (v1 - v0 - 1) * fvox
        q = (q + parent.sample(q)) @ a[:3, :3].T + a[:3, 3]
        q = (q - mt) / mvox
        m0 = np.maximum(np.floor(q.min(axis=0)).astype(int) - margin, 0)
        m1 = np.minimum(np.ceil(q.max(axis=0)).astype(int) + margin + 1, mshape)
        if (m1 <= m0).any():  # nothing of the moving image here
            return sampled(blo, bhi, k1 - k0)
        ml = _level(moving, mlead, m0, m1)
        fs, ms = [fl], [ml]
        for _ in range(stages):  # coarse to fine within the block, while a copy is worth it
            if min(fs[0].data.shape) < 2 * max(MIN_SOLVE_SIZE, window):
                break
            fs.insert(0, _halve(fs[0]))
            ms.insert(0, _halve(ms[0]))
        init = sampled(blo, bhi, _control_shape(blo, bhi, fs[0].voxel_size, settings.grid))
        with _solve_lock:  # one fit at a time: they share the GPU
            grid = solve(
                fs,
                ms,
                a,
                dataclasses.replace(
                    settings,
                    iterations=settings.iterations[:1] * len(fs),
                    window=settings.window[:1] * len(fs),
                ),
                box=(blo, bhi),
                device=device,
                init=init,
                ranges=ranges,
                label=f" level {level} block {tuple(int(i) for i in index[:3])}",
            )[0]
        return grid.values

    def fitted(index: tuple[int, ...]) -> np.ndarray:
        """Block \`\`index\`\`'s fit: kept in \`\`blocks\`\`, fitted once through \`\`_block_queue\`\`
        however many requests want it, and dropped unfitted if they all give up first."""
        hit = blocks.get((key, index))
        if hit is not None:
            return hit

        def fit() -> np.ndarray:
            values = np.ascontiguousarray(compute(index), dtype=np.float32)
            blocks.put((key, index), values)
            return values

        return _block_queue.run((key, index), level, fit)

    nblocks = -(-n // block)

    def read(sl: tuple[slice, ...]) -> np.ndarray:
        """The blended lattice over \`\`sl\`\`: every block whose fit reaches it, weighted."""
        w0 = np.asarray([s.start or 0 for s in sl[:3]])
        w1 = np.asarray([size if s.stop is None else s.stop for s, size in zip(sl[:3], n)])
        acc = np.zeros((*(w1 - w0), 3))
        total = np.zeros(tuple(w1 - w0))
        b0 = np.maximum((w0 - context) // block, 0)
        b1 = np.minimum((w1 - 1 + context) // block, nblocks - 1)
        for index in np.ndindex(*(b1 - b0 + 1)):
            index = tuple(int(v) for v in b0 + index)
            c0, c1, k0, k1 = extent(index)
            i0, i1 = np.maximum(k0, w0), np.minimum(k1, w1)
            if (i1 <= i0).any():
                continue
            weight = tent(i0, i1, c0, c1, context)
            if not weight.any():
                continue
            values = fitted(index)[tuple(slice(a - k, b - k) for a, b, k in zip(i0, i1, k0))]
            box = tuple(slice(a - w, b - w) for a, b, w in zip(i0, i1, w0))
            acc[box] += weight[..., None] * values
            total[box] += weight
        return (acc / total[..., None]).astype(np.float32)[..., sl[3]]

    return VectorField(read, info.shape, 3, Affine.scale_translation(spacing, lo), key)


_FIELDS = ("shape", "chunk_shape", "voxel_size", "units", "axes", "translation")


def _axes_of(info: ArrayInfo, sl: slice) -> ArrayInfo:
    """\`\`info\`\` restricted to the axes \`\`sl\`\`."""
    return ArrayInfo(dtype=info.dtype, **{f: tuple(getattr(info, f)[sl]) for f in _FIELDS})


def _prepend(space: ArrayInfo, lead: ArrayInfo) -> ArrayInfo:
    """Axes of \`\`lead\`\` followed by those of \`\`space\`\`, with \`\`lead\`\`'s dtype."""
    return ArrayInfo(
        dtype=lead.dtype, **{f: tuple(getattr(lead, f)) + tuple(getattr(space, f)) for f in _FIELDS}
    )


def open_register(
    url: str, *, cache_bytes: int = 0, cache: LRUCache | None = None
) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    location, _, query = url[len("register://") :].rpartition("?")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if "fixed" not in q or not location:
        raise ValueError(
            "register:// needs a moving image and fixed=...: register://<moving>?fixed=<fixed>"
            " (the parameters follow the last '?')"
        )
    p = RegisterParams.from_query(q)

    mov = open_source(location, cache_bytes=cache_bytes, cache=cache)
    fix = open_source(p.fixed, cache_bytes=cache_bytes, cache=cache)
    nm, nf = _n_lead(mov, location), _n_lead(fix, p.fixed)
    minfo, finfo = mov.levels[0].info, fix.levels[0].info
    if tuple(minfo.axes[nm:]) != tuple(finfo.axes[nf:]):
        raise ValueError(f"spatial axes differ: moving {minfo.axes}, fixed {finfo.axes}")
    if tuple(minfo.units[nm:]) != tuple(finfo.units[nf:]):
        raise ValueError(
            f"spatial units differ: moving {minfo.units[nm:]}, fixed {finfo.units[nf:]}"
        )
    mlead = _lead(mov, nm, p.moving_channel, "moving")
    flead = _lead(fix, nf, p.fixed_channel, "fixed")

    levels = p.levels or _default_levels(fix, nf)
    if any(not 0 <= i < len(fix.levels) for i in levels):
        raise ValueError(f"levels {levels}: the fixed image has levels 0..{len(fix.levels) - 1}")
    if p.refine > levels[-1]:
        raise ValueError(
            f"refine={p.refine}: {levels[-1]} level(s) lie below level {levels[-1]}, the finest solved"
        )
    if p.refine and p.frames is not None:
        raise ValueError(
            "frames and refine do not combine: refined blocks are fitted after the solve"
        )
    n_all = len(levels) + p.refine

    def per_level(name: str, values: list[int]) -> list[int]:
        """\`\`values\`\` for every level, refined ones included: one value for all, one per
        solved level (the refined ones take the last) or one per level."""
        if len(values) not in {1, len(levels), n_all}:
            counts = " or ".join(str(c) for c in sorted({1, len(levels), n_all}))
            raise ValueError(
                f"{name} needs {counts} values: one, one per level, or one per level with the"
                " refined ones"
            )
        values = values * n_all if len(values) == 1 else values
        return values + values[-1:] * (n_all - len(values))

    iters, windows = per_level("iterations", p.iterations), per_level("window", p.window)
    settings = Settings(
        iterations=tuple(iters[: len(levels)]),
        smooth=p.smooth,
        grid=p.grid,
        window=tuple(windows[: len(levels)]),
    )
    moving_t = p.show == "image" and "t" in minfo.axes  # which the frames replace
    if p.frames is not None and moving_t and (minfo.axes[0] != "t" or minfo.shape[0] != 1):
        raise ValueError(
            f"{location}: frames need a moving image without a time axis, or with one time"
            f" point first; axes {minfo.axes}, shape {minfo.shape}"
        )

    labels = minfo.dtype.kind == "u" and minfo.dtype.itemsize >= 4
    interpolation = p.interpolation or ("nearest" if labels else "linear")
    order = 0 if interpolation == "nearest" else 1

    # Solve (or recall): moving levels nearest each fixed level's voxel size.
    mvox, fvox = _spatial_vox(mov, nm), _spatial_vox(fix, nf)
    mlevels = [int(np.argmin([np.abs(np.log(m / fvox[i])).sum() for m in mvox])) for i in levels]
    coarsest: list[Level] = []  # the coarsest solved level of each image, read once if needed

    def read_coarsest() -> list[Level]:
        if not coarsest:
            coarsest.append(_level(fix.levels[levels[0]], flead))
            coarsest.append(_level(mov.levels[mlevels[0]], mlead))
        return coarsest

    if p.affine == "auto":
        pair = [fix.levels[levels[0]].cache_key(), mov.levels[mlevels[0]].cache_key(), flead, mlead]
        akey = _digest(json.dumps(pair), 16)
        if akey not in _affines:
            with _solve_lock:
                _affines[akey] = find_affine(*read_coarsest(), device=p.device)[0]
        affine = _affines[akey]
    else:
        affine = _affine(p.affine)
    ident = {
        "fixed": [fix.levels[i].cache_key() for i in levels],
        "moving": [mov.levels[j].cache_key() for j in mlevels],
        "lead": [flead, mlead],
        "affine": np.round(affine, 12).tolist(),
        "settings": dataclasses.asdict(settings),
        "frames": p.frames,
    }
    solve_key = _digest(json.dumps(ident, sort_keys=True), 16)
    lo = np.asarray(finfo.translation[nf:], dtype=float) - fvox[0] / 2
    hi = lo + np.asarray(finfo.shape[nf:]) * fvox[0]

    with _solve_lock:
        grids = _solved.get(solve_key)
        if grids is not None:
            _solved.move_to_end(solve_key)
    if grids is None:
        # read outside the lock, so other opens need not wait behind this one's I/O; the
        # affine alone reads nothing
        solving = sum(settings.iterations) > 0
        fixed_levels, moving_levels = [], []
        if solving:
            fixed_levels = [read_coarsest()[0]] + [_level(fix.levels[i], flead) for i in levels[1:]]
            moving_levels = [read_coarsest()[1]] + [
                _level(mov.levels[j], mlead) for j in mlevels[1:]
            ]
        with _solve_lock:  # one solve at a time: they share the GPU
            if solve_key not in _solved:
                log.info("register: %s onto %s at levels %s", location, p.fixed, levels)
                _solved[solve_key] = solve(
                    fixed_levels,
                    moving_levels,
                    affine,
                    settings,
                    box=(lo, hi),
                    device=p.device,
                    snapshots=p.frames or 1,
                )
                if solving:
                    _ranges[solve_key] = (
                        _range(fixed_levels[0].data),
                        _range(moving_levels[0].data),
                    )
                while len(_solved) > KEEP:
                    _ranges.pop(_solved.popitem(last=False)[0], None)
            grids = _solved[solve_key]

    to_moving = Affine(affine[:3])

    def field_of(k: int) -> VectorField:
        g = grids[k]
        values = g.values
        return VectorField(
            lambda sl: values[sl],
            values.shape,
            3,
            Affine.scale_translation(g.spacing, g.origin),
            f"register:{solve_key}:{k}",
        )

    fields = [field_of(k) for k in range(len(grids))]

    def space_chunk(i: int) -> tuple[int, ...]:
        return tuple(p.chunk or fix.levels[i].info.chunk_shape[nf:])

    # Below the solved levels, each refined level's field is fitted block by block on
    # demand from the solved field (nothing is fitted until a chunk is asked for).
    refined: dict[int, VectorField] = {}
    if p.refine:
        ranges = _ranges.get(solve_key)
        if ranges is None:
            ranges = _ranges[solve_key] = tuple(_range(lvl.data) for lvl in read_coarsest())
        for r, i in enumerate(range(levels[-1] - 1, levels[-1] - 1 - p.refine, -1)):
            j = int(np.argmin([np.abs(np.log(m / fvox[i])).sum() for m in mvox]))
            refined[i] = _refined_field(
                fix.levels[i],
                flead,
                mov.levels[j],
                mlead,
                affine,
                fields[0],
                lo,
                hi,
                tuple(p.block),
                dataclasses.replace(
                    settings,
                    iterations=(iters[len(levels) + r],),
                    window=(windows[len(levels) + r],),
                ),
                p.halo,
                levels[-1] - i,
                ranges,
                p.device,
                i,
                _blocks if cache is None else cache,
            )

    def field_for(i: int, k: int) -> VectorField:
        """The field output level \`\`i\`\` is served through: the finest fitted at or above it."""
        if not refined or i >= levels[-1]:
            return fields[k]
        return refined[max(i, min(refined))]

    def level(i: int, k: int) -> Source:
        """Fixed level \`\`i\`\` through field \`\`k\`\`."""
        fl = fix.levels[i].info
        grid = dataclasses.replace(_axes_of(fl, slice(nf, None)), chunk_shape=space_chunk(i))
        to_space = Affine.scale_translation(fvox[i], fl.translation[nf:])
        field = field_for(i, k)
        key = "register:" + _digest(f"register|{field.key}|{i}|{p.show}|{order}|{grid.chunk_shape}")
        if p.show == "field":
            info = _prepend(grid, ArrayInfo((3,), np.float32, (3,), (1.0,), ("",), ("c",)))
            return FieldLevelSource(field, to_space, info, key)

        def mapping_for(j: int, through: VectorField):
            ml = mov.levels[j].info
            to_index = Affine.scale_translation(mvox[j], ml.translation[nm:]).inverse()
            return simplify(Sequence([to_space, Displacements(through), to_moving, to_index]))

        # the moving level is chosen through the solved field: sampling a refined one
        # here would fit blocks before anything is viewed
        j = _pick_level(grid.shape, mapping_for(0, fields[k]), mvox)
        ml = mov.levels[j].info
        info = _prepend(grid, _axes_of(ml, slice(nm)))
        registered = SceneLevelSource(mov.levels[j], mapping_for(j, field), info, nm, order, key)
        if p.show == "image":
            return registered
        dtype = np.result_type(fl.dtype, ml.dtype)
        pair = _prepend(grid, ArrayInfo((2,), dtype, (2,), (1.0,), ("",), ("c",)))
        return PairSource(fix.levels[i], flead, registered, mlead, pair, key + ":pair")

    out: list[Source] = []
    for i in range(len(fix.levels)):
        frames = [level(i, k) for k in range(len(grids))]
        if p.frames is None:
            out.append(frames[0])
            continue
        key = "register:" + _digest("frames|" + "|".join(f.cache_key() for f in frames))
        out.append(FramesSource(frames, moving_t, key))
    # how a viewer should show it: the pair compared in colour, the field as a heat map
    shader = None
    if p.show == "pair":
        shader = compare_shader(*(_display_range(lvl.data) for lvl in read_coarsest()))
    elif p.show == "field":
        moved = np.linalg.norm(grids[-1].values, axis=-1)
        shader = field_shader(max(float(np.percentile(moved, 99)), float(np.mean(fvox[0]))))
    return MultiscaleSource(out, name=f"{mov.name or location}~registered", shader=shader)
`,b=`"""\`\`stitch://\`\` sources: the tiles of a BigStitcher project stitched by interest points and
fused, region by region as they are read; no fused copy is written.

    stitch://<project.xml>?channel=0&model=translation&epsilon=5&threshold=0.005

When opened, each overlap of two tiles' stage positions (grown by \`\`margin\`\`) is read at
\`\`level\`\`, its interest points found, matched between the two tiles and filtered by RANSAC,
and every tile's placement fitted to the kept matches (\`\`chunkmirage.stitching\`\`; the
parameters are \`\`StitchParams\`\`). Each level of the fused volume then has the tiles' voxel
size at that level and covers them all; a region of it is each tile that reaches it sampled
where its placement puts it, blended at the seams. The project's images must be OME-Zarr
(BigStitcher-Spark's \`\`bdv.multimg.zarr\`\`); the outermost transforms named "Stitching
Transform" (a stitching done before) are left out of the stage positions, and the result is
compared with them in the log.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage_contrib import stitching as S

log = logging.getLogger("chunkmirage")
CHUNK = (64, 128, 128)


class FusedLevel(Source):
    """One level of the fused tiles."""

    def __init__(self, tiles, sources, placements, level: int, grid: dict, info: ArrayInfo,
                 blend: float, key: str):
        self.tiles, self.sources, self.placements = tiles, sources, placements
        self.level, self.grid, self._info, self.blend, self._key = level, grid, info, blend, key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        lo, hi = S.scene_box(self.grid, box.start, box.stop)
        blocks = []
        for t, tile in enumerate(self.tiles):
            r = S.region(tile, self.level, lo, hi, self.placements[t])
            if r is None:
                blocks.append(None)
                continue
            start, stop = r
            blocks.append((self.sources[t][self.level].read(Box(tuple(start), tuple(stop))), start))
        return S.fuse(self.tiles, self.placements, self.level, self.grid, box.start, box.stop,
                      blocks, self.blend, self._info.dtype)


def _fetch(url: str) -> str:
    if "://" not in url or url.startswith("file://"):
        with open(url.removeprefix("file://")) as f:
            return f.read()
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def stitch(tiles: list[dict], sources: list[MultiscaleSource], p: S.StitchParams) -> dict:
    """Interest points in every overlap at \`\`p.level\`\`, then \`\`stitching.register\`\`."""
    points = []
    for i, j, lo, hi in S.overlaps(tiles, p.margin):
        both = []
        for t in (i, j):
            r = S.region(tiles[t], p.level, lo, hi)
            if r is None:
                both.append(np.zeros((0, 3)))
                continue
            level = sources[t][p.level]
            block = level.read(Box(tuple(r[0]), tuple(r[1])))
            both.append(S.points_in(tiles[t], p.level, block, r[0], level.info.voxel_size, lo, hi, p))
        points.append(both)
    return S.register(tiles, points, p)


def open_stitch(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    from chunkmirage.pipeline import select_axes
    from chunkmirage.sources.registry import open_source

    location, _, query = url[len("stitch://") :].rpartition("?")
    if not location:
        location, query = query, ""
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    p = S.StitchParams.from_query(q)
    t0 = time.perf_counter()
    tiles = S.tiles_from_bdv(_fetch(location), location.rsplit("/", 1)[0], p.channel)
    sources = []
    for t in tiles:
        ms = open_source(t["url"], cache_bytes=cache_bytes, cache=cache)
        ms = select_axes(ms, {k: v for k, v in t["select"].items() if k in ms[0].info.axes})
        sources.append(ms)
        t["shape"] = [list(level.info.shape) for level in ms]
    if p.level >= (n := min(len(s) for s in sources)):
        raise ValueError(f"stitch:// level={p.level}: the tiles have levels 0..{n - 1}")
    found = stitch(tiles, sources, p)
    placements = [np.asarray(a) for a in found["placements"]]
    kept = [r for r in found["pairs"] if r["kept"]]
    log.info(
        "stitch://: %d of %d overlaps kept (%s inliers) in %.1f s%s", len(kept),
        len(found["pairs"]), ", ".join(str(r["inliers"]) for r in kept), time.perf_counter() - t0,
        "" if (d := S.compare(tiles, placements)) is None else f"; {d:.2f} from the project's stitching",
    )
    base = sources[0][0].info
    levels = []
    for k, grid in enumerate(S.grids(tiles, placements)):
        info = ArrayInfo(
            shape=tuple(grid["shape"]), dtype=base.dtype,
            chunk_shape=tuple(min(c, s) for c, s in zip(CHUNK, grid["shape"])),
            voxel_size=tuple(grid["voxel"]), units=tuple(base.units[-3:]), axes=("z", "y", "x"),
            translation=tuple(grid["origin"]),
        )
        levels.append(FusedLevel(tiles, sources, placements, k, grid, info, p.blend, f"stitch:{url}:{k}"))
    return MultiscaleSource(levels, name=url)
`,x=`"""Procedurally generated sources: arbitrarily large, nothing on disk, exact at every scale.

URL form (accepted by \`\`open_source\`\`)::

    synthetic://blobs?shape=4096,4096,4096&chunk=64,64,64&levels=5&seed=0&voxel_size=8
    synthetic://noise?...        fractal (fBm) value noise
    synthetic://shells?...       hollow spheres (membrane-like)
    synthetic://julia?...        3-D slice of a quaternion Julia set
    synthetic://mandelbulb?shape=1073741824,1073741824,1073741824   the Mandelbulb, to zoom into
    synthetic://blobs+noise?...  sum of kinds

Every voxel is a deterministic function of its world coordinate, so level *i* is simply
the same function sampled with a 2**i voxel spacing: \`\`s1[z, y, x] == s0[2z, 2y, 2x]\`\`.
That makes the pyramid free and exactly self-consistent, unlike averaging. The one
exception is \`\`mandelbulb\`\`, which iterates more the finer the level (as fractal zoomers
do: detail a coarse level could not show needs more iterations to resolve), so a coarse
level's boundary is a little fuller than a fine one's. Its coordinates are float64, so an
array 2**30 voxels across, 23 levels deep, still resolves its finest voxels.

Generation is numpy on ~260k voxels per chunk; numpy releases the GIL for those
operations, so the server's threadpool already computes chunks on all cores.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

_KINDS = ("blobs", "noise", "julia", "shells", "mandelbulb")
F32 = np.float32


def _hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uint32 hash of integer lattice coordinates -> float32 in [0, 1)."""
    h = (
        np.asarray(x).astype(np.uint32) * np.uint32(73856093)
        ^ np.asarray(y).astype(np.uint32) * np.uint32(19349663)
        ^ np.asarray(z).astype(np.uint32) * np.uint32(83492791)
        ^ np.uint32((seed * 2654435761) & 0xFFFFFFFF)
    )
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return h.astype(F32) * F32(1.0 / 4294967296.0)


def _lattice(z, y, x, cell: float):
    """Per-voxel lattice cell indices (int) and fractional offsets, plus the covered ranges."""
    fz, fy, fx = (np.asarray(v, dtype=F32) / F32(cell) for v in (z, y, x))
    iz, iy, ix = (np.floor(f).astype(np.int64) for f in (fz, fy, fx))
    tz, ty, tx = fz - iz, fy - iy, fx - ix
    rng = [(int(i.min()) - 1, int(i.max()) + 2) for i in (iz, iy, ix)]  # +2: room for corner+1
    return (iz, iy, ix), (tz, ty, tx), rng


def _grid(rng, seed):
    """Hash values on the small lattice grid covering \`\`rng\`\` (inclusive lows, exclusive highs)."""
    (z0, z1), (y0, y1), (x0, x1) = rng
    gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    return _hash3(gx, gy, gz, seed), (z0, y0, x0)


def _value_noise(z, y, x, cell: float, seed: int) -> np.ndarray:
    """Trilinear lattice noise. Hashes the (small) lattice once, then gathers 8 corners."""
    (iz, iy, ix), (tz, ty, tx), rng = _lattice(z, y, x, cell)
    vals, (z0, y0, x0) = _grid(rng, seed)
    tz, ty, tx = (t * t * (3 - 2 * t) for t in (tz, ty, tx))  # smoothstep
    iz, iy, ix = iz - z0, iy - y0, ix - x0
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    for dz in (0, 1):
        wz = tz if dz else 1 - tz
        for dy in (0, 1):
            wy = ty if dy else 1 - ty
            for dx in (0, 1):
                wx = tx if dx else 1 - tx
                out += wz * wy * wx * vals[iz + dz, iy + dy, ix + dx]
    return out


def _fbm(z, y, x, seed: int, base_cell: float = 256.0, octaves: int = 2) -> np.ndarray:
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    amp, total, cell = 1.0, 0.0, base_cell
    for o in range(octaves):
        out += F32(amp) * _value_noise(z, y, x, cell, seed + o)
        total += amp
        amp *= 0.5
        cell /= 2
    return out / F32(total)


# Blobs / shells: one lattice cell holds \`per_cell\` random spheres whose influence is cut off
# at \`reach\` <= cell/2. Two evaluation strategies with identical results: loop over cells and
# evaluate each sphere on its footprint (cheap when a chunk spans few cells, i.e. fine
# levels), or gather per voxel from the 8 nearest cells (bounded cost at coarse levels; a
# sphere reaching at most cell/2 can only touch voxels in the 2x2x2 cells nearest to them).
_CELL_LOOP_MAX = 216  # switch strategy above this many cells


def _cell_params(gz, gy, gx, seed: int, k: int, cell: float, kind: str):
    s = seed * 131 + k * 7
    u = [_hash3(gx, gy, gz, s + i) for i in range(5)]
    if kind == "blobs":
        center = ((gz + u[0]) * cell, (gy + u[1]) * cell, (gx + u[2]) * cell)
        radius = 8 + u[3] * (cell / 6 - 8)  # 8 .. cell/6, so reach = 3r <= cell/2
        amp = 90 + u[4] * 130
        reach = 3 * radius
    else:  # shells: centred, larger, one per cell; reach = r + 10 <= cell/2
        center = tuple((g + 0.5 + (uu - 0.5) * 0.4) * cell for g, uu in zip((gz, gy, gx), u[:3]))
        radius = cell * (0.22 + 0.15 * u[3])
        amp = np.full_like(radius, 200)
        reach = radius + 10
    f = lambda a: np.asarray(a, dtype=F32)  # noqa: E731
    return tuple(f(c) for c in center), f(radius), f(amp), f(reach)


def _sphere_field(d2, radius, amp, reach, kind):
    inside = d2 <= reach * reach
    if kind == "blobs":
        return inside * amp * np.exp(-d2 / (2 * radius * radius))
    d = np.sqrt(d2)
    return inside * (amp * np.exp(-((d - radius) ** 2) / F32(2 * 2.5**2)) + F32(40) * (d < radius))


def _spheres(z, y, x, seed: int, kind: str) -> np.ndarray:
    cell, per_cell = (128.0, 2) if kind == "blobs" else (160.0, 1)
    z32, y32, x32 = (np.asarray(v, dtype=F32) for v in (z, y, x))
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    (iz, iy, ix), (tz, ty, tx), rng = _lattice(z32, y32, x32, cell)
    (z0, z1), (y0, y1), (x0, x1) = rng
    gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    n_cells = gz.size
    # nearest-cell base index per voxel: floor(f - 0.5), relative to the grid origin
    nz, ny, nx = (i - (t < 0.5) for i, t in ((iz, tz), (iy, ty), (ix, tx)))
    for k in range(per_cell):
        (cz, cy, cx), rad, amp, reach = _cell_params(gz, gy, gx, seed, k, cell, kind)
        if n_cells <= _CELL_LOOP_MAX:
            # loop over cells, evaluate each sphere only on its footprint
            for i in np.ndindex(gz.shape):
                c0, c1, c2, r, a, rc = cz[i], cy[i], cx[i], rad[i], amp[i], reach[i]
                sl = []
                for coord, c in zip((z32.ravel(), y32.ravel(), x32.ravel()), (c0, c1, c2)):
                    lo = int(np.searchsorted(coord, c - rc))
                    hi = int(np.searchsorted(coord, c + rc, side="right"))
                    if hi <= lo:
                        break
                    sl.append(slice(lo, hi))
                if len(sl) < 3:
                    continue
                zz, yy, xx = z32[sl[0]], y32[:, sl[1]], x32[:, :, sl[2]]
                d2 = (zz - c0) ** 2 + (yy - c1) ** 2 + (xx - c2) ** 2
                out[tuple(sl)] += _sphere_field(d2, r, a, rc, kind)
        else:
            # gather from the 8 nearest cells for every voxel
            for dz in (0, 1):
                jz = nz - z0 + dz
                for dy in (0, 1):
                    jy = ny - y0 + dy
                    for dx in (0, 1):
                        jx = nx - x0 + dx
                        jz_, jy_, jx_ = np.broadcast_arrays(jz, jy, jx)
                        d2 = (
                            (z32 - cz[jz_, jy_, jx_]) ** 2
                            + (y32 - cy[jz_, jy_, jx_]) ** 2
                            + (x32 - cx[jz_, jy_, jx_]) ** 2
                        )
                        out += _sphere_field(
                            d2, rad[jz_, jy_, jx_], amp[jz_, jy_, jx_], reach[jz_, jy_, jx_], kind
                        )
    return out


def _julia(z, y, x, seed: int, scale: float, max_iter: int = 14) -> np.ndarray:
    """Escape-time of a quaternion Julia set sampled on a 3-D slice (w=0). Bright = slow escape."""
    rng = np.random.default_rng(seed)
    c = (rng.uniform(-0.8, 0.4, 4) * np.array([1, 1, 1, 0.3])).astype(F32)
    qx, qy, qz = np.broadcast_arrays(
        *(np.asarray(v, dtype=F32) / F32(scale) - F32(1.4) for v in (x, y, z))
    )
    qx, qy, qz = qx.copy(), qy.copy(), qz.copy()
    qw = np.zeros_like(qx)
    count = np.zeros(qx.shape, dtype=F32)
    alive = np.ones(qx.shape, dtype=bool)
    for _ in range(max_iter):
        nx = qx * qx - qy * qy - qz * qz - qw * qw + c[0]
        ny = 2 * qx * qy + c[1]
        nz = 2 * qx * qz + c[2]
        nw = 2 * qx * qw + c[3]
        alive &= (nx * nx + ny * ny + nz * nz + nw * nw) < 16
        qx, qy, qz, qw = (np.where(alive, v, F32(0)) for v in (nx, ny, nz, nw))
        count += alive
    return F32(255.0 / max_iter) * count


BULB = 1.25  # the Mandelbulb's array spans [-BULB, BULB] on each axis
BULB_POWER = 8.0


def _mandelbulb(start, stop, step: int, full_shape) -> np.ndarray:
    """The power-8 Mandelbulb's escape time: 4 x the (smooth) iteration at which a point
    leaves radius 2, so a colour stays a colour from level to level, 255 for points that
    have not left (inside). The array spans [-1.25, 1.25] on every axis, in float64 from
    integer indices; a level of voxels \`\`v\`\` across iterates 10 + 3 log2(256 / v) times
    (10 for the whole bulb on 256 voxels, a few more for every halving)."""
    across = BULB * 2 / max(full_shape)  # the finest voxel, in bulb units
    iters = int(np.clip(10 + 3 * np.log2(max(full_shape) / (256 * step)), 10, 80))
    axes = [
        ((np.arange(a, b, dtype=np.float64) + 0.5) * step) * across - BULB
        for a, b in zip(start, stop)
    ]
    pz, py, px = np.meshgrid(*axes, indexing="ij")
    x, y, z = px.copy(), py.copy(), pz.copy()
    out = np.full(px.shape, 255.0)
    alive = np.ones(px.shape, dtype=bool)
    for i in range(iters):
        r = np.sqrt(x * x + y * y + z * z)
        gone = alive & (r > 2)
        if gone.any():  # smooth escape: the fraction of an iteration past radius 2
            out[gone] = 4 * (i + 1 - np.log(np.log(r[gone]) / np.log(2)) / np.log(BULB_POWER))
            alive &= ~gone
        if not alive.any():
            break
        ra = r[alive]
        theta = BULB_POWER * np.arccos(np.clip(z[alive] / np.maximum(ra, 1e-300), -1, 1))
        phi = BULB_POWER * np.arctan2(y[alive], x[alive])
        rn = ra**BULB_POWER
        x[alive] = rn * np.sin(theta) * np.cos(phi) + px[alive]
        y[alive] = rn * np.sin(theta) * np.sin(phi) + py[alive]
        z[alive] = rn * np.cos(theta) + pz[alive]
    return np.where(alive, 255, np.clip(out, 0, 254)).astype(np.float32)


def generate(
    kinds: tuple[str, ...], full_shape: tuple[int, ...], level: int, seed: int, start, stop
) -> np.ndarray:
    """Pure function: voxels of \`\`kinds\`\` for box [start, stop) at \`\`level\`\`."""
    step = 2**level
    z = (np.arange(start[0], stop[0]) * step)[:, None, None].astype(F32)
    y = (np.arange(start[1], stop[1]) * step)[None, :, None].astype(F32)
    x = (np.arange(start[2], stop[2]) * step)[None, None, :].astype(F32)
    shape = tuple(b - a for a, b in zip(start, stop))
    out = np.zeros(shape, dtype=F32)
    for kind in kinds:
        if kind == "blobs" or kind == "shells":
            out += _spheres(z, y, x, seed, kind)
        elif kind == "noise":
            out += F32(80) * _fbm(z, y, x, seed)
        elif kind == "julia":
            out += _julia(z, y, x, seed, scale=max(full_shape) / 2.8)
        elif kind == "mandelbulb":
            out += _mandelbulb(start, stop, step, full_shape)
    return np.clip(out, 0, 255).astype(np.uint8)


class SyntheticSource(Source):
    def __init__(
        self,
        kinds: list[str],
        shape,
        chunk_shape,
        level: int,
        seed: int,
        voxel_size: float,
        unit: str,
    ):
        self.kinds = tuple(kinds)
        self.level = level
        self.seed = seed
        self.step = 2**level
        self._full = tuple(int(s) for s in shape)
        self._info = ArrayInfo(
            shape=tuple(-(-s // self.step) for s in self._full),
            dtype=np.uint8,
            chunk_shape=tuple(chunk_shape),
            voxel_size=(voxel_size * self.step,) * 3,
            units=(unit,) * 3,
            axes=("z", "y", "x"),
        )

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return f"synthetic:{'+'.join(self.kinds)}:{self._full}:{self.seed}:s{self.level}"

    def read(self, box: Box) -> np.ndarray:
        return generate(
            self.kinds, self._full, self.level, self.seed, tuple(box.start), tuple(box.stop)
        )


def open_synthetic(url: str) -> MultiscaleSource:
    parts = urlsplit(url)
    kinds = [k for k in parts.netloc.split("+") if k]
    for k in kinds:
        if k not in _KINDS:
            raise ValueError(f"unknown synthetic kind {k!r}; choose from {_KINDS}")
    q = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    shape = tuple(int(s) for s in q.get("shape", "1024,1024,1024").split(","))
    chunk = tuple(int(s) for s in q.get("chunk", "64,64,64").split(","))
    levels = int(q.get("levels", "0")) or max(1, int(np.ceil(np.log2(max(shape) / 256))) + 1)
    seed = int(q.get("seed", "0"))
    voxel_size = float(q.get("voxel_size", "8"))
    unit = q.get("unit", "nm")
    if len(shape) != 3:
        raise ValueError("synthetic sources are 3-D: shape=z,y,x")
    src = [
        SyntheticSource(kinds or ["blobs"], shape, chunk, lvl, seed, voxel_size, unit)
        for lvl in range(levels)
    ]
    return MultiscaleSource(src, name=url)
`,S=`"""Stitching overlapping tiles by interest points, as BigStitcher registers them: blobs found
in each overlap by a difference of Gaussians, matched between neighbouring tiles by the
constellation of their nearest neighbours, the matches filtered by RANSAC, and every tile's
placement fitted to all the kept matches at once. Then the tiles are fused, blended where
they overlap, region by region: nothing is written.

Coordinates are physical, C order \`\`(z, y, x)\`\`. A tile is placed by an affine from its
voxels to the scene (rows \`\`[A | t]\`\`): its stage position (where the microscope put it),
which stitching corrects by a model of its own (translation, rigid or affine) on top.

Each step is a function of arrays and JSON-able values, so the browser engine's workers run
them as they are (numpy and scipy only), and \`\`stitch://\`\` (\`\`sources.stitch\`\`) runs them
when it is opened. \`\`tiles_from_bdv\`\` reads the tiles and their placement from a BigStitcher
(BigDataViewer) project, the XML its tools write.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MIN_MATCHES = {"translation": 1, "rigid": 3, "affine": 4}  # a model's smallest sample
BATCH = 512  # RANSAC hypotheses scored at once


class StitchParams(BaseModel):
    """How tiles are stitched: the interest points, their matching, RANSAC and the fit."""

    channel: int = Field(0, ge=0, description="The tiles' channel (setup attribute) stitched")
    level: int = Field(1, ge=0, description="The tiles' level interest points are found on")
    sigma: float = Field(
        1.8, gt=0, description="Blob size: the difference of Gaussians' smaller sigma, in "
        "voxels of that level along x (scaled along the other axes by their spacing)",
    )
    threshold: float = Field(
        0.005, gt=0, description="Smallest difference-of-Gaussians peak kept, in units of the "
        "tile's intensity range (BigStitcher's threshold)",
    )
    margin: float = Field(
        20.0, ge=0, description="How far tiles may be from their stage positions: the "
        "overlaps searched are grown by this (physical units)",
    )
    neighbors: int = Field(3, ge=2, le=6, description="Nearest neighbours a point's descriptor holds")
    redundancy: int = Field(1, ge=0, le=3, description="Extra neighbours: subsets of them are tried too")
    significance: float = Field(
        3.0, ge=1, description="A match's descriptor must be this many times closer than the "
        "next best candidate's",
    )
    model: Literal["translation", "rigid", "affine"] = Field(
        "translation", description="What each tile may do beyond its stage position"
    )
    epsilon: float = Field(5.0, gt=0, description="RANSAC: largest error of an inlier (physical units)")
    min_inlier_ratio: float = Field(0.1, ge=0, le=1, description="RANSAC: smallest share of inliers")
    min_inliers: int = Field(6, ge=1, description="RANSAC: fewest inliers for a pair to count")
    iterations: int = Field(10000, ge=1, le=200000, description="RANSAC: hypotheses tried")
    seed: int = Field(0, description="RANSAC's random draws")
    blend: float = Field(
        40.0, ge=0, description="Fusion: the band at a tile's edges its weight falls off over "
        "(physical units along y and x)",
    )

    @classmethod
    def from_query(cls, q: dict[str, str]) -> StitchParams:
        if unknown := set(q) - set(cls.model_fields):
            raise ValueError(
                f"unknown stitch:// parameters {sorted(unknown)}; allowed: {sorted(cls.model_fields)}"
            )
        return cls(**q)


# ------------------------------------------------------------------ affines
def to4(a) -> np.ndarray:
    m = np.eye(4)
    m[:3] = np.asarray(a, float).reshape(3, 4)
    return m


def apply(a, p: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return p @ a[:3, :3].T + a[:3, 3]


def compose(*affines) -> np.ndarray:
    """\`\`compose(a, b)(p) == a(b(p))\`\`."""
    m = np.eye(4)
    for a in affines:
        m = m @ to4(a)
    return m[:3]


def invert(a) -> np.ndarray:
    return np.linalg.inv(to4(a))[:3]


# ------------------------------------------------------------------ interest points
def _gaussian(x: np.ndarray, sigma) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma, mode="nearest", truncate=3.0)


def detect(block: np.ndarray, voxel, sigma: float, threshold: float, lo: float, hi: float) -> np.ndarray:
    """Bright blobs in \`\`block\`\`: local maxima of a difference of Gaussians (sigma and
    2^(1/4) sigma, BigStitcher's), at least \`\`threshold\`\` of the intensity range [lo, hi],
    to subvoxel precision. Returns \`\`(n, 3)\`\` voxel positions in \`\`block\`\`; none within a
    voxel of its faces (their maxima are cut off)."""
    from scipy.ndimage import maximum_filter

    if min(block.shape) < 3:
        return np.zeros((0, 3))
    voxel = np.asarray(voxel, float)
    s = sigma * voxel[-1] / voxel  # the same physical size along every axis
    x = (np.asarray(block, np.float32) - lo) / max(hi - lo, 1e-12)
    dog = _gaussian(x, s) - _gaussian(x, s * 2 ** 0.25)
    peak = (dog == maximum_filter(dog, size=3, mode="nearest")) & (dog >= threshold)
    peak[[0, -1]] = peak[:, [0, -1]] = peak[:, :, [0, -1]] = False
    at = np.argwhere(peak)
    if not len(at):
        return np.zeros((0, 3))
    # a parabola through each peak and its two neighbours, per axis
    out = at.astype(float)
    v = dog[tuple(at.T)]
    for a in range(3):
        up, down = at.copy(), at.copy()
        up[:, a] += 1
        down[:, a] -= 1
        f1, f0 = dog[tuple(up.T)], dog[tuple(down.T)]
        curve = f1 + f0 - 2 * v
        out[:, a] += np.where(curve < 0, 0.5 * (f0 - f1) / np.where(curve < 0, curve, -1), 0).clip(-0.5, 0.5)
    return out


def points_in(tile: dict, level: int, block: np.ndarray, start, voxel, lo, hi, p: StitchParams) -> np.ndarray:
    """A tile's interest points in the scene box \`\`[lo, hi]\`\`, from \`\`block\`\` (its level's
    voxels from \`\`start\`\`, as \`\`region\`\` gives them): scene positions at its stage placement."""
    block = np.asarray(block)
    v = detect(block, voxel, p.sigma, p.threshold, float(block.min()), float(block.max()))
    w = apply(compose(tile["stage"], level_to_base(tile, level)), v + np.asarray(start))
    return w[((w >= lo) & (w <= hi)).all(1)]


# ------------------------------------------------------------------ matching
def descriptors(points: np.ndarray, neighbors: int, redundancy: int):
    """Each point's constellation: the offsets to \`\`neighbors\`\` of its nearest
    \`\`neighbors + redundancy\`\` (every such subset, nearest first), concatenated. Unchanged
    by a translation, so tiles placed only roughly still match. Returns the descriptors and
    the point each belongs to."""
    from scipy.spatial import cKDTree

    k = neighbors + redundancy
    if len(points) <= k:
        return np.zeros((0, 3 * neighbors)), np.zeros(0, int)
    _, near = cKDTree(points).query(points, k + 1)
    offsets = points[near[:, 1:]] - points[:, None, :]  # (n, k, 3), nearest first
    subsets = list(itertools.combinations(range(k), neighbors))
    d = np.concatenate([offsets[:, list(s)].reshape(len(points), -1) for s in subsets])
    owner = np.tile(np.arange(len(points)), len(subsets))
    return d, owner


def match(a: np.ndarray, b: np.ndarray, neighbors: int = 3, redundancy: int = 1, significance: float = 3.0) -> np.ndarray:
    """Candidate matches \`\`(i, j)\`\` of points \`\`a[i]\`\` and \`\`b[j]\`\`: \`\`b[j]\`\` is the point
    whose descriptors come closest to one of \`\`a[i]\`\`'s, \`\`significance\`\` times closer than
    any other point's, and the other way round."""
    from scipy.spatial import cKDTree

    da, oa = descriptors(a, neighbors, redundancy)
    db, ob = descriptors(b, neighbors, redundancy)
    if not len(da) or not len(db):
        return np.zeros((0, 2), int)

    def best(dx, ox, dy, oy, nx):
        """For each point of x: its nearest point of y, and how much nearer than the next."""
        k = min(len(dy), 8)
        dist, at = cKDTree(dy).query(dx, k)
        dist, at = dist.reshape(len(dx), k), at.reshape(len(dx), k)
        rows = np.repeat(ox, k)
        cols = oy[at.ravel()]
        dist = dist.ravel()
        # each (x point, y point): the closest of their descriptors
        order = np.lexsort((dist, cols, rows))
        rows, cols, dist = rows[order], cols[order], dist[order]
        first = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])]
        rows, cols, dist = rows[first], cols[first], dist[first]
        order = np.lexsort((dist, rows))
        rows, cols, dist = rows[order], cols[order], dist[order]
        start = np.r_[True, rows[1:] != rows[:-1]]
        idx = np.flatnonzero(start)
        nearest = np.full(nx, -1)
        ratio = np.zeros(nx)
        nearest[rows[idx]] = cols[idx]
        nxt = idx + 1
        has = (nxt < len(rows)) & (np.r_[rows, -1][nxt] == rows[idx])
        second = np.where(has, np.r_[dist, np.inf][np.where(has, nxt, len(dist))], np.inf)
        ratio[rows[idx]] = second / np.maximum(dist[idx], 1e-12)
        return nearest, ratio

    ab, ra = best(da, oa, db, ob, len(a))
    ba, _ = best(db, ob, da, oa, len(b))
    i = np.flatnonzero((ab >= 0) & (ra >= significance))
    i = i[ba[ab[i]] == i]  # mutual
    return np.stack([i, ab[i]], 1)


# ------------------------------------------------------------------ models
def fit(model: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The \`\`model\`\` taking points \`\`p\`\` nearest to \`\`q\`\` (least squares), as \`\`[A | t]\`\`."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if model == "translation":
        return np.hstack([np.eye(3), (q - p).mean(0)[:, None]])
    cp, cq = p.mean(0), q.mean(0)
    if model == "rigid":
        u, _, vt = np.linalg.svd((q - cq).T @ (p - cp))
        r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        return np.hstack([r, (cq - r @ cp)[:, None]])
    x = np.hstack([p - cp, np.ones((len(p), 1))])
    sol = np.linalg.lstsq(x, q - cq, rcond=None)[0]  # (4, 3)
    a = sol[:3].T
    return np.hstack([a, (cq + sol[3] - a @ cp)[:, None]])


def _fit_many(model: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """\`\`fit\`\` of each of a batch of minimal samples \`\`(m, k, 3)\`\`: \`\`(m, 3, 4)\`\`."""
    cp, cq = p.mean(1, keepdims=True), q.mean(1, keepdims=True)
    m = len(p)
    if model == "translation":
        out = np.zeros((m, 3, 4))
        out[:, :, :3] = np.eye(3)
        out[:, :, 3] = (cq - cp)[:, 0]
        return out
    if model == "rigid":
        u, _, vt = np.linalg.svd(np.einsum("mki,mkj->mij", q - cq, p - cp))
        d = np.sign(np.linalg.det(u @ vt))
        u[:, :, 2] *= d[:, None]
        a = u @ vt
    else:
        x = np.concatenate([p - cp, np.ones(p.shape[:2] + (1,))], 2)  # (m, 4, 4)
        x = x + np.eye(4)[None] * 1e-9  # degenerate samples stay finite (and score badly)
        sol = np.linalg.solve(x, q - cq)  # (m, 4, 3)
        a = np.transpose(sol[:, :3], (0, 2, 1))
        cq = cq + sol[:, 3:4]
    t = cq[:, 0] - np.einsum("mij,mj->mi", a, cp[:, 0])
    return np.concatenate([a, t[:, :, None]], 2)


def ransac(p: np.ndarray, q: np.ndarray, model: str, epsilon: float, min_inlier_ratio: float,
           min_inliers: int, iterations: int, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """The \`\`model\`\` that most candidate matches \`\`p[i] -> q[i]\`\` agree with to within
    \`\`epsilon\`\`, from \`\`iterations\`\` random minimal samples, refitted to its inliers until
    they stop changing. Returns the model (\`\`None\`\` if too few agree) and the inliers."""
    n, k = len(p), MIN_MATCHES[model]
    none = np.zeros(n, bool)
    if n < max(k, min_inliers):
        return None, none
    rng = np.random.default_rng(seed)
    best, best_count = none, 0
    for start in range(0, iterations, BATCH):
        m = min(BATCH, iterations - start)
        pick = np.argsort(rng.random((m, n)), 1)[:, :k] if k > 1 else rng.integers(0, n, (m, 1))
        models = _fit_many(model, p[pick], q[pick])
        d = p @ models[:, :, :3].transpose(0, 2, 1) + models[:, None, :, 3] - q  # (m, n, 3)
        err = np.sqrt((d * d).sum(2))
        counts = (err < epsilon).sum(1)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_count, best = int(counts[i]), err[i] < epsilon
    for _ in range(20):  # refit to the inliers until they settle
        if best.sum() < k:
            return None, none
        a = fit(model, p[best], q[best])
        now = np.linalg.norm(apply(a, p) - q, axis=1) < epsilon
        if (now == best).all():
            break
        best = now
    if best.sum() < max(k, min_inliers) or best.mean() < min_inlier_ratio:
        return None, best
    return fit(model, p[best], q[best]), best


def optimize(n: int, links: list[tuple[int, int, np.ndarray, np.ndarray]], model: str,
             fixed: int = 0, rounds: int = 2000, tolerance: float = 1e-4) -> tuple[list[np.ndarray], list[bool]]:
    """Every tile's correction (applied after its stage placement) fitted to all the kept
    matches at once: link \`\`(i, j, p, q)\`\` asks tile \`\`i\`\`'s points \`\`p\`\` to meet tile
    \`\`j\`\`'s \`\`q\`\` (both placed by their stage positions). Each round refits each tile to
    where its neighbours put their ends of its matches. Of each group of tiles that links
    join, one stays where it is: \`\`fixed\`\` in its group, the first tile in the others.
    Returns the corrections and which tiles were joined to another."""
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    models = [identity.copy() for _ in range(n)]
    group = list(range(n))

    def find(t):
        while group[t] != t:
            t = group[t]
        return t

    for i, j, _, _ in links:
        group[find(i)] = find(j)
    anchors = {}
    for t in [fixed, *range(n)]:
        anchors.setdefault(find(t), t)
    joined = [any(t in (i, j) for i, j, _, _ in links) for t in range(n)]
    order = [t for t in range(n) if joined[t] and t not in anchors.values()]
    for _ in range(rounds):
        moved = 0.0
        for t in order:
            p, q = [], []
            for i, j, pi, qj in links:
                if i == t:
                    p.append(pi)
                    q.append(apply(models[j], qj))
                elif j == t:
                    p.append(qj)
                    q.append(apply(models[i], pi))
            if not p:
                continue
            p, q = np.concatenate(p), np.concatenate(q)
            new = fit(model, p, q)
            moved = max(moved, float(np.abs(apply(new, p) - apply(models[t], p)).max()))
            models[t] = new
        if moved < tolerance:
            break
    return models, joined


# ------------------------------------------------------------------ the whole registration
def overlaps(tiles: list[dict], margin: float) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Pairs of tiles whose stage placements overlap, and the overlap's scene box (grown by
    \`\`margin\`\`), as \`\`(i, j, lo, hi)\`\`."""
    boxes = [bounds(t["stage"], t["shape"][0]) for t in tiles]
    out = []
    for i, j in itertools.combinations(range(len(tiles)), 2):
        lo = np.maximum(boxes[i][0], boxes[j][0])
        hi = np.minimum(boxes[i][1], boxes[j][1])
        if (hi > lo).all():
            out.append((i, j, lo - margin, hi + margin))
    return out


def bounds(affine, shape) -> tuple[np.ndarray, np.ndarray]:
    """The scene box an affine puts a volume of \`\`shape\`\` voxels in (voxel centres at
    integers, so a voxel's extent is +-0.5)."""
    corners = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in shape])))
    w = apply(affine, corners)
    return w.min(0), w.max(0)


def region(tile: dict, level: int, lo, hi, placement=None) -> tuple[list[int], list[int]] | None:
    """The voxels \`\`[start, stop)\`\` of a tile's level holding the scene box \`\`[lo, hi]\`\` (a
    voxel more on each side, for interpolation) where \`\`placement\`\` (default: its stage
    position) puts it, or \`\`None\`\` if it holds none of it."""
    to_scene = compose(tile["stage"] if placement is None else placement, level_to_base(tile, level))
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    v = apply(invert(to_scene), corners)
    start = np.maximum(np.floor(v.min(0)) - 1, 0).astype(int)
    stop = np.minimum(np.ceil(v.max(0)) + 2, tile["shape"][level]).astype(int)
    if (stop <= start).any():
        return None
    return start.tolist(), stop.tolist()


def level_to_base(tile: dict, level: int) -> np.ndarray:
    """A tile level's voxels to its level-0 voxels: each level halves (or so) its shape;
    its voxel centres sit at the centres of the level-0 voxels they cover."""
    f = np.array(tile["shape"][0], float) / np.array(tile["shape"][level], float)
    f = np.round(f) if np.allclose(f, np.round(f), atol=0.05) else f
    return np.hstack([np.diag(f), ((f - 1) / 2)[:, None]])


def register(tiles: list[dict], points: list[tuple[np.ndarray, np.ndarray]], p: StitchParams,
             fixed: int = 0) -> dict:
    """Matches, RANSAC and the global fit from each overlapping pair's interest points
    (\`\`points[k]\`\`: pair \`\`k\`\` of \`\`overlaps\`\`, both tiles' points in scene coordinates at
    their stage placement). JSON-able: per pair the candidates, inliers and errors; per
    tile its correction and its final placement."""
    pairs = overlaps(tiles, p.margin)
    links, report = [], []
    for k, (i, j, _, _) in enumerate(pairs):
        a, b = (np.asarray(x, float).reshape(-1, 3) for x in points[k])
        m = match(a, b, p.neighbors, p.redundancy, p.significance)
        model, inliers = ransac(a[m[:, 0]], b[m[:, 1]], p.model, p.epsilon, p.min_inlier_ratio,
                                p.min_inliers, p.iterations, p.seed)
        kept = m[inliers] if model is not None else m[:0]
        if model is not None:
            links.append((i, j, a[kept[:, 0]], b[kept[:, 1]]))
        report.append({
            "tiles": [i, j], "points": [len(a), len(b)], "candidates": len(m),
            "inliers": int(len(kept)), "kept": model is not None,
            # the matches, scene positions of both ends at the stage placement
            "a": a[m[:, 0]].round(3).tolist(), "b": b[m[:, 1]].round(3).tolist(),
            "inlier": inliers.tolist(),
        })
    corrections, placed = optimize(len(tiles), links, p.model, fixed)
    for r in report:
        i, j = r["tiles"]
        if r["kept"]:
            sel = np.array(r["inlier"], bool)
            a, b = np.array(r["a"])[sel], np.array(r["b"])[sel]
            e = np.linalg.norm(apply(corrections[i], a) - apply(corrections[j], b), axis=1)
            r["error"] = {"mean": float(e.mean()), "max": float(e.max())}
    return {
        "pairs": report,
        "corrections": [c.tolist() for c in corrections],
        "placed": placed,
        "placements": [compose(c, t["stage"]).tolist() for c, t in zip(corrections, tiles)],
    }


def compare(tiles: list[dict], placements: list) -> float | None:
    """How far the found placements are from the tiles' reference ones (a registration done
    before), as the root-mean-square distance of the tiles' centres once the two are
    lined up on average (stitching fixes the scene only up to a common shift)."""
    if not all(t.get("reference") is not None for t in tiles):
        return None
    c = [np.array(t["shape"][0], float)[None] / 2 - 0.5 for t in tiles]
    ours = np.concatenate([apply(a, x) for a, x in zip(placements, c)])
    theirs = np.concatenate([apply(t["reference"], x) for t, x in zip(tiles, c)])
    d = ours - theirs
    return float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()))


# ------------------------------------------------------------------ fusion
def level_voxel(tile: dict, level: int) -> np.ndarray:
    """A tile level's voxel size in the scene, at its stage placement."""
    return np.linalg.norm(compose(tile["stage"], level_to_base(tile, level))[:, :3], axis=0)


def grids(tiles: list[dict], placements: list) -> list[dict]:
    """The fused volume's levels, one per level every tile has: the first tile's voxel size
    there, and the origin and shape covering every tile where \`\`placements\`\` put them."""
    lo, hi = zip(*(bounds(a, t["shape"][0]) for a, t in zip(placements, tiles)))
    lo, hi = np.min(lo, 0), np.max(hi, 0)
    out = []
    for k in range(min(len(t["shape"]) for t in tiles)):
        voxel = level_voxel(tiles[0], k)
        shape = np.ceil((hi - lo) / voxel - 1e-6).astype(int)
        out.append({"shape": shape.tolist(), "voxel": voxel.tolist(), "origin": (lo + voxel / 2).tolist()})
    return out


def scene_box(grid: dict, out_lo, out_hi) -> tuple[np.ndarray, np.ndarray]:
    """The scene box that voxels \`\`[out_lo, out_hi)\`\` of a fused level cover."""
    voxel, origin = np.asarray(grid["voxel"]), np.asarray(grid["origin"])
    return origin + (np.asarray(out_lo) - 0.5) * voxel, origin + (np.asarray(out_hi) - 0.5) * voxel


def _weight(v: np.ndarray, shape, band: np.ndarray) -> np.ndarray:
    """A tile's blending weight at its voxel positions \`\`v\`\` (n, 3): 1 inside, falling off as
    a half cosine over \`\`band\`\` voxels from each face, 0 outside."""
    w = np.ones(len(v))
    for a in range(3):
        inside = (v[:, a] >= -0.5) & (v[:, a] <= shape[a] - 0.5)
        w = np.where(inside, w, 0)
        if band[a] > 0:
            d = np.minimum(v[:, a] + 0.5, shape[a] - 0.5 - v[:, a])
            w = w * np.where(d < band[a], 0.5 - 0.5 * np.cos(np.pi * np.clip(d, 0, None) / band[a]), 1)
    return w


def fuse(tiles: list[dict], placements: list, level: int, grid: dict, out_lo, out_hi,
         blocks: list[tuple[np.ndarray, list[int]] | None], blend: float, dtype) -> np.ndarray:
    """Voxels \`\`[out_lo, out_hi)\`\` of the fused volume on \`\`grid\`\`: each tile sampled
    (trilinearly) where its placement puts it, from \`\`blocks[t]\`\` (its level's voxels from
    a start, or \`\`None\`\` where it has none of the region), averaged with weights that fall
    off near its edges, so the seams do not show."""
    from scipy.ndimage import map_coordinates

    out_lo, out_hi = np.asarray(out_lo), np.asarray(out_hi)
    shape = tuple(out_hi - out_lo)
    idx = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(out_lo, out_hi)], indexing="ij"), -1)
    w_scene = np.asarray(grid["origin"]) + idx.reshape(-1, 3) * np.asarray(grid["voxel"])
    total = np.zeros(len(w_scene))
    weight = np.zeros(len(w_scene))
    for t, got in enumerate(blocks):
        if got is None:
            continue
        block, start = got
        tile = tiles[t]
        to_scene = compose(placements[t], level_to_base(tile, level))
        v = apply(invert(to_scene), w_scene)
        vox = np.abs(np.linalg.det(np.asarray(to_scene)[:, :3])) ** (1 / 3)
        band = np.array([0.0, blend, blend]) / vox  # along y and x: tiles share their z range
        w = _weight(v, tile["shape"][level], band)
        hit = w > 0
        if not hit.any():
            continue
        local = (v[hit] - np.asarray(start)).T
        total[hit] += w[hit] * map_coordinates(np.asarray(block, np.float32), local, order=1, mode="nearest")
        weight[hit] += w[hit]
    out = np.where(weight > 0, total / np.maximum(weight, 1e-12), 0).reshape(shape)
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)


# ------------------------------------------------------------------ BigStitcher projects
def _affine_xyz(text: str) -> np.ndarray:
    """A BigDataViewer affine (12 numbers, rows of x, y, z) in C order (z, y, x)."""
    m = np.array([float(v) for v in text.split()]).reshape(3, 4)
    a = np.zeros((3, 4))
    a[:, :3] = m[::-1, :3][:, ::-1]
    a[:, 3] = m[::-1, 3]
    return a


def tiles_from_bdv(xml: str, base: str, channel: int = 0, drop: str = "Stitching Transform") -> list[dict]:
    """The tiles of one channel of a BigStitcher project (its XML text; \`\`base\`\`: the URL
    of the folder it is in), each with its stage placement: every view transform but the
    outermost ones named \`\`drop\`\` (the stitching found before, kept as the reference). Reads
    projects whose images are OME-Zarr (BigStitcher-Spark's \`\`bdv.multimg.zarr\`\` loader),
    one group per setup and time point."""
    root = ET.fromstring(xml)
    loader = root.find("SequenceDescription/ImageLoader")
    if loader is None or loader.get("format") != "bdv.multimg.zarr":
        raise ValueError(f"only OME-Zarr BigStitcher projects are read (bdv.multimg.zarr), not {loader.get('format') if loader is not None else 'none'}")
    zarr = loader.findtext("zarr", "").strip()
    folder = zarr if "://" in zarr or zarr.startswith("/") else f"{base.rstrip('/')}/{zarr}"
    groups = {(g.get("setup"), g.get("tp")): g.get("path") for g in loader.iter("zgroup")}
    tp = min({tp for _, tp in groups}, key=int)
    regs = {r.get("setup"): r for r in root.iter("ViewRegistration") if r.get("timepoint") == tp}
    tiles = []
    for vs in root.iter("ViewSetup"):
        attrs = vs.find("attributes")
        if attrs is None or int(attrs.findtext("channel", "0")) != channel:
            continue
        sid = vs.findtext("id").strip()
        size = [int(v) for v in vs.findtext("size").split()][::-1]  # x y z -> z y x
        transforms = regs[sid].findall("ViewTransform")
        names = [t.findtext("Name", "") for t in transforms]
        affines = [_affine_xyz(t.findtext("affine")) for t in transforms]
        n = 0
        while n < len(names) and names[n] == drop:
            n += 1
        tiles.append({
            "name": f"tile {attrs.findtext('tile', sid)}", "setup": int(sid),
            "url": f"{folder}/{groups[(sid, tp)]}", "select": {"t": 0, "c": 0},
            "shape": [size], "stage": compose(*affines[n:]).tolist(),
            "reference": compose(*affines).tolist() if n else None,
        })
    if not tiles:
        raise ValueError(f"no tiles of channel {channel}")
    return tiles
`,C=`"""Following one object through a time series of label images, frame by frame, as the
frames are read: in each next frame it is the label that overlaps it most. Labels need not
keep their ids from frame to frame (a segmentation done frame by frame rarely does); an
object that moves less than its own size between frames is followed by its overlap.

Each frame reads only a box around the object (its last bounding box, grown by a margin),
so following one nucleus through a whole time-lapse reads a sliver of it. A nucleus that
divides first collapses (its envelope breaks down in mitosis, and the segmentation loses
it); new nuclei then appear near where it was, as labels nothing covered the frame before
(\`\`newborns\`\`), and are taken for its daughters and followed in turn, so a track becomes a
lineage. That is a guess from where and when they appear: the segmentation does not say
which nucleus a new one came from, and in a dense colony a neighbour's daughter can be
taken for this one's. \`\`step\`\` and
\`\`newborns\`\` are the work of one frame, numpy only, which the browser engine's workers run
as they are; \`\`follow\`\` and \`\`lineage\`\` loop them over a source for Python callers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

MIN_OVERLAP = 0.2  # of the object's voxels: less and it is lost (it left, or the labels failed)
DIVIDED = 0.65  # a volume this fraction of the frame before's, or less: a division
NEWBORN = 0.4  # a label no label of the frame before covers this share of: new (a daughter, say)
GAP = 24  # frames searched for daughters after a nucleus collapses into mitosis and is lost
WITHIN = 18.0  # physical units (µm) from the mother that daughters are looked for


def measure(block: np.ndarray, label: int, start, voxel) -> dict | None:
    """Label \`\`label\`\` in \`\`block\`\` (whose voxel 0 is \`\`start\`\` of the level): its volume
    (voxels times their size), centroid and bounding box (level voxels, \`\`[lo, hi)\`\`), and
    whether it touches the block's faces (the block may not hold all of it)."""
    at = np.argwhere(np.asarray(block) == label)
    if not len(at):
        return None
    lo, hi = at.min(0), at.max(0) + 1
    shape = np.asarray(block.shape)
    touches = bool(((lo == 0) & (np.asarray(start) > 0)).any() or (hi == shape).any())
    start = np.asarray(start)
    return {
        "label": int(label),
        "volume": float(len(at) * np.prod(voxel)),
        "centroid": (at.mean(0) + start).tolist(),
        "lo": (lo + start).tolist(),
        "hi": (hi + start).tolist(),
        "touches": touches,
    }


def step(prev: np.ndarray, label: int, nxt: np.ndarray, start, voxel) -> dict | None:
    """The object labelled \`\`label\`\` in \`\`prev\`\`, in the next frame \`\`nxt\`\` (the same box of
    the level, from \`\`start\`\`): the label it overlaps most there, measured, with the overlap
    (its share of the object's voxels). \`\`None\`\` if under \`\`MIN_OVERLAP\`\`."""
    mask = np.asarray(prev) == label
    if not mask.any():
        return None
    under = np.asarray(nxt)[mask]
    under = under[under > 0]
    if len(under) < MIN_OVERLAP * mask.sum():
        return None
    ids, counts = np.unique(under, return_counts=True)
    best = int(ids[np.argmax(counts)])
    found = measure(nxt, best, start, voxel)
    if found is not None:
        found["overlap"] = float(counts.max() / mask.sum())
    return found


def box_around(record: dict, shape, margin) -> tuple[list[int], list[int]]:
    """The box the next frame is read over: the object's bounding box grown by \`\`margin\`\`
    voxels, within the level."""
    lo = np.maximum(np.asarray(record["lo"]) - margin, 0)
    hi = np.minimum(np.asarray(record["hi"]) + margin, shape)
    return lo.tolist(), hi.tolist()


def follow(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, t0: int,
           label: int, frames: range, margin) -> Iterator[tuple[int, dict]]:
    """The object labelled \`\`label\`\` at frame \`\`t0\`\`, through \`\`frames\`\` (counting up or down
    from \`\`t0\`\`); \`\`read(t, lo, hi)\`\` gives a frame's labels over a box of the level. Yields
    \`\`(t, record)\`\` for \`\`t0\`\` and each frame it is found in, until it is lost; a record
    says \`\`divided\`\` when its volume fell to \`\`DIVIDED\`\` of the frame before's."""
    margin = np.asarray(margin)
    # the whole object at t0: start from a box around its voxels in the whole frame
    first = measure(read(t0, [0] * len(shape), list(shape)), label, [0] * len(shape), voxel)
    if first is None:
        raise ValueError(f"no label {label} at frame {t0}")
    yield t0, first
    record, t = first, t0
    for nt in frames:
        lo, hi = box_around(record, shape, margin)
        prev, nxt = read(t, lo, hi), read(nt, lo, hi)
        found = step(prev, record["label"], nxt, lo, voxel)
        if found is not None and found["touches"]:  # the box cut it: read all of it
            lo, hi = box_around(found, shape, 2 * margin)
            found = step(read(t, lo, hi), record["label"], read(nt, lo, hi), lo, voxel)
        if found is None:
            return
        found["divided"] = found["volume"] <= DIVIDED * record["volume"]
        yield nt, found
        record, t = found, nt


def newborns(prev: np.ndarray, nxt: np.ndarray, start, voxel, centre, within: float = WITHIN,
             min_volume: float = 0.0) -> list[dict]:
    """Labels of \`\`nxt\`\` that are new: covered less than \`\`NEWBORN\`\` by any one label of the
    frame before, \`\`prev\`\` (the same box, from \`\`start\`\`); within \`\`within\`\` (physical units)
    of \`\`centre\`\` (level voxels) and of \`\`min_volume\`\` or more. After a mitosis, which hides
    the nucleus for a few frames, these are likely its daughters (a guess: nothing says which
    nucleus a new one came from). Nearest first."""
    nxt, prev, voxel = np.asarray(nxt), np.asarray(prev), np.asarray(voxel)
    out = []
    for i in np.unique(nxt[nxt > 0]):
        mask = nxt == i
        under = prev[mask]
        under = under[under > 0]
        if len(under) and np.bincount(under).max() >= NEWBORN * mask.sum():
            continue
        found = measure(nxt, int(i), start, voxel)
        far = np.linalg.norm((np.asarray(found["centroid"]) - np.asarray(centre)) * voxel)
        if far <= within and found["volume"] >= min_volume:
            found["distance"] = float(far)
            out.append(found)
    return sorted(out, key=lambda f: f["distance"])


def mother_of(branch: dict[int, dict], lost: int) -> dict | None:
    """If a branch lost at frame \`\`lost\`\` had collapsed first (its last volume \`\`DIVIDED\`\` of
    its largest in the dozen frames before, or less), the record at that largest: the
    nucleus that went into mitosis. \`\`None\`\` if it was just lost."""
    recent = [branch[t] for t in sorted(branch) if lost - 12 <= t < lost]
    if not recent:
        return None
    biggest = max(recent, key=lambda r: r["volume"])
    return biggest if recent[-1]["volume"] <= DIVIDED * biggest["volume"] else None


def daughters(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, mother: dict,
              lost: int, frames: int, gap: int = GAP, within: float = WITHIN) -> list[tuple[int, dict]]:
    """Up to two likely daughters of \`\`mother\`\` (a record, see \`\`mother_of\`\`): the first new
    nuclei (\`\`newborns\`\`) near it, looked for from frame
    \`\`lost\`\` for \`\`gap\`\` frames in a box \`\`within\`\` around it: \`\`(frame, record)\`\` where each
    first appears."""
    voxel = np.asarray(voxel)
    reach = np.ceil(within / voxel).astype(int)
    lo = np.maximum(np.round(mother["centroid"]).astype(int) - reach, 0).tolist()
    hi = np.minimum(np.round(mother["centroid"]).astype(int) + reach + 1, shape).tolist()
    found: list[tuple[int, dict]] = []
    for t in range(lost, min(lost + gap, frames)):
        for d in newborns(read(t - 1, lo, hi), read(t, lo, hi), lo, voxel, mother["centroid"], within, 0.1 * mother["volume"]):
            found.append((t, d))
            if len(found) == 2:
                return found
    return found


def lineage(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, t0: int,
            label: int, frames: int, margin, max_branches: int = 8) -> list[dict[int, dict]]:
    """The object labelled \`\`label\`\` at frame \`\`t0\`\` followed forward and back (\`\`follow\`\`),
    and forward through its divisions: the daughters of each nucleus that collapses and is
    lost are followed too. Returns branches (frame -> record), the first the object's own;
    each daughter's first record says \`\`mother\`\`, the branch it came from."""
    main = dict(follow(read, shape, voxel, t0, label, range(t0 + 1, frames), margin))
    main.update(follow(read, shape, voxel, t0, label, range(t0 - 1, -1, -1), margin))
    branches, todo = [main], [0]
    while todo and len(branches) < max_branches:
        k = todo.pop(0)
        last = max(branches[k])
        mother = mother_of(branches[k], last + 1)
        if mother is None or last + 1 >= frames:
            continue
        for t, d in daughters(read, shape, voxel, mother, last + 1, frames):
            if len(branches) >= max_branches:
                break
            branch = dict(follow(read, shape, voxel, t, d["label"], range(t + 1, frames), margin))
            branch[t]["mother"] = k
            branches.append(branch)
            todo.append(len(branches) - 1)
    return branches
`;const w=`https://cdn.jsdelivr.net/pyodide/v0.28.3/full/`,T={...Object.fromEntries(Object.entries(/* @__PURE__ */ Object.assign({"../../src/chunkmirage/cache.py":e,"../../src/chunkmirage/core.py":t,"../../src/chunkmirage/fused.py":n,"../../src/chunkmirage/meshes.py":r,"../../src/chunkmirage/ops/__init__.py":i,"../../src/chunkmirage/ops/base.py":a,"../../src/chunkmirage/ops/code.py":o,"../../src/chunkmirage/ops/downsample.py":s,"../../src/chunkmirage/sources/base.py":c,"../../src/chunkmirage_contrib/__init__.py":l,"../../src/chunkmirage_contrib/ops/__init__.py":u,"../../src/chunkmirage_contrib/ops/combine.py":d,"../../src/chunkmirage_contrib/ops/filters.py":f,"../../src/chunkmirage_contrib/ops/pointwise.py":p,"../../src/chunkmirage_contrib/ops/segment.py":m,"../../src/chunkmirage_contrib/ops/terrain.py":h,"../../src/chunkmirage_contrib/registration.py":g,"../../src/chunkmirage_contrib/sources/__init__.py":_,"../../src/chunkmirage_contrib/sources/geotiff.py":v,"../../src/chunkmirage_contrib/sources/register.py":y,"../../src/chunkmirage_contrib/sources/stitch.py":b,"../../src/chunkmirage_contrib/sources/synthetic.py":x,"../../src/chunkmirage_contrib/stitching.py":S,"../../src/chunkmirage_contrib/tracking.py":C})).map(([e,t])=>[e.slice(e.indexOf(`/src/`)+5),t])),"chunkmirage/__init__.py":`"""chunkmirage's engine, for the browser."""
`,"chunkmirage/sources/__init__.py":`"""The sources the browser computes itself."""
`},E=self;let D=null,O=null,k=null;async function A(e=[]){D=await(await import(
/* @vite-ignore */
`${w}pyodide.mjs`)).loadPyodide({indexURL:w}),D.setStdout({batched:e=>{k&&E.postMessage({print:e,...k})}}),await D.loadPackage([`numpy`,`pydantic`,...e]);for(let[e,t]of Object.entries(T))D.FS.mkdirTree(`/chunkmirage/${e.slice(0,e.lastIndexOf(`/`))}`),D.FS.writeFile(`/chunkmirage/${e}`,t);D.runPython(`import sys; sys.path.insert(0, "/chunkmirage")

import json
import numpy as np
from chunkmirage import fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import op_from_spec
import chunkmirage_contrib.ops  # noqa: F401  registers the shipped ops

VIEWS = {}
SOURCES = {}

def describe(url):
    from chunkmirage_contrib.sources.synthetic import open_synthetic
    ms = SOURCES.setdefault(url, open_synthetic(url))
    i = ms[0].info
    return json.dumps({
        "dtype": i.dtype.name, "channels": 1,
        "axes": [{"name": a, "unit": u} for a, u in zip(i.axes, i.units)],
        "levels": [{"shape": list(l.info.shape), "voxel": list(l.info.voxel_size), "origin": list(l.info.translation)} for l in ms],
    })

def _info(shape, dtype, chunk, voxel, origin=None, unit=""):
    n = len(shape)
    axes = (("c",) if n == 4 else ()) + ("z", "y", "x")
    return ArrayInfo(shape=tuple(shape), dtype=np.dtype(dtype), chunk_shape=tuple(shape[: n - 3]) + tuple(chunk),
                     voxel_size=(1.0,) * (n - 3) + tuple(voxel), units=(unit,) * n, axes=axes,
                     translation=(0.0,) * (n - 3) + tuple(origin or (0.0,) * 3))

def plan(view, ops, shape, dtype, chunk, voxel, source=None):
    ops = [op_from_spec(s) for s in json.loads(ops)]
    info = _info(list(shape), dtype, list(chunk), list(voxel))
    if any(op.input_voxel_size() is not None for op in ops) or any(abs(r - 1) > 1e-9 for r in fused.scale(info, ops)):
        raise ValueError("this page runs ops on every level, each on its level's grid: one that reads at a "
                         "voxel size of its own, or changes the grid (downsample), runs from Python")
    out, lead, halo = fused.plan(info, ops)
    if source:
        describe(source)
    VIEWS[view] = (ops, dtype, list(chunk), source)
    added = list(out.shape[: out.ndim - len(halo)])  # leading axes the ops add (a model's channels)
    controls = {i: [c.model_dump() for c in op.directives.controls] for i, op in enumerate(ops) if op.name == "python"}
    return json.dumps({"dtype": out.dtype.name, "kind": out.kind, "lead": lead, "halo": list(halo), "ndim": out.ndim,
                       "added": added, "controls": controls})

def _mesh(m, result, box, shape, dtype, chunk, voxel, origin, unit):
    """legacy: a fragment (Neuroglancer's encoding). octree: from the whole coarsest level,
    the multi-resolution nodes (level, z, y, x each), header (count, fragment bytes,
    quantization bits) first, then the index. node: a node's mesh, header (vertices,
    triangles) then both as uint32, for the page to encode with Draco."""
    import struct
    from chunkmirage import meshes
    mode, levels = m.pop("mode", "legacy"), m.pop("levels", None)
    m.pop("raw", None)
    spec = meshes.MeshSpec(**{k: v for k, v in m.items() if k not in ("core_lo", "core_hi")})
    info = _info(list(shape), dtype, chunk, list(voxel), list(origin), unit)
    if mode == "mask":  # part of the coarsest level: inside or not, and the surface band, on
        # the part's core (the box the page asked for has a border of meshes.BAND around it)
        core = Box(tuple(m.pop("core_lo")), tuple(m.pop("core_hi")))
        grown = [min(a, meshes.BAND) for a in core.start]
        if any(g < meshes.BAND and b + g != a for g, a, b in zip(grown, core.start, box.start)):
            raise ValueError("a mask part needs a border of meshes.BAND voxels")
        inside = np.asarray(result) >= spec.threshold
        return np.ascontiguousarray(inside[core.slices()], np.uint8).tobytes() + meshes.surface_band(inside, core).astype(np.uint8).tobytes()
    if mode == "node":
        v, f = meshes.multires_fragment(spec, result, box, info, chunk)
        return struct.pack("<II", len(v), len(f)) + v.astype("<u4").tobytes() + f.astype("<u4").tobytes()
    return meshes.fragment(spec, result, box, info)

def octree(band, shape, mesh, chunk, unit):
    """A multi-resolution mesh's nodes (level, z, y, x each; header: their count, the
    fragments' size and the quantization bits) and index, from the whole coarsest level's
    surface band (uint8, 1 near the surface; meshes.surface_band)."""
    import struct
    from chunkmirage import meshes
    m = json.loads(mesh)
    m.pop("mode", None)
    levels = m.pop("levels")
    spec = meshes.MeshSpec(**{**m, "threshold": 1})  # the mask is inside or not
    infos = [_info(l["shape"], "uint8", list(chunk), l["voxel"], l["origin"], unit) for l in levels]
    block = np.frombuffer(band.to_py(), np.uint8).reshape(tuple(shape))
    lods = meshes.lod_levels(infos, spec)
    nodes = meshes.multires_nodes(spec, block, infos, list(chunk))
    flat = np.array([[lv, *n] for lv, ns in zip(lods, nodes) for n in ns], "<i4").reshape(-1, 4)
    head = struct.pack("<III", len(flat), meshes.FRAGMENT_BYTES, meshes.BITS)
    return head + flat.tobytes() + meshes.multires_index(nodes, infos, lods, list(chunk))

def compute(view, level, data, read_shape, in_lo, in_hi, out_lo, out_hi, full_shape, voxel, origin=None, mesh=None, unit=""):
    ops, dtype, chunk, source = VIEWS[view]
    full = tuple(full_shape)
    lead = len(full) - 3
    in_box = Box((0,) * lead + tuple(in_lo), full[:lead] + tuple(in_hi))
    if mesh is not None and json.loads(mesh).get("raw"):  # the page's own mask: no ops to run
        result = np.frombuffer(data.to_py(), np.uint8).reshape(tuple(read_shape))
        return _mesh(json.loads(mesh), result, Box(tuple(out_lo), tuple(out_hi)), full[lead:], "uint8", chunk, voxel, origin, unit)
    if data is None:  # a source computed here: the padded block, as a server stage reads it
        block = SOURCES[source][level].read_padded(in_box, edge=True)
    else:
        block = np.frombuffer(data.to_py(), dtype=np.dtype(dtype)).reshape(tuple(read_shape))
        block = fused.pad_edge(block, in_box, full)
    out, _, halo = fused.plan(_info(full, dtype, chunk, list(voxel)), ops)  # the level's voxels: for_level
    added = out.shape[: out.ndim - len(halo)]  # leading axes the ops add, whole in every chunk
    out_box = Box((0,) * len(added) + tuple(out_lo), tuple(added) + tuple(out_hi))
    result = fused.run(ops, block, in_box, out_box, out, halo)
    if mesh is not None:  # a mesh of these voxels, not a chunk
        return _mesh(json.loads(mesh), result, Box(tuple(out_lo), tuple(out_hi)), full[lead:], out.dtype.name, chunk, voxel, origin, unit)
    # a zarr chunk is always whole: one at the edge of the array is padded with the fill value
    result = np.pad(result, [(0, 0)] * len(added) + [(0, c - n) for c, n in zip(chunk, result.shape[len(added):])])
    return np.ascontiguousarray(result).astype(result.dtype.newbyteorder("<"), copy=False).tobytes()


def call(fn, args, arrays):
    """One step of a page's work in chunkmirage_contrib.stitching (the stitch page) or
    chunkmirage_contrib.tracking (track_*: the track page): JSON in, JSON (or a chunk's bytes) out,
    arrays as raw bytes."""
    if fn.startswith("track_"):
        return _track(fn, json.loads(args), arrays)
    from chunkmirage_contrib import stitching as S
    a = json.loads(args)
    p = S.StitchParams(**a.get("params", {}))
    if fn == "tiles":
        return json.dumps(S.tiles_from_bdv(a["xml"], a["base"], a["channel"]))
    if fn == "overlaps":  # each overlap, and the region of each of its two tiles that holds it
        out = []
        for i, j, lo, hi in S.overlaps(a["tiles"], p.margin):
            out.append({"tiles": [i, j], "lo": lo.tolist(), "hi": hi.tolist(),
                        "regions": [S.region(a["tiles"][t], p.level, lo, hi) for t in (i, j)]})
        return json.dumps(out)
    if fn == "points":
        block = np.frombuffer(arrays[0].to_py(), np.dtype(a["dtype"])).reshape(a["shape"])
        pts = S.points_in(a["tile"], p.level, block, a["start"], a["voxel"], a["lo"], a["hi"], p)
        return json.dumps(pts.round(3).tolist())
    if fn == "register":
        found = S.register(a["tiles"], a["points"], p, a.get("fixed", 0))
        found["grids"] = S.grids(a["tiles"], found["placements"])
        found["reference"] = S.compare(a["tiles"], found["placements"])
        return json.dumps(found)
    if fn == "regions":  # where each tile holds voxels [lo, hi) of a fused level
        lo, hi = S.scene_box(a["grid"], a["lo"], a["hi"])
        return json.dumps([S.region(t, a["level"], lo, hi, pl) for t, pl in zip(a["tiles"], a["placements"])])
    if fn == "fuse":
        blocks, k = [], 0
        for r in a["regions"]:
            if r is None:
                blocks.append(None)
                continue
            shape = [b - s for s, b in zip(*r)]
            blocks.append((np.frombuffer(arrays[k].to_py(), np.dtype(a["dtype"])).reshape(shape), r[0]))
            k += 1
        out = S.fuse(a["tiles"], a["placements"], a["level"], a["grid"], a["lo"], a["hi"], blocks, p.blend, a["dtype"])
        out = np.pad(out, [(0, c - n) for c, n in zip(a["chunk"], out.shape)])  # zarr chunks are whole
        return np.ascontiguousarray(out).astype(out.dtype.newbyteorder("<"), copy=False).tobytes()
    raise ValueError(f"no stitching step {fn}")

def _track(fn, a, arrays):
    from chunkmirage_contrib import tracking as T
    block = lambda k: np.frombuffer(arrays[k].to_py(), np.dtype(a["dtype"])).reshape(a["shape"])
    if fn == "track_measure":  # the object at its first frame, in a block from a.start
        return json.dumps(T.measure(block(0), a["label"], a["start"], a["voxel"]))
    if fn == "track_step":  # the object from the frame before (block 0) in the next (block 1)
        return json.dumps(T.step(block(0), a["label"], block(1), a["start"], a["voxel"]))
    if fn == "track_newborns":  # new labels near a mother lost to mitosis: blocks of the frame before and this
        return json.dumps(T.newborns(block(0), block(1), a["start"], a["voxel"], a["centre"], a["within"], a["min_volume"]))
    raise ValueError(f"no tracking step {fn}")
`)}async function j(e,t){D||await(O??=A(t));let n={};for(let[t,r]of Object.entries(e)){for(let e of r.ops)e.op===`python`&&typeof e.code==`string`&&await D.loadPackagesFromImports(e.code);let e=JSON.parse(D.globals.get(`plan`)(t,JSON.stringify(r.ops),r.shape,r.dtype,r.chunk,r.voxel,r.source));if(e.halo.length!==3||e.added.length>1)throw Error(`view ${t}: a viewer shows three-axis volumes, with one channel axis at most; its ops leave ${e.ndim} axes`);n[t]=e}return n}async function M(e){D||await(O??=A());let t=(e.arrays??[]).map(e=>new Uint8Array(e)),n=await F(()=>D.globals.get(`call`)(e.fn,e.args,t));if(typeof n==`string`)return JSON.parse(n);let r=n.toJs();return n.destroy(),r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}const N={scipy:`scipy`,skimage:`scikit-image`},P=/* @__PURE__ */ new Map;async function F(e){for(let t=0;;t++)try{return e()}catch(e){let n=String(e?.message??``),r=(/No module named '(\w+)/.exec(n)??/`(\w+)` install you are using seems to be broken/.exec(n))?.[1];if(!r||!N[r]||t>1)throw e;P.has(r)||P.set(r,D.loadPackage([N[r]])),await P.get(r),D.runPython(`import sys\nfor m in [m for m in sys.modules if m == "${r}" or m.startswith("${r}.")]: del sys.modules[m]`)}}const I=`https://cdn.jsdelivr.net/gh/google/draco@1.5.7/javascript/`;let L=null;function R(){return L??=(async()=>{let e=await(await fetch(`${I}draco_encoder.js`)).text();return Function(`${e}\nreturn DracoEncoderModule;`)()({locateFile:e=>I+e})})()}async function z(e,t,n){let r=new DataView(e.buffer,e.byteOffset,8),i=r.getUint32(0,!0),a=r.getUint32(4,!0),o=new Uint8Array(t);if(!a)return o.buffer;let s=await R(),c=new Uint32Array(e.slice(8,8+i*12).buffer),l=new Uint32Array(e.slice(8+i*12).buffer),u=new s.Encoder,d=new s.MeshBuilder,f=new s.Mesh,p=new s.DracoInt8Array;try{d.AddFacesToMesh(f,a,l),d.AddFloatAttributeToMesh(f,s.POSITION,i,3,new Float32Array(c)),u.SetAttributeExplicitQuantization(s.POSITION,n,3,[0,0,0],2**n-1),u.SetSpeedOptions(3,3);let e=u.EncodeMeshToDracoBuffer(f,p);if(e>t)throw Error(`a mesh fragment took ${e} bytes, over ${t}`);for(let t=0;t<e;t++)o[t]=p.GetValue(t)}finally{s.destroy(p),s.destroy(f),s.destroy(d),s.destroy(u)}return o.buffer}async function B(e){let t=e.mesh?JSON.stringify({...e.mesh,size:void 0,bits:void 0}):void 0;k={view:e.view,level:e.level,at:e.outLo};let n;try{n=await F(()=>D.globals.get(`compute`)(e.view,e.level,e.data?new Uint8Array(e.data):void 0,e.readShape,e.inLo,e.inHi,e.outLo,e.outHi,e.full,e.voxel,e.origin,t,e.unit??``))}finally{k=null}let r=n.toJs();return n.destroy(),e.mesh?.mode===`node`?z(r,e.mesh.size,e.mesh.bits):r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}E.onmessage=async({data:e})=>{try{if(e.type===`describe`)D||await(O??=A()),E.postMessage({reqId:e.reqId,value:JSON.parse(D.globals.get(`describe`)(e.source))});else if(e.type===`plan`)E.postMessage({reqId:e.reqId,value:await j(e.views,e.packages)});else if(e.type===`octree`){let t=await F(()=>D.globals.get(`octree`)(new Uint8Array(e.band),e.shape,JSON.stringify(e.mesh),e.chunk,e.unit)),n=t.toJs();t.destroy();let r=n.buffer.slice(n.byteOffset,n.byteOffset+n.byteLength);E.postMessage({reqId:e.reqId,value:r},[r])}else if(e.type===`call`){let t=await M(e);E.postMessage({reqId:e.reqId,value:t},t instanceof ArrayBuffer?[t]:[])}else if(e.type===`compute`){let t=await B(e);E.postMessage({reqId:e.reqId,value:t},[t])}}catch(t){let n=t?.message??t?.name??JSON.stringify(t);E.postMessage({reqId:e.reqId,error:String(n)})}};