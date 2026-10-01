"""Dataset paths: where a container ends, how to join onto a path, what is there.

A dataset path is a local directory or an http(s)/s3 URL, and may run on
past its container into the group or array inside it
(``/data/x.zarr/recon-1/em/s0``). ``gs://`` and ``precomputed://`` paths
are neuroglancer precomputed volumes: ``precomputed://`` is followed by a
local directory or, as neuroglancer writes it, a URL
(``precomputed://gs://bucket/volume``).

Only the local filesystem is probed (for ``.zgroup``/``.zarray``/
``zarr.json``); nothing here opens a store.
"""

import os
import re
from typing import Optional, Tuple

ZARR_JSON = "zarr.json"

_REMOTE_PREFIXES = ("http://", "https://", "s3://")
_PRECOMPUTED_PREFIXES = ("precomputed://", "gs://")


def is_remote(path: str) -> bool:
    """An http(s) or s3 URL (read as zarr v2 through fsspec)."""
    return path.startswith(_REMOTE_PREFIXES)


def is_precomputed(path: str) -> bool:
    return path.startswith(_PRECOMPUTED_PREFIXES)


def normalize_path(path: str) -> str:
    """Remove shell-escape backslashes from a filesystem path.

    Users often copy-paste paths from a terminal where spaces are escaped
    (e.g. ``/path/to/file\\ name.zarr``).  YAML preserves the literal
    backslashes, but the filesystem expects plain spaces.
    """
    if is_remote(path):
        return path
    return path.replace("\\ ", " ")


def join(base: str, *parts: str) -> str:
    """Join path components; a URL is joined with "/" whatever the OS."""
    if is_remote(base):
        return "/".join([base.rstrip("/"), *parts])
    return os.path.join(base, *parts)


def is_v3_container(path: str) -> bool:
    """True if ``path`` is a directory with a ``zarr.json`` at its root."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, ZARR_JSON))


def find_v3_container(path: str) -> Optional[str]:
    """The nearest directory at or above ``path`` holding a ``zarr.json``.

    None if there is none (a v2 store), ``path`` is a URL (zarr v3 is read
    from the local filesystem only), or ``path`` does not exist: a missing
    level is not the group above it, whose zarr.json would otherwise answer
    for it.
    """
    if "://" in path or not os.path.exists(path):
        return None
    current = os.path.normpath(path)
    while current and current != os.path.dirname(current):
        if is_v3_container(current):
            return current
        current = os.path.dirname(current)
    return None


def is_zarr_container(path: str) -> bool:
    """A local zarr (v2 or v3) group or array, with or without a .zarr suffix."""
    if is_remote(path):
        return False
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zarray"))
        or os.path.exists(os.path.join(path, ".zattrs"))
        or is_v3_container(path)
    )


def suffix_format(path: str) -> Optional[str]:
    """"zarr" or "n5", whichever of .zarr/.n5 comes last in ``path``; else None."""
    if ".zarr" not in path and ".n5" not in path:
        return None
    return "zarr" if path.rfind(".zarr") > path.rfind(".n5") else "n5"


def _split_container(path: str) -> Tuple[str, str, bool]:
    """``split_container``, and whether the split was at a .zarr/.n5 suffix."""
    extension = suffix_format(path)
    if extension is not None:
        splitter = "." + extension
        container, inner = path.rsplit(splitter, 1)
        if inner.startswith("/"):
            inner = inner[1:]
        return container + splitter, inner, True

    # No .zarr or .n5 suffix: walk up to the directory that is the container.
    if is_remote(path):
        raise RuntimeError(f"Remote URL must contain .zarr or .n5 in the path: {path}")
    # Prefer .zgroup (the container root) over .zarray (a leaf array).
    current = os.path.normpath(path)
    parts = []
    fallback = None  # the first .zarray-only directory, if no .zgroup is found
    while current and current != os.path.dirname(current):
        if os.path.isdir(current):
            if os.path.exists(os.path.join(current, ".zgroup")):
                return current, "/".join(reversed(parts)), False
            if fallback is None and os.path.exists(os.path.join(current, ".zarray")):
                fallback = (current, list(parts))
        current, part = os.path.split(current)
        parts.append(part)

    if fallback is not None:
        container, parts = fallback
        return container, "/".join(reversed(parts)), False

    raise RuntimeError(f"Could not find a zarr or n5 container in path: {path}")


def split_container(path: str) -> Tuple[str, str]:
    """``(container, path inside it)``.

    The container ends at the last ``.zarr``/``.n5`` in ``path``; without
    either suffix it is the nearest directory up from ``path`` with a
    ``.zgroup`` (or, failing that, the first with a ``.zarray``).
    ``/data/x.zarr/em/s0`` is ``("/data/x.zarr", "em/s0")``.
    """
    container, inner, _ = _split_container(path)
    return container, inner


def precomputed_scale(path: str) -> Tuple[str, Optional[int]]:
    """``(volume, scale index)`` for a precomputed path: a last component
    ``s<N>`` names scale N of the volume above it; a path without one is
    the volume itself, with index None."""
    match = re.fullmatch(r"(.+)/s(\d+)", path)
    return (match[1], int(match[2])) if match else (path, None)


def precomputed_kvstore(path: str) -> Tuple[object, int]:
    """``(kvstore, scale_index)`` for a precomputed path, as tensorstore opens it.

    A trailing ``/s<N>`` names the scale (``precomputed_scale``), else it is
    scale 0. ``precomputed://`` followed by a URL -- neuroglancer's way of
    writing a cloud source, ``precomputed://gs://bucket/volume`` or
    ``precomputed://https://host/volume`` -- is that URL, as is a bare
    ``gs://`` path; tensorstore reads each through its own kvstore. Any other
    ``precomputed://`` path is a local directory.

    ``gs://`` is only ever precomputed here: zarr and N5 are read through
    fsspec, which has no gs:// support installed, so a ``gs://`` path with a
    ``.zarr``/``.n5`` suffix raises ValueError rather than being opened as a
    precomputed volume that isn't there.
    """
    explicit = path.startswith("precomputed://")
    location = path[len("precomputed://"):] if explicit else path
    if not explicit and suffix_format(location) is not None:
        bucket_path = location[len("gs://"):]
        raise ValueError(
            f"{path}: zarr and N5 are not read from gs:// (a gs:// path is a neuroglancer "
            f"precomputed volume); for a public bucket, give "
            f"https://storage.googleapis.com/{bucket_path} instead"
        )
    location, scale_index = precomputed_scale(location)
    scale_index = scale_index or 0
    if "://" in location:
        return location, scale_index
    return {"driver": "file", "path": os.path.normpath("/" + location.lstrip("/"))}, scale_index

