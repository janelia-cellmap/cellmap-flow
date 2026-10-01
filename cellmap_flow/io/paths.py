"""Dataset paths: where a container ends, how to join onto a path, what is there.

A dataset path is a local directory or an http(s), s3 or gs URL, and may run
on past its container into the group or array inside it
(``/data/x.zarr/recon-1/em/s0``). A ``precomputed://`` path, and a ``gs://``
path with no ``.zarr``/``.n5`` in it, is a neuroglancer precomputed volume:
``precomputed://`` is followed by a local directory or, as neuroglancer
writes it, a URL (``precomputed://gs://bucket/volume``).

What is at a path is probed through ``io.store`` (for ``.zgroup``,
``.zarray``, ``zarr.json``) only where a path has no ``.zarr``/``.n5``
suffix to split it at; the local v3 checks look at the filesystem.
"""

import os
import re
from typing import Optional, Tuple

ZARR_JSON = "zarr.json"

_REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://")


def is_remote(path: str) -> bool:
    """An http(s), s3 or gs URL."""
    return path.startswith(_REMOTE_PREFIXES)


def is_precomputed(path: str) -> bool:
    """A ``precomputed://`` path, or a ``gs://`` one that names no zarr or
    N5 container (neuroglancer's cloud volumes are precomputed)."""
    if path.startswith("precomputed://"):
        return True
    return path.startswith("gs://") and suffix_format(path) is None


def _location(path: str) -> str:
    """``path`` without a leading ``precomputed://``: a local path or a URL."""
    return path[len("precomputed://"):] if path.startswith("precomputed://") else path


def normalize_path(path: str) -> str:
    """Remove shell-escape backslashes from a filesystem path.

    Users often copy-paste paths from a terminal where spaces are escaped
    (e.g. ``/path/to/file\\ name.zarr``).  YAML preserves the literal
    backslashes, but the filesystem expects plain spaces.

    A URL -- http(s), s3, gs, or one after ``precomputed://`` -- is returned
    as it is: it is not a filesystem path.
    """
    if "://" in _location(path):
        return path
    return path.replace("\\ ", " ")


def join(base: str, *parts: str) -> str:
    """Join path components; a URL is joined with "/" whatever the OS."""
    if "://" in base:
        return "/".join([base.rstrip("/"), *parts])
    return os.path.join(base, *parts)


def parent(path: str) -> str:
    """The directory above ``path``, a URL's as a local path's; a URL's root
    (``s3://bucket``) is its own."""
    if "://" in path:
        path = path.rstrip("/")
        return path if path == _url_root(path) else path.rsplit("/", 1)[0]
    return os.path.dirname(os.path.normpath(path))


def _url_root(path: str) -> str:
    """``scheme://host`` or ``scheme://bucket`` of a URL: nothing is above it."""
    scheme, _, rest = path.partition("://")
    return f"{scheme}://{rest.split('/', 1)[0]}"


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


def split_container(path: str) -> Tuple[str, str]:
    """``(container, path inside it)``.

    The container ends at the last ``.zarr``/``.n5`` in ``path``; without
    either suffix it is the nearest directory up from ``path`` with a
    ``.zgroup`` (or, failing that, the first with a ``.zarray``), on disk or
    at a URL alike.
    ``/data/x.zarr/em/s0`` is ``("/data/x.zarr", "em/s0")``.
    """
    extension = suffix_format(path)
    if extension is not None:
        splitter = "." + extension
        container, inner = path.rsplit(splitter, 1)
        if inner.startswith("/"):
            inner = inner[1:]
        return container + splitter, inner

    # No .zarr or .n5 suffix: walk up to the directory that is the container.
    # Prefer .zgroup (the container root) over .zarray (a leaf array).
    if is_remote(path):
        from cellmap_flow.io.store import exists

        current, top = path.rstrip("/"), _url_root(path)
        has = exists
    else:
        current, top = os.path.normpath(path), None

        def has(directory, name):
            return os.path.isdir(directory) and os.path.exists(os.path.join(directory, name))

    parts = []
    fallback = None  # the first .zarray-only directory, if no .zgroup is found
    while current and current != top and current != os.path.dirname(current):
        if has(current, ".zgroup"):
            return current, "/".join(reversed(parts))
        if fallback is None and has(current, ".zarray"):
            fallback = (current, list(parts))
        parts.append(current.rsplit("/", 1)[1] if top else os.path.basename(current))
        current = parent(current)

    if fallback is not None:
        container, parts = fallback
        return container, "/".join(reversed(parts))

    raise RuntimeError(f"Could not find a zarr or n5 container in path: {path}")


def precomputed_scale(path: str) -> Tuple[str, Optional[int]]:
    """``(volume, scale index)`` for a precomputed path: a last component
    ``s<N>`` names scale N of the volume above it; a path without one is
    the volume itself, with index None."""
    match = re.fullmatch(r"(.+)/s(\d+)", path)
    return (match[1], int(match[2])) if match else (path, None)


def precomputed_volume(path: str) -> Tuple[str, int]:
    """``(volume location, scale index)`` for a precomputed path: the volume's
    local directory or URL, which ``io.store`` reads, and the scale a trailing
    ``/s<N>`` names (``precomputed_scale``), else 0.

    ``precomputed://`` followed by a URL -- neuroglancer's way of writing a
    cloud source, ``precomputed://gs://bucket/volume`` or
    ``precomputed://https://host/volume`` -- is that URL, as is a bare
    ``gs://`` path. Any other ``precomputed://`` path is a local directory.
    """
    location, scale_index = precomputed_scale(_location(path))
    if "://" not in location:
        location = os.path.normpath("/" + location.lstrip("/"))
    return location, scale_index or 0
