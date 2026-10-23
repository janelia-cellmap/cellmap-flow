"""Dataset paths as users write them.

A dataset path is a local directory or an http(s), s3 or gs URL, with
``precomputed://`` in front of a neuroglancer precomputed volume's. worldarray
reads them as they are (``worldarray.store``: ``normalize_path``, ``is_remote``,
``join``, ``parent``, ``basename``). The one rule of cellmap-flow's own is the
shell-escaped space.
"""


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
