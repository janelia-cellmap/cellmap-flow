"""Deprecated: launching jobs is ``cellmap_flow.jobs.launch``.

Kept only for ``install_cleanup_handlers``, which the docs' script example
(docs/source/scripts.rst) imported from here; it warns, and goes in the
release after 0.3.0.
"""

import warnings

_MOVED = {"install_cleanup_handlers": "cellmap_flow.jobs.launch"}


def __getattr__(name):
    if name not in _MOVED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"cellmap_flow.utils.bsub_utils.{name} is deprecated; "
        f"import it from {_MOVED[name]}",
        DeprecationWarning,
        stacklevel=2,
    )
    import importlib

    return getattr(importlib.import_module(_MOVED[name]), name)
