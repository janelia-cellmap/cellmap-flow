"""Deprecated: ``Config`` is ``cellmap_flow.models.models_config.Config``.

Kept only for ``Config``, which docs/source/plugins.rst told plugin authors
to import from here; it warns, and goes in the release after 0.3.0.
"""

import warnings

_MOVED = {"Config": "cellmap_flow.models.models_config"}


def __getattr__(name):
    if name not in _MOVED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"cellmap_flow.utils.serialize_config.{name} is deprecated; "
        f"import it from {_MOVED[name]}",
        DeprecationWarning,
        stacklevel=2,
    )
    import importlib

    return getattr(importlib.import_module(_MOVED[name]), name)
