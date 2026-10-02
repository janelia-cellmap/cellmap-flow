"""Deprecated: the Hugging Face listing is ``cellmap_flow.models.hf_catalog``.

Kept only for ``list_huggingface_models`` and ``refresh_huggingface_models``,
which docs/source/huggingface.rst imported from here; they warn, and go in
the release after 0.3.0. The model form's helpers are
``cellmap_flow.models.registry``'s.
"""

import warnings

_MOVED = {
    "list_huggingface_models": "cellmap_flow.models.hf_catalog",
    "refresh_huggingface_models": "cellmap_flow.models.hf_catalog",
}


def __getattr__(name):
    if name not in _MOVED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"cellmap_flow.models.model_registry.{name} is deprecated; "
        f"import it from {_MOVED[name]}",
        DeprecationWarning,
        stacklevel=2,
    )
    import importlib

    return getattr(importlib.import_module(_MOVED[name]), name)
