"""Import paths the docs used to show still work for one release, and warn.

Each row is a name docs/ told users to import from a module that has since
been dissolved; the docs name its new home now, and scripts and plugins
written against the old docs get a DeprecationWarning pointing there. The
aliases go in the release after 0.3.0 (cleanup_review/WRAPPERS.md).
"""

import importlib

import pytest

# old module, name, new module
ALIASES = {
    "install_cleanup_handlers": ("cellmap_flow.utils.bsub_utils", "install_cleanup_handlers", "cellmap_flow.jobs.launch"),
    "list_huggingface_models": ("cellmap_flow.models.model_registry", "list_huggingface_models", "cellmap_flow.models.hf_catalog"),
    "refresh_huggingface_models": ("cellmap_flow.models.model_registry", "refresh_huggingface_models",
                                   "cellmap_flow.models.hf_catalog"),
    "Config": ("cellmap_flow.utils.serialize_config", "Config", "cellmap_flow.models.models_config"),
}


@pytest.mark.parametrize("alias", ALIASES)
def test_an_old_documented_import_still_works_and_warns(alias):
    old, name, new = ALIASES[alias]
    with pytest.warns(DeprecationWarning, match=new):
        value = getattr(importlib.import_module(old), name)
    assert value is getattr(importlib.import_module(new), name)
