"""A stand-in for cellmap_models' ``model_export.mia_model``, which knows the
networks mia-train trains, and the folders such a model comes in: no test
imports mia-train, downloads a model or traces one. A config says what it is
as the MIA team's config.json does, by its ``model``, ``head`` and
``algorithm``; one without a head (a SimMIM pretraining checkpoint) is not
one the adapter serves."""

import json
import sys
import types

MIA_FILES = ("config.json", "model.safetensors")

SEGMENTATION = {"model": {"name": "dinov3_vit3d"}, "head": {"name": "UNETRHead"}, "algorithm": {"name": "semantic_seg"}}
PRETRAINING = {"model": {"name": "dinov3_vit3d"}, "algorithm": {"name": "simmim"}}


def install_fake_adapter(monkeypatch):
    """The fake ``cellmap_models.model_export.mia_model``, in sys.modules for
    the test: its MIA_FILES and is_mia_config. Its export is
    ``configs/mia.export_in_subprocess``'s to fake, which runs it in a
    process of its own."""
    module = types.ModuleType("cellmap_models.model_export.mia_model")
    module.MIA_FILES = MIA_FILES
    module.is_mia_config = lambda config: isinstance(config, dict) and "head" in config
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def write_mia_folder(path, config=SEGMENTATION):
    """A folder as a mia-train model comes: its config.json and (placeholder) weights."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"\0" * 8)
    return path
