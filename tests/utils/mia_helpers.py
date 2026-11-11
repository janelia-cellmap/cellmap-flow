"""A stand-in for cellmap_models' ``model_export.mia_model``, which knows the
networks mia-train trains, and the folders such a model comes in: no test
imports mia-train, downloads a model or traces one. A config says what it is
as the MIA team's config.json does, by its ``model``, ``head`` and
``algorithm``; one without a head (a SimMIM pretraining checkpoint) is not
one the adapter serves."""

import importlib
import json
import sys
import types

MIA_FILES = ("config.json", "model.safetensors")

SEGMENTATION = {"model": {"name": "dinov3_vit3d"}, "head": {"name": "UNETRHead"}, "algorithm": {"name": "semantic_seg"}}
PRETRAINING = {"model": {"name": "dinov3_vit3d"}, "algorithm": {"name": "simmim"}}

# What mia_metadata says a semantic model's export will declare, from its
# config.json: read in [0, 1] (its intensity_range), two classes as uint8
# probabilities x 255 (its summed softmax, in the export), run at the data's
# own voxel size.
DECLARED = {
    "model_name": "lmd", "framework": "torch", "spatial_dims": 3, "in_channels": 1, "out_channels": 2,
    "channels_names": ["mito", "er"], "input_voxel_size": [8] * 3, "output_voxel_size": [8] * 3,
    "input_shape": [8] * 3, "output_shape": [8] * 3, "inference_input_shape": [8] * 3,
    "inference_output_shape": [8] * 3, "input_normalization": {"method": "min_max", "min": 0.0, "max": 1.0},
    "inference_output_activation": {"name": "softmax", "apply_after_model": False, "dim": 1},
    "output_dtype": "uint8", "native_voxel_size": True,
}
# And an affinity model's: six affinities, its sigmoid in the export, each
# channel's offset declared.
OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 0, 0], [0, 10, 0], [0, 0, 10]]
AFFINITY_DECLARED = {
    **DECLARED, "out_channels": 6, "channels_names": [f"object_aff_{'_'.join(map(str, o))}" for o in OFFSETS],
    "inference_output_activation": {"name": "sigmoid", "apply_after_model": False}, "output_dtype": "float32",
    "affinity_offsets": OFFSETS, "native_voxel_size": False,
}


def model_metadata(**fields):
    """cellmap_models' ModelMetadata of ``fields``, as mia_metadata answers,
    keeping the fields the cellmap_models installed does not know (CI's
    PyPI release has no output_dtype or native_voxel_size)."""
    from pydantic import ConfigDict

    from cellmap_models.model_export.generate_metadata import ModelMetadata

    class Metadata(ModelMetadata):
        model_config = ConfigDict(extra="allow")

    return Metadata(**fields)


def install_fake_adapter(monkeypatch):
    """The fake ``cellmap_models.model_export.mia_model``, in sys.modules for
    the test: its MIA_FILES, is_mia_config, and mia_metadata, which answers
    its ``declared`` (DECLARED, unless a test says otherwise) of any model.
    Its export is ``configs/mia.export_in_subprocess``'s
    to fake, which runs it in a process of its own."""
    # Its package first (for model_metadata), whose __init__ imports the real
    # module's names when the cellmap_models installed has one.
    importlib.import_module("cellmap_models.model_export")
    module = types.ModuleType("cellmap_models.model_export.mia_model")
    module.MIA_FILES = MIA_FILES
    module.is_mia_config = lambda config: isinstance(config, dict) and "head" in config
    module.declared = DECLARED  # what its mia_metadata answers of any model
    module.mia_metadata = lambda source, revision=None, classes=None: model_metadata(**module.declared)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def write_mia_folder(path, config=SEGMENTATION):
    """A folder as a mia-train model comes: its config.json and (placeholder) weights."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"\0" * 8)
    return path
