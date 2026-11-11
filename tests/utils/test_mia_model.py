"""The mia type: a model trained with mia-train, recognized from what is
pasted, converted once into a cellmap-models export in the cache and served
as that export; and what such exports declare (affinity offsets, a native
voxel size, a uint8 output) used where cellmap-flow guessed. cellmap_models'
adapter is faked (tests/utils/mia_helpers.py) and its export is a tiny real
one: no test imports mia-train, downloads a model or traces a ViT."""

import ast
import json
from pathlib import Path

import pytest
import torch

from cellmap_flow.models.registry import build_model
from tests.finetune.test_export_web import _cellmap_folder
from tests.utils.mia_helpers import MIA_FILES, PRETRAINING, SEGMENTATION, install_fake_adapter, write_mia_folder

COMMIT = "1ceca0ef" + "0" * 32
# What the adapter declares of an affinity model's six channels.
OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 0, 0], [0, 10, 0], [0, 0, 10]]


@pytest.fixture
def adapter(monkeypatch):
    return install_fake_adapter(monkeypatch)


def _export(path, channels, **declared):
    """A cellmap-models export of a 1x1x1 convolution to ``channels``, 8³
    voxels in and out, its metadata.json declaring ``declared`` too."""
    _cellmap_folder(path, torch.nn.Conv3d(1, len(channels), 1), (8,) * 3, (8,) * 3)
    metadata = json.loads((path / "metadata.json").read_text())
    metadata.update(out_channels=len(channels), channels_names=channels, input_voxel_size=[8] * 3,
                    output_voxel_size=[8] * 3, **declared)
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def test_a_mia_repo_or_folder_resolves_to_the_mia_type_and_a_pretraining_one_is_refused(adapter, monkeypatch, tmp_path):
    from cellmap_flow.models import resolve as resolve_module

    configs = {"ai-hhmi/lmd-vitl": SEGMENTATION, "ai-hhmi/simmim-vitl": PRETRAINING}
    monkeypatch.setattr(resolve_module, "_hf_files", lambda repo, revision: ["README.md", *MIA_FILES])
    monkeypatch.setattr(resolve_module, "_hf_config", lambda repo, revision: configs[repo])

    resolved = resolve_module.resolve("ai-hhmi/lmd-vitl@v2")
    assert (resolved.type, resolved.params, resolved.name, resolved.env) == (
        "mia", {"repo": "ai-hhmi/lmd-vitl", "revision": "v2"}, "lmd-vitl", "mia")
    with pytest.raises(ValueError):
        resolve_module.resolve("ai-hhmi/simmim-vitl")

    folder = write_mia_folder(tmp_path / "lmd")
    assert resolve_module.resolve(str(folder)).entry() == {"type": "mia", "path": str(folder), "name": "lmd"}
    with pytest.raises(ValueError):
        resolve_module.resolve(str(write_mia_folder(tmp_path / "simmim", PRETRAINING)))
    # Said outright, not looked up.
    assert resolve_module.resolve("mia:ai-hhmi/other", online=False).entry() == {
        "type": "mia", "repo": "ai-hhmi/other", "name": "other"}


def test_a_mia_model_is_converted_once_and_served_as_its_export(adapter, monkeypatch, tmp_path, ome_pyramid):
    """A semantic model, as the adapter exports one: uint8 probabilities x
    255, run at the data's own voxel size."""
    import huggingface_hub

    from cellmap_flow.models.configs import mia
    from cellmap_flow.portable.label_it import BYTE_UNIT
    from cellmap_flow.portable.rank import needed_postprocess

    monkeypatch.setenv(mia.MIA_CACHE_ENV, str(tmp_path / "cache"))
    snapshot = write_mia_folder(tmp_path / "hub" / "snapshots" / COMMIT)
    downloaded = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, revision=None, allow_patterns=None: downloaded.append(repo) or str(snapshot))
    # Hugging Face's local cache, which the dashboard reads the commit from.
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, filename, revision=None: str(snapshot / filename) if downloaded else None)
    converted = []

    def export(source, out):
        converted.append(source)
        _export(Path(out), ["mito", "er"], native_voxel_size=True, output_dtype="uint8")

    monkeypatch.setattr(mia, "export_in_subprocess", export)
    entry = {"type": "mia", "repo": "ai-hhmi/lmd-vitl", "name": "lmd"}

    # Before any server has converted it, the dashboard knows nothing of it,
    # and asking downloads nothing.
    dashboard = build_model(entry)
    assert dashboard.declared_geometry() is None and dashboard.finetune_modes() == ("lora", "full")
    assert not downloaded

    first, second = build_model(entry), build_model(entry)
    assert isinstance(first.config.model, torch.jit.ScriptModule)
    assert isinstance(second.config.model, torch.jit.ScriptModule)
    assert converted == [str(snapshot)]
    assert first.cellmap_model.folder_path == str(tmp_path / "cache" / "ai-hhmi--lmd-vitl" / COMMIT / mia.device_kind())
    assert build_model(first.to_dict()).launch_entry == first.launch_entry

    # Once converted, the dashboard reads it from the cache: its channels,
    # and that a uint8 output cannot be finetuned; its server reports the
    # output as probabilities x 255, which want no sigmoid.
    assert dashboard.declared_geometry()["channels"] == ["mito", "er"]
    assert dashboard.finetune_modes() == () and dashboard.finetune_refusal()
    assert first.config.output_class == BYTE_UNIT
    assert needed_postprocess(BYTE_UNIT, []) is None

    # Served on 4 nm data, it runs at 4 nm, unless its entry says otherwise.
    data = ome_pyramid(levels=((4, 0), (8, 2)))
    native, told = build_model(entry), build_model({**entry, "run_voxel_size": 8})
    for model in (native, told):
        model.adopt_native_voxel_size(data)
    assert native.geometry.input_voxel_size == (4, 4, 4) and told.geometry.input_voxel_size == (8, 8, 8)
    assert converted == [str(snapshot)]


def test_the_offsets_an_export_declares_win_over_its_channel_names(tmp_path):
    from cellmap_flow.dashboard.routes.finetune.common import model_target
    from cellmap_flow.models.configs.cellmap import CellMapModelConfig
    from cellmap_flow.serving.probe import UNIT, review_postprocess

    names = [f"object_aff_{'_'.join(map(str, offset))}" for offset in OFFSETS]
    model = CellMapModelConfig(folder_path=str(_export(tmp_path / "aff", names, affinity_offsets=OFFSETS)))

    # Training and Label it: all six, the long-range ones too, where the
    # names alone are taken to be three nearest neighbours.
    assert model_target(model, lambda: names) == ("affinities", OFFSETS)
    # The model advice's chain into a segmentation: the watershed's neighbourhood.
    advice = review_postprocess(UNIT, [], out_channels=6, channels_names=names,
                                affinity_offsets=model.declared_affinity_offsets())
    assert ast.literal_eval(advice["params"]["AffinityPostprocessor"]["neighborhood"]) == OFFSETS
