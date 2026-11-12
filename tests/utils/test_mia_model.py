"""The mia type: a model trained with mia-train, recognized from what is
pasted, converted once into a cellmap-models export in the cache and served
as that export; and what such exports declare (affinity offsets, a native
voxel size, a uint8 output) used where cellmap-flow guessed. cellmap_models'
adapter is faked (tests/utils/mia_helpers.py) and its export is a tiny real
one: no test imports mia-train, downloads a model or traces a ViT."""

import ast
import json
import os
import time
from pathlib import Path

import pytest
import torch

from cellmap_flow.models.registry import build_model
from tests.finetune.test_export_web import _cellmap_folder
from tests.utils.mia_helpers import (
    AFFINITY_DECLARED, DECLARED, MIA_FILES, OFFSETS, PRETRAINING, SEGMENTATION, install_fake_adapter, write_mia_folder,
)

COMMIT = "1ceca0ef" + "0" * 32


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
    # The same two files with a config.json that is no mia-train model's:
    # the folder is what its other files say.
    zoo = write_mia_folder(tmp_path / "zoo", {"architectures": ["UNet"]})
    (zoo / "rdf.yaml").write_text("")
    assert resolve_module.resolve(str(zoo)).type == "bioimage"
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

    # Before any server has converted it, the dashboard knows what its export
    # will declare (mia_metadata, from its config.json), and asking downloads
    # nothing.
    dashboard = build_model(entry)
    assert dashboard.declared_geometry()["channels"] == ["mito", "er"] and dashboard.finetune_modes() == ()
    assert not downloaded

    first, second = build_model(entry), build_model(entry)
    assert isinstance(first.config.model, torch.jit.ScriptModule)
    assert isinstance(second.config.model, torch.jit.ScriptModule)
    assert converted == [str(snapshot)]
    assert first.cellmap_model.folder_path == str(tmp_path / "cache" / "ai-hhmi--lmd-vitl" / COMMIT / mia.converter_version() / mia.device_kind())
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


def test_a_plan_that_could_not_be_read_is_read_again_a_minute_on_and_never_guessed_finetunable(
        adapter, monkeypatch, tmp_path):
    """The Hub times out once as Submit builds the config the session keeps:
    until the plan is read, nothing is declared and the model is not offered
    for finetuning (an affinity model, which can be); a minute on, it is
    read, and is."""
    import huggingface_hub

    from cellmap_flow.models.configs import mia

    monkeypatch.setenv(mia.MIA_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, filename, revision=None: None)
    adapter.declared = AFFINITY_DECLARED
    plan, failures = adapter.mia_metadata, [TimeoutError("the Hub timed out")]

    def metadata(source, revision=None):
        if failures:
            raise failures.pop()
        return plan(source)

    adapter.mia_metadata = metadata
    clock = [1000.0]
    monkeypatch.setattr(mia.time, "monotonic", lambda: clock[0])

    model = mia.MiaModelConfig(repo="ai-hhmi/lmd-aff")
    assert model.planned_metadata() is None and model.declared_affinity_offsets() is None
    assert model.finetune_modes() == () and model.finetune_refusal()
    clock[0] += mia.MiaModelConfig.PLAN_RETRY_SECONDS
    assert model.planned_metadata() is not None and model.declared_affinity_offsets() == OFFSETS
    assert model.finetune_modes() == ("lora", "full") and model.finetune_refusal() is None


def test_a_conversion_removes_killed_conversions_folders_and_other_converters_exports(monkeypatch, tmp_path):
    """A conversion killed outright left its work folder; an older converter
    left its export for this device. Converting removes both, and keeps a
    conversion under way and the other device's export."""
    from cellmap_flow.models.configs import mia

    def export(folder):
        folder.mkdir(parents=True)
        for name in ("metadata.json", "model.ts"):
            (folder / name).write_text("{}")
        return folder

    monkeypatch.setenv(mia.MIA_CACHE_ENV, str(tmp_path))
    model = tmp_path / "ai-hhmi--lmd-vitl" / COMMIT
    killed, running = model / "new" / ".cuda.converting-1", model / "old" / ".cpu.converting-2"
    for work in (killed, running):
        (work / "export").mkdir(parents=True)
    hours_ago = time.time() - 2 * mia.STALE_WORK_SECONDS
    for path in (killed / "export", killed):
        os.utime(path, (hours_ago, hours_ago))
    old_cuda, old_cpu = export(model / "old" / "cuda"), export(model / "old" / "cpu")
    monkeypatch.setattr(mia, "export_in_subprocess", lambda source, out: export(Path(out)))

    mia.convert_once(tmp_path / "source", model / "new" / "cuda")
    assert mia.is_export(model / "new" / "cuda") and not killed.exists() and running.exists()
    assert not old_cuda.exists() and mia.is_export(old_cpu)


def test_blockwise_refuses_a_mia_model_with_an_overlap_before_converting_it(adapter, monkeypatch, tmp_path, raw_zarr):
    """Its overlap is in what its export will declare: the master refuses
    it without converting the model only to find it there."""
    import huggingface_hub
    import yaml

    from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor
    from cellmap_flow.config.yaml import ConfigError
    from cellmap_flow.models.configs import mia

    monkeypatch.setenv(mia.MIA_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, filename, revision=None: None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: pytest.fail("downloaded"))
    adapter.declared = {**DECLARED, "overlap": 2}
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump({
        "data_path": raw_zarr(), "output_path": str(tmp_path / "out.zarr"), "task_name": "t", "workers": 1,
        "charge_group": "grp", "queue": "gpu_h100",
        "models": [{"type": "mia", "repo": "ai-hhmi/lmd-vitl", "name": "lmd"}],
    }))
    with pytest.raises(ConfigError, match="overlap"):
        CellMapFlowBlockwiseProcessor(str(task), create=True)


# What the model advice suggests after each, from what its server says its
# output is (the uint8 export's declared probabilities x 255; the sigmoid its
# network ends in), for the user to Apply: Submit sets none of it.
ARGMAX = [{"name": "ArgmaxPostprocessor", "threshold": 127.5}]
WATERSHED = [{"name": "DefaultPostprocessor", "clip_min": 0.0, "clip_max": 1.0, "bias": 0.0, "multiplier": 255.0},
             {"name": "AffinityPostprocessor", "bias": 0.5, "neighborhood": str(OFFSETS)}]
SEMANTIC_SERVED = {"available": True, "output_class": "byte_unit", "output_class_from": "declared", "output_channels": 2}
AFFINITY_SERVED = {"available": True, "output_class": "unit", "output_class_from": "graph", "output_op": "Sigmoid",
                   "output_channels": 6}
THEIRS = [{"name": "ThresholdPostprocessor", "threshold": 0.3}]


@pytest.mark.parametrize("declared, served, chain, suggested", [
    pytest.param(DECLARED, SEMANTIC_SERVED, [], ARGMAX, id="semantic: its classes' argmax"),
    pytest.param(AFFINITY_DECLARED, AFFINITY_SERVED, [], WATERSHED, id="affinities: the watershed at its offsets"),
    pytest.param(AFFINITY_DECLARED, AFFINITY_SERVED, THEIRS, None, id="a chain with steps is left alone"),
])
def test_submit_sets_what_a_mia_model_declares_and_the_advice_suggests_the_rest(
        adapter, dashboard, monkeypatch, tmp_path, declared, served, chain, suggested):
    """Ticked in the Hugging Face list and submitted before any server has
    converted it, a mia model's start sets the chain's input normalization
    from what its export will declare (mia_metadata): the raw data's range
    scaled onto [0, 1], as a cellmap-models export's min_max [0, 1] is,
    before its layer is drawn. Its postprocessing is not set: its export
    declares its activation in it, so nothing is to be applied after it, and
    what its output wants is the model advice's suggestion, to Apply. The
    page is told whose the input normalization is, after an Apply that
    sends it as the page writes it too. Not a weight is fetched for it."""
    import huggingface_hub

    from cellmap_flow.dashboard.routes import model_advice
    from cellmap_flow.dashboard.services import launch
    from cellmap_flow.dashboard.state import get_session
    from cellmap_flow.models import hf_catalog
    from cellmap_flow.models.configs import mia
    from cellmap_flow.pipeline_spec import PipelineSpec
    from tests.utils.test_dashboard_launch import _InlineThread, _Job

    repo = "ai-hhmi/lmd-vitl"
    adapter.declared = declared
    monkeypatch.setenv(mia.MIA_CACHE_ENV, str(tmp_path / "cache"))  # no export of it there
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: pytest.fail("downloaded"))
    monkeypatch.setattr(hf_catalog, "cached_huggingface_models", lambda: {repo: {hf_catalog.LISTED_TYPE: "mia"}})
    monkeypatch.setattr(model_advice, "raw_dtype_and_range", lambda path: ("uint8", (0.0, 255.0)))
    drawn_through = []
    monkeypatch.setattr(launch, "run_model",
                        lambda config, st_data: drawn_through.append(PipelineSpec.from_url_blob(st_data)[0]))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    session = get_session()
    session.set_pipeline(PipelineSpec([], chain))
    session.jobs, session.dataset_path, session.models_config = [], "/data/raw.zarr", []

    assert dashboard.post("/api/models", json={"selected_hf_models": [repo]}).status_code == 200
    unit = [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}]
    assert session.pipeline_spec == PipelineSpec(unit, chain) and drawn_through == [PipelineSpec(unit, chain)]

    # Once its server answers, the advice suggests what its output wants.
    (name,) = [model.name for model in session.models_config]
    session.jobs = [_Job(name)]
    monkeypatch.setattr(model_advice, "_fetch_probe", lambda host: served)
    if suggested:
        review = dashboard.get("/api/model_advice").get_json()["models"][0]["postprocess_review"]
        assert [{"name": step, **review["params"].get(step, {})} for step in review["suggest"]] == suggested

    loaded = dashboard.get("/api/models/loaded").get_json()
    assert (loaded["input_norm_from"], loaded["postprocess_from"]) == (name, None)
    page = [{"name": "MinMaxNormalizer", "min_value": "0.0", "max_value": "255.0", "invert": "false"}]
    assert dashboard.put("/api/pipeline", json={"input_norm": page, "postprocess": chain}).status_code == 200
    assert dashboard.get("/api/models/loaded").get_json()["input_norm_from"] == name


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
