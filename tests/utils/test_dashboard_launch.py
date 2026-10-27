"""The dashboard's model launcher (dashboard.services.launch): a model added
or ticked and submitted, the Models tab's Loaded models list with its ✕, the
exported config and the job logs. Its commands are in test_launch_command, its
layers in test_layer_sources_snapshot."""

import shlex
import threading
from types import SimpleNamespace

import neuroglancer
import pytest
import yaml
import zarr

from cellmap_flow.dashboard.services import launch
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import launch as jobs_launch
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.models.bioimage_catalog import bioimage_entry
from cellmap_flow.models.registry import build_model
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.process_chain import process_chain


class _Job:
    def __init__(self, name):
        self.model_name, self.host, self.killed = name, f"http://{name}:1", False

    def kill(self):
        self.killed = True


class _InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.mark.parametrize("resample", [True, False])
def test_the_resample_box_reaches_the_servers_and_the_exported_config(viewer, dashboard, monkeypatch,
                                                                     placeholder_cellmap_folders, resample):
    """The Models tab's box is the session's setting: the servers it submits
    get --resample, and Export Config keeps it for `cellmap_flow yaml`."""
    commands = []
    monkeypatch.setattr(jobs_launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.jobs, session.dataset_path = [], "/data/raw.zarr"
    session.model_catalog = {"catalog": {"mito": "/models/mito"}}

    response = dashboard.post("/api/models", json={"selected_models": ["mito"], "resample": resample})
    assert response.status_code == 200 and session.resample is resample
    assert len(commands) == 1 and ("--resample" in commands[0].split()) is resample
    exported = yaml.safe_load(dashboard.get("/api/export-config").data)
    assert exported.get("resample", False) is resample


@pytest.mark.parametrize("cluster", [True, False])
def test_the_exported_config_has_cluster_settings_only_where_there_is_a_cluster(dashboard, monkeypatch, cluster):
    """A Mac (no bsub) exported queue: gpu_h100, the site's default, which
    nothing there read; without a cluster they are left out, and said so."""
    from cellmap_flow.jobs import lsf

    monkeypatch.setattr(lsf, "available", lambda: cluster)
    session = get_session()
    session.dataset_path, session.queue, session.charge_group = "/data/raw.zarr", "gpu_h100", "my_lab"
    text = dashboard.get("/api/export-config").data.decode()
    exported = yaml.safe_load(text)
    assert ("queue" in exported and "charge_group" in exported) is cluster
    assert ("no cluster" in text) is not cluster
    assert exported["data_path"] == "/data/raw.zarr"


def test_add_builds_a_pasted_models_entry_and_runs_it_in_its_environment(viewer, dashboard, monkeypatch, tmp_path):
    """POST /api/models/add: what the Models tab's Add sends once resolve's
    needs are filled in."""
    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    commands = []
    monkeypatch.setattr(jobs_launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    from cellmap_flow.dashboard.routes import model_resolve

    monkeypatch.setattr(model_resolve, "threading", launch.threading, raising=False)
    import threading as _threading

    monkeypatch.setattr(_threading, "Thread", lambda target, args=(), daemon=None: _InlineThread(target, args))
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.jobs, session.dataset_path = [], "/data/raw.zarr"

    answer = dashboard.post("/api/models/add", json={"entry": {"type": "script", "script_path": str(script), "name": "mine"}})
    assert answer.status_code == 200 and answer.get_json()["name"] == "mine"
    assert [m.name for m in session.models_config] == ["mine"]
    assert len(commands) == 1 and "serve" in commands[0] and str(script) in commands[0]

    session.jobs = [_Job("mine")]
    assert dashboard.post("/api/models/add", json={"entry": {"type": "script", "script_path": str(script),
                                                             "name": "mine"}}).status_code == 409
    bad = dashboard.post("/api/models/add", json={"entry": {"type": "nope", "name": "x"}})
    assert bad.status_code == 400 and "nope" in bad.get_json()["error"]


# What an onnx export's manifest declares (cellmap/onnx-web-models).
DECLARED = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
            {"name": "LambdaNormalizer", "expression": "2*x-1"}]


@pytest.mark.parametrize("chain", [[], [{"name": "LambdaNormalizer", "expression": "x/255"}]],
                         ids=["no input normalization", "the user's"])
def test_a_model_started_without_input_normalization_sets_the_one_it_declares(dashboard, monkeypatch, tmp_path,
                                                                              chain):
    """Its start sets them, as the chain's setter sets a chain, before its
    layer is drawn, so the layer is drawn through them; the postprocessing
    stays. A chain that has input normalizers is the user's, and is left as
    it is. The page reads the chain back, and whose declaration it is, from
    the Loaded models list."""
    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    model = build_model({"type": "script", "script_path": str(script)}, "mine")
    model.declared_input_norm = lambda data_range=None: DECLARED  # as an onnx model's manifest declares them
    drawn_through = []
    monkeypatch.setattr(launch, "run_model",
                        lambda config, st_data: drawn_through.append(PipelineSpec.from_url_blob(st_data)[0]))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    session = get_session()
    postprocess = [{"name": "SigmoidPostprocessor"}]
    session.set_pipeline(PipelineSpec(chain, postprocess))
    session.jobs, session.dataset_path, session.models_config = [], "/data/raw.zarr", [model]

    assert dashboard.post("/api/models", json={"selected_models": ["mine"]}).status_code == 200
    expected = chain or DECLARED
    assert session.pipeline_spec == PipelineSpec(expected, postprocess)
    assert drawn_through == [PipelineSpec(expected, postprocess)]
    session.jobs = [_Job("mine")]
    loaded = dashboard.get("/api/models/loaded").get_json()
    assert loaded["pipeline"] == {"input_norm": expected, "postprocess": postprocess}
    assert loaded["input_norm_from"] == (None if chain else "mine")


def test_a_model_added_without_input_normalization_sets_the_one_it_declares_too(dashboard, monkeypatch, tmp_path):
    """The Models tab's Add (POST /api/models/add) starts a model as Submit
    does: with no input normalization set, the chain takes the one the model
    declares before its layer is drawn."""
    from cellmap_flow.models import registry

    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    built = registry.build_model

    def declaring(entry, name):
        model = built(entry, name)
        model.declared_input_norm = lambda data_range=None: DECLARED
        return model

    monkeypatch.setattr(registry, "build_model", declaring)
    drawn_through = []
    monkeypatch.setattr(launch, "run_model",
                        lambda config, st_data: drawn_through.append(PipelineSpec.from_url_blob(st_data)[0]))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    import threading as _threading

    monkeypatch.setattr(_threading, "Thread", lambda target, args=(), daemon=None: _InlineThread(target, args))
    session = get_session()
    postprocess = [{"name": "SigmoidPostprocessor"}]
    session.set_pipeline(PipelineSpec([], postprocess))
    session.jobs, session.dataset_path = [], "/data/raw.zarr"

    answer = dashboard.post("/api/models/add", json={"entry": {"type": "script", "script_path": str(script), "name": "mine"}})
    assert answer.status_code == 200
    assert session.pipeline_spec == PipelineSpec(DECLARED, postprocess)
    assert drawn_through == [PipelineSpec(DECLARED, postprocess)]


# What cellmap-models' newer exports declare in metadata.json
# (cellmap/mito-aff-unetiso-setup-53), and uint8 raw scaled onto it: the
# CellMap convention's chain, which a CellMap model declaring none gets.
MIN_MAX = {"method": "min_max", "min": -1.0, "max": 1.0}
SIGNED = [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False},
          {"name": "LambdaNormalizer", "expression": "x*2-1"}]


@pytest.mark.parametrize("metadata, declared", [
    ({"framework": "torch", "input_normalization": MIN_MAX}, True),
    ({"framework": "torch"}, False),  # cellmap/mito-aff-unet-setup-16's: nothing declared
], ids=["declared in metadata.json", "the CellMap convention"])
def test_a_hugging_face_model_started_without_input_normalization_sets_minus_one_to_one(dashboard, monkeypatch,
                                                                                       metadata, declared):
    """A cellmap-models export's metadata.json declares the range the model
    reads, [-1, 1]; a CellMap model that declares none was trained on it too,
    by convention. Either's start sets the raw scaled onto it before its
    layer is drawn, and the page is told which of the two it was."""
    from cellmap_flow.dashboard.routes import model_advice
    from cellmap_flow.models.configs.huggingface import HuggingFaceModelConfig

    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", lambda self: metadata)
    monkeypatch.setattr(model_advice, "raw_dtype_and_range", lambda path: ("uint8", (0.0, 255.0)))
    drawn_through = []
    monkeypatch.setattr(launch, "run_model",
                        lambda config, st_data: drawn_through.append(PipelineSpec.from_url_blob(st_data)[0]))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    session = get_session()
    postprocess = [{"name": "SigmoidPostprocessor"}]
    session.set_pipeline(PipelineSpec([], postprocess))
    session.jobs, session.dataset_path, session.models_config = [], "/data/raw.zarr", []

    answer = dashboard.post("/api/models", json={"selected_hf_models": ["cellmap/mito-aff-unetiso-setup-53"]})
    assert answer.status_code == 200
    assert session.pipeline_spec == PipelineSpec(SIGNED, postprocess)
    assert drawn_through == [PipelineSpec(SIGNED, postprocess)]
    (name,) = [model.name for model in session.models_config]
    session.jobs = [_Job(name)]
    loaded = dashboard.get("/api/models/loaded").get_json()
    assert (loaded["input_norm_from"], loaded["input_norm_declared"]) == (name, declared)


def test_job_logs_keep_a_job_that_failed_to_start_until_it_is_started_again(dashboard, monkeypatch):
    """Its log, the one being read, vanished from the page when it died."""
    from cellmap_flow.jobs import launch as jobs_launch

    class Job:
        job_id, host = "78", None

        def __init__(self, name):
            self.model_name = name

        def get_status(self):
            return None

        def peek(self):
            return "ValueError: give voxel_size"

    monkeypatch.setattr(jobs_launch, "_starting", set())
    monkeypatch.setattr(jobs_launch, "_failed", {})
    get_session().jobs = []
    failing = Job("impartial_shrimp")
    with pytest.raises(JobStartError), jobs_launch._while_starting(failing):
        raise JobStartError("ended (failed) without reporting a server address")
    (job,) = dashboard.get("/api/job-logs").get_json()["jobs"]
    assert (job["model_name"], job["status"], job["log"]) == (
        "impartial_shrimp", "failed to start", "ValueError: give voxel_size")

    with jobs_launch._while_starting(Job("impartial_shrimp")):
        (job,) = dashboard.get("/api/job-logs").get_json()["jobs"]
        assert job["status"] == "starting"
    assert jobs_launch.failed_starts() == []


def _finetune_jobs_model(monkeypatch, status):
    """A finetune job serving mito_finetuned_1 (add_finetuned_layer's job and
    layer), the job ``status`` (a JobStatus), in the session's job manager."""
    from cellmap_flow.dashboard import finetune_layers

    monkeypatch.setattr(finetune_layers, "fetch_model_info", lambda url: {})
    monkeypatch.setattr(finetune_layers, "prediction_layer",
                        lambda *a, **k: neuroglancer.ImageLayer(source="zarr://http://gpu1:9000/x"))
    training = SimpleNamespace(inference_server_url="http://gpu1:9000", lsf_job=SimpleNamespace(job_id="4242"),
                               model_name="mito", finetuned_model_name=None, params={}, status=status, job_id="ft-1")
    finetune_layers.add_finetuned_layer(training, "mito_finetuned_1")
    training.finetuned_model_name = "mito_finetuned_1"
    get_session().finetune_job_manager = SimpleNamespace(jobs={"ft-1": training})
    return training


def test_the_loaded_models_say_where_each_came_from_and_whether_it_can_be_removed(viewer, dashboard, monkeypatch,
                                                                                 tmp_path):
    """One model from each place the Models tab picks them, a finetune's, and
    one still starting; with what the tab's rows know each by, to untick them."""
    from cellmap_flow.finetune.job_manager.state import JobStatus
    from cellmap_flow.models.models_config import CellposeModelConfig, HuggingFaceModelConfig

    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    session = get_session()
    session.model_catalog = {"Mito": {"mito": "/models/mito"}, "Nuclei": {"nuc": "/models/nuc"}}
    session.models_config = [
        HuggingFaceModelConfig(repo="cellmap/fly_organelles_run08_438000", name="fly_organelles_run08_438000"),
        build_model(bioimage_entry("affable-shark", [8, 8, 8], "affable_shark")),
        CellposeModelConfig(pretrained_model="cpsam_v2", voxel_size=[64, 64, 64], output="probability",
                            name="cellpose_sam_v2_probability"),
        build_model({"type": "script", "script_path": str(script)}, "mine"),
    ]
    session.jobs = [_Job(name) for name in ("mito", "fly_organelles_run08_438000", "affable_shark",
                                            "cellpose_sam_v2_probability", "mine")]
    _finetune_jobs_model(monkeypatch, JobStatus.WAITING_FOR_RESTART)
    release = threading.Event()
    launch._launch("nuc", release.wait, 5)
    try:
        models = {m["name"]: m for m in dashboard.get("/api/models/loaded").get_json()["models"]}
    finally:
        release.set()
    assert {name: (m["source"], m["detail"], m["status"], m["removable"]) for name, m in models.items()} == {
        "mito": ("catalog", "Mito", "running", True),
        "fly_organelles_run08_438000": ("huggingface", "cellmap/fly_organelles_run08_438000", "running", True),
        "affable_shark": ("bioimage", "affable-shark", "running", True),
        "cellpose_sam_v2_probability": ("cellpose", "cpsam_v2, probability", "running", True),
        "mine": ("added", "script", "running", True),
        "mito_finetuned_1": ("finetuned", "from mito", "running", False),
        "nuc": ("catalog", "Nuclei", "starting", True),
    }
    assert list(models) == ["mito", "fly_organelles_run08_438000", "affable_shark", "cellpose_sam_v2_probability",
                            "mine", "mito_finetuned_1", "nuc"]
    assert models["fly_organelles_run08_438000"]["hf_id"] == "cellmap/fly_organelles_run08_438000"
    assert models["affable_shark"]["zoo_id"] == "affable-shark"
    assert models["cellpose_sam_v2_probability"]["cellpose"] == {"model": "cpsam_v2", "output": "probability",
                                                                  "do_3d": False}
    assert "Finetune tab" in models["mito_finetuned_1"]["note"]
    assert all(m["note"] is None for name, m in models.items() if name != "mito_finetuned_1")
    assert {m["source"] for m in models.values()} == set(launch.SOURCES)  # the keys models-tab.js labels


def test_remove_stops_that_model_alone_and_leaves_the_others_and_unsubmitted_ticks_alone(
        viewer, dashboard, monkeypatch, placeholder_cellmap_folders):
    """On the page nuc is ticked and sam unticked, neither submitted: the ✕
    of mito sends mito's name alone, so neither of them changes."""
    launched = []
    monkeypatch.setattr(launch, "run_model", lambda config, st: launched.append(config.name))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.model_catalog = {"catalog": {"mito": "/models/mito", "sam": "/models/sam", "nuc": "/models/nuc"}}
    mito, sam = _Job("mito"), _Job("sam")
    session.jobs = [mito, sam]
    with viewer.txn() as s:
        for name in ("mito", "sam"):
            s.layers[name] = neuroglancer.ImageLayer(source=f"zarr://http://{name}/x")

    answer = dashboard.post("/api/models/remove", json={"name": "mito", "selected_models": ["nuc"]})
    assert answer.status_code == 200
    assert answer.get_json()["removed"] == "mito"
    assert [m["name"] for m in answer.get_json()["models"]] == ["sam"]
    assert mito.killed and not sam.killed and launched == []
    assert session.jobs == [sam] and [layer.name for layer in viewer.state.layers] == ["sam"]

    # Ticked again and submitted, it is started again.
    launch.update_run_models(["mito", "sam"])
    assert launched == ["mito"]

    assert dashboard.post("/api/models/remove", json={"name": "ghost"}).status_code == 404
    assert dashboard.post("/api/models/remove", json={"name": " "}).status_code == 400
    assert not sam.killed


def test_export_config_leaves_out_a_model_taken_off(viewer, dashboard):
    """The ✕ takes the model's config with it: a config exported after it,
    loaded with `cellmap_flow yaml`, starts only the models that were kept."""
    from cellmap_flow.models.models_config import HuggingFaceModelConfig

    session = get_session()
    session.dataset_path = "/data/raw.zarr"
    session.models_config = [HuggingFaceModelConfig(repo=f"cellmap/{name}", name=name) for name in ("mito", "nuc")]
    session.jobs = [_Job("mito"), _Job("nuc")]

    def exported():
        return [m["name"] for m in yaml.safe_load(dashboard.get("/api/export-config").data)["models"]]

    assert exported() == ["mito", "nuc"]
    assert dashboard.post("/api/models/remove", json={"name": "mito"}).status_code == 200
    assert exported() == ["nuc"]


def test_a_model_added_by_reference_and_ticked_again_is_started_again_from_its_config(viewer, monkeypatch, tmp_path):
    """No catalog entry names it, so its ticked box did nothing once it had
    been stopped: removed, or unticked and submitted."""
    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    started = []
    monkeypatch.setattr(launch, "run_model", lambda config, st: started.append(config.name))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.model_catalog = {"catalog": {"mito": "/models/mito"}}
    session.models_config = [build_model({"type": "script", "script_path": str(script)}, "mine")]
    session.jobs = []

    launch.update_run_models(["mine"])
    assert started == ["mine"]
    session.jobs = [_Job("mine")]
    launch.update_run_models(["mine"])  # running already: not started twice
    assert started == ["mine"]


def test_a_models_scale_picks_the_level_its_server_reads(viewer, monkeypatch, tmp_path):
    """A config with `scale` launched from the Models tab was served at the
    session's path, the scale ignored; the CLIs applied it. Every launcher
    now reads the level the scale names (serving.launch.launch)."""
    commands = []
    monkeypatch.setattr(jobs_launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    group = tmp_path / "raw.zarr"
    zarr.open_group(str(group), mode="w").create_dataset("s1", shape=(4, 4, 4), dtype="u1")
    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    session = get_session()
    session.dataset_path, session.jobs = str(group), []
    session.models_config = [build_model({"type": "script", "script_path": str(script), "scale": "s1"}, "mine")]

    launch.update_run_models(["mine"])

    (argv,) = [shlex.split(c) for c in commands]
    assert argv[argv.index("-d") + 1] == str(group / "s1")
