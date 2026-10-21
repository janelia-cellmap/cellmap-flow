"""The dashboard's model launcher (dashboard.services.launch): what a failed
launch leaves, a model taken off and put back on, and the Models tab's
Loaded models list with its ✕. Its commands are in test_launch_command, its
layers in test_layer_sources_snapshot."""

import logging
import threading
import time
from types import SimpleNamespace

import neuroglancer
import pytest
import yaml

from cellmap_flow.dashboard.services import launch
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.lsf import BsubTimeoutError, LSFJob
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.models.bioimage_catalog import bioimage_entry
from cellmap_flow.process_chain import process_chain


@pytest.mark.parametrize("error", [pytest.param(JobStartError("no GPU queue took it"), id="no-queue-took-it"),
                                   pytest.param(BsubTimeoutError("bsub hung"), id="bsub-timed-out")])
@pytest.mark.parametrize("start", [lambda: launch.run_model("/models/mito", "mito", "blob"),
                                   lambda: launch.run_hf_model("cellmap/mito", "mito", "blob"),
                                   lambda: launch.run_bioimage_model(
                                       bioimage_entry("affable-shark", [8, 8, 8], "mito"), "blob"),
                                   lambda: launch.run_cellpose_model(
                                       {"pretrained_model": "cpsam_v2", "voxel_size": [64, 64, 64],
                                        "output": "probability", "name": "cellpose_sam_v2"}, "blob")],
                         ids=["catalog", "hf", "bioimage", "cellpose"])
def test_a_failed_launch_is_logged_not_raised_and_adds_no_layer(viewer, monkeypatch, caplog, error, start):
    """It runs on a dashboard thread, whose exceptions reach only stderr."""
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(launch, "start_hosts", fail)
    session = get_session()
    session.jobs, session.dataset_path = [], "/data/raw.zarr"
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.dashboard.services.launch"):
        start()
    assert get_session().jobs == [] and len(viewer.state.layers) == 0
    assert any(str(error) in r.getMessage() for r in caplog.records)


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


def test_a_model_taken_off_is_killed_forgotten_and_can_be_started_again(viewer, monkeypatch):
    launched = []
    monkeypatch.setattr(launch, "run_model", lambda path, name, st: launched.append(name))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    kept, dropped = _Job("mito"), _Job("nuc")
    get_session().jobs = [kept, dropped]
    process_chain().input_norms, process_chain().postprocess = [], []
    get_session().model_catalog = {"catalog": {"mito": "/models/mito", "nuc": "/models/nuc"}}
    with viewer.txn() as s:
        for name in ("mito", "nuc"):
            s.layers[name] = neuroglancer.ImageLayer(source=f"zarr://http://{name}/x")
    get_session().shader_controls["nuc"] = {"contrast": {"range": [0, 1]}}

    launch.update_run_models(["mito"])
    assert dropped.killed and not kept.killed and get_session().jobs == [kept]
    assert [layer.name for layer in viewer.state.layers] == ["mito"]
    # Kept for when it is drawn again (the ✕ forgets it).
    assert get_session().shader_controls["nuc"] == {"contrast": {"range": [0, 1]}}
    # Selecting it again must start it again, not be a silent no-op.
    launch.update_run_models(["mito", "nuc"])
    assert launched == ["nuc"]


def test_a_second_submit_while_a_model_is_still_starting_does_not_start_it_again(viewer, monkeypatch):
    """Two impartial_shrimp jobs once ran side by side: Submit again (Resample
    toggled) before the first had a server, and it was started twice."""
    import threading

    release, launched = threading.Event(), []

    def slow_start(path, name, st):
        launched.append(name)
        release.wait(5)

    monkeypatch.setattr(launch, "run_model", slow_start)
    get_session().jobs = []
    process_chain().input_norms, process_chain().postprocess = [], []
    get_session().model_catalog = {"catalog": {"mito": "/models/mito"}}
    try:
        launch.update_run_models(["mito"])
        launch.update_run_models(["mito"])
        assert launched == ["mito"] and "mito" in launch.running_names()
    finally:
        release.set()
    for _ in range(100):  # its launch thread returns
        if "mito" not in launch.running_names():
            break
        time.sleep(0.05)
    assert "mito" not in launch.running_names()


def test_submit_on_the_models_tab_leaves_a_finetune_jobs_server_running(viewer, monkeypatch):
    """The Finetune tab owns that job. The Models tab, rendered before its
    server came up, has no box for it, and bkilling it ended the training."""
    from cellmap_flow.dashboard import finetune_layers

    monkeypatch.setattr(finetune_layers, "fetch_model_info", lambda url: {})
    monkeypatch.setattr(finetune_layers, "prediction_layer",
                        lambda *a, **k: neuroglancer.ImageLayer(source="zarr://http://gpu1:9000/x"))
    killed = []
    monkeypatch.setattr(LSFJob, "kill", lambda self: killed.append(self.job_id))
    training = SimpleNamespace(inference_server_url="http://gpu1:9000", lsf_job=SimpleNamespace(job_id="4242"),
                               model_name="mito", finetuned_model_name=None, params={}, job_id="ft-1")
    finetune_layers.add_finetuned_layer(training, "mito_finetuned_1")
    monkeypatch.setattr(launch, "run_model", lambda path, name, st: None)
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    get_session().model_catalog = {"catalog": {"nuc": "/models/nuc"}}

    launch.update_run_models(["nuc"])
    assert killed == [] and [j.model_name for j in get_session().jobs] == ["mito_finetuned_1"]
    assert "mito_finetuned_1" in viewer.state.layers


@pytest.mark.parametrize("resample", [True, False])
def test_the_resample_box_reaches_the_servers_and_the_exported_config(viewer, dashboard, monkeypatch, resample):
    """The Models tab's box is the session's setting: the servers it submits
    get --resample, and Export Config keeps it for `cellmap_flow yaml`."""
    commands = []
    monkeypatch.setattr(launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
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
    monkeypatch.setattr(launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
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


def test_job_logs_list_a_job_that_is_still_starting(dashboard, monkeypatch):
    """From Submit until a server answers the page said no job had been submitted."""
    from cellmap_flow.jobs import launch as jobs_launch

    class Starting:
        model_name, job_id, host = "impartial_shrimp", "77", None

        def get_status(self):
            return None

        def peek(self):
            return "Installing environment..."

    monkeypatch.setattr(jobs_launch, "_starting", {Starting()})
    get_session().jobs = []
    (job,) = dashboard.get("/api/job-logs").get_json()["jobs"]
    assert (job["model_name"], job["status"], job["log"]) == ("impartial_shrimp", "starting", "Installing environment...")


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

    for i in range(jobs_launch.FAILED_STARTS_KEPT + 2):
        with pytest.raises(RuntimeError), jobs_launch._while_starting(Job(f"m{i}")):
            raise RuntimeError("died")
    names = [j.model_name for j in jobs_launch.failed_starts()]
    assert names == [f"m{i}" for i in range(2, jobs_launch.FAILED_STARTS_KEPT + 2)]


def test_a_launch_that_raises_is_reported_in_the_log(viewer, monkeypatch, caplog):
    """Launch threads' exceptions reached only the terminal."""
    def broken(*args, **kwargs):
        raise RuntimeError("no such environment")

    monkeypatch.setattr(launch, "server_command_for", broken)
    get_session().dataset_path = "/data/raw.zarr"
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.dashboard.services.launch"):
        launch.run_hf_model("cellmap/mito", "mito", "blob")
    assert any("no such environment" in r.getMessage() for r in caplog.records)


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


def _wait_for(condition, seconds=5):
    for _ in range(int(seconds / 0.02)):
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_the_loaded_models_say_where_each_came_from_and_whether_it_can_be_removed(viewer, dashboard, monkeypatch,
                                                                                 tmp_path):
    """One model from each place the Models tab picks them, a finetune's, and
    one still starting; with what the tab's rows know each by, to untick them."""
    from cellmap_flow.finetune.job_manager.state import JobStatus
    from cellmap_flow.models.models_config import (
        BioModelConfig, CellposeModelConfig, HuggingFaceModelConfig,
    )
    from cellmap_flow.models.registry import build_model

    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    session = get_session()
    session.model_catalog = {"Mito": {"mito": "/models/mito"}, "Nuclei": {"nuc": "/models/nuc"}}
    session.models_config = [
        HuggingFaceModelConfig(repo="cellmap/fly_organelles_run08_438000", name="fly_organelles_run08_438000"),
        BioModelConfig(**bioimage_entry("affable-shark", [8, 8, 8], "affable_shark")),
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


def test_remove_stops_that_model_alone_and_leaves_the_others_and_unsubmitted_ticks_alone(viewer, dashboard,
                                                                                       monkeypatch):
    """On the page nuc is ticked and sam unticked, neither submitted: the ✕
    of mito sends mito's name alone, so neither of them changes."""
    launched = []
    monkeypatch.setattr(launch, "run_model", lambda path, name, st: launched.append(name))
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

    missing = dashboard.post("/api/models/remove", json={"name": "ghost"})
    assert missing.status_code == 404 and missing.get_json()["error"] == "No model named ghost is running"
    nameless = dashboard.post("/api/models/remove", json={"name": " "})
    assert nameless.status_code == 400 and "name is required" in nameless.get_json()["error"]
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


def test_a_finetune_jobs_model_is_removed_only_once_its_job_has_ended(viewer, dashboard, monkeypatch):
    """While the job is alive its model is its server, which Restart trains
    again: the Finetune tab stops it. Once the job has ended its server has
    too, and ✕ takes the layer left pointing at it away, killing nothing."""
    from cellmap_flow.finetune.job_manager.state import JobStatus

    killed = []
    monkeypatch.setattr(LSFJob, "kill", lambda self: killed.append(self.job_id))
    training = _finetune_jobs_model(monkeypatch, JobStatus.WAITING_FOR_RESTART)

    refused = dashboard.post("/api/models/remove", json={"name": "mito_finetuned_1"})
    assert refused.status_code == 409 and "Cancel" in refused.get_json()["error"]
    assert killed == [] and [j.model_name for j in get_session().jobs] == ["mito_finetuned_1"]
    assert "mito_finetuned_1" in viewer.state.layers

    training.status = JobStatus.CANCELLED
    (entry,) = dashboard.get("/api/models/loaded").get_json()["models"]
    assert (entry["removable"], entry["status"]) == (True, "ended") and "has ended (CANCELLED)" in entry["note"]
    answer = dashboard.post("/api/models/remove", json={"name": "mito_finetuned_1"})
    assert answer.status_code == 200 and answer.get_json()["models"] == []
    assert killed == [] and get_session().jobs == [] and "mito_finetuned_1" not in viewer.state.layers


@pytest.mark.parametrize("finetuned", [False, True], ids=["a running model", "an ended finetune job's model"])
def test_a_removed_models_shader_is_forgotten_so_a_later_layer_of_its_name_starts_fresh(viewer, dashboard,
                                                                                       monkeypatch, finetuned):
    from cellmap_flow.finetune.job_manager.state import JobStatus

    session = get_session()
    name = "mito_finetuned_1" if finetuned else "mito"
    if finetuned:
        _finetune_jobs_model(monkeypatch, JobStatus.CANCELLED)
    else:
        session.jobs = [_Job(name)]
    session.shaders[name], session.shader_controls[name] = "void main() {}", {"contrast": {"range": [0, 1]}}

    assert dashboard.post("/api/models/remove", json={"name": name}).status_code == 200
    assert name not in session.shaders and name not in session.shader_controls


def test_a_model_removed_while_starting_is_stopped_as_it_comes_up_and_can_be_submitted_again(viewer, dashboard,
                                                                                           monkeypatch):
    """Its job waiting in a queue is killed at once; what its launch still
    starts is killed as it comes up, and gets no layer. A Submit meanwhile
    starts it again, and that one is kept."""
    from cellmap_flow.jobs import launch as jobs_launch

    release, started = threading.Event(), []

    def slow_start(command, job_name, **kwargs):
        release.wait(5)
        job = _Job(job_name)
        started.append(job)
        jobs_launch.started_jobs().append(job)  # as start_hosts records it
        return job

    monkeypatch.setattr(launch, "start_hosts", slow_start)
    monkeypatch.setattr(launch, "prediction_layer", lambda *a, **k: neuroglancer.ImageLayer(source="zarr://http://m/x"))
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.jobs, session.dataset_path = [], "/data/raw.zarr"
    session.model_catalog = {"catalog": {"mito": "/models/mito"}}
    queued = _Job("mito")
    jobs_launch._starting.add(queued)

    launch.update_run_models(["mito"])
    try:
        assert [(m["name"], m["status"]) for m in dashboard.get("/api/models/loaded").get_json()["models"]] == [
            ("mito", "starting")]
        answer = dashboard.post("/api/models/remove", json={"name": "mito"})
        assert answer.status_code == 200 and answer.get_json()["models"] == []
        assert queued.killed and "mito" not in launch.running_names()
        launch.update_run_models(["mito"])  # ticked again and submitted
    finally:
        release.set()
    assert _wait_for(lambda: len(started) == 2 and not launch._launching)  # both launches returned
    # The two came up in either order: the withdrawn one's job is killed and forgotten, the other kept.
    (kept,) = [job for job in started if not job.killed]
    assert session.jobs == [kept] and "mito" in viewer.state.layers


def test_a_model_added_by_reference_and_ticked_again_is_started_again_from_its_config(viewer, monkeypatch, tmp_path):
    """No catalog entry names it, so its ticked box did nothing once it had
    been stopped: removed, or unticked and submitted."""
    from cellmap_flow.models.registry import build_model

    script = tmp_path / "m.py"
    script.write_text("x = 1\n")
    started = []
    monkeypatch.setattr(launch, "run_model_config", lambda config, st: started.append(config.name))
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
