"""The dashboard's model launcher (dashboard.services.launch): what a failed
launch leaves, and a model taken off and put back on. Its commands are in
test_launch_command, its layers in test_layer_sources_snapshot."""

import logging
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
                                       bioimage_entry("affable-shark", [8, 8, 8], "mito"), "blob")],
                         ids=["catalog", "hf", "bioimage"])
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
    def __init__(self, target, args=()):
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

    launch.update_run_models(["mito"])
    assert dropped.killed and not kept.killed and get_session().jobs == [kept]
    assert [layer.name for layer in viewer.state.layers] == ["mito"]
    # Selecting it again must start it again, not be a silent no-op.
    launch.update_run_models(["mito", "nuc"])
    assert launched == ["nuc"]


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
                               model_name="mito", finetuned_model_name=None, params={})
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


def test_a_launch_that_raises_is_reported_in_the_log(viewer, monkeypatch, caplog):
    """Launch threads' exceptions reached only the terminal."""
    def broken(*args, **kwargs):
        raise RuntimeError("no such environment")

    monkeypatch.setattr(launch, "server_command_for", broken)
    get_session().dataset_path = "/data/raw.zarr"
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.dashboard.services.launch"):
        launch.run_hf_model("cellmap/mito", "mito", "blob")
    assert any("no such environment" in r.getMessage() for r in caplog.records)
