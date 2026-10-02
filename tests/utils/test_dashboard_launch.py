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
from cellmap_flow.process_chain import process_chain


@pytest.mark.parametrize("error", [pytest.param(JobStartError("no GPU queue took it"), id="no-queue-took-it"),
                                   pytest.param(BsubTimeoutError("bsub hung"), id="bsub-timed-out")])
@pytest.mark.parametrize("start", [lambda: launch.run_model("/models/mito", "mito", "blob"),
                                   lambda: launch.run_hf_model("cellmap/mito", "mito", "blob")], ids=["catalog", "hf"])
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
