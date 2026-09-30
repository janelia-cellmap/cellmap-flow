"""The dashboard's model launcher (dashboard.services.launch): what a failed
launch leaves, and a model taken off and put back on. Its commands are in
test_launch_command, its layers in test_layer_sources_snapshot."""

import logging

import neuroglancer
import pytest

from cellmap_flow.dashboard.services import launch
from cellmap_flow.globals import g
from cellmap_flow.jobs.lsf import BsubTimeoutError
from cellmap_flow.jobs.spec import JobStartError


@pytest.mark.parametrize("error", [pytest.param(JobStartError("no GPU queue took it"), id="no-queue-took-it"),
                                   pytest.param(BsubTimeoutError("bsub hung"), id="bsub-timed-out")])
@pytest.mark.parametrize("start", [lambda: launch.run_model("/models/mito", "mito", "blob"),
                                   lambda: launch.run_hf_model("cellmap/mito", "mito", "blob")], ids=["catalog", "hf"])
def test_a_failed_launch_is_logged_not_raised_and_adds_no_layer(viewer, monkeypatch, caplog, error, start):
    """It runs on a dashboard thread, whose exceptions reach only stderr."""
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(launch, "start_hosts", fail)
    g.jobs, g.dataset_path = [], "/data/raw.zarr"
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.dashboard.services.launch"):
        start()
    assert g.jobs == [] and len(viewer.state.layers) == 0
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
    g.jobs, g.input_norms, g.postprocess = [kept, dropped], [], []
    g.model_catalog = {"catalog": {"mito": "/models/mito", "nuc": "/models/nuc"}}
    with viewer.txn() as s:
        for name in ("mito", "nuc"):
            s.layers[name] = neuroglancer.ImageLayer(source=f"zarr://http://{name}/x")

    launch.update_run_models(["mito"])
    assert dropped.killed and not kept.killed and g.jobs == [kept]
    assert [layer.name for layer in viewer.state.layers] == ["mito"]
    # Selecting it again must start it again, not be a silent no-op.
    launch.update_run_models(["mito", "nuc"])
    assert launched == ["nuc"]
