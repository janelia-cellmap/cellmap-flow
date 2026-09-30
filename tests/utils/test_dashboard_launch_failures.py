"""A model that fails to start is reported, and never becomes a layer."""

import contextlib
import logging

import pytest

from cellmap_flow.globals import g
from cellmap_flow.utils.bsub_utils import BsubTimeoutError, JobStartError


class _Viewer:
    def __init__(self):
        self.state = type("State", (), {"layers": {}})()

    @contextlib.contextmanager
    def txn(self):
        yield self.state


@pytest.mark.parametrize("error", [JobStartError("no GPU queue took it"), BsubTimeoutError("bsub hung")])
@pytest.mark.parametrize("launcher", ["run_model", "run_hf_model"])
def test_a_failed_launch_is_logged_not_raised(monkeypatch, caplog, error, launcher):
    import cellmap_flow.models.run as run

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(run, "start_hosts", fail)
    g.viewer = _Viewer()
    g.jobs = []
    g.dataset_path = "/data/raw.zarr"

    target = "/models/mito" if launcher == "run_model" else "cellmap/mito"
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.models.run"):
        getattr(run, launcher)(target, "mito", "blob")

    assert g.jobs == []
    assert g.viewer.state.layers == {}
    assert any(str(error) in r.getMessage() for r in caplog.records)
