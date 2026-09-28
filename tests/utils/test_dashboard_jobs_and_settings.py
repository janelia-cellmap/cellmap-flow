"""Dashboard job bookkeeping and settings endpoints."""

import contextlib

import pytest
from flask import Flask

from cellmap_flow.globals import g


class FakeJob:
    def __init__(self, name):
        self.model_name = name
        self.host = f"http://{name}:1"
        self.killed = False

    def kill(self):
        self.killed = True


class FakeViewer:
    def __init__(self):
        self.state = type("State", (), {"layers": {}})()

    @contextlib.contextmanager
    def txn(self):
        yield self.state


def test_killed_jobs_are_forgotten_and_can_be_relaunched(monkeypatch):
    import cellmap_flow.models.run as run

    launched = []
    monkeypatch.setattr(run, "run_model", lambda path, name, st: launched.append(name))
    monkeypatch.setattr(run.threading, "Thread", _InlineThread)

    kept, dropped = FakeJob("mito"), FakeJob("nuc")
    g.jobs = [kept, dropped]
    g.viewer = FakeViewer()
    g.viewer.state.layers = {"mito": object(), "nuc": object()}
    g.model_catalog = {"catalog": {"mito": "/models/mito", "nuc": "/models/nuc"}}
    g.input_norms, g.postprocess = [], []

    run.update_run_models(["mito"])
    assert dropped.killed and not kept.killed
    assert g.jobs == [kept]
    assert "nuc" not in g.viewer.state.layers

    # Selecting it again must start it again, not be a silent no-op.
    run.update_run_models(["mito", "nuc"])
    assert launched == ["nuc"]


class _InlineThread:
    def __init__(self, target, args=()):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def client(monkeypatch):
    from cellmap_flow.dashboard.routes.models import models_bp
    from cellmap_flow.dashboard.routes.pipeline import pipeline_bp

    monkeypatch.setattr(type(g), "save_server_config", lambda self: None)
    app = Flask(__name__)
    app.register_blueprint(models_bp)
    app.register_blueprint(pipeline_bp)
    return app.test_client()


@pytest.mark.parametrize("bad", [None, "", "twelve"])
def test_blockwise_config_rejects_bad_numbers_with_400(client, bad):
    g.nb_workers = 14
    payload = {"nb_cores_master": 4, "nb_cores_worker": 12, "nb_workers": bad}
    response = client.post("/api/blockwise-config", json=payload)
    assert response.status_code == 400
    # The settings forms read exactly this shape.
    body = response.get_json()
    assert body["success"] is False and "nb_workers" in body["error"]
    assert g.nb_workers == 14, "nothing may be applied on a rejected request"


def test_blockwise_config_accepts_numeric_strings(client):
    payload = {"nb_cores_master": "4", "nb_cores_worker": "12", "nb_workers": "3"}
    response = client.post("/api/blockwise-config", json=payload)
    assert response.status_code == 200
    assert g.nb_workers == 3


@pytest.mark.parametrize("bad", ["", "lots", None])
def test_server_config_rejects_bad_numbers_with_400(client, bad):
    g.queue, g.nb_workers = "gpu_h100", 14
    response = client.post(
        "/api/server-config", json={"queue": "gpu_a100", "nb_workers": bad}
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False and "nb_workers" in body["error"]
    assert (g.queue, g.nb_workers) == ("gpu_h100", 14)


def test_server_config_without_a_json_body_is_a_400(client):
    response = client.post("/api/server-config", data="not json")
    assert response.status_code == 400
    assert response.get_json()["success"] is False
