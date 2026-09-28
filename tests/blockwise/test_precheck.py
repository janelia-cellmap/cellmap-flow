"""The dashboard's blockwise precheck has no side effects.

It constructed CellMapFlowBlockwiseProcessor(create=True), which created the
output arrays, loaded each model into the dashboard process, and overwrote
the dashboard's live g.input_norms and g.postprocess with the task's.
"""

import os

import pytest

from cellmap_flow.blockwise.blockwise_processor import precheck
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.utils.config_utils import ConfigError

JSON_DATA = {
    "input_norm": {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}},
    "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}},
}


@pytest.fixture
def no_model_loading(monkeypatch):
    def refuse(self):
        raise AssertionError("the precheck loaded a model")

    monkeypatch.setattr(ScriptModelConfig, "_get_config", refuse)


@pytest.fixture
def dashboard_state():
    g.input_norms = ["the dashboard's own"]
    g.postprocess = ["chain"]
    yield
    assert g.input_norms == ["the dashboard's own"]
    assert g.postprocess == ["chain"]


def test_a_valid_task_passes_and_nothing_changes(
    raw_array, model_script, task_yaml, tmp_path, no_model_loading, dashboard_state
):
    path = task_yaml(raw_array(), model_script(), json_data=JSON_DATA)

    summary = precheck(path)

    assert summary["models"] == ["m"]
    assert not os.path.exists(tmp_path / "out.zarr"), "the output must not be created"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"output_path": "/somewhere/out"}, ".zarr"),
        ({"workers": 0}, "workers"),
        ({"track_progress": True}, "tmp_dir"),
        ({"model_mode": "SOMETIMES"}, "SOMETIMES"),
        ({"output_channels": ["a", "a"]}, "duplicated"),
        ({"json_data": {"input_norm": {}}}, "json_data"),
    ],
)
def test_bad_settings_are_reported(raw_array, model_script, task_yaml, no_model_loading, overrides, message):
    with pytest.raises(ConfigError, match=message):
        precheck(task_yaml(raw_array(), model_script(), **overrides))


def test_a_missing_data_path_is_reported(model_script, task_yaml, tmp_path, no_model_loading):
    with pytest.raises(ConfigError, match="does not exist"):
        precheck(task_yaml(str(tmp_path / "nope.zarr" / "raw"), model_script()))


def test_the_processor_rejects_the_same_settings(raw_array, model_script, task_yaml):
    from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor

    with pytest.raises(ConfigError, match="tmp_dir"):
        CellMapFlowBlockwiseProcessor(
            task_yaml(raw_array(), model_script(), track_progress=True), create=True
        )


@pytest.fixture
def client():
    from cellmap_flow.dashboard.app import app

    return app.test_client()


def test_the_route_checks_without_side_effects(
    client, raw_array, model_script, task_yaml, tmp_path, no_model_loading, dashboard_state
):
    path = task_yaml(raw_array(), model_script(), json_data=JSON_DATA)

    response = client.post("/api/blockwise/precheck", json={"yaml_paths": [path]})

    assert response.get_json() == {"success": True, "message": "success"}
    assert not os.path.exists(tmp_path / "out.zarr")


def test_the_route_answers_a_config_error(client, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("charge_group: g\nmodels: {}\n")  # no data_path

    response = client.post("/api/blockwise/precheck", json={"yaml_paths": [str(path)]})

    body = response.get_json()
    assert body["success"] is False
    assert "data_path" in body["error"]
