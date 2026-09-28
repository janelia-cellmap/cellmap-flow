"""/api/blockwise/generate writes the pipeline's chain as an ordered list.

It wrote json_data as a dict keyed by step name, so a chain that used the
same step twice (two Lambda steps, say) kept only the last one and the
blockwise run computed something other than what the dashboard showed.
"""

import copy

import pytest
import yaml

from cellmap_flow.globals import g

from tests.blockwise.test_submit_route import PIPELINE


@pytest.fixture
def client(tmp_path):
    from cellmap_flow.dashboard.app import app

    g.blockwise_tasks_dir = str(tmp_path / "tasks")
    return app.test_client()


def test_a_step_used_twice_reaches_the_task_yaml_twice_and_in_order(client):
    pipeline = copy.deepcopy(PIPELINE)
    pipeline["normalizers"] = [
        {"name": "LambdaNormalizer", "params": {"expression": "x*2"}},
        {"name": "LambdaNormalizer", "params": {"expression": "x-1"}},
    ]
    pipeline["postprocessors"] = [
        {"name": "ThresholdPostprocessor", "params": {"threshold": 0.5}},
    ]

    body = client.post("/api/blockwise/generate", json={"pipeline": pipeline}).get_json()
    (task_path,) = body["task_paths"]
    with open(task_path) as f:
        json_data = yaml.safe_load(f)["json_data"]

    assert json_data["input_norm"] == [
        {"name": "LambdaNormalizer", "expression": "x*2"},
        {"name": "LambdaNormalizer", "expression": "x-1"},
    ]
    assert json_data["postprocess"] == [{"name": "ThresholdPostprocessor", "threshold": 0.5}]
