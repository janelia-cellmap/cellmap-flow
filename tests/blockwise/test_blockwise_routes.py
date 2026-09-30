"""The dashboard's blockwise routes: the task YAML generated, the precheck, and
what submitting reports. The master's bsub argv is pinned in
test_bsub_argv_snapshot; LSF here is conftest's ``fake_lsf``."""

import copy
import os
import re

import pytest
import yaml

from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig

PIPELINE = {
    "inputs": [{"params": {"dataset_path": "/data/raw.zarr/raw"}}],
    "outputs": [{"params": {"dataset_path": "/out/pred"}}],
    "models": [{"name": "m", "params": {"type": "script", "script_path": "/s.py"}}],
    "blockwise_config": [{"params": {"charge_group": "grp", "queue": "gpu_h100", "nb_workers": 2,
                                     "nb_cores_worker": 12, "nb_cores_master": 4, "tmp_dir": "/scratch/progress"}}],
}
ACCEPTED = "Your request was accepted.\nJob <5150> is submitted to default queue <local>.\n"


@pytest.fixture
def tasks(tmp_path):
    g.blockwise_tasks_dir, g.walltime = str(tmp_path / "tasks"), "12:00"
    return tmp_path / "tasks"


def test_the_task_yaml_holds_the_chain_in_order_and_the_walltime(dashboard, tasks):
    """It wrote json_data as a dict keyed by step name, so a chain using a step
    twice kept only the last one, and blockwise computed something else than
    the dashboard showed."""
    pipeline = copy.deepcopy(PIPELINE)
    pipeline["normalizers"] = [{"name": "LambdaNormalizer", "params": {"expression": "x*2"}},
                               {"name": "LambdaNormalizer", "params": {"expression": "x-1"}}]
    pipeline["postprocessors"] = [{"name": "ThresholdPostprocessor", "params": {"threshold": 0.5}}]
    (path,) = dashboard.post("/api/blockwise/generate", json={"pipeline": pipeline}).get_json()["task_paths"]
    task = yaml.safe_load(open(path))
    assert task["json_data"]["input_norm"] == [{"name": "LambdaNormalizer", "expression": "x*2"},
                                               {"name": "LambdaNormalizer", "expression": "x-1"}]
    assert task["json_data"]["postprocess"] == [{"name": "ThresholdPostprocessor", "threshold": 0.5}]
    assert task["walltime"] == "12:00", "the workers get the master's"


def test_the_precheck_passes_a_task_without_side_effects(dashboard, raw_zarr, pooling_model, task_yaml, tmp_path,
                                                        monkeypatch):
    """It built the processor (create=True): it created the outputs, loaded each
    model into the dashboard, and replaced the dashboard's own chain."""
    def refuse(self):
        raise AssertionError("the precheck loaded a model")

    monkeypatch.setattr(ScriptModelConfig, "_get_config", refuse)
    g.input_norms, g.postprocess = ["the dashboard's own"], ["chain"]
    body = dashboard.post("/api/blockwise/precheck", json={"yaml_paths": [task_yaml(raw_zarr(), pooling_model())]})
    assert body.get_json() == {"success": True, "message": "success"}
    assert not os.path.exists(tmp_path / "out.zarr")
    assert (g.input_norms, g.postprocess) == (["the dashboard's own"], ["chain"])


def test_the_precheck_answers_a_config_error(dashboard, tmp_path):
    (tmp_path / "bad.yaml").write_text("charge_group: g\nmodels: {}\n")  # no data_path
    body = dashboard.post("/api/blockwise/precheck", json={"yaml_paths": [str(tmp_path / "bad.yaml")]}).get_json()
    assert body["success"] is False and "data_path" in body["error"]


@pytest.mark.parametrize("yaml_paths, as_given", [
    pytest.param("prechecked", True, id="prechecked-yamls-as-they-are"),
    pytest.param(None, False, id="none-given"),
    pytest.param([], False, id="empty"),
    pytest.param("not-a-list", False, id="not-a-list"),
    pytest.param(["/no/such/task.yaml"], False, id="missing"),
])
def test_submit_runs_the_prechecked_yamls_or_else_generates_them(dashboard, tasks, fake_lsf, yaml_paths, as_given):
    fake_lsf.answers["bsub"] = [ACCEPTED]
    payload = {"pipeline": PIPELINE}
    if yaml_paths == "prechecked":
        (generated,) = dashboard.post("/api/blockwise/generate", json={"pipeline": PIPELINE}).get_json()["task_paths"]
        # Under a name generate never uses, so a regenerated file cannot pass for it.
        yaml_paths = [os.path.join(os.path.dirname(generated), "prechecked_task.yaml")]
        os.rename(generated, yaml_paths[0])
    if yaml_paths is not None:
        payload["yaml_paths"] = yaml_paths
    before = sorted(tasks.glob("*.yaml"))

    body = dashboard.post("/api/blockwise/submit", json=payload).get_json()

    assert body["success"], body
    (argv,) = fake_lsf.commands("bsub")
    if as_given:
        assert body["task_paths"] == yaml_paths and argv[-1] == yaml_paths[0]
        assert sorted(tasks.glob("*.yaml")) == before, "nothing is regenerated"
    else:
        (task,) = body["task_paths"]
        assert os.path.dirname(task) == str(tasks) and argv[-1] == task


@pytest.mark.parametrize("typed, stem", [
    pytest.param("nuc cerebellum", "nuc_cerebellum", id="spaces"),
    pytest.param("  a/b\\c:d  ", "a_b_c_d", id="unsafe-characters"),  # a file name and an LSF job name
    pytest.param("", "cellmap_flow", id="none-typed"),
])
def test_the_typed_job_name_names_the_task_and_its_master(dashboard, tasks, fake_lsf, typed, stem):
    fake_lsf.answers["bsub"] = [ACCEPTED]
    generated = dashboard.post("/api/blockwise/generate", json={"pipeline": PIPELINE, "job_name": typed}).get_json()
    task = generated["task_name"]
    (path,) = generated["task_paths"]
    assert re.fullmatch(rf"{stem}_\d{{8}}_\d{{6}}", task) and os.path.basename(path) == f"{task}.yaml"
    assert yaml.safe_load(open(path))["task_name"] == task
    body = dashboard.post("/api/blockwise/submit", json={"pipeline": PIPELINE, "yaml_paths": [path],
                                                         "task_name": task}).get_json()
    (argv,) = fake_lsf.commands("bsub")
    assert argv[argv.index("-J") + 1] == body["task_name"] == task


@pytest.mark.parametrize("answer, expected", [
    pytest.param(ACCEPTED, {"success": True, "job_id": "5150"}, id="accepted"),
    pytest.param((255, "", "Project grp is not valid\n"),
                 {"success": False, "error": "LSF error: Project grp is not valid\n"}, id="refused"),
    # bsub said yes, so the task is taken to be queued.
    pytest.param("Your request was accepted.\n", {"success": True, "job_id": "unknown"}, id="accepted-without-an-id"),
])
def test_what_bsub_answered_is_what_submit_reports(dashboard, tasks, fake_lsf, answer, expected):
    fake_lsf.answers["bsub"] = [answer]
    body = dashboard.post("/api/blockwise/submit", json={"pipeline": PIPELINE, "job_name": "my run"}).get_json()
    assert {k: body.get(k) for k in expected} == expected and len(fake_lsf.commands("bsub")) == 1
    if expected.get("job_id") == "5150":  # the master's log, where its %J became the id
        assert body["log_path"] == os.path.join(str(tasks), f"{body['task_name']}_5150.log")
