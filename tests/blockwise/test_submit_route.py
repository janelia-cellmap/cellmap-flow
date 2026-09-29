"""/api/blockwise/submit builds a sound bsub command, without calling LSF.

It ran a bare ``python`` (whatever the job's PATH found first, not the
dashboard's environment) with no -W and no -o, so the master got its queue's
default run limit and LSF mailed its output.
"""

import os
import re
import subprocess
import sys

import pytest
import yaml

from cellmap_flow.dashboard.routes import blockwise as route
from cellmap_flow.globals import g

PIPELINE = {
    "inputs": [{"params": {"dataset_path": "/data/raw.zarr/raw"}}],
    "outputs": [{"params": {"dataset_path": "/out/pred"}}],
    "models": [{"name": "m", "params": {"type": "script", "script_path": "/s.py"}}],
    "blockwise_config": [
        {
            "params": {
                "charge_group": "grp",
                "queue": "gpu_h100",
                "nb_workers": 2,
                "nb_cores_worker": 12,
                "nb_cores_master": 4,
                "tmp_dir": "/scratch/progress",
            }
        }
    ],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from cellmap_flow.dashboard.app import app

    g.blockwise_tasks_dir = str(tmp_path / "tasks")
    g.walltime = "12:00"
    return app.test_client()


@pytest.fixture
def bsub(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv, 0, "Your request was accepted.\nJob <5150> is submitted to default queue <local>.\n", ""
        )

    monkeypatch.setattr(route.subprocess, "run", run)
    return calls


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_the_master_runs_this_interpreter_with_a_walltime_and_a_log(client, bsub, tmp_path):
    response = client.post("/api/blockwise/submit", json={"pipeline": PIPELINE, "job_name": "my run"})

    body = response.get_json()
    assert body["success"], body
    assert body["job_id"] == "5150"
    (argv,) = bsub
    assert argv[argv.index("-m") - 1] == sys.executable
    assert _flag(argv, "-m") == "cellmap_flow.blockwise.multiple_cli"
    assert _flag(argv, "-W") == "12:00"
    assert body["task_name"].startswith("my_run_") and _flag(argv, "-J") == body["task_name"]
    log = _flag(argv, "-o")
    assert log == os.path.join(str(tmp_path / "tasks"), f"{body['task_name']}_%J.log")
    assert body["log_path"] == log.replace("%J", "5150")
    assert "-q" not in argv, "the master is a CPU job; the queue is the workers'"
    assert "-gpu" not in argv
    assert "bash" not in argv, "run directly: nothing in the command needs a shell"


def _task_files(tmp_path):
    return sorted((tmp_path / "tasks").glob("*.yaml"))


def test_the_prechecked_yamls_are_submitted_as_they_are(client, bsub, tmp_path):
    generated = client.post("/api/blockwise/generate", json={"pipeline": PIPELINE}).get_json()
    # Under a name generate would never use, so a regenerated file cannot be
    # mistaken for it even within the same second.
    (original,) = generated["task_paths"]
    checked = os.path.join(os.path.dirname(original), "prechecked_task.yaml")
    os.rename(original, checked)
    paths = [checked]
    before = _task_files(tmp_path)

    body = client.post(
        "/api/blockwise/submit", json={"pipeline": PIPELINE, "yaml_paths": paths}
    ).get_json()

    assert body["success"], body
    assert body["task_paths"] == paths
    (argv,) = bsub
    assert argv[-len(paths):] == paths
    assert _task_files(tmp_path) == before, "nothing is regenerated"


@pytest.mark.parametrize("typed, stem", [
    ("nuc cerebellum", "nuc_cerebellum"),
    ("  a/b\\c:d  ", "a_b_c_d"),  # safe as a file name and an LSF job name
    ("", "cellmap_flow"),
])
def test_the_typed_job_name_names_the_task_and_its_master(client, bsub, typed, stem):
    generated = client.post(
        "/api/blockwise/generate", json={"pipeline": PIPELINE, "job_name": typed}
    ).get_json()
    task = generated["task_name"]
    assert re.fullmatch(rf"{stem}_\d{{8}}_\d{{6}}", task)
    (path,) = generated["task_paths"]
    assert os.path.basename(path) == f"{task}.yaml"
    assert yaml.safe_load(open(path))["task_name"] == task

    body = client.post(
        "/api/blockwise/submit", json={"pipeline": PIPELINE, "yaml_paths": [path], "task_name": task}
    ).get_json()
    (argv,) = bsub
    assert _flag(argv, "-J") == body["task_name"] == task


@pytest.mark.parametrize("yaml_paths", [None, [], "not-a-list", ["/no/such/task.yaml"]])
def test_otherwise_the_task_is_generated_as_before(client, bsub, tmp_path, yaml_paths):
    payload = {"pipeline": PIPELINE}
    if yaml_paths is not None:
        payload["yaml_paths"] = yaml_paths

    body = client.post("/api/blockwise/submit", json=payload).get_json()

    assert body["success"], body
    (task,) = body["task_paths"]
    assert os.path.dirname(task) == str(tmp_path / "tasks")
    assert bsub[0][-1] == task


def test_the_workers_get_the_same_walltime_through_the_task_yaml(client, bsub):
    body = client.post("/api/blockwise/submit", json={"pipeline": PIPELINE}).get_json()
    (task,) = body["task_paths"]
    assert yaml.safe_load(open(task))["walltime"] == "12:00"


@pytest.mark.parametrize("returncode, stdout, stderr, expected", [
    (255, "", "Project grp is not valid\n", {"success": False, "error": "LSF error: Project grp is not valid\n"}),
    # As before the move: bsub said yes, so the task is taken to be queued.
    (0, "Your request was accepted.\n", "", {"success": True, "job_id": "unknown"}),
])
def test_what_bsub_answered_is_what_the_route_reports(client, monkeypatch, returncode, stdout, stderr, expected):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if kwargs.get("check") and returncode:
            raise subprocess.CalledProcessError(returncode, argv, stdout, stderr)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(route.subprocess, "run", run)
    body = client.post("/api/blockwise/submit", json={"pipeline": PIPELINE}).get_json()

    assert len(calls) == 1
    assert {k: body.get(k) for k in expected} == expected
