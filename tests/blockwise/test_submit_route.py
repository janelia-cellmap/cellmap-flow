"""/api/blockwise/submit builds a sound bsub command, without calling LSF.

It ran a bare ``python`` (whatever the job's PATH found first, not the
dashboard's environment) with no -W and no -o, so the master got its queue's
default run limit and LSF mailed its output.
"""

import os
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
    log = _flag(argv, "-o")
    assert log == os.path.join(str(tmp_path / "tasks"), "my_run_%J.log")
    assert body["log_path"] == log.replace("%J", "5150")
    assert "-q" not in argv, "the master is a CPU job; the queue is the workers'"


def test_the_workers_get_the_same_walltime_through_the_task_yaml(client, bsub):
    body = client.post("/api/blockwise/submit", json={"pipeline": PIPELINE}).get_json()
    (task,) = body["task_paths"]
    assert yaml.safe_load(open(task))["walltime"] == "12:00"
