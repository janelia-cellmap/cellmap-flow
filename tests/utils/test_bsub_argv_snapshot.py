"""What each job submission hands to bsub, and a local run to Popen, pinned.

Four places build bsub commands: start_hosts for inference servers, the
finetune job manager, blockwise spawn_worker, and the dashboard's blockwise
master. These snapshots were recorded against the code as it stood before
they moved onto one builder in cellmap_flow.jobs, so anything that changes
what reaches LSF shows up here as a diff, not as a job that behaves
differently on the cluster.

Every subprocess call is recorded, not only bsub: the ``which bsub`` probe
and the ``bjobs -J`` taken before a submission (to recognise the job if bsub
times out) are part of what a submission does. For each call the snapshot
keeps the argv, the environment as a difference from this process's (an
empty dict means it inherits ours; LSF copies the submitting environment
into the job), and the timeout.

Nothing reaches LSF and no process starts: subprocess.run and
subprocess.Popen are fakes, and each test checks they were called.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellmap_flow.globals import g
from cellmap_flow.utils import bsub_utils

BSUB_ANSWER = "Job <4242> is submitted to queue <gpu_a100>.\n"


class Recorder:
    """Stands in for subprocess.run and Popen, and keeps what they were given."""

    def __init__(self, monkeypatch, tmp_path, bsub_installed=True):
        self.tmp = str(tmp_path)
        self.bsub_installed = bsub_installed
        self.runs = []
        self.popens = []
        monkeypatch.setattr(subprocess, "run", self.run)
        monkeypatch.setattr(subprocess, "Popen", self.popen)

    def normalize(self, value):
        if isinstance(value, dict):
            return {self.normalize(k): self.normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.normalize(v) for v in value]
        if not isinstance(value, str):
            return value
        # Longest first: the interpreter lives under the prefix.
        value = value.replace(sys.executable, "<python>").replace(sys.prefix, "<prefix>")
        value = value.replace(self.tmp, "<tmp>")
        value = re.sub(r"\d{8}_\d{6}", "<ts>", value)
        # tempfile.mkstemp's random part of a local run's log name.
        return re.sub(r"_local_[^/]*\.log$", "_local_<random>.log", value)

    def env_delta(self, env):
        if env is None:
            return {}
        delta = {k: v for k, v in env.items() if os.environ.get(k) != v}
        delta.update({k: "<removed>" for k in os.environ if k not in env})
        return self.normalize(delta)

    def run(self, argv, **kwargs):
        argv = list(argv)
        self.runs.append({
            "argv": self.normalize(argv),
            "env": self.env_delta(kwargs.get("env")),
            "timeout": kwargs.get("timeout"),
        })
        if argv[:2] == ["which", "bsub"]:
            out = b"/usr/bin/bsub\n" if self.bsub_installed else b""
            return subprocess.CompletedProcess(argv, 0 if out else 1, out, b"")
        if argv[0] == "bjobs":
            return subprocess.CompletedProcess(argv, 255, "", f"Job <{argv[-1]}> is not found\n")
        if argv[0] == "bsub":
            return subprocess.CompletedProcess(argv, 0, BSUB_ANSWER, "")
        raise AssertionError(f"unexpected command {argv}")

    def popen(self, args, **kwargs):
        self.popens.append({
            "args": self.normalize(list(args)),
            "env": self.env_delta(kwargs.get("env")),
            "stderr_to_stdout": kwargs.get("stderr") == subprocess.STDOUT,
            "stdin_devnull": kwargs.get("stdin") == subprocess.DEVNULL,
            "start_new_session": kwargs.get("start_new_session"),
            "shell": kwargs.get("shell", False),
        })
        return SimpleNamespace(pid=31337, poll=lambda: None, returncode=None)


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "server_logs"
    monkeypatch.setattr(bsub_utils, "SERVER_LOG_DIR", path)
    return path


def _split(argv):
    """bsub's flags as {flag: value}, and the command that follows them."""
    assert argv[0] == "bsub"
    flags, i = {}, 1
    while i < len(argv) and argv[i].startswith("-"):
        flags[argv[i]] = argv[i + 1]
        i += 2
    return flags, argv[i:]


SERVER_COMMAND = "cellmap_flow_server script -s /models/m.py -d /data/raw.zarr"


# --- (a) an inference server, through start_hosts ---------------------------


def test_a_server_submission(monkeypatch, tmp_path, log_dir):
    rec = Recorder(monkeypatch, tmp_path)
    g.walltime = "12:00"
    g.jobs = []

    job = bsub_utils.start_hosts(
        SERVER_COMMAND, queue="gpu_a100", charge_group="grp", job_name="mito model",
        wait_for_host=False, cycle_queues=False,
    )

    assert rec.runs == [
        {"argv": ["which", "bsub"], "env": {}, "timeout": 5},
        {"argv": ["bjobs", "-a", "-noheader", "-J", "mito model"], "env": {}, "timeout": 10},
        {
            "argv": [
                "bsub", "-J", "mito model", "-o", "<tmp>/server_logs/mito_model_%J.log",
                "-P", "grp", "-q", "gpu_a100", "-gpu", "num=1", "-n", "4", "-W", "12:00",
                "bash", "-c", SERVER_COMMAND,
            ],
            "env": {},
            "timeout": 30,
        },
    ]
    assert rec.popens == []
    assert job.job_id == "4242" and job.queue == "gpu_a100"
    assert job.log_file == log_dir / "mito_model_4242.log"
    assert g.jobs == [job]


# --- (b) a finetune run, through the job manager ------------------------------


class _ScriptModel:
    """Geometry given outright, so submitting never resolves or builds it."""

    cli_name = "script"
    name = "mito"
    script_path = "/models/mito.py"
    channels = ["mito"]
    input_voxel_size = [8, 8, 8]
    output_voxel_size = [16, 16, 16]


class _Thread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


def _submit_finetune(monkeypatch, tmp_path):
    from cellmap_flow.finetune import finetune_job_manager as fjm

    session = tmp_path / "session"
    corrections = session / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text("{}")
    (corrections / "_virtual_sources.json").write_text(json.dumps({
        "kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr",
    }))
    monkeypatch.setattr(fjm.threading, "Thread", _Thread)
    g.walltime = "12:00"
    return fjm.FinetuneJobManager().submit_finetuning_job(
        model_config=_ScriptModel(), corrections_path=corrections, output_base=session,
        queue="gpu_a100", charge_group="my_lab",
    )


FINETUNE_COMMAND = (
    "LD_LIBRARY_PATH=<prefix>/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "
    "stdbuf -oL <python> -m cellmap_flow.finetune.finetune_cli --model-type script "
    "--model-script /models/mito.py --corrections <tmp>/session/corrections "
    "--output-dir <tmp>/session/runs/mito_<ts> --model-name mito --channels mito "
    "--input-voxel-size 8 8 8 --output-voxel-size 16 16 16 --lora-r 8 --lora-alpha 16 "
    "--num-epochs 10 --batch-size 8 --learning-rate 0.0001 --loss-type combined "
    "--auto-serve --serve-data-path /data/raw.zarr --no-augment "
    "--models-dir <tmp>/session/models --queue gpu_a100 --charge-group my_lab "
    "2>&1 | stdbuf -oL tee <tmp>/session/runs/mito_<ts>/training_log.txt"
)


def test_a_finetune_submission(monkeypatch, tmp_path, log_dir):
    rec = Recorder(monkeypatch, tmp_path)

    job = _submit_finetune(monkeypatch, tmp_path)

    assert rec.runs == [
        {"argv": ["which", "bsub"], "env": {}, "timeout": 5},
        {"argv": ["bjobs", "-a", "-noheader", "-J", "finetune_mito_<ts>"], "env": {}, "timeout": 10},
        {
            "argv": [
                "bsub", "-J", "finetune_mito_<ts>", "-o", "<tmp>/server_logs/finetune_mito_<ts>_%J.log",
                "-P", "my_lab", "-q", "gpu_a100", "-gpu", "num=1", "-n", "4", "-W", "12:00",
                "bash", "-c", FINETUNE_COMMAND,
            ],
            "env": {},
            "timeout": 30,
        },
    ]
    assert rec.popens == []
    assert job.lsf_job.job_id == "4242"
    metadata = json.loads((job.output_dir / "metadata.json").read_text())
    assert metadata["lsf_job_id"] == "4242"
    assert rec.normalize(metadata["command"]) == FINETUNE_COMMAND


# --- (c) a blockwise worker, through spawn_worker -----------------------------


def test_a_blockwise_worker_submission(monkeypatch, tmp_path):
    from cellmap_flow.blockwise.blockwise_processor import spawn_worker

    rec = Recorder(monkeypatch, tmp_path)

    spawn_worker(
        "predict_mt", "/tasks/t.yaml", "grp", "gpu_h100", ncpu=12, walltime="24:00",
        log_dir=tmp_path / "daisy_logs",
    )()

    assert rec.runs == [
        {
            "argv": [
                "bsub", "-J", "predict_mt", "-o", "<tmp>/daisy_logs/predict_mt_%J.log",
                "-P", "grp", "-q", "gpu_h100", "-gpu", "num=1", "-n", "12", "-W", "24:00",
                "bash", "-c", "cellmap_flow_blockwise /tasks/t.yaml --client",
            ],
            "env": {},
            # An over-ratio request is held for minutes before bsub answers.
            "timeout": None,
        },
    ]


# --- (d) the dashboard's blockwise master -------------------------------------

PIPELINE = {
    "inputs": [{"params": {"dataset_path": "/data/raw.zarr/raw"}}],
    "outputs": [{"params": {"dataset_path": "/out/pred"}}],
    "models": [{"name": "m", "params": {"type": "script", "script_path": "/s.py"}}],
    "blockwise_config": [{"params": {
        "charge_group": "grp", "queue": "gpu_h100", "nb_workers": 2,
        "nb_cores_worker": 12, "nb_cores_master": 4, "tmp_dir": "/scratch/progress",
    }}],
}


def test_the_blockwise_master_submission(monkeypatch, tmp_path):
    from cellmap_flow.dashboard.app import app

    g.blockwise_tasks_dir = str(tmp_path / "tasks")
    g.walltime = "12:00"
    rec = Recorder(monkeypatch, tmp_path)

    body = app.test_client().post(
        "/api/blockwise/submit", json={"pipeline": PIPELINE, "job_name": "my run"}
    ).get_json()

    assert body["success"], body
    (call,) = rec.runs
    # The flag order is not part of the contract; the flags and the command are.
    flags, command = _split(call["argv"])
    assert flags == {
        "-J": "my run",
        "-n": "4",
        "-P": "grp",
        "-W": "12:00",
        "-o": "<tmp>/tasks/my_run_%J.log",
    }
    assert command == [
        "<python>", "-m", "cellmap_flow.blockwise.multiple_cli", "<tmp>/tasks/cellmap_flow_<ts>.yaml",
    ]
    assert call["env"] == {} and call["timeout"] is None
    assert body["job_id"] == "4242"
    assert rec.normalize(body["log_path"]) == "<tmp>/tasks/my_run_4242.log"


# --- local runs, when there is no bsub ----------------------------------------


def test_a_local_server_run(monkeypatch, tmp_path, log_dir):
    rec = Recorder(monkeypatch, tmp_path)
    g.jobs = []

    job = bsub_utils.start_hosts(SERVER_COMMAND, job_name="mito model", local=True, wait_for_host=False)

    assert rec.runs == []
    assert rec.popens == [{
        "args": ["cellmap_flow_server", "script", "-s", "/models/m.py", "-d", "/data/raw.zarr"],
        # A python child block-buffers a file otherwise.
        "env": {"PYTHONUNBUFFERED": "1"},
        "stderr_to_stdout": True,
        "stdin_devnull": True,
        "start_new_session": True,
        "shell": False,
    }]
    assert rec.normalize(str(job.log_file)) == "<tmp>/server_logs/mito_model_local_<random>.log"
    assert Path(job.log_file).exists()
    assert g.jobs == [job]


def test_a_local_finetune_run(monkeypatch, tmp_path, log_dir):
    rec = Recorder(monkeypatch, tmp_path, bsub_installed=False)

    job = _submit_finetune(monkeypatch, tmp_path)

    assert rec.runs == [{"argv": ["which", "bsub"], "env": {}, "timeout": 5}]
    assert rec.popens == [{
        # The command sets LD_LIBRARY_PATH and pipes through tee: it needs a shell.
        "args": ["bash", "-c", FINETUNE_COMMAND],
        "env": {"PYTHONUNBUFFERED": "1"},
        "stderr_to_stdout": True,
        "stdin_devnull": True,
        "start_new_session": True,
        "shell": False,
    }]
    # It tees its own log; a second copy under server_logs would be noise.
    assert str(job.lsf_job.log_file) == os.devnull
    assert not log_dir.exists() or not list(log_dir.iterdir())
    metadata = json.loads((job.output_dir / "metadata.json").read_text())
    assert metadata["lsf_job_id"] == "PID:31337"
