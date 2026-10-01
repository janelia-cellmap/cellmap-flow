"""What each job submission hands to bsub, and a local run to Popen, pinned.

Recorded against the code before the four bsub builders (start_hosts, the
finetune job manager, spawn_worker, the dashboard's blockwise master) moved
onto one in cellmap_flow.jobs. Every subprocess call is kept, with its argv,
its environment as a difference from ours ({} = inherits it, which LSF then
copies into the job) and its timeout. subprocess.run and Popen are fakes.
"""

import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import launch
from cellmap_flow.jobs.settings import launcher_settings

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
        # The random part of a ready file's name (jobs.ready.ready_path).
        value = re.sub(r"_[0-9a-f]{8}\.ready$", "_<token>.ready", value)
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
        })
        return SimpleNamespace(pid=31337, poll=lambda: None, returncode=None)


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "server_logs"
    monkeypatch.setattr(launch, "SERVER_LOG_DIR", path)
    return path


def _split(argv):
    """bsub's flags as {flag: value}, and the command that follows them."""
    assert argv[0] == "bsub"
    flags, i = {}, 1
    while i < len(argv) and argv[i].startswith("-"):
        flags[argv[i]] = argv[i + 1]
        i += 2
    return flags, argv[i:]


# What serving.launch.server_command builds for a script model.
SERVER_COMMAND = """cellmap_flow serve --model '{"type":"script","script_path":"/models/m.py"}' -d /data/raw.zarr"""


# --- (a) an inference server, through start_hosts ---------------------------


def test_a_server_submission(monkeypatch, tmp_path, log_dir):
    rec = Recorder(monkeypatch, tmp_path)
    launcher_settings().walltime = "12:00"

    launch.start_hosts(
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
            # Where the server is to write its address (jobs/ready.py).
            "env": {"CELLMAP_FLOW_READY_FILE": "<tmp>/server_logs/mito_model_<token>.ready"},
            "timeout": 30,
        },
    ]


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
    from cellmap_flow.finetune.job_manager import manager

    session = tmp_path / "session"
    corrections = session / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text("{}")
    (corrections / "_virtual_sources.json").write_text(json.dumps({
        "kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr",
    }))
    monkeypatch.setattr(manager.threading, "Thread", _Thread)
    return manager.FinetuneJobManager().submit_finetuning_job(
        model_config=_ScriptModel(), corrections_path=corrections, output_base=session,
        queue="gpu_a100", charge_group="my_lab", walltime="12:00",
    )


FINETUNE_COMMAND = (
    "set -o pipefail; LD_LIBRARY_PATH=<prefix>/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "
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
    # What a later dashboard finds the job by (rehydrate_session).
    assert json.loads((job.output_dir / "metadata.json").read_text())["lsf_job_id"] == "4242"


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
                "bash", "-c", "cellmap_flow blockwise /tasks/t.yaml --client",
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

    get_session().blockwise_tasks_dir = str(tmp_path / "tasks")
    launcher_settings().walltime = "12:00"
    rec = Recorder(monkeypatch, tmp_path)

    body = app.test_client().post(
        "/api/blockwise/submit", json={"pipeline": PIPELINE, "job_name": "my run"}
    ).get_json()

    assert body["success"], body
    (call,) = rec.runs
    # The flag order is not part of the contract; the flags and the command are.
    flags, command = _split(call["argv"])
    assert flags == {
        "-J": "my_run_<ts>",
        "-n": "4",
        "-P": "grp",
        "-W": "12:00",
        "-o": "<tmp>/tasks/my_run_<ts>_%J.log",
    }
    assert command == [
        "<python>", "-m", "cellmap_flow.blockwise.multiple_cli", "<tmp>/tasks/my_run_<ts>.yaml",
    ]
    assert call["env"] == {} and call["timeout"] is None


# --- local runs, when there is no bsub ----------------------------------------


def test_a_local_server_run(monkeypatch, tmp_path, log_dir):
    rec = Recorder(monkeypatch, tmp_path)

    job = launch.start_hosts(SERVER_COMMAND, job_name="mito model", local=True, wait_for_host=False)

    assert rec.runs == []
    assert rec.popens == [{
        "args": ["cellmap_flow", "serve", "--model", '{"type":"script","script_path":"/models/m.py"}',
                 "-d", "/data/raw.zarr"],
        # A python child block-buffers a file otherwise.
        "env": {"PYTHONUNBUFFERED": "1"},
        "stderr_to_stdout": True,
        "stdin_devnull": True,
        "start_new_session": True,
    }]
    assert rec.normalize(str(job.log_file)) == "<tmp>/server_logs/mito_model_local_<random>.log"


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
    }]
    # It tees its own log; a second copy under server_logs would be noise.
    assert str(job.lsf_job.log_file) == os.devnull
