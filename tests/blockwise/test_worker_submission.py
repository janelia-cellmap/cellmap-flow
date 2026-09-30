"""How the blockwise master submits its workers to LSF.

The worker bsub had no -W, so the GPU queues killed every worker at two
hours; all workers wrote to the same daisy_logs/out.out and out.err; and the
bsub result was never checked, so a refused submission left the master
waiting forever for a worker that would never connect.
"""

import subprocess

import pytest

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor, spawn_worker
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.utils import bsub_utils


@pytest.fixture
def bsub(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if argv[0] == "bsub":
            return subprocess.CompletedProcess(argv, 0, "Job <77> is submitted to queue <gpu_h100>.\n", "")
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(jobs_lsf.subprocess, "run", run)
    return calls


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_a_worker_gets_a_walltime_and_its_own_log(bsub, tmp_path):
    spawn_worker("predict_mt", "/tasks/t.yaml", "grp", "gpu_h100", ncpu=12, walltime="24:00", log_dir=tmp_path)()

    (argv, kwargs), = [c for c in bsub if c[0][0] == "bsub"]
    assert _flag(argv, "-W") == "24:00"
    assert _flag(argv, "-o") == str(tmp_path / "predict_mt_%J.log")
    assert "-e" not in argv, "stdout and stderr share the per-job log"
    assert _flag(argv, "-P") == "grp" and _flag(argv, "-q") == "gpu_h100"
    assert _flag(argv, "-n") == "12" and _flag(argv, "-gpu") == "num=1"
    assert argv[-1] == "cellmap_flow_blockwise /tasks/t.yaml --client"
    assert kwargs.get("timeout") is None, "an over-ratio bsub is held for minutes; wait for it"


def test_a_worker_without_a_walltime_gets_the_default(bsub, tmp_path):
    spawn_worker("w", "/t.yaml", "grp", "gpu_h100", log_dir=tmp_path)()
    argv = next(c[0] for c in bsub if c[0][0] == "bsub")
    assert _flag(argv, "-W") == bsub_utils.DEFAULT_WALLTIME


def test_a_refused_worker_submission_raises(monkeypatch, tmp_path):
    def refuse(argv, **kwargs):
        raise subprocess.CalledProcessError(255, argv, "", "Project grp is not valid")

    monkeypatch.setattr(jobs_lsf.subprocess, "run", refuse)
    with pytest.raises(subprocess.CalledProcessError):
        spawn_worker("w", "/t.yaml", "grp", "gpu_h100", log_dir=tmp_path)()


def test_the_yaml_walltime_reaches_the_workers(raw_array, model_script, task_yaml):
    path = task_yaml(raw_array(), model_script(), walltime="36:00")
    assert CellMapFlowBlockwiseProcessor(path, create=True).walltime == "36:00"


def test_without_one_the_saved_walltime_is_used(raw_array, model_script, task_yaml, monkeypatch):
    monkeypatch.setattr(blockwise_processor.g, "walltime", "10:00")
    path = task_yaml(raw_array(), model_script())
    assert CellMapFlowBlockwiseProcessor(path, create=True).walltime == "10:00"


def test_only_the_workers_run_the_model(raw_array, model_script, task_yaml, monkeypatch):
    from cellmap_flow.models.models_config import ModelConfig

    def refuse(*args, **kwargs):
        raise AssertionError("the master ran the model")

    monkeypatch.setattr(blockwise_processor, "Inferencer", refuse)
    monkeypatch.setattr(ModelConfig, "_validate_model_shapes", refuse)
    master = CellMapFlowBlockwiseProcessor(task_yaml(raw_array(), model_script()), create=True)
    assert master.inferencers == [] and master.output_arrays, "it still creates the outputs"
    with pytest.raises(RuntimeError, match="worker"):
        master.process_fn(None)


def test_a_worker_checks_its_model_on_the_warmup_only(raw_array, model_script, task_yaml, monkeypatch):
    from cellmap_flow.models.models_config import ModelConfig

    def refuse(*args, **kwargs):
        raise AssertionError("a forward besides the warmup")

    monkeypatch.setattr(ModelConfig, "_validate_model_shapes", refuse)
    path = task_yaml(raw_array(), model_script())
    CellMapFlowBlockwiseProcessor(path, create=True)  # creates the outputs a worker opens
    (inferencer,) = CellMapFlowBlockwiseProcessor(path, create=False).inferencers
    assert inferencer.output_class is not None, "the warmup forward ran"
