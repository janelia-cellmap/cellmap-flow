"""The trainer's stdout markers, the log reading, and the job manager's listeners.

The markers are a protocol between two processes that can be different
versions (a training job outlives a dashboard upgrade), so the constants and
patterns are pinned here. What the dashboard does about a finished
iteration -- a viewer layer, a pipeline-builder model -- is now a listener.
"""

import re
from datetime import datetime
from pathlib import Path

from cellmap_flow.finetune import finetuned_model_templates, markers
from cellmap_flow.finetune.finetune_job_manager import (
    FinetuneJob,
    FinetuneJobManager,
    JobStatus,
    trainer_outputs_from_log,
)
from cellmap_flow.finetune.job_log import LogTailer
from cellmap_flow.utils.web_utils import IP_PATTERN

URL = "http://node7:8123"
SERVER_LINE = f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}\n"


def _append(path, text):
    with open(path, "a") as f:
        f.write(text)


def test_the_tailer_hands_out_whole_lines_and_starts_over_on_a_new_file(tmp_path):
    log = tmp_path / "training_log.txt"
    log.write_text("Starting epoch 3 of 10...\nEpoch 3/10 - Lo")
    tail = LogTailer(log)

    assert tail.read() == "Starting epoch 3 of 10...\n"
    _append(log, "ss: 0.25\nTRAINING_ITERATION_COM")
    assert tail.read() == "Epoch 3/10 - Loss: 0.25\n"

    log.write_text("new\n")  # shorter: replaced, not appended to
    assert tail.read() == "new\n", "the held-back half line belonged to the old file"


def test_the_markers_and_patterns_are_the_ones_both_sides_always_used(capsys):
    patterns = {
        "EPOCH_START_RE": (r"Starting\s+epoch\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE),
        "EPOCH_SUMMARY_RE": (r"Epoch\s+(\d+)/(\d+)\s*-\s*Loss:\s*([\d.]+)", re.IGNORECASE),
        "ITERATION_COMPLETE_RE": (r"TRAINING_ITERATION_COMPLETE:\s+(\S+)", 0),
        "MODEL_YAML_RE": (r"^.*?FINETUNED_MODEL_YAML:\s*(.+?)\s*$", re.MULTILINE),
        "STATUS_MARKER_RE": (r"TRAINING_DIVERGED|RESTARTING_TRAINING|WAITING_FOR_RESTART", 0),
        "SERVER_URL_RE": (re.escape(IP_PATTERN[0]) + r"(.+?)" + re.escape(IP_PATTERN[1]), 0),
    }
    for name, (pattern, flags) in patterns.items():
        compiled = getattr(markers, name)
        assert (compiled.pattern, compiled.flags & (re.IGNORECASE | re.MULTILINE)) == (pattern, flags), name

    # What the trainer prints today, until it prints through markers.emit.
    here = Path(markers.__file__).parent
    trainer = (here / "finetune_cli.py").read_text() + (here / "lora_trainer.py").read_text()
    for marker in (
        markers.TRAINING_ITERATION_COMPLETE, markers.RESTART_FAILED, markers.INFERENCE_SERVER_FAILED,
        markers.TRAINING_DIVERGED, markers.RESTARTING_TRAINING, markers.WAITING_FOR_RESTART,
    ):
        assert marker in trainer, marker
    assert markers.FINETUNED_MODEL_YAML == finetuned_model_templates.FINETUNED_MODEL_YAML_MARKER

    # And what emit prints, the job manager reads.
    markers.emit(markers.FINETUNED_MODEL_YAML, "/s/models/m_1.yaml")
    markers.emit(markers.TRAINING_ITERATION_COMPLETE, "m_1")
    markers.emit(markers.WAITING_FOR_RESTART)
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "WAITING_FOR_RESTART"
    assert trainer_outputs_from_log(out) == ("m_1", "/s/models/m_1.yaml")


def _job(tmp_path):
    out = tmp_path / "runs" / "r"
    out.mkdir(parents=True, exist_ok=True)
    return FinetuneJob(
        job_id="j", lsf_job=None, model_name="m", output_dir=out, params={},
        status=JobStatus.RUNNING, created_at=datetime.now(), log_file=out / "training_log.txt",
    )


class Recording:
    def __init__(self):
        self.events = []

    def on_server_ready(self, job, url, model_name):
        self.events.append(("server ready", url, model_name))

    def on_iteration_complete(self, job, model_name):
        # The previous name is still on the job while listeners run.
        self.events.append(("iteration complete", model_name, job.finetuned_model_name))


def test_listeners_hear_of_the_server_and_each_iteration(tmp_path):
    manager = FinetuneJobManager()
    manager.remove_listener(manager.viewer_listener)
    heard = Recording()
    manager.add_listener(heard)
    job = _job(tmp_path)
    job.log_file.write_text("TRAINING_ITERATION_COMPLETE: m_finetuned_1\n" + SERVER_LINE)

    manager._parse_inference_server_ready(job, job.log_file.read_text())
    manager._parse_training_restart(job, "")
    _append(job.log_file, "RESTARTING_TRAINING\nTRAINING_ITERATION_COMPLETE: m_finetuned_2\n")
    manager._parse_training_restart(job, "")

    assert heard.events == [
        ("server ready", URL, "m_finetuned_1"),
        ("iteration complete", "m_finetuned_1", None),
        ("iteration complete", "m_finetuned_2", "m_finetuned_1"),
    ]
    assert job.finetuned_model_name == "m_finetuned_2"


def test_the_default_listener_adds_the_layer_and_the_model_and_failures_stay_contained(
    tmp_path, monkeypatch, caplog
):
    manager = FinetuneJobManager()
    calls = []

    def add_layer(job, model_name):
        calls.append(("layer", model_name))
        if model_name.endswith("_2"):
            raise RuntimeError("no viewer")

    class Broken:
        def on_iteration_complete(self, job, model_name):
            raise RuntimeError("listener bug")

    monkeypatch.setattr(manager, "_add_finetuned_neuroglancer_layer", add_layer)
    monkeypatch.setattr(
        manager, "_register_finetune_model_config", lambda job, name: calls.append(("config", name))
    )
    manager.add_listener(Broken())
    job = _job(tmp_path)
    job.log_file.write_text("TRAINING_ITERATION_COMPLETE: m_finetuned_1\n" + SERVER_LINE)

    manager._parse_inference_server_ready(job, job.log_file.read_text())
    _append(job.log_file, "TRAINING_ITERATION_COMPLETE: m_finetuned_2\n")
    manager._parse_training_restart(job, "")

    assert calls == [
        ("layer", "m_finetuned_1"), ("config", "m_finetuned_1"),
        # The layer failing does not skip the config, nor does another listener.
        ("layer", "m_finetuned_2"), ("config", "m_finetuned_2"),
    ]
    assert job.finetuned_model_name == "m_finetuned_2", "and the name moves on regardless"
    assert "no viewer" in caplog.text and "listener bug" in caplog.text
