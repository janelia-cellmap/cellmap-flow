"""How the job manager follows a training log, and who it tells.

The trainer's markers and the patterns that find them now live in
finetune/markers.py, the whole-lines-only log reading in job_log.LogTailer,
and what the dashboard does about a finished iteration -- a viewer layer,
a pipeline-builder model -- is a listener, so other code can be told too.
"""

import re
from datetime import datetime
from pathlib import Path

import pytest

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


# --- LogTailer ------------------------------------------------------------------


def _append(path, text):
    with open(path, "a") as f:
        f.write(text)


def test_only_whole_lines_are_handed_out(tmp_path):
    log = tmp_path / "training_log.txt"
    log.write_text("Starting epoch 3 of 10...\nEpoch 3/10 - Lo")
    tail = LogTailer(log)

    assert tail.read() == "Starting epoch 3 of 10...\n"
    assert tail.read() == ""
    _append(log, "ss: 0.25\nTRAINING_ITERATION_COM")
    assert tail.read() == "Epoch 3/10 - Loss: 0.25\n"
    _append(log, "PLETE: m_1\n")
    assert tail.read() == "TRAINING_ITERATION_COMPLETE: m_1\n"


def test_a_log_that_shrinks_is_read_again_from_the_start(tmp_path):
    log = tmp_path / "training_log.txt"
    log.write_text("one\ntwo\nthr")
    tail = LogTailer(log)
    assert tail.read() == "one\ntwo\n"

    log.write_text("new\n")

    assert tail.read() == "new\n", "the held-back 'thr' belonged to the old file"


def test_a_log_that_cannot_be_read_yet_loses_nothing(tmp_path):
    log = tmp_path / "training_log.txt"
    tail = LogTailer(log)
    with pytest.raises(OSError):
        tail.read()

    log.write_text("first\n")
    assert tail.read() == "first\n"


# --- markers --------------------------------------------------------------------


def test_the_patterns_are_the_ones_the_job_manager_always_used():
    expected = {
        "EPOCH_START_RE": (r"Starting\s+epoch\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE),
        "EPOCH_SUMMARY_RE": (r"Epoch\s+(\d+)/(\d+)\s*-\s*Loss:\s*([\d.]+)", re.IGNORECASE),
        "ITERATION_COMPLETE_RE": (r"TRAINING_ITERATION_COMPLETE:\s+(\S+)", 0),
        "MODEL_YAML_RE": (r"^.*?FINETUNED_MODEL_YAML:\s*(.+?)\s*$", re.MULTILINE),
        "STATUS_MARKER_RE": (r"TRAINING_DIVERGED|RESTARTING_TRAINING|WAITING_FOR_RESTART", 0),
        "SERVER_URL_RE": (re.escape(IP_PATTERN[0]) + r"(.+?)" + re.escape(IP_PATTERN[1]), 0),
    }
    for name, (pattern, flags) in expected.items():
        compiled = getattr(markers, name)
        assert compiled.pattern == pattern, name
        assert compiled.flags & (re.IGNORECASE | re.MULTILINE) == flags, name


def test_the_markers_are_the_ones_the_trainer_prints():
    here = Path(markers.__file__).parent
    trainer = (here / "finetune_cli.py").read_text() + (here / "lora_trainer.py").read_text()
    for marker in (
        markers.TRAINING_ITERATION_COMPLETE,
        markers.RESTART_FAILED,
        markers.INFERENCE_SERVER_FAILED,
        markers.TRAINING_DIVERGED,
        markers.RESTARTING_TRAINING,
        markers.WAITING_FOR_RESTART,
    ):
        assert marker in trainer, marker
    assert markers.FINETUNED_MODEL_YAML == finetuned_model_templates.FINETUNED_MODEL_YAML_MARKER


def test_what_emit_prints_the_job_manager_reads(capsys):
    markers.emit(markers.FINETUNED_MODEL_YAML, "/s/models/m_1.yaml")
    markers.emit(markers.TRAINING_ITERATION_COMPLETE, "m_1")
    markers.emit(markers.WAITING_FOR_RESTART)

    out = capsys.readouterr().out
    assert out == (
        "FINETUNED_MODEL_YAML: /s/models/m_1.yaml\n"
        "TRAINING_ITERATION_COMPLETE: m_1\n"
        "WAITING_FOR_RESTART\n"
    )
    assert trainer_outputs_from_log(out) == ("m_1", "/s/models/m_1.yaml")
    assert markers.STATUS_MARKER_RE.findall(out) == ["WAITING_FOR_RESTART"]


# --- listeners ------------------------------------------------------------------


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


def test_a_listener_hears_of_the_server_and_of_each_iteration(tmp_path):
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


def test_the_default_listener_still_adds_the_layer_and_registers_the_model(tmp_path, monkeypatch):
    manager = FinetuneJobManager()
    calls = []

    def add_layer(job, model_name):
        calls.append(("layer", model_name))
        job.finetuned_model_name = model_name

    monkeypatch.setattr(manager, "_add_finetuned_neuroglancer_layer", add_layer)
    monkeypatch.setattr(
        manager, "_register_finetune_model_config", lambda job, name: calls.append(("config", name))
    )
    job = _job(tmp_path)
    job.log_file.write_text("TRAINING_ITERATION_COMPLETE: m_finetuned_1\n" + SERVER_LINE)

    manager._parse_inference_server_ready(job, job.log_file.read_text())
    _append(job.log_file, "TRAINING_ITERATION_COMPLETE: m_finetuned_2\n")
    manager._parse_training_restart(job, "")

    assert calls == [
        ("layer", "m_finetuned_1"), ("config", "m_finetuned_1"),
        ("layer", "m_finetuned_2"), ("config", "m_finetuned_2"),
    ]
    assert job.inference_server_url == URL


def test_one_failure_does_not_stop_the_rest(tmp_path, monkeypatch, caplog):
    manager = FinetuneJobManager()
    registered = []

    def broken_layer(job, model_name):
        raise RuntimeError("no viewer")

    class Broken:
        def on_iteration_complete(self, job, model_name):
            raise RuntimeError("listener bug")

    monkeypatch.setattr(manager, "_add_finetuned_neuroglancer_layer", broken_layer)
    monkeypatch.setattr(
        manager, "_register_finetune_model_config", lambda job, name: registered.append(name)
    )
    heard = Recording()
    manager.add_listener(Broken())
    manager.add_listener(heard)
    job = _job(tmp_path)
    job.log_file.write_text("TRAINING_ITERATION_COMPLETE: m_finetuned_1\n")

    manager._parse_training_restart(job, "")

    assert registered == ["m_finetuned_1"], "the layer failing does not skip the config"
    assert heard.events == [("iteration complete", "m_finetuned_1", None)]
    assert job.finetuned_model_name == "m_finetuned_1", "and the name moves on regardless"
    assert "no viewer" in caplog.text and "listener bug" in caplog.text
