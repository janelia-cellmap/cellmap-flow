"""FinetuneJobManager: the command a job runs, what the monitor reads from its
log, what it tells listeners and the viewer, and finding jobs again after a
dashboard restart. What the trainer prints is test_finetune_cli's."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import neuroglancer
import pytest

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager, JobStatus, finetune_export_kwargs
from cellmap_flow.finetune.model_loading import decode_model_entry
from cellmap_flow.globals import g
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.spec import JobStatus as LSF
from cellmap_flow.utils.web_utils import IP_PATTERN

URL = "http://node7:8123"


class _Script:
    cli_name, name, script_path, channels = "script", "m", "/s.py", ["mito"]


class _Exported:
    """What export_merged produces: no flags of its own, so it goes as its entry."""

    cli_name, name = "cellmap", "exported"

    def to_dict(self):
        return {"type": "cellmap", "folder_path": "/models/exported", "name": "exported"}


class _Hub:
    """No geometry of its own; building it (weights, torch.export, CUDA) in the dashboard is not allowed."""

    cli_name, name, repo, revision = "huggingface", "hub", "org/hub", None

    @property
    def config(self):
        raise AssertionError("submit built the model")


class _Bio:
    cli_name, name = "bioimage", "bio"


GEOMETRY = SimpleNamespace(input_voxel_size=[8] * 3, output_voxel_size=[4] * 3, channels=["nuc"])


@pytest.mark.parametrize("config, geometry, manifest, settings, present, absent", [
    (_Script(), None, None, dict(queue="gpu_a100", charge_group="my_lab", distillation_lambda=0.0,
                           distillation_scope="all"),
     # 0 is passed: left out, the trainer makes it 1.0 when good regions exist.
     ["--model-type script --model-script /s.py", "--queue gpu_a100", "--charge-group my_lab",
      "--distillation-lambda 0.0", "--channels mito --input-voxel-size 16 16 16 --output-voxel-size 16 16 16"],
     ["--distillation-all-voxels"]),
    # A manifest without the raw path: served on what the volume's own attrs name.
    (_Exported(), None, {"kind": "volume_zarr_v1"}, dict(distillation_scope="all"),
     ["--model-type cellmap", "--distillation-all-voxels"], ["--distillation-lambda"]),
    (_Hub(), GEOMETRY, None, {}, ["--repo org/hub", "--channels nuc --input-voxel-size 8 8 8 --output-voxel-size 4 4 4"], []),
], ids=["script", "exported", "geometry from its server"])
def test_submit_writes_the_command_the_trainer_runs(local_jobs, session, monkeypatch, config, geometry,
                                                    manifest, settings, present, absent):
    from cellmap_flow.utils import model_geometry

    asked, resolve = [], model_geometry.resolve_model_geometry
    monkeypatch.setattr(model_geometry, "resolve_model_geometry",
                        lambda name, c: asked.append(name) or geometry or resolve(name, c))
    base = session(manifest=manifest)
    job = FinetuneJobManager().submit_finetuning_job(
        model_config=config, corrections_path=base / "corrections", output_base=base, **settings
    )

    metadata = json.loads((job.output_dir / "metadata.json").read_text())
    command = metadata["command"]
    # The console script of this interpreter, and tee line-buffered too: stdio
    # block-buffers a file, so the log (and the dashboard) got 5-10 epochs at once.
    assert f"{sys.executable} -m cellmap_flow.finetune.finetune_cli" in command
    assert f"| stdbuf -oL tee {job.log_file}" in command and "stdbuf -oL python -m" not in command
    assert f"--models-dir {base / 'models'}" in command  # the session's, next to runs/
    assert "--auto-serve --serve-data-path /data/raw.zarr" in command
    assert all(part in command for part in present) and not any(part in command for part in absent)
    if hasattr(config, "to_dict"):
        tokens = command.split()
        assert decode_model_entry(tokens[tokens.index("--model-entry") + 1]) == config.to_dict()
    assert asked == [config.name], "the geometry is looked up once, not per field"
    # What a dashboard started later finds the job by; its own log is the tee'd one.
    assert (metadata["lsf_job_id"], metadata["status"]) == ("PID:77", "PENDING")
    assert job.corrections_path == base / "corrections"
    assert local_jobs.runs[0]["log_file"] == os.devnull


@pytest.mark.parametrize("config, manifest, error", [
    (_Bio(), True, "cannot be finetuned"),  # argparse would exit 2 on the GPU node
    (_Script(), False, "_virtual_sources.json"),  # a crop zarr alone: nothing the trainer reads
])
def test_submit_refuses_what_the_trainer_cannot_train(local_jobs, session, config, manifest, error):
    base = session()
    if not manifest:
        (base / "corrections" / "_virtual_sources.json").unlink()
    with pytest.raises(ValueError, match=error):
        FinetuneJobManager().submit_finetuning_job(
            model_config=config, corrections_path=base / "corrections", output_base=base
        )
    assert local_jobs.runs == [] and not (base / "runs").exists()


def _lsf(*replies, **fields):
    """A scheduler job that reports ``replies`` in turn."""
    replies = iter(replies)
    return SimpleNamespace(get_status=lambda: next(replies), **fields)


def _monitor(manager, job, chunks, monkeypatch):
    """monitor_job over a log that grows by one chunk per poll; the job's
    (status, epoch, loss) at each poll."""
    job.log_file.write_text(chunks[0])
    rest, seen = iter(chunks[1:]), []

    def sleep(seconds):
        seen.append((job.status.value, job.current_epoch, job.latest_loss))
        with open(job.log_file, "a") as f:
            f.write(next(rest, ""))

    monkeypatch.setattr(fjm.time, "sleep", sleep)
    manager.monitor_job(job)
    return seen


FULL, LORA = ["full_finetune/model_state_dict.pt"], ["lora_adapter/adapter_model.safetensors", "lora_adapter/adapter_config.json"]
COMPLETED = ["FINETUNED_MODEL_YAML: /s/models/m_1.yaml\nTRAINING_ITERATION_COMPLETE: m_1\n"
             "RESTARTING_TRAINING\nFINETUNED_MODEL_YAML: /s/models/m_2.yaml\nTRAINING_ITERATION_COMPLETE: m_2\n"]


@pytest.mark.parametrize("chunks, seen, final, name, exported", [
    # One loss per epoch, from its summary line, even split across two reads.
    pytest.param(["Starting epoch 1 of 10...\n  Batch 1/3 - Loss: 0.9\nEpoch 1/10 - Lo",
                  "ss: 0.5 - Supervised: 0.5\nStarting epoch 2 of 10...\nEpoch 2/10 - Loss: 0.25 - Sup\n",
                  "  Batch 1/3 - Loss: 0.123\n"],
                 [("RUNNING", 1, None), ("RUNNING", 2, 0.25), ("RUNNING", 2, 0.25)], "FAILED", None, FULL,
                 id="progress"),
    # The last status marker decides, and LSF saying RUNNING does not undo waiting.
    pytest.param(["Epoch 3/10 - Loss: nan\nTRAINING_DIVERGED\nWAITING_FOR_RESTART\n",
                  "RESTARTING_TRAINING\nTRAINING_DIVERGED\n", "WAITING_FOR_RESTART\nRESTARTING_TRAINING\n",
                  "TRAINING_ITERATION_COMPLETE: m_1\nWAITING_FOR_RESTART\n"],
                 [("WAITING_FOR_RESTART", 0, None), ("WAITING_FOR_RESTART", 0, None), ("RUNNING", 0, None),
                  ("WAITING_FOR_RESTART", 0, None)], "FAILED", "m_1", FULL, id="waiting for a restart"),
    # The name and YAML are the trainer's: the manager made up its own, never
    # found that YAML, and wrote a second one.
    pytest.param(COMPLETED, [("RUNNING", 0, None)], "COMPLETED", "m_2", FULL, id="completed, full"),
    pytest.param(COMPLETED, [("RUNNING", 0, None)], "COMPLETED", "m_2", LORA, id="completed, LoRA"),
])
def test_the_monitor_follows_the_log(make_job, monkeypatch, tmp_path, chunks, seen, final, name, exported):
    job = make_job("PENDING", lsf_job=_lsf(*[LSF.RUNNING] * len(chunks), LSF[final]))
    (job.output_dir / "metadata.json").write_text(json.dumps({"status": "PENDING"}))
    for export in exported:  # complete_job checks that the export is there
        (job.output_dir / export).parent.mkdir(exist_ok=True)
        (job.output_dir / export).write_bytes(b"")
    manager = FinetuneJobManager()
    manager.remove_listener(manager.viewer_listener)

    assert _monitor(manager, job, chunks, monkeypatch) == seen
    metadata = json.loads((job.output_dir / "metadata.json").read_text())
    assert job.status.value == metadata["status"] == final
    assert job.finetuned_model_name == metadata["finetuned_model_name"] == name
    if final == "COMPLETED":
        assert job.model_yaml_path == Path(metadata["model_yaml_path"]) == Path("/s/models/m_2.yaml")
        assert not (tmp_path / "models").exists()


class _Killable:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True

    def get_status(self):
        return LSF.FAILED if self.killed else LSF.RUNNING


@pytest.mark.parametrize("cancel", ["through the manager", "racing the poll"])
def test_a_cancelled_job_stays_cancelled(make_job, monkeypatch, cancel):
    """LSF reports the kill as EXIT, and the next poll overwrote CANCELLED with FAILED."""
    monkeypatch.setattr(fjm.time, "sleep", lambda s: None)
    manager, job = FinetuneJobManager(), make_job(lsf_job=_Killable())
    manager.jobs[job.job_id] = job
    if cancel == "through the manager":
        assert manager.cancel_job(job.job_id)
    else:  # killed, and not yet marked cancelled
        job.cancel_requested = job.lsf_job.killed = True
    manager.monitor_job(job)
    assert job.status == JobStatus.CANCELLED


def test_each_iteration_reaches_the_listeners_and_the_viewer(make_job, monkeypatch):
    """The layer comes once the server is up (it had the source zarr://None/...),
    for a local run too (a LocalJob has no job_id), shows the output's own range
    ([0, 255] made a sigmoid's look black), and each iteration replaces it. A
    listener that raises stops neither the others nor the monitor."""
    from cellmap_flow.post.postprocessors import SigmoidPostprocessor
    from cellmap_flow.utils import server_info

    monkeypatch.setattr(server_info, "fetch_model_info", lambda *a, **k: {"output_class": None})
    for key, value in dict(viewer=neuroglancer.Viewer(), jobs=[], models_config=[], input_norms=[],
                           postprocess=[SigmoidPostprocessor()]).items():
        monkeypatch.setattr(g, key, value, raising=False)
    heard = []

    class Recording:
        def on_server_ready(self, job, url, model_name):
            heard.append(("server", url, model_name))

        def on_iteration_complete(self, job, model_name):
            heard.append(("iteration", model_name, job.finetuned_model_name))  # the previous name, still

    class Broken:
        def on_iteration_complete(self, job, model_name):
            raise RuntimeError("listener bug")

    manager = FinetuneJobManager()
    for listener in (Recording(), Broken(), manager.viewer_listener):  # the viewer's after a broken one
        manager.remove_listener(listener)
        manager.add_listener(listener)
    job = make_job(lsf_job=_lsf(*[LSF.RUNNING] * 3, LSF.FAILED, process=SimpleNamespace(pid=99)))
    _monitor(manager, job, ["TRAINING_ITERATION_COMPLETE: m_finetuned_1\n",
                            f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}\n",
                            "RESTARTING_TRAINING\nTRAINING_ITERATION_COMPLETE: m_finetuned_2\n"], monkeypatch)

    assert heard == [("iteration", "m_finetuned_1", None), ("server", URL, "m_finetuned_1"),
                     ("iteration", "m_finetuned_2", "m_finetuned_1")]
    assert [layer.name for layer in g.viewer.state.layers] == ["m_finetuned_2"]
    assert "range=[0, 1]" in g.viewer.state.layers["m_finetuned_2"].shader
    assert [j.job_id for j in g.jobs] == ["local"]
    assert [c.name for c in g.models_config] == ["m_finetuned_2"]


def _run(session, name, **metadata):
    run = session / "runs" / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "job_id": name, "model_name": "m", "created_at": datetime.now().isoformat(),
        "corrections_path": str(session / "corrections"), "params": {"num_epochs": 5}, **metadata,
    }))
    return run


def test_jobs_still_on_the_cluster_are_picked_up_again(local_jobs, session, monkeypatch):
    """Jobs lived only in the dashboard's memory: after a restart a running job
    could not be seen or cancelled, and waited for a restart until walltime."""
    base = session()
    _run(base, "alive", lsf_job_id="101", status="WAITING_FOR_RESTART")
    done = _run(base, "done_meanwhile", lsf_job_id="102", status="RUNNING")
    _run(base, "finished", lsf_job_id="103", status="COMPLETED")
    _run(base, "local", lsf_job_id="PID:4", status="RUNNING")
    _run(base, "from_before_this_was_recorded", status="RUNNING")
    asked = []
    monkeypatch.setattr(jobs_lsf, "statuses", lambda ids: asked.append(sorted(ids)) or {
        "101": LSF.RUNNING, "102": LSF.FAILED})
    manager = FinetuneJobManager()

    assert manager.rehydrate_session(base) == 1
    job = manager.jobs["alive"]
    assert isinstance(job.lsf_job, LSFJob) and job.lsf_job.job_id == "101"
    assert (job.status, job.corrections_path, job.total_epochs) == (JobStatus.RUNNING, base / "corrections", 5)
    assert local_jobs.monitors == [job]
    assert asked == [["101", "102"]], "one bjobs call; finished and local runs are not asked about"
    assert json.loads((done / "metadata.json").read_text())["status"] == "FAILED"
    # Idempotent, and with the other one recorded as final there is nothing to ask about.
    assert manager.rehydrate_session(base) == 0 and len(asked) == 1


@pytest.mark.parametrize("log, status, detail", [
    ("TRAINING_ITERATION_COMPLETE: m_1\nWAITING_FOR_RESTART\nTRAINING_ITERATION_COMPLETE: m_2\n",
     "COMPLETED", "finished 2 iteration(s), the last m_2"),
    ("Starting epoch 1 of 5...\nTraceback (most recent call last):\n", "FAILED", "how is not known"),
    (None, "FAILED", "how is not known"),
])
def test_a_job_lsf_has_forgotten_is_judged_by_its_log(session, monkeypatch, log, status, detail):
    """The trainer never says "done" (it waits for restarts until stopped), so a
    finished iteration is the evidence that a model was delivered."""
    run = _run(session(), "purged", lsf_job_id="501", status="WAITING_FOR_RESTART")
    if log is not None:
        (run / "training_log.txt").write_text(log)
    monkeypatch.setattr(jobs_lsf, "statuses", lambda ids: {"501": None})

    assert FinetuneJobManager().rehydrate_session(run.parent.parent) == 0
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == status and detail in metadata["status_detail"] and "501" in metadata["status_detail"]


@pytest.mark.parametrize("status, restarted", [("WAITING_FOR_RESTART", True), ("COMPLETED", False)])
def test_only_a_job_waiting_for_a_restart_is_restarted(make_job, status, restarted):
    """Only a job whose server was marked ready could be restarted, which a
    restart resets and a diverged iteration never sets. A COMPLETED trainer has
    exited: the request went nowhere, and the job showed RUNNING for ever."""
    manager = FinetuneJobManager()
    job = make_job(status, inference_server_ready=True)
    manager.jobs[job.job_id] = job
    signal = job.output_dir / "restart_signal.json"  # no server to send it to
    if restarted:
        manager.restart_finetuning_job(job.job_id, {"learning_rate": 5e-5})
        assert json.loads(signal.read_text())["params"] == {"learning_rate": 5e-5}
        assert job.status == JobStatus.RUNNING
    else:
        with pytest.raises(ValueError):
            manager.restart_finetuning_job(job.job_id, {})
        assert not signal.exists()


@pytest.mark.parametrize("on_disk, params, served_from", [
    (None, {"lora_r": 64}, "lora_adapter"),
    (None, {"lora_r": 0}, "full_finetune/model_state_dict.pt"),
    ("full_finetune/model_state_dict.pt", {"lora_r": 64}, "full_finetune/model_state_dict.pt"),
    ("lora_adapter/adapter_config.json", {"lora_r": 0}, "lora_adapter"),
])
def test_what_a_run_is_served_from_is_what_it_exported(tmp_path, on_disk, params, served_from):
    """Decided by what is on disk, then the job's rank: a viewer pointed at an adapter a rank-0 run never made."""
    if on_disk:
        (tmp_path / on_disk).parent.mkdir(parents=True)
        (tmp_path / on_disk).write_bytes(b"")
    (path,) = finetune_export_kwargs(tmp_path, params).values()
    assert path == str(tmp_path / served_from)
