"""FinetuneJobManager: the command a job runs, what the monitor reads from its
log, what it tells listeners, and finding jobs again after a dashboard
restart. What the trainer prints is test_finetune_cli's, and what the
dashboard's listener does test_finetune_layers'."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
import requests

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager
from cellmap_flow.finetune.job_manager.persistence import finetune_export_kwargs
from cellmap_flow.finetune.job_manager.state import JobStatus
from cellmap_flow.finetune.model_loading import decode_model_entry
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.spec import JobStatus as LSF
from cellmap_flow.serving.protocol import IP_PATTERN

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


@pytest.fixture
def submit(local_jobs, session, monkeypatch):
    """``submit(config, geometry=None, manifest=None, **settings)``: a job submitted
    from a session, locally. ``geometry`` is what the model's server reports
    (None: the real lookup, which finds nothing for these configs). Returns the
    job, its metadata.json and command, and the models the geometry was asked for."""
    from cellmap_flow.models import geometry_cache

    record = SimpleNamespace(asked=[], runs=local_jobs.runs)
    resolve = geometry_cache.resolve_model_geometry

    def run(config, geometry=None, manifest=None, **settings):
        monkeypatch.setattr(geometry_cache, "resolve_model_geometry",
                            lambda name, c: record.asked.append(name) or geometry or resolve(name, c))
        record.base = session(manifest=manifest)
        record.job = FinetuneJobManager().submit_finetuning_job(
            model_config=config, corrections_path=record.base / "corrections", output_base=record.base, **settings
        )
        record.metadata = json.loads((record.job.output_dir / "metadata.json").read_text())
        record.command = record.metadata["command"]
        return record

    return run


def test_a_job_runs_this_interpreters_trainer_and_logs_as_it_goes(submit):
    """tee writes through stdio, which block-buffers a file: the log, and the
    dashboard, got 5-10 epochs at once. And the command tees its own log, so a
    local run keeps no second copy."""
    job = submit(_Script())
    assert f"{sys.executable} -m cellmap_flow.finetune.finetune_cli" in job.command
    assert f"| stdbuf -oL tee {job.job.log_file}" in job.command and "stdbuf -oL python -m" not in job.command
    assert job.runs[0]["log_file"] == os.devnull


def test_the_job_writes_its_yamls_into_the_session_with_its_queue_and_charge_group(submit):
    """The trainer writes each iteration's serving YAML, into the session's
    models/ next to runs/, and a model served from one runs where this job did."""
    job = submit(_Script(), queue="gpu_a100", charge_group="my_lab")
    assert f"--models-dir {job.base / 'models'}" in job.command
    assert "--queue gpu_a100" in job.command and "--charge-group my_lab" in job.command
    assert (job.metadata["queue"], job.metadata["charge_group"]) == ("gpu_a100", "my_lab")


@pytest.mark.parametrize("manifest", [
    pytest.param(None, id="the manifest's raw data"),
    pytest.param({"kind": "volume_zarr_v1"}, id="else what the volume's attrs name"),
])
def test_the_job_serves_the_data_it_trains_on(submit, manifest):
    assert "--auto-serve --serve-data-path /data/raw.zarr" in submit(_Script(), manifest=manifest).command


def test_a_dashboard_started_later_can_find_the_job(submit):
    """Jobs lived only in the dashboard's memory, and metadata.json did not
    record the scheduler's id: after a restart a running job was lost."""
    job = submit(_Script())
    assert (job.metadata["lsf_job_id"], job.metadata["status"]) == ("PID:77", "PENDING")
    assert job.job.corrections_path == job.base / "corrections"


@pytest.mark.parametrize("config, flags", [
    pytest.param(_Script(), "--model-type script --model-script /s.py", id="a script"),
    pytest.param(_Hub(), "--model-type huggingface --repo org/hub", id="a Hugging Face repo"),
])
def test_the_command_names_the_model_the_way_its_type_takes(submit, config, flags):
    assert flags in submit(config, geometry=GEOMETRY).command


def test_a_model_without_flags_of_its_own_goes_as_its_entry(submit):
    """What export_merged produces (cellmap), and finetuned models, reach the trainer as their to_dict()."""
    tokens = submit(_Exported()).command.split()
    assert tokens[tokens.index("--model-type") + 1] == "cellmap"
    assert decode_model_entry(tokens[tokens.index("--model-entry") + 1]) == _Exported().to_dict()


def test_the_geometry_comes_from_the_models_server_without_building_it(submit):
    """For a script, Hugging Face or DaCapo model, reading model_config.config
    built the model in the dashboard: weights download, torch.export, CUDA."""
    job = submit(_Hub(), geometry=GEOMETRY)
    assert "--channels nuc --input-voxel-size 8 8 8 --output-voxel-size 4 4 4" in job.command
    assert job.asked == ["hub"], "looked up once, not once per field"


def test_what_a_model_does_not_say_of_its_geometry_is_the_named_default(submit):
    """The script says its channels and nothing else: 16 nm is a guess, and the log names it."""
    command = submit(_Script()).command
    assert "--channels mito --input-voxel-size 16 16 16 --output-voxel-size 16 16 16" in command


@pytest.mark.parametrize("settings, present, absent", [
    # Left out, the weight is "unset", which the trainer makes 1.0 when good
    # regions exist: "0 (Disabled)" could not be expressed.
    pytest.param(dict(distillation_lambda=0.0, distillation_scope="all"), ["--distillation-lambda 0.0"],
                 ["--distillation-all-voxels"], id="an explicit 0 is passed, and no scope with it"),
    pytest.param(dict(distillation_scope="all"), ["--distillation-all-voxels"], ["--distillation-lambda"],
                 id="unset is left to the trainer"),
    pytest.param(dict(distillation_lambda=0.5, distillation_scope="all"),
                 ["--distillation-lambda 0.5", "--distillation-all-voxels"], [], id="a weight and its scope"),
])
def test_the_distillation_flags(submit, settings, present, absent):
    command = submit(_Script(), **settings).command
    assert all(flag in command for flag in present) and not any(flag in command for flag in absent)


@pytest.mark.parametrize("config, manifest, error", [
    pytest.param(_Bio(), True, "cannot be finetuned", id="a type the trainer cannot load"),  # argparse exit 2
    pytest.param(_Script(), False, "_virtual_sources.json", id="a session without a manifest"),
])
def test_submit_refuses_what_the_trainer_cannot_train(local_jobs, session, config, manifest, error):
    """Refused before anything is submitted, not on the GPU node after queueing."""
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


def _monitor(manager, job, chunks, monkeypatch, observe=None):
    """monitor_job over a log that grows by one chunk per poll; what ``observe()``
    returns at each poll (default: the job's status, epoch and loss)."""
    job.log_file.write_text(chunks[0])
    rest, seen = iter(chunks[1:]), []
    observe = observe or (lambda: (job.status.value, job.current_epoch, job.latest_loss))

    def sleep(seconds):
        seen.append(observe())
        with open(job.log_file, "a") as f:
            f.write(next(rest, ""))

    monkeypatch.setattr(fjm.time, "sleep", sleep)
    manager.monitor_job(job)
    return seen


FULL = ["full_finetune/model_state_dict.pt"]
LORA = ["lora_adapter/adapter_model.safetensors", "lora_adapter/adapter_config.json"]


@pytest.fixture
def monitored(make_job, monkeypatch):
    """``monitored(chunks, final="FAILED", exported=FULL)``: a pending job monitored
    while LSF says RUNNING for each chunk, then ``final``. Returns the job, its
    metadata.json and its (status, epoch, loss) at each poll."""

    def run(chunks, final="FAILED", exported=FULL):
        job = make_job("PENDING", lsf_job=_lsf(*[LSF.RUNNING] * len(chunks), LSF[final]))
        (job.output_dir / "metadata.json").write_text(json.dumps({"status": "PENDING"}))
        for export in exported:  # complete_job checks that the export is there
            (job.output_dir / export).parent.mkdir(exist_ok=True)
            (job.output_dir / export).write_bytes(b"")
        seen = _monitor(FinetuneJobManager(), job, chunks, monkeypatch)
        return SimpleNamespace(job=job, seen=seen, metadata=json.loads((job.output_dir / "metadata.json").read_text()))

    return run


def test_the_monitor_reads_one_loss_per_epoch(monitored):
    """From the epoch's summary line, even split across two reads; per-batch
    losses are running means mid-epoch, and are not read."""
    run = monitored(["Starting epoch 1 of 10...\n  Batch 1/3 - Loss: 0.9\nEpoch 1/10 - Lo",
                     "ss: 0.5 - Supervised: 0.5\nStarting epoch 2 of 10...\nEpoch 2/10 - Loss: 0.25 - Sup\n",
                     "  Batch 1/3 - Loss: 0.123\n"])
    assert [(epoch, loss) for _, epoch, loss in run.seen] == [(1, None), (2, 0.25), (2, 0.25)]
    assert run.job.total_epochs == 10


def test_the_last_status_marker_in_the_log_decides(monitored):
    """A restart after a divergence in the same chunk leaves the job running,
    and the reverse leaves it waiting; LSF saying RUNNING does not undo waiting.
    A job diverged in a later iteration could never be restarted."""
    run = monitored(["Epoch 3/10 - Loss: nan\nTRAINING_DIVERGED\nWAITING_FOR_RESTART\n",
                     "RESTARTING_TRAINING\nTRAINING_DIVERGED\n", "WAITING_FOR_RESTART\nRESTARTING_TRAINING\n",
                     "TRAINING_ITERATION_COMPLETE: m_1\nWAITING_FOR_RESTART\n"])
    assert [status for status, _, _ in run.seen] == [
        "WAITING_FOR_RESTART", "WAITING_FOR_RESTART", "RUNNING", "WAITING_FOR_RESTART"]


def test_the_final_status_is_recorded_for_a_dashboard_started_later(monitored):
    run = monitored(["Starting epoch 1 of 5...\n"])
    assert run.job.status.value == run.metadata["status"] == "FAILED"


@pytest.mark.parametrize("exported", [pytest.param(FULL, id="a full finetune"),
                                      pytest.param(LORA, id="a LoRA adapter")])
def test_a_completed_job_takes_the_trainers_name_and_yaml(monitored, tmp_path, exported):
    """The manager made up its own name (the job's creation time, not the
    iteration's), never found that YAML, and wrote a second one."""
    run = monitored(["FINETUNED_MODEL_YAML: /s/models/m_1.yaml\nTRAINING_ITERATION_COMPLETE: m_1\n"
                     "RESTARTING_TRAINING\nFINETUNED_MODEL_YAML: /s/models/m_2.yaml\n"
                     "TRAINING_ITERATION_COMPLETE: m_2\n"], final="COMPLETED", exported=exported)
    assert run.job.status.value == run.metadata["status"] == "COMPLETED"
    assert run.job.finetuned_model_name == run.metadata["finetuned_model_name"] == "m_2"
    assert run.job.model_yaml_path == Path(run.metadata["model_yaml_path"]) == Path("/s/models/m_2.yaml")
    assert not (tmp_path / "models").exists(), "no YAML of its own"


class _Killable:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True

    def get_status(self):
        return LSF.FAILED if self.killed else LSF.RUNNING


@pytest.mark.parametrize("cancel", [pytest.param("through the manager", id="cancelled through the manager"),
                                    pytest.param("racing the poll", id="killed before it was marked cancelled")])
def test_a_cancelled_job_stays_cancelled(make_job, monkeypatch, cancel):
    """LSF reports the kill as EXIT, and the next poll overwrote CANCELLED with FAILED."""
    monkeypatch.setattr(fjm.time, "sleep", lambda s: None)
    manager, job = FinetuneJobManager(), make_job(lsf_job=_Killable())
    manager.jobs[job.job_id] = job
    if cancel == "through the manager":
        assert manager.cancel_job(job.job_id)
    else:
        job.cancel_requested = job.lsf_job.killed = True
    manager.monitor_job(job)
    assert job.status == JobStatus.CANCELLED


SERVER = f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}\n"
# Iteration 1 completes before its server is up (the trainer announces the
# model first), then the server comes up, then a restart completes iteration 2.
ITERATIONS = ["TRAINING_ITERATION_COMPLETE: m_finetuned_1\n", SERVER,
              "RESTARTING_TRAINING\nTRAINING_ITERATION_COMPLETE: m_finetuned_2\n"]


@pytest.mark.parametrize("chunks, events", [
    pytest.param(ITERATIONS, [("iteration", "m_finetuned_1", None), ("server", URL, "m_finetuned_1", "m_finetuned_1"),
                              ("iteration", "m_finetuned_2", "m_finetuned_1")], id="the server after its model"),
    # Iteration 1 was announced again as it came up: only the dashboard's own
    # listener used to set the job's name.
    pytest.param(["TRAINING_ITERATION_COMPLETE: m_finetuned_1\n" + SERVER, ITERATIONS[2]],
                 [("server", URL, "m_finetuned_1", None), ("iteration", "m_finetuned_2", "m_finetuned_1")],
                 id="the server and its model in one read"),
])
def test_listeners_hear_of_the_server_and_of_each_iteration(make_job, monkeypatch, chunks, events):
    """What the dashboard does about a job is a listener. While listeners run
    the job still has its model's previous name, so one can replace that
    model's layer; a listener that raises stops neither the others nor the
    monitor; and one added twice is told once."""
    heard = []

    class Recording:
        def on_server_ready(self, job, url, model_name):
            heard.append(("server", url, model_name, job.finetuned_model_name))

        def on_iteration_complete(self, job, model_name):
            heard.append(("iteration", model_name, job.finetuned_model_name))

    class Broken:
        def on_iteration_complete(self, job, model_name):
            raise RuntimeError("listener bug")

    manager, recording = FinetuneJobManager(), Recording()
    manager.add_listener(Broken())
    manager.add_listener(recording)
    manager.add_listener(recording)
    job = make_job(lsf_job=_lsf(*[LSF.RUNNING] * len(chunks), LSF.FAILED))
    _monitor(manager, job, chunks, monkeypatch)
    assert heard == events
    assert job.finetuned_model_name == "m_finetuned_2"


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
    pytest.param("TRAINING_ITERATION_COMPLETE: m_1\nWAITING_FOR_RESTART\nTRAINING_ITERATION_COMPLETE: m_2\n",
                 "COMPLETED", "finished 2 iteration(s), the last m_2", id="it finished iterations"),
    pytest.param("Starting epoch 1 of 5...\nTraceback (most recent call last):\n", "FAILED", "how is not known",
                 id="it finished none"),
    pytest.param(None, "FAILED", "how is not known", id="it left no log"),
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


CREATED = "2026-01-01T12:00:00"
PARAMS = {"lora_r": 8, "lora_alpha": 16, "num_epochs": 5, "learning_rate": 0.0001}
TWO_ITERATIONS = ("FINETUNED_MODEL_YAML: /s/models/m_finetuned_1.yaml\nTRAINING_ITERATION_COMPLETE: m_finetuned_1\n"
                  f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}\nWAITING_FOR_RESTART\nRESTARTING_TRAINING\n"
                  "TRAINING_ITERATION_COMPLETE: m_finetuned_2\nWAITING_FOR_RESTART\n")

# One session's runs: the status and LSF job id their metadata.json records,
# what bjobs says of them ("not found": LSF has purged the job; None: bjobs
# says nothing of it), and their training log.
RUNS = {
    "cancelled": ("CANCELLED", "113", None, None),
    "completed": ("COMPLETED", "111", None, None),
    "done_meanwhile": ("RUNNING", "105", "DONE", None),
    "exited_meanwhile": ("RUNNING", "106", "EXIT", None),
    "failed": ("FAILED", "112", None, None),
    "known": ("RUNNING", "115", None, None),  # this dashboard already follows it
    "local": ("RUNNING", "PID:4", None, None),
    "pending": ("PENDING", "102", "PEND", None),
    "purged_after_two": ("WAITING_FOR_RESTART", "107", "not found", TWO_ITERATIONS),
    "purged_mid_epoch": ("RUNNING", "108", "not found", "Starting epoch 1 of 5...\n"),
    "purged_without_a_log": ("PENDING", "109", "not found", None),
    "running": ("RUNNING", "101", "RUN", "Starting epoch 2 of 5...\n"),
    "suspended": ("RUNNING", "104", "USUSP", None),
    "unanswered": ("RUNNING", "110", None, None),
    "unrecorded": ("RUNNING", None, None, None),  # from before metadata.json kept the LSF id
    "waiting": ("WAITING_FOR_RESTART", "103", "RUN", TWO_ITERATIONS),
}
# What metadata.json records of each run, as submit and the monitor leave it.
# "from_an_old_dashboard" has only what the oldest dashboards wrote.
RECORD = {
    "model_name": "m", "model_type": "script", "model_script": "/s.py", "model_entry": None,
    "params": PARAMS, "created_at": CREATED, "queue": "gpu_h100", "charge_group": "cellmap",
    "command": "python -m cellmap_flow.finetune.finetune_cli ...",
}

# What the manager makes of them. The jobs it follows again, as it rebuilds
# them: their status from LSF's answer, the rest from metadata.json, and what
# only the log says (the server, the model, the epoch) left for the monitor.
REATTACHED = {
    "from_an_old_dashboard": dict(lsf_job_id="114", model_name="", params={}, status="RUNNING", created_at="<now>",
                                  total_epochs=10, corrections_path=None),
    "pending": dict(lsf_job_id="102", status="PENDING"),
    "running": dict(lsf_job_id="101", status="RUNNING"),
    "suspended": dict(lsf_job_id="104", status="RUNNING"),
    "waiting": dict(lsf_job_id="103", status="RUNNING"),
}
# And what it writes into the others' metadata.json, merged into the rest.
RECORDED = {
    "done_meanwhile": {"status": "COMPLETED"},
    "exited_meanwhile": {"status": "FAILED"},
    "purged_after_two": {"status": "COMPLETED", "status_detail":
                         "LSF no longer knows job 107; it had finished 2 iteration(s), the last m_finetuned_2"},
    "purged_mid_epoch": {"status": "FAILED", "status_detail":
                         "LSF no longer knows job 108; it ended while no dashboard was watching, and how is not known"},
    "purged_without_a_log": {"status": "FAILED", "status_detail":
                             "LSF no longer knows job 109; it ended while no dashboard was watching, and how is not known"},
}


def test_rehydrating_a_session_snapshot(fake_lsf, local_jobs, tmp_path):
    """A dashboard started later finds its jobs from each run's metadata.json and
    one bjobs call. Jobs outlive dashboard upgrades, so this reading of the file
    is a format: pinned here for every state a run can be left in."""
    session = tmp_path / "s"
    written = {}
    for name, (status, lsf_job_id, _, log) in RUNS.items():
        run = session / "runs" / name
        run.mkdir(parents=True)
        written[name] = {"job_id": name, **RECORD, "corrections_path": str(session / "corrections"),
                         "output_dir": str(run), "models_dir": str(session / "models"),
                         "lsf_job_id": lsf_job_id, "status": status}
        if name == "waiting":
            written[name].update(inference_server_url=URL, finetuned_model_name="m_finetuned_2",
                                 model_yaml_path="/s/models/m_finetuned_1.yaml")
        if log is not None:
            (run / "training_log.txt").write_text(log)
    written["from_an_old_dashboard"] = {"job_id": "from_an_old_dashboard", "lsf_job_id": "114", "status": "RUNNING"}
    written["no_job_id"] = {"lsf_job_id": "116", "status": "RUNNING"}
    for name, metadata in written.items():
        (session / "runs" / name).mkdir(parents=True, exist_ok=True)
        (session / "runs" / name / "metadata.json").write_text(json.dumps(metadata))
    (session / "runs" / "unreadable").mkdir()
    (session / "runs" / "unreadable" / "metadata.json").write_text('{"job_id": "unreadable", "lsf_')

    said = {**{lsf_job_id: stat for lsf_job_id, stat in (RUNS[n][1:3] for n in RUNS) if stat}, "114": "RUN"}
    fake_lsf.answers["bjobs"] = [
        (255, "".join(f"{i}  me  {stat}  gpu_h100  login1  h10u05  finetune_m  Jan 1 12:00\n"
                      for i, stat in said.items() if stat != "not found"),
         "".join(f"Job <{i}> is not found\n" for i, stat in said.items() if stat == "not found")),
        "110  me  DONE  gpu_h100  login1  h10u05  finetune_m  Jan 1 12:00\n",
    ]
    manager = FinetuneJobManager()
    manager.jobs["known"] = known = SimpleNamespace(job_id="known")

    assert manager.rehydrate_session(session) == len(REATTACHED)

    # One bjobs call, for the runs whose record is not final, in the order of their directories.
    assert fake_lsf.calls == [["bjobs", "-noheader", "105", "106", "114", "102", "107", "108", "109", "101",
                               "104", "110", "103"]]
    rebuilt = {}
    for job_id, job in manager.jobs.items():
        if job is known:
            continue
        seen = job.to_dict()
        assert isinstance(job.lsf_job, LSFJob) and (job.lsf_job.job_id, job.lsf_job.model_name) == (
            seen["lsf_job_id"], written[job_id].get("model_name"))
        if "created_at" not in written[job_id]:
            assert abs(datetime.now() - job.created_at).total_seconds() < 60
            seen["created_at"] = "<now>"
        rebuilt[job_id] = seen
    run = str(session / "runs" / "{}")
    assert rebuilt == {job_id: {
        "job_id": job_id, "model_name": "m", "output_dir": run.format(job_id), "params": PARAMS,
        "created_at": CREATED, "log_file": run.format(job_id) + "/training_log.txt", "finetuned_model_name": None,
        "model_yaml_path": None, "current_epoch": 0, "total_epochs": 5, "latest_loss": None,
        "inference_server_url": None, "inference_server_ready": False,
        "corrections_path": str(session / "corrections"), **fields,
    } for job_id, fields in REATTACHED.items()}
    assert [job.job_id for job in local_jobs.monitors] == list(REATTACHED), "each is monitored again"
    for name, metadata in written.items():
        on_disk = json.loads((session / "runs" / name / "metadata.json").read_text())
        assert on_disk == {**metadata, **RECORDED.get(name, {})}, name

    # Asked again, only the run bjobs said nothing of is asked about; it has finished since.
    assert manager.rehydrate_session(session) == 0
    assert fake_lsf.calls[1:] == [["bjobs", "-noheader", "110"]]
    assert json.loads((session / "runs" / "unanswered" / "metadata.json").read_text())["status"] == "COMPLETED"
    # And with nothing unfinished left, LSF is asked nothing.
    assert manager.rehydrate_session(session) == 0 and len(fake_lsf.calls) == 2


@pytest.mark.parametrize("status, server, sent", [
    pytest.param("WAITING_FOR_RESTART", "answers", "over HTTP", id="waiting, its server answers: over HTTP"),
    pytest.param("WAITING_FOR_RESTART", "refuses", "in the signal file",
                 id="waiting, its server refuses: through the signal file"),
    pytest.param("WAITING_FOR_RESTART", None, "in the signal file", id="waiting, with no server: through the signal file"),
    pytest.param("COMPLETED", "answers", None, id="completed: the trainer has exited"),
])
def test_only_a_job_waiting_for_a_restart_is_restarted(make_job, monkeypatch, status, server, sent):
    """Only a job whose server was marked ready could be restarted, which a
    restart resets and a diverged iteration never sets. A COMPLETED trainer has
    exited: the request went nowhere, and the job showed RUNNING for ever.
    The request goes to the job's server with the job's token, else into a
    file the trainer also watches: a protocol with jobs already running."""
    manager = FinetuneJobManager()
    job = make_job(status, inference_server_ready=True, inference_server_url=URL if server else None)
    manager.jobs[job.job_id] = job
    (job.output_dir / "restart_token").write_text("the-job-token")  # as submit writes it
    posted = []

    def post(url, json=None, headers=None, timeout=None):
        posted.append((url, json, headers))
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"success": server == "answers", "error": "refused"})

    monkeypatch.setattr(requests, "post", post)
    signal = job.output_dir / "restart_signal.json"
    if sent is None:
        with pytest.raises(ValueError):
            manager.restart_finetuning_job(job.job_id, {})
        assert not posted and not signal.exists()
        return
    manager.restart_finetuning_job(job.job_id, {"learning_rate": 5e-5})
    request = {"restart": True, "timestamp": ANY, "params": {"learning_rate": 5e-5}}
    assert posted == ([(f"{URL}/__control__/restart", request, {"X-Restart-Token": "the-job-token"})] if server else [])
    if sent == "in the signal file":
        assert json.loads(signal.read_text()) == request
    else:
        assert not signal.exists()
    assert (job.status, job.params["learning_rate"]) == (JobStatus.RUNNING, 5e-5)


@pytest.mark.parametrize("on_disk, params, served_from", [
    pytest.param(None, {"lora_r": 64}, "lora_adapter", id="nothing yet, a LoRA job"),
    pytest.param(None, {"lora_r": 0}, "full_finetune/model_state_dict.pt", id="nothing yet, a full finetune"),
    pytest.param("full_finetune/model_state_dict.pt", {"lora_r": 64}, "full_finetune/model_state_dict.pt",
                 id="full weights on disk win over the rank"),
    pytest.param("lora_adapter/adapter_config.json", {"lora_r": 0}, "lora_adapter",
                 id="an adapter on disk wins over the rank"),
])
def test_what_a_run_is_served_from_is_what_it_exported(tmp_path, on_disk, params, served_from):
    """Decided by what is on disk, then the job's rank: a viewer pointed at an adapter a rank-0 run never made."""
    if on_disk:
        (tmp_path / on_disk).parent.mkdir(parents=True)
        (tmp_path / on_disk).write_bytes(b"")
    (path,) = finetune_export_kwargs(tmp_path, params).values()
    assert path == str(tmp_path / served_from)
