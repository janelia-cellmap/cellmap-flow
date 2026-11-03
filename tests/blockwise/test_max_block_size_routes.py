"""The dashboard's max-block-size routes: check the cache, submit a probe
job to the pipeline's blockwise-config queue, and poll it.

LSF here is conftest's ``fake_lsf``; the probe itself never runs -- these
tests are about the routes' plumbing (what gets submitted, what gets
cached, when polling says "done"), not the probe's own search, which
tests/blockwise/test_max_block_size.py covers."""

import copy
import json

import pytest

from cellmap_flow.blockwise import max_block_size_cache
from cellmap_flow.blockwise.max_block_size import BlockSizeCandidate, MaxBlockSizeResult
from cellmap_flow.dashboard.routes import blockwise as blockwise_routes
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.spec import JobStatus

SETTINGS = {"charge_group": "grp", "queue": "gpu_h100", "nb_workers": 2, "nb_cores_worker": 12,
            "nb_cores_master": 4, "tmp_dir": "/scratch/progress"}
ACCEPTED = "Your request was accepted.\nJob <5150> is submitted to default queue <local>.\n"


@pytest.fixture(autouse=True)
def _isolated_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(max_block_size_cache, "CACHE_PATH", str(tmp_path / "max_block_size_cache.json"))
    monkeypatch.setattr(blockwise_routes, "_ENDED_AT", {})


@pytest.fixture
def no_grace(monkeypatch):
    """An ended job's missing result is final at once."""
    monkeypatch.setattr(blockwise_routes, "RESULT_GRACE_SECONDS", {})


@pytest.fixture
def tasks(tmp_path):
    session = get_session()
    session.blockwise_tasks_dir, session.walltime = str(tmp_path / "tasks"), "12:00"
    return tmp_path / "tasks"


def _pipeline(script_path, **changes):
    base = {
        "models": [{"name": "m", "params": {"type": "script", "script_path": script_path}}],
        "blockwise_config": [{"params": SETTINGS}],
    }
    return {**copy.deepcopy(base), **copy.deepcopy(changes)}


def _result():
    return MaxBlockSizeResult(
        gpu_name="Fake H100",
        total_memory_bytes=80 * 1024 ** 3,
        candidates=[BlockSizeCandidate(1, (56, 56, 56, 2), True, peak_bytes=1024)],
    )


def _result_as_json():
    """_result().to_dict(), round-tripped through JSON: a dashboard answer's
    tuples come back as lists, which plain equality would otherwise fail on."""
    return json.loads(json.dumps(_result().to_dict()))


def test_check_reports_not_cached_when_there_is_nothing_yet(dashboard, model_script):
    script = model_script()
    answer = dashboard.post(
        "/api/blockwise/max-block-size/check", json={"pipeline": _pipeline(script)}
    ).get_json()
    assert answer == {"cached": False}


def test_check_finds_what_store_put_in_the_cache(dashboard, model_script):
    from cellmap_flow.models.models_config import ScriptModelConfig

    script = model_script()
    max_block_size_cache.store(ScriptModelConfig(script, name="m"), "gpu_h100", _result())

    answer = dashboard.post(
        "/api/blockwise/max-block-size/check", json={"pipeline": _pipeline(script)}
    ).get_json()
    assert answer["cached"] is True
    assert answer["result"] == _result_as_json()


def test_check_without_a_queue_is_not_cached_with_an_error(dashboard, model_script):
    script = model_script()
    answer = dashboard.post(
        "/api/blockwise/max-block-size/check",
        json={"pipeline": _pipeline(script, blockwise_config=[{"params": {**SETTINGS, "queue": ""}}])},
    ).get_json()
    assert answer == {"cached": False, "error": "Set a queue in the Blockwise Config node first"}


def test_compute_submits_a_one_gpu_job_on_the_requested_queue(dashboard, model_script, fake_lsf, tasks):
    fake_lsf.answers["bsub"] = [ACCEPTED]
    script = model_script()

    answer = dashboard.post(
        "/api/blockwise/max-block-size/compute", json={"pipeline": _pipeline(script)}
    ).get_json()

    assert answer["success"], answer
    assert answer["job_id"] == "5150"
    assert answer["queue"] == "gpu_h100"
    assert answer["model_entry"] == {"type": "script", "script_path": script, "name": "m"}

    (bsub_argv,) = fake_lsf.commands("bsub")
    assert "-q" in bsub_argv and "gpu_h100" in bsub_argv
    assert "-gpu" in bsub_argv  # gpus=1, unlike the master job in submit_blockwise_task
    assert "max-block-size" in bsub_argv
    model_index = bsub_argv.index("--model") + 1
    assert json.loads(bsub_argv[model_index]) == answer["model_entry"]
    # The probe caches its own answer under this queue (max_block_size_cli
    # --queue), for when the builder has stopped polling by the time it ends.
    assert bsub_argv[bsub_argv.index("--queue") + 1] == "gpu_h100"


def test_compute_accepts_the_queue_as_a_list_the_builder_actually_sends(dashboard, model_script, fake_lsf, tasks):
    """The Blockwise Config node's Queue field is always a list by the time
    it reaches here (the dashboard's Queues type supports several,
    comma-separated, tried in order of freeness): JobSpec.queue wants a
    single string, so compute must resolve it to one itself rather than
    pass the list straight through (that failed with "sequence item 8:
    expected str instance, list found" building the bsub command line)."""
    fake_lsf.answers["bsub"] = [ACCEPTED]
    script = model_script()

    answer = dashboard.post(
        "/api/blockwise/max-block-size/compute",
        json={"pipeline": _pipeline(script, blockwise_config=[{"params": {**SETTINGS, "queue": ["gpu_h200"]}}])},
    ).get_json()

    assert answer["success"], answer
    assert answer["queue"] == "gpu_h200"
    (bsub_argv,) = fake_lsf.commands("bsub")
    assert all(isinstance(arg, str) for arg in bsub_argv)
    assert "-q" in bsub_argv and "gpu_h200" in bsub_argv


def test_compute_with_several_queues_probes_the_first_named(dashboard, model_script, fake_lsf, tasks):
    """Several queues ("try gpu_h100, then gpu_h200") name a fallback order
    for where a job may land, not several GPU types to size a block for at
    once; the first named is the one probed and cached, deliberately not
    resolved through current LSF load (jobs.queues.candidates), which would
    make the cached answer depend on when it happened to be asked."""
    fake_lsf.answers["bsub"] = [ACCEPTED]
    script = model_script()

    answer = dashboard.post(
        "/api/blockwise/max-block-size/compute",
        json={"pipeline": _pipeline(
            script, blockwise_config=[{"params": {**SETTINGS, "queue": ["gpu_h100", "gpu_h200"]}}]
        )},
    ).get_json()

    assert answer["success"], answer
    assert answer["queue"] == "gpu_h100"
    assert fake_lsf.commands("bjobs") == []  # no LSF query to decide: gpu_h100 is just taken as named


def test_check_also_accepts_the_queue_as_a_list(dashboard, model_script):
    script = model_script()
    answer = dashboard.post(
        "/api/blockwise/max-block-size/check",
        json={"pipeline": _pipeline(script, blockwise_config=[{"params": {**SETTINGS, "queue": ["gpu_h200"]}}])},
    ).get_json()
    assert answer == {"cached": False}


def test_compute_without_a_queue_fails_before_submitting_anything(dashboard, model_script, fake_lsf):
    script = model_script()
    answer = dashboard.post(
        "/api/blockwise/max-block-size/compute",
        json={"pipeline": _pipeline(script, blockwise_config=[{"params": {**SETTINGS, "queue": ""}}])},
    ).get_json()
    assert answer == {"success": False, "error": "Set a queue in the Blockwise Config node first"}
    assert fake_lsf.calls == []


def test_status_reads_and_caches_the_result_once_the_file_exists(dashboard, model_script, tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_result().to_dict()))
    entry = {"type": "script", "script_path": model_script()}

    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(result_path), "model_entry": entry, "queue": "gpu_h100"},
    ).get_json()

    assert answer == {"done": True, "result": _result_as_json()}
    from cellmap_flow.models.models_config import ScriptModelConfig
    assert max_block_size_cache.load_cached(ScriptModelConfig(entry["script_path"], name="m"), "gpu_h100") == (
        _result_as_json()
    )


def test_status_keeps_polling_while_the_job_is_still_running(dashboard, fake_lsf, tmp_path):
    fake_lsf.answers["bjobs"] = ["5150   user    RUN   gpu_h100    host  job 1\n"]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json")},
    ).get_json()
    assert answer == {"done": False}


def test_status_keeps_polling_while_the_job_is_pending(dashboard, fake_lsf, tmp_path):
    fake_lsf.answers["bjobs"] = ["5150   user    PEND   gpu_h100    -  job 1\n"]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json")},
    ).get_json()
    assert answer == {"done": False}


def test_status_keeps_polling_when_bjobs_itself_could_not_be_run(dashboard, fake_lsf, tmp_path):
    """"Could not ask" (statuses leaves the id out) is not "not found": the
    job may well be running, so this is no reason to give up on it."""
    fake_lsf.answers["bjobs"] = [FileNotFoundError("bjobs")]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json")},
    ).get_json()
    assert answer == {"done": False}


def test_status_stops_once_the_job_is_done_without_a_result(dashboard, fake_lsf, tmp_path, no_grace):
    """A probe that exits 0 without its result file is over all the same:
    polling it as "not done" went on forever in the builder."""
    fake_lsf.answers["bjobs"] = ["5150   user    DONE   gpu_h100    host  job 1\n"]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json"),
              "log_path": "/logs/max_block_size_5150.log"},
    ).get_json()
    assert answer["done"] is True
    assert "5150" in answer["error"] and "COMPLETED" in answer["error"]
    assert "/logs/max_block_size_5150.log" in answer["error"]


def test_a_done_jobs_missing_result_is_waited_for_before_it_is_called_missing(dashboard, fake_lsf, tmp_path):
    """The result is written on a compute node seconds before LSF says DONE,
    and NFS here can go on answering "no such file" for most of a minute
    after: a real result (11:35:22) was called missing by the poll after
    DONE (11:35:26). So an ended job's result is waited for a while -- and
    taken the moment it shows."""
    fake_lsf.answers["bjobs"] = ["5150   user    DONE   gpu_h100    host  job 1\n"] * 3
    result_path = tmp_path / "late.json"
    body = {"job_id": "5150", "result_path": str(result_path)}

    answer = dashboard.post("/api/blockwise/max-block-size/status", json=body).get_json()
    assert answer["done"] is False
    assert "5150" in answer["note"] and "COMPLETED" in answer["note"] and "waiting" in answer["note"]
    assert dashboard.post("/api/blockwise/max-block-size/status", json=body).get_json()["done"] is False

    result_path.write_text(json.dumps(_result().to_dict()))
    answer = dashboard.post("/api/blockwise/max-block-size/status", json=body).get_json()
    assert answer == {"done": True, "result": _result_as_json()}
    assert blockwise_routes._ENDED_AT == {}  # nothing kept for a job that is over


def test_a_done_jobs_result_is_taken_from_the_cache_the_probe_wrote(dashboard, fake_lsf, model_script, tmp_path):
    """The probe caches its own answer (max_block_size_cli --queue): an entry
    stored since the job was submitted is its answer, however late its file
    is to show here. One from before (what Recompute is replacing) is not."""
    from cellmap_flow.models.models_config import ScriptModelConfig

    fake_lsf.answers["bjobs"] = ["5150   user    DONE   gpu_h100    host  job 1\n"] * 2
    script = model_script()
    entry = {"type": "script", "script_path": script, "name": "m"}
    max_block_size_cache.store(ScriptModelConfig(script, name="m"), "gpu_h100", _result())
    stored = max_block_size_cache.stored_at(ScriptModelConfig(script, name="m"), "gpu_h100")
    body = {"job_id": "5150", "result_path": str(tmp_path / "late.json"), "model_entry": entry, "queue": "gpu_h100"}

    # Submitted after the entry was stored: the entry is an older probe's.
    answer = dashboard.post("/api/blockwise/max-block-size/status", json={**body, "submitted_at": stored + 5}).get_json()
    assert answer["done"] is False and "waiting" in answer["note"]
    # Submitted before: the entry is this job's.
    answer = dashboard.post("/api/blockwise/max-block-size/status", json={**body, "submitted_at": stored - 5}).get_json()
    assert answer == {"done": True, "result": _result_as_json()}


def test_status_reports_an_error_once_lsf_says_the_job_failed(dashboard, fake_lsf, tmp_path, no_grace):
    fake_lsf.answers["bjobs"] = ["5150   user    EXIT   gpu_h100    host  job 1\n"]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json")},
    ).get_json()
    assert answer["done"] is True
    assert "5150" in answer["error"] and "FAILED" in answer["error"]


def test_a_job_lsf_has_forgotten_gets_no_grace(dashboard, fake_lsf, tmp_path):
    """It ended long ago: anything it wrote has long been visible."""
    fake_lsf.answers["bjobs"] = [(0, "", "Job <5150> is not found\n")]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json")},
    ).get_json()
    assert answer["done"] is True and "LSF no longer knows it" in answer["error"]


def test_status_reports_an_error_once_lsf_no_longer_knows_the_job(dashboard, fake_lsf, tmp_path):
    fake_lsf.answers["bjobs"] = [(0, "", "Job <5150> is not found\n")]
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status",
        json={"job_id": "5150", "result_path": str(tmp_path / "never_written.json")},
    ).get_json()
    assert answer["done"] is True
    assert "5150" in answer["error"]


def test_cancel_bkills_the_probe_job(dashboard, fake_lsf):
    fake_lsf.answers["bkill"] = ["Job <5150> is being terminated\n"]
    answer = dashboard.post("/api/blockwise/max-block-size/cancel", json={"job_id": "5150"}).get_json()
    assert answer == {"success": True}
    assert fake_lsf.commands("bkill") == [["bkill", "5150"]]


def test_cancel_of_a_job_that_already_ended_is_a_success_too(dashboard, fake_lsf):
    """The user wanted it gone, and it is: not an error to show them."""
    fake_lsf.answers["bkill"] = [(255, "", "Job <5150>: Job has already finished\n")]
    answer = dashboard.post("/api/blockwise/max-block-size/cancel", json={"job_id": "5150"}).get_json()
    assert answer == {"success": True}


def test_cancel_reports_what_bkill_refused(dashboard, fake_lsf):
    fake_lsf.answers["bkill"] = [(255, "", "Job <5150>: User permission denied\n")]
    answer = dashboard.post("/api/blockwise/max-block-size/cancel", json={"job_id": "5150"}).get_json()
    assert answer["success"] is False
    assert "5150" in answer["error"] and "User permission denied" in answer["error"]


def test_cancel_without_a_job_id_is_refused_without_calling_lsf(dashboard, fake_lsf):
    answer = dashboard.post("/api/blockwise/max-block-size/cancel", json={}).get_json()
    assert answer["success"] is False and "job_id" in answer["error"]
    assert fake_lsf.calls == []


def test_status_without_a_job_id_stops_polling_rather_than_looping_forever(dashboard, tmp_path):
    answer = dashboard.post(
        "/api/blockwise/max-block-size/status", json={"result_path": str(tmp_path / "x.json")}
    ).get_json()
    assert answer["done"] is True
    assert "job_id" in answer["error"]
