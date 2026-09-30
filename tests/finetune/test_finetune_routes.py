"""The dashboard's finetune routes, through the Flask client: what submit and
restart send the job manager and write into the session's manifest, the jobs
list after a dashboard restart, resuming a session, and MinIO URLs behind a
proxy. What the volume routes write is pinned by test_volume_snapshot."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.globals import g

OFFSETS = "offsets = [[1, 0, 0], [0, 1, 0]]\nmodel = None\n"
CROP = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The dashboard's test client, a model "m" (a script with affinity offsets) and billing to "my_lab"."""
    from cellmap_flow.dashboard.app import app

    (tmp_path / "model.py").write_text(OFFSETS)
    for key, value in dict(models_config=[SimpleNamespace(name="m", script_path=str(tmp_path / "model.py"))],
                           charge_group="my_lab", annotation_volumes={}, output_sessions={}).items():
        monkeypatch.setattr(g, key, value, raising=False)
    return app.test_client()


@pytest.fixture
def trainable_session(tmp_path, annotation_volume):
    """``trainable_session(labels, crops)``: <tmp>/base/<session>/corrections, whose manifest names a volume."""

    def make(labels, crops=()):
        volume = annotation_volume(labels, crops=crops)
        corrections = tmp_path / "base" / "20260101_120000" / "corrections"
        corrections.mkdir(parents=True)
        (corrections / "_virtual_sources.json").write_text(json.dumps(
            {"kind": "volume_zarr_v1", "volume_zarr_path": volume.path, "raw_dataset_path": volume.raw}))
        (tmp_path / "base" / "20260102_090000").mkdir()  # newer, and nothing to train on
        return corrections

    return make


def _labels(**boxes):
    labels = np.zeros((32,) * 3, np.uint8)
    for value, where in boxes.values():
        labels[where] = value
    return labels


CROPPED = (_labels(crop=(1, np.s_[0:16, 0:16, 0:16]), fg=(2, np.s_[4:8, 4:8, 4:8])), [CROP])
PAINTED = (_labels(stroke=(2, np.s_[3, 3, 3:6])), [])
STROKE_BESIDE = (_labels(crop=(1, np.s_[0:16, 0:16, 0:16]), stroke=(2, np.s_[20, 20, 20])), [CROP])
ABSENT = object()


@pytest.fixture
def submit(client, trainable_session):
    """``submit(volume=CROPPED, via_base_path=False, **request)``: POST
    /api/finetune/submit for model "m", on a session over ``volume``. Returns
    the status, what the job manager was asked for (None if nothing), and the
    session's manifest afterwards."""

    def run(volume=CROPPED, via_base_path=False, **request):
        corrections = trainable_session(*volume)
        asked = []
        g.finetune_job_manager = SimpleNamespace(jobs={}, submit_finetuning_job=lambda **kw: asked.append(kw)
                                                 or SimpleNamespace(job_id="j", output_dir=corrections, lsf_job=None))
        path = corrections.parent.parent if via_base_path else corrections
        response = client.post("/api/finetune/submit",
                               json={"model_name": "m", "corrections_path": str(path), **request})
        return SimpleNamespace(status=response.status_code, sent=asked[0] if asked else None, corrections=corrections,
                               manifest=json.loads((corrections / "_virtual_sources.json").read_text()))

    return run


@pytest.mark.parametrize("volume, request_data, sent", [
    pytest.param(CROPPED, {}, dict(mask_unannotated=False, loss_type="mse"), id="imported crops are dense"),
    # A distance target needs 3D boundaries, which scribbles do not have.
    pytest.param(PAINTED, {"output_type": "distance"}, dict(mask_unannotated=True, loss_type="margin",
                                                            output_type="binary"), id="a painted session"),
    pytest.param(STROKE_BESIDE, {}, dict(mask_unannotated=True, loss_type="margin"),
                 id="a stroke beside the crops"),
])
def test_submit_trains_scribbles_as_scribbles(submit, volume, request_data, sent):
    """Scribbles were detected from per-chunk extracts that no session has any
    more, so it never fired: painted sessions trained as dense labels, with
    unannotated voxels taken for background. It is read from the volume now."""
    job = submit(volume, **request_data)
    assert {key: job.sent[key] for key in sent} == sent


def test_submit_reads_the_affinity_offsets_from_the_models_script(submit):
    job = submit()
    assert (job.sent["output_type"], job.sent["offsets"]) == ("affinities", "[[1, 0, 0], [0, 1, 0]]")


def test_submit_bills_the_dashboards_charge_group(submit):
    """Every finetune job billed "cellmap", the job manager's default."""
    assert submit().sent["charge_group"] == "my_lab"


@pytest.mark.parametrize("value, status, epochs", [
    pytest.param("", 200, 10, id="blank gets the default"),
    pytest.param("25", 200, 25, id="a number"),
    pytest.param("ten", 400, None, id="not a number"),
    pytest.param("2.5", 400, None, id="not a whole number"),
])
def test_a_number_field_is_read_as_a_number(submit, value, status, epochs):
    """An emptied field reached the trainer as "--num-epochs None", failing on
    the cluster; a field that is not a number is a 400 here instead."""
    job = submit(num_epochs=value)
    assert job.status == status
    assert (job.sent or {}).get("num_epochs") == epochs


@pytest.mark.parametrize("request_data, manifest", [
    pytest.param({"rehearsal_fraction": "0.5", "patches_per_epoch": 0},
                 dict(rehearsal_fraction=0.5, patches_per_epoch=None), id="a fraction, and patches on auto"),
    # 0 turns rehearsal off for this run, without dropping the regions.
    pytest.param({"rehearsal_fraction": 0}, dict(rehearsal_fraction=0.0), id="rehearsal off"),
    pytest.param({"rehearsal_fraction": ""}, dict(rehearsal_fraction=ABSENT), id="blank leaves it alone"),
])
def test_the_runs_overrides_reach_the_manifest(submit, request_data, manifest):
    job = submit(**request_data)
    assert job.status == 200
    assert {key: job.manifest.get(key, ABSENT) for key in manifest} == manifest


@pytest.mark.parametrize("fraction", [pytest.param(1.5, id="out of range"), pytest.param("abc", id="not a number")])
def test_a_rehearsal_fraction_that_is_not_one_is_refused(submit, fraction):
    job = submit(rehearsal_fraction=fraction)
    assert job.status == 400 and job.sent is None and "rehearsal_fraction" not in job.manifest


def test_submit_after_a_dashboard_restart_finds_the_session_on_disk(submit):
    """The dashboard forgot its sessions and made a new, empty one for the base
    path: "Corrections path does not exist". The newest session there with
    something to train on is used instead."""
    job = submit(via_base_path=True)
    assert job.sent["corrections_path"] == job.corrections


@pytest.mark.parametrize("registered, written", [
    # A painted volume and no manifest: it trained on a per-chunk copy, or not at all.
    pytest.param("this session's", True, id="the session's own volume"),
    pytest.param("another session's", False, id="another session's volume"),
    pytest.param("an incomplete", False, id="an incomplete record"),  # better none than one the trainer chokes on
])
def test_submit_backfills_the_manifest_of_a_session_from_before_it(client, tmp_path, registered, written):
    corrections = tmp_path / "session" / "corrections"
    corrections.mkdir(parents=True)
    volume = {"zarr_path": str(corrections / "vol.zarr"), "dataset_path": "/nrs/raw.zarr/em/s0",
              "input_size": [178] * 3, "output_size": [56] * 3, "input_voxel_size": [16] * 3,
              "output_voxel_size": [16] * 3, "corrections_dir": str(corrections)}
    if registered == "another session's":
        volume["corrections_dir"] = str(tmp_path / "elsewhere" / "corrections")
    if registered == "an incomplete":
        volume["output_size"] = volume["dataset_path"] = None
    g.annotation_volumes = {"vol": volume}
    g.finetune_job_manager = SimpleNamespace(jobs={}, submit_finetuning_job=lambda **kw: SimpleNamespace(
        job_id="j", output_dir=corrections, lsf_job=None))
    client.post("/api/finetune/submit", json={"model_name": "m", "corrections_path": str(corrections)})

    manifest = corrections / "_virtual_sources.json"
    assert manifest.exists() == written
    if written:
        assert json.loads(manifest.read_text())["input_size_voxels"] == [178] * 3


@pytest.fixture
def restart(client, local_jobs, session, monkeypatch):
    """``restart(pulled=0, **request)``: POST /api/finetune/job/<id>/restart for
    a job submitted through the dashboard's own job manager, with MinIO sync
    pulling ``pulled`` volumes. Returns the response body, the syncs asked for,
    what the trainer is sent, and the session."""
    from cellmap_flow.dashboard.routes.finetune import training
    from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

    def run(pulled=0, **request):
        base = session()
        manager = g.finetune_job_manager  # made when first asked for, as in the dashboard
        assert isinstance(manager, FinetuneJobManager)
        job = manager.submit_finetuning_job(model_config=g.models_config[0], corrections_path=base / "corrections",
                                            output_base=base)
        record = SimpleNamespace(syncs=[], sent=[], base=base)
        monkeypatch.setattr(training, "sync_all_annotations_from_minio",
                            lambda force=True: record.syncs.append(force) or pulled)
        monkeypatch.setattr(manager, "restart_finetuning_job",
                            lambda job_id, updated_params: record.sent.append(updated_params) or job)
        record.body = client.post(f"/api/finetune/job/{job.job_id}/restart", json=request).get_json()
        return record

    return run


@pytest.mark.parametrize("pulled, synced", [
    pytest.param(2, 2, id="new annotations"),
    pytest.param(0, 0, id="nothing new: a parameters-only restart"),
    pytest.param(-1, 0, id="MinIO is not running"),
])
def test_a_restart_pulls_the_new_annotations_first(restart, pulled, synced):
    """The trainer rebuilds its data from the volume on disk, and only the sync
    puts the browser's strokes there: a session with a manifest skipped it, and
    trained on the old annotations."""
    run = restart(pulled=pulled)
    assert run.syncs == [False], "a diff of the chunks, whether or not there is a manifest"
    assert run.body["annotations_synced"] == synced


def test_a_restart_refreshes_the_manifest_of_the_jobs_session(restart):
    """The job did not record its corrections dir, so a restart never refreshed
    the manifest: its new settings, shown in the dialog, were dropped."""
    run = restart(patches_per_epoch=7)
    assert json.loads((run.base / "corrections" / "_virtual_sources.json").read_text())["patches_per_epoch"] == 7


def test_a_restart_sends_the_trainer_its_own_flags(restart):
    """The trainer's flag is --no-augment (the toggle was dropped), --offsets is
    JSON (a list killed the restart), and the scope is --distillation-all-voxels."""
    run = restart(augment=True, offsets=[[1, 0, 0]], distillation_scope="all", loss_type="margin")
    assert run.sent == [{"augment": True, "no_augment": False, "offsets": "[[1, 0, 0]]",
                         "distillation_all_voxels": True, "loss_type": "margin"}]


def test_the_jobs_list_looks_for_jobs_in_the_saved_output_path(client, session, monkeypatch):
    """After a dashboard restart this dashboard has made no sessions yet; the
    output path saved in the user prefs is where its jobs are."""
    from cellmap_flow.dashboard.routes.finetune import common

    base = session()
    looked = []
    g.finetune_job_manager = SimpleNamespace(jobs={}, rehydrate_session=looked.append, list_jobs=lambda: [])
    monkeypatch.setattr(common, "load_user_prefs", lambda: {"outputPath": str(base.parent)})
    assert client.get("/api/finetune/jobs").get_json()["success"]
    assert looked == [str(base)]


def _zarr(path, **attrs):
    path.mkdir(parents=True)
    (path / ".zattrs").write_text(json.dumps(attrs))


def test_resuming_offers_the_volume_the_session_trains_on_first(client, tmp_path):
    """Resume took whichever zarr os.listdir() gave first, crop zarrs and legacy
    per-chunk extracts included, and counted the extracts' chunks: 0 for a
    painted session."""
    corrections = tmp_path / "out" / "20260101_000000" / "corrections"
    _zarr(corrections / "aaa_crop.zarr", dataset_path="/raw")  # a crop, listed first
    _zarr(corrections / "vol-old.zarr", type="annotation_volume")
    time.sleep(0.01)
    _zarr(corrections / "vol-new.zarr", type="annotation_volume")
    _zarr(corrections / "vol-x_chunk_1.zarr", type="annotation_volume")  # a legacy extract
    s0 = corrections / "vol-new.zarr" / "annotation" / "s0"
    s0.mkdir(parents=True)
    for key in (".zarray", "0.0.0", "0.0.1"):
        (s0 / key).write_bytes(b"x")
    (corrections / "_virtual_sources.json").write_text(json.dumps({"volume_zarr_path": str(corrections / "vol-old.zarr")}))
    _zarr(tmp_path / "out" / "20250101_000000" / "corrections" / "v_chunk_0_0_0.zarr", source="sparse_volume")

    sessions = client.post("/api/finetune/list-existing-sessions", json={"output_path": str(tmp_path / "out")}
                           ).get_json()["sessions"]
    assert [s["session_id"] for s in sessions] == ["20260101_000000"], "extracts alone are nothing to resume"
    assert [v["volume_id"] for v in sessions[0]["volumes"]] == ["vol-old", "vol-new"]
    assert sessions[0]["chunk_count"] == 2


MINIO = "http://10.0.0.5:9000/annotations/vol-1.zarr"
PROXY = "https://gateway.example.org/minio/annotations/vol-1.zarr"


TEMPLATE = "{proto}://{host}/minio"


@pytest.mark.parametrize("template, headers, expected", [
    pytest.param(None, {"X-Forwarded-Host": "gateway.example.org"}, MINIO, id="opt-in only"),
    pytest.param(TEMPLATE, {}, MINIO, id="direct access: no proxy, no rewrite"),
    pytest.param(TEMPLATE, None, MINIO, id="outside a request"),
    pytest.param(TEMPLATE, {"X-Forwarded-Host": "gateway.example.org, inner", "X-Forwarded-Proto": "https"},
                 PROXY, id="the first proxy of a chain, and its protocol"),
    pytest.param(TEMPLATE, {"X-Forwarded-Host": "gateway.example.org"}, PROXY.replace("https", "http"),
                 id="no forwarded protocol: the request's"),
    pytest.param("https://gateway.example.org/minio/", {"X-Forwarded-Host": "anything"}, PROXY,
                 id="a fixed proxy URL"),
])
def test_minio_urls_are_rewritten_only_behind_a_configured_proxy(monkeypatch, template, headers, expected):
    from flask import Flask

    from cellmap_flow.dashboard.routes.finetune.common import rewrite_minio_url_for_proxy

    if template is None:
        monkeypatch.delenv("CELLMAP_FLOW_MINIO_PROXY_URL", raising=False)
    else:
        monkeypatch.setenv("CELLMAP_FLOW_MINIO_PROXY_URL", template)
    if headers is None:
        assert rewrite_minio_url_for_proxy(MINIO) == expected
        return
    with Flask(__name__).test_request_context(base_url="http://node:5000", headers=headers):
        assert rewrite_minio_url_for_proxy(MINIO) == expected
