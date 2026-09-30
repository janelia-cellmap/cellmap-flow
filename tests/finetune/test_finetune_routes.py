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
    from cellmap_flow.dashboard import finetune_utils
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import common

    # The tests-only aliases (WRAPPERS.md): W3-A's Session facade replaces them.
    monkeypatch.setattr(finetune_utils, "output_sessions", {})
    monkeypatch.setattr(common, "output_sessions", finetune_utils.output_sessions)
    (tmp_path / "model.py").write_text(OFFSETS)
    for key, value in dict(models_config=[SimpleNamespace(name="m", script_path=str(tmp_path / "model.py"))],
                           charge_group="my_lab", annotation_volumes={}).items():
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


@pytest.mark.parametrize("volume, request_data, status, sent, manifest", [
    (CROPPED, {}, 200, dict(mask_unannotated=False, loss_type="mse", charge_group="my_lab", num_epochs=10,
                            output_type="affinities", offsets="[[1, 0, 0], [0, 1, 0]]"), {}),
    # Painted sessions were never recognised, so scribbles trained as dense labels.
    (PAINTED, {"output_type": "distance"}, 200,
     dict(mask_unannotated=True, output_type="binary", loss_type="margin"), {}),
    (STROKE_BESIDE, {}, 200, dict(mask_unannotated=True, loss_type="margin"), {}),
    # An emptied field reached the trainer as "--num-epochs None", on the cluster.
    (CROPPED, {"num_epochs": ""}, 200, dict(num_epochs=10), {}),
    (CROPPED, {"num_epochs": "25"}, 200, dict(num_epochs=25), {}),
    (CROPPED, {"num_epochs": "ten"}, 400, None, {}),
    (CROPPED, {"num_epochs": "2.5"}, 400, None, {}),
    # 0 is a choice (rehearsal off for this run), blank leaves the manifest alone.
    (CROPPED, {"rehearsal_fraction": "0.5", "patches_per_epoch": 0}, 200, {},
     dict(rehearsal_fraction=0.5, patches_per_epoch=None)),
    (CROPPED, {"rehearsal_fraction": 0}, 200, {}, dict(rehearsal_fraction=0.0)),
    (CROPPED, {"rehearsal_fraction": ""}, 200, {}, dict(rehearsal_fraction=ABSENT)),
    (CROPPED, {"rehearsal_fraction": 1.5}, 400, None, dict(rehearsal_fraction=ABSENT)),
    (CROPPED, {"rehearsal_fraction": "abc"}, 400, None, dict(rehearsal_fraction=ABSENT)),
    # A dashboard restart forgot its sessions: the base path finds the one on disk.
    (CROPPED, {"corrections_path": "<base>"}, 200, dict(corrections_path="<corrections>"), {}),
], ids=["crops", "painted", "a stroke beside the crops", "blank number", "number", "not a number",
        "not a whole number", "overrides", "rehearsal off", "blank rehearsal", "rehearsal out of range",
        "rehearsal not a number", "the base path"])
def test_what_submit_sends_the_job_manager(client, trainable_session, volume, request_data, status, sent, manifest):
    corrections = trainable_session(*volume)
    submitted = []
    g.finetune_job_manager = SimpleNamespace(jobs={}, submit_finetuning_job=lambda **kw: submitted.append(kw)
                                             or SimpleNamespace(job_id="j", output_dir=corrections, lsf_job=None))
    paths = {"<base>": str(corrections.parent.parent), "<corrections>": corrections}
    data = {"model_name": "m", "corrections_path": str(corrections), **request_data}
    response = client.post("/api/finetune/submit", json={k: paths.get(v, v) for k, v in data.items()})

    assert response.status_code == status, response.get_json()
    if sent is not None:
        (kwargs,) = submitted
        assert {key: kwargs[key] for key in sent} == {k: paths.get(v, v) for k, v in sent.items()}
    written = json.loads((corrections / "_virtual_sources.json").read_text())
    assert {key: written.get(key, ABSENT) for key in manifest} == manifest


@pytest.mark.parametrize("registered, written", [
    ("this session's", True),  # a painted volume, and no manifest: trained on a per-chunk copy, or not at all
    ("another session's", False),
    ("an incomplete", False),  # better no manifest than one the trainer chokes on
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


@pytest.mark.parametrize("pulled, synced", [(2, 2), (0, 0), (-1, 0)])  # -1: MinIO is not running
def test_a_restart_pulls_the_new_annotations_first(client, local_jobs, session, monkeypatch, pulled, synced):
    """The trainer rebuilds its data from the volume on disk, and only the sync
    puts the browser's strokes there: a session with a manifest skipped it, and
    trained on the old annotations. The job used to forget its corrections dir,
    so the new settings never reached the manifest; and the trainer's flags are
    --no-augment and JSON --offsets, not what the form sends."""
    from cellmap_flow.dashboard.routes.finetune import training
    from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

    base = session()
    manager = g.finetune_job_manager  # the dashboard's own, made when first asked for
    assert isinstance(manager, FinetuneJobManager)
    job = manager.submit_finetuning_job(model_config=g.models_config[0], corrections_path=base / "corrections",
                                        output_base=base)
    syncs, restarts = [], []
    monkeypatch.setattr(training, "sync_all_annotations_from_minio", lambda force=True: syncs.append(force) or pulled)
    monkeypatch.setattr(manager, "restart_finetuning_job", lambda job_id, updated_params: restarts.append(
        updated_params) or job)
    response = client.post(f"/api/finetune/job/{job.job_id}/restart", json={
        "patches_per_epoch": 7, "augment": True, "offsets": [[1, 0, 0]], "distillation_scope": "all",
        "loss_type": "margin"})

    assert response.get_json()["annotations_synced"] == synced and syncs == [False]
    assert json.loads((base / "corrections" / "_virtual_sources.json").read_text())["patches_per_epoch"] == 7
    assert restarts == [{"augment": True, "no_augment": False, "offsets": "[[1, 0, 0]]",
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


@pytest.mark.parametrize("template, headers, expected", [
    (None, {"X-Forwarded-Host": "gateway.example.org"}, MINIO),  # opt-in only
    ("{proto}://{host}/minio", {}, MINIO),  # direct access: no proxy, no rewrite
    ("{proto}://{host}/minio", None, MINIO),  # outside a request
    ("{proto}://{host}/minio", {"X-Forwarded-Host": "gateway.example.org, inner", "X-Forwarded-Proto": "https"}, PROXY),
    ("{proto}://{host}/minio", {"X-Forwarded-Host": "gateway.example.org"}, PROXY.replace("https", "http")),
    ("https://gateway.example.org/minio/", {"X-Forwarded-Host": "anything"}, PROXY),
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
