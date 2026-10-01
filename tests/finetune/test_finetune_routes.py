"""The dashboard's finetune routes, through the Flask client: what submit and
restart send the job manager and write into the session's manifest, the jobs
list after a dashboard restart, resuming a session, MinIO URLs behind a
proxy, and what every route answers. What the volume routes write is pinned
by test_volume_snapshot."""

import json
import time
from types import SimpleNamespace
from unittest.mock import ANY

import numpy as np
import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.settings import launcher_settings

OFFSETS = "offsets = [[1, 0, 0], [0, 1, 0]]\nmodel = None\n"
CROP = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}


class _Script:
    """A script model, as the job manager tells a model's type (by its class's cli_name)."""

    cli_name = "script"

    def __init__(self, name, script_path):
        self.name, self.script_path = name, script_path


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The dashboard's test client, a model "m" (a script with affinity offsets), billing to "my_lab"
    and a walltime of 10:00."""
    from cellmap_flow.dashboard.app import app

    (tmp_path / "model.py").write_text(OFFSETS)
    for key, value in dict(models_config=[_Script("m", str(tmp_path / "model.py"))], annotation_volumes={},
                           output_sessions={}).items():
        monkeypatch.setattr(get_session(), key, value)
    for key, value in dict(charge_group="my_lab", walltime="10:00").items():
        monkeypatch.setattr(launcher_settings(), key, value)
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
def submit(client, trainable_session, monkeypatch):
    """``submit(volume=CROPPED, via_base_path=False, pulled=None, **request)``:
    POST /api/finetune/submit for model "m", on a session over ``volume``,
    with MinIO sync pulling the labels ``pulled`` into the volume, if given.
    Returns the status and answer, what the job manager was asked for (None if
    nothing), the syncs asked for, the listeners it was given, and the
    session's manifest afterwards."""
    import zarr

    from cellmap_flow.dashboard.routes.finetune import training

    def run(volume=CROPPED, via_base_path=False, pulled=None, **request):
        corrections = trainable_session(*volume)
        syncs = []

        def sync(force=True):
            syncs.append(force)
            if pulled is None:
                return 0
            manifest = json.loads((corrections / "_virtual_sources.json").read_text())
            zarr.open_group(manifest["volume_zarr_path"])["annotation"]["s0"][:] = pulled
            return 1

        monkeypatch.setattr(training, "sync_all_annotations_from_minio", sync)
        asked, listeners = [], []
        get_session().finetune_job_manager = SimpleNamespace(
            jobs={}, add_listener=listeners.append,
            submit_finetuning_job=lambda **kw: asked.append(kw) or SimpleNamespace(
                job_id="j", output_dir=corrections, lsf_job=None))
        path = corrections.parent.parent if via_base_path else corrections
        response = client.post("/api/finetune/submit",
                               json={"model_name": "m", "corrections_path": str(path), **request})
        return SimpleNamespace(status=response.status_code, body=response.get_json(), sent=asked[0] if asked else None,
                               syncs=syncs, corrections=corrections, listeners=listeners,
                               manifest=json.loads((corrections / "_virtual_sources.json").read_text()))

    return run


@pytest.mark.parametrize("volume, request_data, sent", [
    pytest.param(CROPPED, {}, dict(mask_unannotated=False, loss_type="mse"), id="imported crops are dense"),
    # A distance target needs 3D boundaries, which scribbles do not have. Only
    # the side of 0.5 is asked for: the form's margin 0.3 pushed painted voxels
    # to 0.7, 2.5 voxels inside a distance model's boundary, and its
    # distillation of 0.01 held nothing else in place.
    pytest.param(PAINTED, {"output_type": "distance", "loss_type": "margin", "margin": 0.3, "distillation_lambda": 0.01},
                 dict(mask_unannotated=True, loss_type="margin", output_type="binary", margin=0.5,
                      distillation_lambda=0.5), id="a painted session"),
    pytest.param(PAINTED, {"output_type": "distance", "loss_type": "margin", "distillation_lambda": 10},
                 dict(margin=0.5, distillation_lambda=10), id="a painted session asking for more distillation"),
    pytest.param(STROKE_BESIDE, {}, dict(mask_unannotated=True, loss_type="margin", distillation_lambda=0.5),
                 id="a stroke beside the crops"),
    # The CLI takes a distance target only with bce, and a soft target is not smoothed.
    pytest.param(CROPPED, {"output_type": "distance", "loss_type": "margin"},
                 dict(mask_unannotated=False, loss_type="bce", label_smoothing=0.0, output_type="distance"),
                 id="a distance model on imported crops"),
])
def test_submit_trains_scribbles_as_scribbles(submit, volume, request_data, sent):
    """Scribbles were detected from per-chunk extracts that no session has any
    more, so it never fired: painted sessions trained as dense labels, with
    unannotated voxels taken for background. It is read from the volume now."""
    job = submit(volume, **request_data)
    assert {key: job.sent[key] for key in sent} == sent


def test_submit_says_when_a_distance_model_is_trained_as_binary(submit):
    """The form keeps showing margin 0.3 and distillation 0.01, so the answer says what was used."""
    job = submit(PAINTED, output_type="distance", loss_type="margin", margin=0.3, distillation_lambda=0.01)
    assert "margin 0.5 and distillation 0.5" in job.body["note"]


def test_submit_reads_the_strokes_still_in_minio(submit):
    """Whether a session is sparse was read from the volume on disk, which lags
    the browser's strokes (they go to MinIO) by up to the periodic sync's 30 s:
    a stroke painted just before Submit trained as dense labels."""
    job = submit(CROPPED, pulled=STROKE_BESIDE[0])
    assert job.syncs == [False], "a diff of the chunks"
    assert job.sent["mask_unannotated"] is True


def test_a_submit_sends_the_job_manager_the_forms_defaults(submit):
    """With the affinity offsets from the model's script: output_type and
    offsets are sent only for affinity models, which say so there."""
    job = submit()
    assert job.body == {"success": True, "job_id": "j", "lsf_job_id": None, "output_dir": str(job.corrections),
                        "tensorboard_command": f"tensorboard --logdir {job.corrections.parent}",
                        "output_type": "affinities", "message": "Finetuning job submitted successfully"}
    assert job.sent == dict(
        model_config=get_session().models_config[0], corrections_path=job.corrections, lora_r=8, num_epochs=10,
        batch_size=2, learning_rate=1e-4, output_base=job.corrections.parent, checkpoint_path_override=None,
        auto_serve=True,
        mask_unannotated=False, loss_type="mse", label_smoothing=0.1, distillation_lambda=None,
        distillation_scope="unlabeled", margin=0.3, balance_classes=False, augment=False, queue="gpu_h100",
        charge_group="my_lab", walltime="10:00", output_type="affinities", select_channel=None,
        offsets="[[1, 0, 0], [0, 1, 0]]",
    )


@pytest.mark.parametrize("geometry, offsets", [
    pytest.param(SimpleNamespace(channels="mito_aff"), "[[1, 0, 0]]", id="one channel, named by a string"),
    pytest.param(SimpleNamespace(channel_names=("x_aff", "y_aff")), "[[1, 0, 0], [0, 1, 0]]",
                 id="a ModelGeometry's channel names"),
])
def test_an_affinity_model_is_told_by_its_channel_names(submit, tmp_path, monkeypatch, geometry, offsets):
    """For a model whose script names no offsets. A string was iterated letter
    by letter, and a ModelGeometry's channel_names were not read: both
    trained as binary."""
    from cellmap_flow.models import geometry_cache

    (tmp_path / "plain.py").write_text("model = None\n")
    monkeypatch.setattr(get_session(), "models_config", [_Script("m", str(tmp_path / "plain.py"))])
    monkeypatch.setattr(geometry_cache, "resolve_model_geometry", lambda name, config: geometry)
    job = submit()
    assert (job.sent["output_type"], job.sent["offsets"]) == ("affinities", offsets)


@pytest.mark.parametrize("dashboards, request_data, billed", [
    pytest.param("my_lab", {}, "my_lab", id="the dashboard's"),
    pytest.param("my_lab", {"charge_group": "other_lab"}, "other_lab", id="the request's"),
    pytest.param("", {}, "cellmap", id="the site's, when the dashboard has none"),
])
def test_submit_bills_the_dashboards_charge_group(submit, dashboards, request_data, billed):
    """Every finetune job billed "cellmap", the job manager's default."""
    launcher_settings().charge_group = dashboards
    assert submit(**request_data).sent["charge_group"] == billed


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


def test_submit_gives_the_manifest_the_dashboards_chain(submit):
    """The trainer normalizes its input as the dashboard's servers do, and the
    finetuned model's YAML postprocesses as they do, only if the manifest
    carries the session's chain."""
    from cellmap_flow.pipeline_spec import PipelineSpec

    norm = [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}]
    post = [{"name": "SigmoidPostprocessor"}]
    get_session().set_pipeline(PipelineSpec(norm, post))
    job = submit()
    spec = get_session().pipeline_spec
    assert (job.manifest["input_norm"], job.manifest["postprocess"]) == (list(spec.input_norm), list(spec.postprocess))
    assert (job.manifest["input_norm"], job.manifest["postprocess"]) == (norm, post)


@pytest.mark.parametrize("request_data", [
    pytest.param({"rehearsal_fraction": 1.5}, id="a fraction out of range"),
    pytest.param({"rehearsal_fraction": "abc"}, id="a fraction that is not a number"),
    # The manifest was given the dashboard's chains before the counts were read.
    pytest.param({"num_epochs": "ten"}, id="a count that is not a number"),
])
def test_a_refused_submit_leaves_the_session_as_it_was(submit, request_data):
    job = submit(**request_data)
    assert job.status == 400 and job.sent is None
    assert set(job.manifest) == {"kind", "volume_zarr_path", "raw_dataset_path"}


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
    get_session().annotation_volumes = {"vol": volume}
    get_session().finetune_job_manager = SimpleNamespace(jobs={}, add_listener=lambda listener: None,
                                                         submit_finetuning_job=lambda **kw: SimpleNamespace(
                                                             job_id="j", output_dir=corrections, lsf_job=None))
    client.post("/api/finetune/submit", json={"model_name": "m", "corrections_path": str(corrections)})

    manifest = corrections / "_virtual_sources.json"
    assert manifest.exists() == written
    if written:
        assert json.loads(manifest.read_text())["input_size_voxels"] == [178] * 3


@pytest.fixture
def restart(client, local_jobs, session, annotation_volume, monkeypatch):
    """``restart(pulled=0, job_status="WAITING_FOR_RESTART", job_params={},
    volume=None, **request)``: POST /api/finetune/job/<id>/restart for a job
    submitted through the dashboard's own job manager, now in ``job_status``
    and with ``job_params`` over its params, on a session whose manifest names
    a volume holding ``volume`` (labels, crops) if given, with MinIO sync
    pulling ``pulled`` volumes. Returns the response's status and body, the
    syncs asked for, what the trainer is sent, and the session."""
    from cellmap_flow.dashboard.routes.finetune import training
    from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager
    from cellmap_flow.finetune.job_manager.state import JobStatus

    def run(pulled=0, job_status="WAITING_FOR_RESTART", job_params={}, volume=None, **request):
        if volume is not None:
            labels, crops = volume
            volume = annotation_volume(labels, crops=crops)
            base = session(manifest={"kind": "volume_zarr_v1", "volume_zarr_path": volume.path,
                                     "raw_dataset_path": volume.raw})
        else:
            base = session()
        manager = get_session().finetune_job_manager  # made when first asked for, as in the dashboard
        assert isinstance(manager, FinetuneJobManager)
        job = manager.submit_finetuning_job(model_config=get_session().models_config[0],
                                            corrections_path=base / "corrections", output_base=base)
        job.status = JobStatus(job_status)
        job.params.update(job_params)
        record = SimpleNamespace(syncs=[], sent=[], base=base)
        monkeypatch.setattr(training, "sync_all_annotations_from_minio",
                            lambda force=True: record.syncs.append(force) or pulled)
        monkeypatch.setattr(manager, "restart_finetuning_job",
                            lambda job_id, updated_params: record.sent.append(updated_params) or job)
        response = client.post(f"/api/finetune/job/{job.job_id}/restart", json=request)
        record.status, record.body = response.status_code, response.get_json()
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
                         "distillation_all_voxels": True, "loss_type": "margin",
                         # The job's target, and the session's sparsity (see the test below).
                         "output_type": "binary", "label_smoothing": 0.0, "mask_unannotated": False}]


# What the Finetune tab sends on Restart with its form at the defaults.
FORM = {"lora_r": 8, "num_epochs": 10, "batch_size": 2, "learning_rate": 1e-4, "loss_type": "margin",
        "distillation_lambda": 0.01, "distillation_scope": "unlabeled", "balance_classes": False,
        "augment": False, "label_smoothing": 0.1, "margin": 0.3}


@pytest.mark.parametrize("job_params, volume, request_data, sent", [
    pytest.param({"output_type": "distance"}, None, FORM,
                 dict(output_type="distance", loss_type="bce", label_smoothing=0.0, mask_unannotated=False),
                 id="a distance model, from the unchanged form"),
    pytest.param({"output_type": "binary"}, PAINTED, {**FORM, "loss_type": "mse"},
                 dict(loss_type="margin", distillation_lambda=0.5, mask_unannotated=True),
                 id="mse on scribbles"),
    # Strokes painted since submit, over a distance model's imported crops.
    pytest.param({"output_type": "distance"}, STROKE_BESIDE, FORM,
                 dict(output_type="binary", loss_type="margin", distillation_lambda=0.5, margin=0.5,
                      mask_unannotated=True),
                 id="a distance model whose session has become sparse"),
])
def test_a_restart_trains_what_submit_would(restart, job_params, volume, request_data, sent):
    """Restart sent the form as it was, undoing what submit had chosen: a
    distance model's next iteration got the form's margin loss, which the
    trainer refuses for a distance target, so every restart failed until the
    user picked BCE by hand."""
    run = restart(job_params=job_params, volume=volume, **request_data)
    assert run.status == 200
    assert {key: run.sent[0].get(key) for key in sent} == sent


@pytest.mark.parametrize("job_status, request_data, status, error", [
    # It failed with a 500, as if the dashboard were broken.
    pytest.param("WAITING_FOR_RESTART", {"rehearsal_fraction": "abc"}, 400,
                 "rehearsal_fraction must be a number between 0 and 1", id="an override that is not one"),
    # The Finetune tab offered Restart for a finished job that had been serving.
    pytest.param("COMPLETED", {"patches_per_epoch": 7}, 409,
                 "is in state COMPLETED - can only restart a job that is waiting", id="a job that has finished"),
])
def test_a_refused_restart_changes_nothing(restart, job_status, request_data, status, error):
    """A restart the job could not take still wrote the form's settings into the
    session's manifest, which later submits inherited, and pulled from MinIO,
    and then answered 500."""
    run = restart(job_status=job_status, **request_data)
    assert (run.status, run.body["success"]) == (status, False) and error in run.body["error"]
    assert run.sent == [] and run.syncs == []
    manifest = json.loads((run.base / "corrections" / "_virtual_sources.json").read_text())
    assert not {"patches_per_epoch", "rehearsal_fraction"} & set(manifest)


@pytest.fixture
def jobs_list(client, session, monkeypatch):
    """GET /api/finetune/jobs from a dashboard started after the job in
    ``session()``: the output path is only in the user prefs. Returns the
    sessions the job manager was asked to look in, and the listeners it was
    given, in the order it got them."""
    from cellmap_flow.dashboard.routes.finetune import common

    base = session()
    asked = SimpleNamespace(base=base, calls=[])
    get_session().finetune_job_manager = SimpleNamespace(
        jobs={}, list_jobs=lambda: [], rehydrate_session=lambda path: asked.calls.append(("look in", path)),
        add_listener=lambda listener: asked.calls.append(("listener", type(listener))))
    monkeypatch.setattr(common, "load_user_prefs", lambda: {"outputPath": str(base.parent)})
    assert client.get("/api/finetune/jobs").get_json()["success"]
    return asked


def test_the_jobs_list_looks_for_jobs_in_the_saved_output_path(jobs_list):
    """After a dashboard restart this dashboard has made no sessions yet; the
    output path saved in the user prefs is where its jobs are."""
    assert [path for what, path in jobs_list.calls if what == "look in"] == [str(jobs_list.base)]


def test_the_viewer_follows_the_jobs_the_dashboard_finds(jobs_list):
    """The job manager has no viewer code: the dashboard's listener is what
    adds a job's layer, and it has to be there before a found job's monitor
    starts."""
    from cellmap_flow.dashboard.finetune_layers import FinetuneLayerListener

    assert jobs_list.calls[0] == ("listener", FinetuneLayerListener)


def test_the_viewer_follows_the_jobs_the_dashboard_submits(submit):
    from cellmap_flow.dashboard.finetune_layers import FinetuneLayerListener

    assert [type(listener) for listener in submit().listeners] == [FinetuneLayerListener]


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


# --- What every route answers ----------------------------------------------------------
#
# The finetune tab shows these answers, error messages included, and scripts
# call the /api/viewer ones. Paths are written <tmp>, a session's timestamp
# <session> and a time <time>.

STOP = ("Stop requested. Training will exit after the current epoch; the inference server will "
        "then start so you can restart with updated parameters.")
NOT_RESTARTABLE = ("Job j is in state RUNNING - can only restart a job that is waiting for a restart "
                   "(its training iteration has finished or diverged)")
NO_GOOD_REGIONS_SESSION = ("Could not work out where to save good regions for this session, so the mark was "
                           "discarded. Create or resume an annotation volume first.")
NO_CORRECTIONS_PATH = "corrections_path is required. Please specify the output path where annotation crops are saved."
CROPS = "crops:\n  - path: /data/crop.zarr\n"


def _refused(error):
    return {"success": False, "error": error}


class _Starting(str):
    """A message that starts with this; the rest is a library's."""

    def __eq__(self, other):
        return isinstance(other, str) and other.startswith(self)

    __hash__ = str.__hash__


# make_job's job "j" as the routes list it.
JOB = {"job_id": "j", "lsf_job_id": None, "model_name": "m", "output_dir": "<tmp>/runs/r", "params": {},
       "status": "RUNNING", "created_at": "<time>", "log_file": "<tmp>/runs/r/training_log.txt",
       "finetuned_model_name": None, "model_yaml_path": None, "current_epoch": 0, "total_epochs": 10,
       "inference_server_url": None, "inference_server_ready": False, "corrections_path": None, "loss": None,
       "progress_percent": 0.0}


class _Killable:
    job_id = "4242"

    def kill(self):
        pass


GEOMETRY = SimpleNamespace(read_shape=[96] * 3, write_shape=[64] * 3, input_voxel_size=[8] * 3,
                           output_voxel_size=[16] * 3, output_channels=1)


def _given(situation, world, monkeypatch):
    """Change the world ``routes`` sets up to ``situation``."""
    from cellmap_flow.finetune.job_manager.state import JobStatus
    from cellmap_flow.models import geometry_cache

    job = world.job
    if situation == "no viewer":
        get_session().viewer = None
    elif situation == "the viewer has a position":
        with get_session().viewer.txn() as s:
            s.position = [1, 2, 3]
    elif situation == "no models":
        get_session().models_config = []
    elif situation == "its geometry is known":
        monkeypatch.setattr(geometry_cache, "model_geometry_config", lambda name: GEOMETRY)
    elif situation == "a saved pipeline":
        get_session().builder_model_configs = {"m": {"write_shape": [64] * 3, "output_voxel_size": [16] * 3,
                                                     "output_channels": 2}}
    elif situation == "an empty session":
        (world.tmp / "s" / "corrections").mkdir(parents=True)
    elif situation == "no output dir":
        job.output_dir = world.tmp / "gone"
    elif situation == "on LSF":
        job.lsf_job = _Killable()
    elif situation == "waiting":
        job.status = JobStatus.WAITING_FOR_RESTART
    else:
        assert situation is None, situation


@pytest.fixture
def routes(client, viewer, make_job, tmp_path, monkeypatch):
    """``routes(method, url, body=None, situation=None)``: (status, JSON) of a
    request. The world: model "m", a viewer, no annotation session, MinIO not
    running, the user prefs under tmp_path, and a job manager holding job "j"
    (running, not on LSF, no log yet). ``situation`` changes it (see
    _given)."""
    import re

    from cellmap_flow.dashboard.routes.finetune import common
    from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager

    monkeypatch.setattr(common, "USER_PREFS_FILE", str(tmp_path / "user_prefs.json"))
    world = SimpleNamespace(tmp=tmp_path, job=make_job())
    manager = FinetuneJobManager()
    manager.jobs[world.job.job_id] = world.job
    monkeypatch.setattr(get_session(), "finetune_job_manager", manager)
    monkeypatch.setattr(get_session(), "minio_state", dict(get_session().minio_state, process=None, ip=None, port=None))

    def names(value):
        if isinstance(value, dict):
            return {k: names(v) for k, v in value.items()}
        if isinstance(value, list):
            return [names(v) for v in value]
        if not isinstance(value, str):
            return value
        value = re.sub(r"\d{4}-\d\d-\d\dT[\d:.]+", "<time>", value.replace(str(tmp_path), "<tmp>"))
        return re.sub(r"\d{8}_\d{6}", "<session>", value)

    def request(method, url, body=None, situation=None):
        _given(situation, world, monkeypatch)
        if isinstance(body, dict):
            body = {k: v.replace("<tmp>", str(tmp_path)) if isinstance(v, str) else v for k, v in body.items()}
            response = getattr(client, method)(url, json=body)
        else:
            response = getattr(client, method)(url, data=body, content_type="text/plain" if body else None)
        return response.status_code, names(response.get_json(silent=True))

    request.job = world.job
    return request


ANSWERS = [
    # training
    pytest.param("get", "/api/finetune/jobs", None, None, 200, {"success": True, "jobs": [JOB]},
                 id="jobs"),
    pytest.param("get", "/api/finetune/job/j/status", None, None, 200, {"success": True, **JOB},
                 id="status"),
    pytest.param("get", "/api/finetune/job/nope/status", None, None, 404, _refused("Job not found"),
                 id="status of an unknown job"),
    pytest.param("get", "/api/finetune/job/j/logs", None, None, 200,
                 {"success": True, "logs": "Log file not yet created...", "offset": 0}, id="logs before the log"),
    pytest.param("get", "/api/finetune/job/nope/logs", None, None, 404, _refused("Job not found"),
                 id="logs of an unknown job"),
    pytest.param("post", "/api/finetune/job/j/cancel", None, "on LSF", 200,
                 {"success": True, "message": "Job j cancelled"}, id="cancel"),
    pytest.param("post", "/api/finetune/job/j/cancel", None, None, 400, _refused("Failed to cancel job"),
                 id="cancel a job not on LSF"),
    pytest.param("post", "/api/finetune/job/nope/cancel", None, None, 400, _refused("Failed to cancel job"),
                 id="cancel an unknown job"),
    pytest.param("post", "/api/finetune/job/j/stop-early", None, None, 200, {"success": True, "message": STOP},
                 id="stop early"),
    pytest.param("post", "/api/finetune/job/j/stop-early", None, "no output dir", 400,
                 _refused("Job output dir missing: <tmp>/gone"), id="stop early without an output dir"),
    pytest.param("post", "/api/finetune/job/nope/stop-early", None, None, 404, _refused("Job nope not found"),
                 id="stop an unknown job early"),
    pytest.param("post", "/api/finetune/job/j/restart", {}, "waiting", 200,
                 {"success": True, "job_id": "j", "annotations_synced": 0,
                  "message": "Restart request sent. No new annotations to pull; training will restart on the "
                             "same GPU."}, id="restart"),
    pytest.param("post", "/api/finetune/job/j/restart", {}, None, 409, _refused(NOT_RESTARTABLE),
                 id="restart a job that is not waiting"),
    pytest.param("post", "/api/finetune/job/nope/restart", {}, None, 404, _refused("Job nope not found"),
                 id="restart an unknown job"),
    pytest.param("post", "/api/finetune/submit", {}, None, 400, _refused("model_name is required"),
                 id="submit without a model"),
    pytest.param("post", "/api/finetune/submit", {"model_name": "m"}, None, 400, _refused(NO_CORRECTIONS_PATH),
                 id="submit without a corrections path"),
    pytest.param("post", "/api/finetune/submit", {"model_name": "x", "corrections_path": "<tmp>/c"}, None, 404,
                 _refused("Model x not found"), id="submit an unknown model"),
    pytest.param("post", "/api/finetune/submit",
                 {"model_name": "x", "corrections_path": "<tmp>/c", "num_epochs": "ten"}, None, 400,
                 _refused("num_epochs must be a whole number, got 'ten'"), id="submit a bad number"),
    pytest.param("post", "/api/finetune/submit", {"model_name": "m", "corrections_path": "<tmp>/none"}, None, 400,
                 _refused("Corrections path does not exist: <tmp>/none/<session>/corrections. Please create "
                          "annotation crops first."), id="submit without corrections"),
    # annotation volumes and the user's settings
    pytest.param("get", "/api/finetune/models", None, "a saved pipeline", 200,
                 {"models": [{"name": "m", "write_shape": [64] * 3, "output_voxel_size": [16] * 3,
                              "output_channels": 2}], "selected_model": "m"}, id="models"),
    pytest.param("post", "/api/finetune/create-volume", {"model_name": "m"}, "no models", 400,
                 _refused("No models loaded"), id="create a volume without models"),
    pytest.param("post", "/api/finetune/create-volume", {"model_name": "x"}, None, 404,
                 _refused("Model x not found"), id="create a volume for an unknown model"),
    pytest.param("post", "/api/finetune/create-volume", {}, None, 400, _refused("model_name is required"),
                 id="create a volume without a model"),
    pytest.param("post", "/api/finetune/create-volume", {"model_name": "m"}, "its geometry is known", 400,
                 _refused("No dataset path configured"), id="create a volume without data"),
    pytest.param("get", "/api/finetune/user-prefs", None, None, 200, {"success": True, "prefs": {}},
                 id="user prefs"),
    pytest.param("post", "/api/finetune/user-prefs", {"outputPath": "/out", "unset": None}, None, 200,
                 {"success": True, "prefs": {"outputPath": "/out"}}, id="set user prefs"),
    pytest.param("post", "/api/finetune/load-crops", {}, None, 400, _refused("Missing 'yaml' field"),
                 id="load crops without a yaml"),
    pytest.param("post", "/api/finetune/load-crops", {"yaml": CROPS}, None, 400, _refused("Missing 'model_name' field"),
                 id="load crops without a model"),
    pytest.param("post", "/api/finetune/load-crops", {"yaml": "crops: []", "model_name": "m"}, None, 400,
                 _refused("No crops listed in YAML"), id="load no crops"),
    pytest.param("post", "/api/finetune/load-crops", {"yaml": "crops: [\n", "model_name": "m"}, None, 400,
                 _refused(_Starting("YAML parse error: while parsing a flow node")),
                 id="load crops from a yaml that does not parse"),
    pytest.param("post", "/api/finetune/load-crops", {"yaml": "crops:\n  - {}\n", "model_name": "m"}, None, 400,
                 {"success": False, "error": "YAML validation failed",
                  "details": [{"loc": ["crops", 0, "path"], "msg": "Field required"}]},
                 id="load invalid crops, saying where and what but not echoing the input"),
    pytest.param("post", "/api/finetune/load-crops", {"yaml": CROPS, "model_name": "x"}, None, 404,
                 _refused("Model x not found"), id="load crops for an unknown model"),
    pytest.param("post", "/api/finetune/load-crops", {"yaml": CROPS, "model_name": "m"}, None, 400,
                 _refused("No raw dataset path configured"), id="load crops without data"),
    pytest.param("get", "/api/finetune/load-crops-progress", None, None, 400,
                 _refused("Missing 'load_id' query param"), id="crop progress without an id"),
    pytest.param("get", "/api/finetune/load-crops-progress?load_id=nope", None, None, 404,
                 _refused("Unknown load_id nope"), id="crop progress of an unknown load"),
    pytest.param("get", "/api/finetune/read-yaml", None, None, 400, _refused("Missing 'path' query param"),
                 id="read a yaml without a path"),
    pytest.param("post", "/api/finetune/list-existing-sessions", {}, None, 400, _refused("output_path required"),
                 id="list sessions without a path"),
    pytest.param("post", "/api/finetune/list-existing-sessions", {"output_path": "<tmp>/none"}, None, 200,
                 {"success": True, "sessions": []}, id="list the sessions of a path that is not there"),
    pytest.param("post", "/api/finetune/load-existing-volume", {"output_path": "<tmp>/o"}, None, 400,
                 _refused("source_session_path and output_path required"), id="resume without a session"),
    pytest.param("post", "/api/finetune/load-existing-volume",
                 {"source_session_path": "<tmp>/s", "output_path": "<tmp>/o"}, None, 404,
                 _refused("No corrections found in <tmp>/s"), id="resume a session without corrections"),
    pytest.param("post", "/api/finetune/load-existing-volume",
                 {"source_session_path": "<tmp>/s", "output_path": "<tmp>/o"}, "an empty session", 404,
                 _refused("No annotation volume found in <tmp>/s/corrections"), id="resume a session without a volume"),
    pytest.param("get", "/api/finetune/load-existing-volume-progress", None, None, 400,
                 _refused("Missing 'load_id' query param"), id="resume progress without an id"),
    pytest.param("get", "/api/finetune/load-existing-volume-progress?load_id=nope", None, None, 404,
                 _refused("Unknown load_id nope"), id="resume progress of an unknown load"),
    # the viewer's overlays
    pytest.param("post", "/api/finetune/add-to-viewer", {"crop_id": "c", "minio_url": "http://m:9000/a/c.zarr"},
                 None, 200, {"success": True, "message": "Layer added to viewer", "layer_name": "annotation_c"},
                 id="add a volume's layer"),
    pytest.param("post", "/api/finetune/add-to-viewer", {"crop_id": "c"}, "no viewer", 400,
                 _refused("Viewer not initialized"), id="add a volume's layer without a viewer"),
    pytest.param("post", "/api/finetune/sync-annotations", {}, None, 400, _refused("MinIO not initialized"),
                 id="sync without MinIO"),
    pytest.param("post", "/api/finetune/refresh-annotated-regions", {}, None, 200, {"success": True, "count": 0},
                 id="refresh the annotated regions"),
    pytest.param("post", "/api/finetune/refresh-annotated-regions", {}, "no viewer", 400,
                 _refused("Viewer not initialized"), id="refresh the annotated regions without a viewer"),
    pytest.param("get", "/api/finetune/good-regions", None, None, 200, {"success": True, "regions": [], "count": 0},
                 id="good regions"),
    pytest.param("post", "/api/finetune/good-regions/mark-view", {}, "the viewer has a position", 409,
                 _refused(NO_GOOD_REGIONS_SESSION), id="mark a good region without a session"),
    pytest.param("post", "/api/finetune/good-regions/mark-view", {}, None, 400, _refused("Viewer has no position"),
                 id="mark a good region where the viewer has no position"),
    pytest.param("post", "/api/finetune/good-regions/mark-view", None, "no viewer", 400,
                 _refused("Viewer not initialized"), id="mark a good region without a viewer"),
    pytest.param("post", "/api/finetune/good-regions/delete", {"id": "nope"}, None, 404, _refused("No region nope"),
                 id="delete an unknown good region"),
    pytest.param("post", "/api/viewer/add-image-layer", {"name": "n"}, None, 400, _refused("Missing path or name"),
                 id="add an image layer without a path"),
    pytest.param("post", "/api/viewer/add-segmentation-layer", {"path": "/p", "name": "n"}, "no viewer", 400,
                 _refused("viewer not initialized"), id="add a segmentation layer without a viewer"),
    pytest.param("post", "/api/viewer/remove-layer", {}, None, 400, _refused("Missing name"),
                 id="remove a layer without a name"),
    pytest.param("post", "/api/viewer/rename-layer", {"old_name": "a", "new_name": "a"}, None, 200,
                 {"success": True, "renamed": False, "old_name": "a", "new_name": "a", "reload_page": False},
                 id="rename a layer to its name"),
]


@pytest.mark.parametrize("method, url, body, situation, status, answer", ANSWERS)
def test_what_each_route_answers(routes, method, url, body, situation, status, answer):
    assert routes(method, url, body, situation) == (status, answer)


def test_a_stop_early_leaves_the_trainer_a_signal(routes):
    routes("post", "/api/finetune/job/j/stop-early")
    signal = json.loads((routes.job.output_dir / "stop_signal.json").read_text())
    assert signal == {"requested_at": ANY, "reason": "user_requested_stop_early"}


@pytest.mark.parametrize("start, body, progress, phase", [
    pytest.param("/api/finetune/load-crops", {"yaml": CROPS, "model_name": "x"}, "/api/finetune/load-crops-progress",
                 "setup", id="loading crops"),
    pytest.param("/api/finetune/load-existing-volume", {}, "/api/finetune/load-existing-volume-progress",
                 "starting", id="resuming a session"),
])
def test_a_long_request_reports_its_progress_as_it_goes(routes, start, body, progress, phase):
    """The tab polls under the load_id it sent; each step is recorded as the
    request reaches it, up to the one that failed here."""
    assert routes("post", start, {"load_id": "L1", **body})[0] in (400, 404)
    status, answer = routes("get", f"{progress}?load_id=L1")
    assert status == 200 and answer["success"]
    assert {k: answer["progress"][k] for k in ("phase", "done")} == {"phase": phase, "done": False}
    assert {"created_at", "updated_at"} <= set(answer["progress"])
