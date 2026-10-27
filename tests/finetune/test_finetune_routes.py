"""The dashboard's finetune routes, through the Flask client: what submit and
restart send the job manager and write into the session's manifest, finding
the session after a dashboard restart, resuming a session, MinIO URLs behind
a proxy, and the status the job routes answer with."""

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


@pytest.fixture
def submit(client, trainable_session, monkeypatch):
    """``submit(volume=CROPPED, via_base_path=False, pulled=None, **request)``:
    POST /api/finetune/submit for model "m", on a session over ``volume``,
    with MinIO sync pulling the labels ``pulled`` into the volume, if given.
    Returns the status and answer, what the job manager was asked for (None if
    nothing), the syncs asked for, and the session's manifest afterwards."""
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
        asked = []
        get_session().finetune_job_manager = SimpleNamespace(
            jobs={}, add_listener=lambda listener: None,
            submit_finetuning_job=lambda **kw: asked.append(kw) or SimpleNamespace(
                job_id="j", output_dir=corrections, lsf_job=None))
        path = corrections.parent.parent if via_base_path else corrections
        response = client.post("/api/finetune/submit",
                               json={"model_name": "m", "corrections_path": str(path), **request})
        return SimpleNamespace(status=response.status_code, body=response.get_json(), sent=asked[0] if asked else None,
                               syncs=syncs, corrections=corrections,
                               manifest=json.loads((corrections / "_virtual_sources.json").read_text()))

    return run


@pytest.mark.parametrize("volume, request_data, sent", [
    pytest.param(CROPPED, {}, dict(mask_unannotated=False, loss_type="margin"), id="imported crops are dense"),
    # A distance target needs 3D boundaries, which scribbles do not have; they
    # bound the distance instead. The form's distillation of 0.01 held nothing
    # else in place.
    pytest.param(PAINTED, {"output_type": "distance", "loss_type": "margin", "margin": 0.3, "distillation_lambda": 0.01},
                 dict(mask_unannotated=True, loss_type="interval", output_type="distance", label_smoothing=0.0,
                      distillation_lambda=0.5), id="a painted session"),
    pytest.param(STROKE_BESIDE, {"loss_type": "mse"}, dict(mask_unannotated=True, loss_type="margin", distillation_lambda=0.5),
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
    assert {key: getattr(job.sent["options"], key) for key in sent} == sent


def test_submit_reads_the_strokes_still_in_minio(submit):
    """Whether a session is sparse was read from the volume on disk, which lags
    the browser's strokes (they go to MinIO) by up to the periodic sync's 30 s:
    a stroke painted just before Submit trained as dense labels."""
    job = submit(CROPPED, pulled=STROKE_BESIDE[0])
    assert job.syncs == [False], "a diff of the chunks"
    assert job.sent["options"].mask_unannotated is True


def test_an_affinity_model_is_told_by_the_offsets_its_script_names(submit):
    """With the affinity offsets from the model's script: output_type and
    offsets are sent only for affinity models, which say so there."""
    job = submit()
    assert (job.sent["options"].output_type, job.sent["options"].offsets) == ("affinities", "[[1, 0, 0], [0, 1, 0]]")


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


def test_submit_after_a_dashboard_restart_finds_the_session_on_disk(submit):
    """The dashboard forgot its sessions and made a new, empty one for the base
    path: "Corrections path does not exist". The newest session there with
    something to train on is used instead."""
    job = submit(via_base_path=True)
    assert job.sent["corrections_path"] == job.corrections


@pytest.fixture
def restart(client, local_jobs, session, monkeypatch):
    """``restart(pulled=0, **request)``: POST /api/finetune/job/<id>/restart for
    a job submitted through the dashboard's own job manager and now waiting
    for a restart, with MinIO sync pulling ``pulled`` volumes. Returns the
    response's body and the syncs asked for."""
    from cellmap_flow.dashboard.routes.finetune import training
    from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager
    from cellmap_flow.finetune.job_manager.state import JobStatus

    def run(pulled=0, **request):
        base = session()
        manager = get_session().finetune_job_manager  # made when first asked for, as in the dashboard
        assert isinstance(manager, FinetuneJobManager)
        job = manager.submit_finetuning_job(model_config=get_session().models_config[0],
                                            corrections_path=base / "corrections", output_base=base)
        job.status = JobStatus.WAITING_FOR_RESTART
        record = SimpleNamespace(syncs=[])
        monkeypatch.setattr(training, "sync_all_annotations_from_minio",
                            lambda force=True: record.syncs.append(force) or pulled)
        monkeypatch.setattr(manager, "restart_finetuning_job", lambda job_id, updated_params: job)
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


class _Declares:
    """A model that says ``declared`` of its geometry, and that is never to be built."""

    def __init__(self, name, declared=None):
        self.name, self.declared = name, declared

    def declared_geometry(self):
        return self.declared

    @property
    def config(self):
        raise AssertionError(f"{self.name} was built to read its geometry")


def test_the_model_list_takes_what_a_model_declares_and_never_builds_it(client, monkeypatch):
    """The Finetune tab asks every two seconds while a model's server is not
    up, and each answer built the model in the dashboard to read its sizes:
    25 s on its CPU for a 3 GB TorchScript."""
    from cellmap_flow.dashboard.services import launch

    declared = {"write_shape": [960] * 3, "output_voxel_size": [16] * 3, "output_channels": 3}
    monkeypatch.setattr(get_session(), "models_config",
                        [_Declares("onnx", declared), _Declares("starting"), _Declares("stopped")])
    monkeypatch.setattr(launch, "running_names", lambda: {"onnx", "starting"})
    models = {m["name"]: m for m in client.get("/api/finetune/models").get_json()["models"]}
    assert {k: models["onnx"][k] for k in declared} == declared
    # Without a geometry, why not: its server is not up yet, or it is not running.
    assert set(models["starting"]) == set(models["stopped"]) == {"name", "status", "finetune_modes"}
    assert models["starting"]["status"] != models["stopped"]["status"]


ORGANELLES = ["all_mem", "organelle", "mito", "er", "nucleus", "pm", "vs", "ld"]


class _Organelles(_Declares):
    """fly_organelles' eight channels, each a class the binary target trains:
    declared, and named in its metadata.json, as a Hugging Face model's are."""

    def __init__(self, name):
        super().__init__(name, {"write_shape": [448] * 3, "output_voxel_size": [8] * 3, "output_channels": 8,
                                "channels": ORGANELLES})

    def _load_metadata(self):
        return {"channels_names": ORGANELLES}


def test_the_model_list_says_which_models_train_one_channel_at_a_time(client, monkeypatch):
    """An affinity model trains every channel at once; the paint is about one
    of fly_organelles' classes. Told without building either."""
    affinities = {"write_shape": [960] * 3, "output_voxel_size": [16] * 3, "output_channels": 3,
                  "channels": ["mito_aff_z", "mito_aff_y", "mito_aff_x"]}
    monkeypatch.setattr(get_session(), "models_config", [_Declares("setup_54", affinities), _Organelles("organelles")])
    models = {m["name"]: m for m in client.get("/api/finetune/models").get_json()["models"]}
    assert (models["setup_54"]["channel_choice"], models["organelles"]["channel_choice"]) == (False, True)
    assert models["organelles"]["channels"] == ORGANELLES


@pytest.mark.parametrize("pick, refused", [pytest.param({}, True, id="no channel: refused"),
                                           pytest.param({"select_channel": 2}, False, id="a channel: trained")])
def test_submit_asks_which_channel_a_model_of_classes_trains(submit, monkeypatch, pick, refused):
    """Without one, each of the eight was taught the paint: er and nucleus learned "mito"."""
    monkeypatch.setattr(get_session(), "models_config", [_Organelles("organelles")])
    job = submit(model_name="organelles", **pick)
    assert (job.status, job.sent is None) == ((400, True) if refused else (200, False))
    if not refused:
        assert job.sent["options"].select_channel == 2


MINIO = "http://10.0.0.5:9000/annotations/vol-1.zarr"
PROXY ="https://gateway.example.org/minio/annotations/vol-1.zarr"


@pytest.mark.parametrize("template, headers, expected", [
    pytest.param(None, {"X-Forwarded-Host": "gateway.example.org"}, MINIO, id="opt-in only"),
    pytest.param("{proto}://{host}/minio", {}, MINIO, id="direct access: no proxy, no rewrite"),
    pytest.param("{proto}://{host}/minio", {"X-Forwarded-Host": "gateway.example.org, inner",
                                            "X-Forwarded-Proto": "https"},
                 PROXY, id="the first proxy of a chain, and its protocol"),
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
    with Flask(__name__).test_request_context(base_url="http://node:5000", headers=headers):
        assert rewrite_minio_url_for_proxy(MINIO) == expected


class _Killable:
    job_id = "4242"

    def kill(self):
        pass


@pytest.fixture
def routes(client, viewer, make_job, tmp_path, monkeypatch):
    """``routes(method, url, body=None, situation=None)``: the status and
    whether the answer says success. The world: model "m", a viewer, no
    annotation session, MinIO not running, the user prefs under tmp_path, and
    a job manager holding job "j" (running, not on LSF, no log yet).
    ``situation`` changes it: "on LSF" or "waiting" (for a restart)."""
    from cellmap_flow.dashboard.routes.finetune import common
    from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager
    from cellmap_flow.finetune.job_manager.state import JobStatus

    monkeypatch.setattr(common, "USER_PREFS_FILE", str(tmp_path / "user_prefs.json"))
    job = make_job()
    manager = FinetuneJobManager()
    manager.jobs[job.job_id] = job
    monkeypatch.setattr(get_session(), "finetune_job_manager", manager)
    monkeypatch.setattr(get_session(), "minio_state", dict(get_session().minio_state, process=None, ip=None, port=None))

    def request(method, url, body=None, situation=None):
        if situation == "on LSF":
            job.lsf_job = _Killable()
        elif situation == "waiting":
            job.status = JobStatus.WAITING_FOR_RESTART
        if body is None:
            response = getattr(client, method)(url)
        else:
            body = {k: v.replace("<tmp>", str(tmp_path)) if isinstance(v, str) else v for k, v in body.items()}
            response = getattr(client, method)(url, json=body)
        return response.status_code, (response.get_json(silent=True) or {}).get("success")

    request.job = job
    return request


# The finetune tab shows a refusal's error and stops; a 500 would read as the
# dashboard being broken. A refusal's wording is the route's own.
STATUSES = [
    pytest.param("get", "/api/finetune/jobs", None, None, 200, id="jobs"),
    pytest.param("get", "/api/finetune/job/nope/status", None, None, 404, id="status of an unknown job"),
    pytest.param("post", "/api/finetune/job/j/cancel", None, "on LSF", 200, id="cancel"),
    pytest.param("post", "/api/finetune/job/j/stop-early", None, None, 200, id="stop early"),
    pytest.param("post", "/api/finetune/job/nope/stop-early", None, None, 404, id="stop an unknown job early"),
    pytest.param("post", "/api/finetune/job/j/stop-early", None, "waiting", 400, id="stop a job that is not training"),
    pytest.param("post", "/api/finetune/job/j/restart", {}, "waiting", 200, id="restart"),
    pytest.param("post", "/api/finetune/job/j/restart", {}, None, 409, id="restart a job that is not waiting"),
    pytest.param("post", "/api/finetune/submit", {}, None, 400, id="submit without a model"),
    pytest.param("post", "/api/finetune/submit", {"model_name": "x", "corrections_path": "<tmp>/c"}, None, 404,
                 id="submit an unknown model"),
    pytest.param("post", "/api/finetune/load-existing-volume",
                 {"source_session_path": "<tmp>/s", "output_path": "<tmp>/o"}, None, 404,
                 id="resume a session without corrections"),
    pytest.param("post", "/api/finetune/sync-annotations", {}, None, 400, id="sync without MinIO"),
]


@pytest.mark.parametrize("method, url, body, situation, status", STATUSES)
def test_each_route_answers_with_its_status(routes, method, url, body, situation, status):
    assert routes(method, url, body, situation) == (status, status == 200)


def test_a_stop_early_leaves_the_trainer_a_signal(routes):
    routes("post", "/api/finetune/job/j/stop-early")
    signal = json.loads((routes.job.output_dir / "stop_signal.json").read_text())
    assert signal == {"requested_at": ANY, "reason": "user_requested_stop_early"}


@pytest.mark.parametrize("start, body, progress, phase", [
    pytest.param("/api/finetune/load-crops", {"yaml": "crops:\n  - path: /data/crop.zarr\n", "model_name": "x"},
                 "/api/finetune/load-crops-progress", "setup", id="loading crops"),
    pytest.param("/api/finetune/load-existing-volume", {}, "/api/finetune/load-existing-volume-progress",
                 "starting", id="resuming a session"),
])
def test_a_long_request_reports_its_progress_as_it_goes(routes, client, start, body, progress, phase):
    """The tab polls under the load_id it sent; each step is recorded as the
    request reaches it, up to the one that failed here."""
    assert routes("post", start, {"load_id": "L1", **body})[0] in (400, 404)
    response = client.get(f"{progress}?load_id=L1")
    assert response.status_code == 200 and response.get_json()["success"]
    assert {k: response.get_json()["progress"][k] for k in ("phase", "done")} == {"phase": phase, "done": False}
