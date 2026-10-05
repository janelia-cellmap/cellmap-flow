"""The AI-annotate routes, from settings to an accepted (and undone) annotation.

The model is the ``fake`` provider: the pixels darker than the plane's
median within a disk around the click, which over conftest's raw is its
dark disk at the volume's centre. Jobs run in the request that starts them
(``sync_jobs``). The plane is XY at z = 8 unless a test says otherwise, so
the write box is z 8, y and x 0..16.
"""

import base64
import io
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate import secrets, staging
from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.dashboard.routes.finetune import ai_annotate as routes
from cellmap_flow.dashboard.state import get_session
from tests.ai_annotate.conftest import CENTRE_NM

API = "/api/finetune/ai-annotate"
SETTINGS = {"provider": "fake", "model": "fake-threshold", "label_key": "mito"}
PLANE = (8, slice(0, 16), slice(0, 16))
CENTRE = (8, 8, 8)  # in the dark disk: foreground
CORNER = (8, 0, 0)  # bright: background


FAKE_DESTINATION = "nowhere (fake provider, runs locally)"


def _settings(dashboard, **extra):
    """Save the settings; an acknowledgement names the destination the page showed, as the page does."""
    if extra.get("acknowledge"):
        extra.setdefault("destination", FAKE_DESTINATION)
    response = dashboard.post(f"{API}/settings", json={**SETTINGS, **extra})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _run(dashboard, **body):
    return dashboard.post(f"{API}/run", json={"point_nm": CENTRE_NM, "depth_axis": 0, **body})


def _ready(dashboard):
    """Acknowledge, run at the centre and return the ready status."""
    _settings(dashboard, acknowledge=True)
    response = _run(dashboard)
    assert response.status_code == 200, response.get_json()
    status = dashboard.get(f"{API}/status").get_json()
    assert status["status"] == "ready", status
    assert status["annotate_id"] == response.get_json()["annotate_id"]
    return status


def _audit_events(corrections_dir):
    path = f"{corrections_dir}/ai_annotate_log.jsonl"
    with open(path) as f:
        return [json.loads(line)["event"] for line in f]


def _corrections():
    return get_session().annotation_volumes["vol-1"]["corrections_dir"]


# --- Off unless configured -------------------------------------------------------


def test_without_a_config_file_it_is_off_and_says_how_to_turn_it_on(dashboard, ai_config, ai_volume, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setenv("CELLMAP_FLOW_AI_ANNOTATE_CONFIG", str(tmp_path / "missing.yaml"))

    config = dashboard.get(f"{API}/config").get_json()
    assert config["success"] and config["enabled"] is False
    assert "docs/ai_annotate.md" in config["reason"]
    assert config["keybinding"] == "Shift+G"

    for route, body in (("settings", SETTINGS), ("run", {})):
        response = dashboard.post(f"{API}/{route}", json=body)
        assert response.status_code == 403, route
        assert "docs/ai_annotate.md" in response.get_json()["error"]


def test_a_malformed_config_is_reported_without_its_values(dashboard, ai_config, tmp_path):
    path = ai_config()
    path.write_text("enabled: true\nproviders:\n  fake:\n    type: fake\n    models: [fake-threshold]\n"
                    "    api_key: sk-do-not-echo-1234\n")

    config = dashboard.get(f"{API}/config").get_json()

    assert config["enabled"] is False and "api_key" in config["reason"]
    assert "sk-do-not-echo-1234" not in json.dumps(config)


def test_the_config_lists_what_the_page_needs(dashboard, ai_config, ai_volume):
    ai_config(daily_call_limit=7)

    config = dashboard.get(f"{API}/config").get_json()

    assert config["enabled"] is True
    assert config["providers"] == [{"id": "fake", "type": "fake", "models": ["fake-threshold"],
                                    "destination": "nowhere (fake provider, runs locally)"}]
    assert (config["default_provider"], config["daily_call_limit"], config["calls_today"]) == ("fake", 7, 0)
    mito = next(o for o in config["organelles"] if o["key"] == "mito")
    assert mito["name"] == "mitochondria" and "mitochondria" in mito["prompt"]
    assert config["settings"] is None and config["acknowledged"] == []


# --- Settings and the checks before a run --------------------------------------


@pytest.mark.parametrize("body", [
    pytest.param({**SETTINGS, "model": "gemini-3-pro-image"}, id="model-not-listed"),
    pytest.param({**SETTINGS, "provider": "vertex"}, id="provider-not-listed"),
])
def test_a_provider_or_model_the_config_does_not_list_is_refused(dashboard, ai_config, ai_volume, body):
    ai_config()

    response = dashboard.post(f"{API}/settings", json=body)

    assert response.status_code == 400 and not response.get_json()["success"]
    assert get_session().ai_annotate["settings"] is None


def test_settings_bind_shift_g_in_the_viewer(dashboard, ai_config, ai_volume, viewer):
    ai_config()
    _settings(dashboard)
    _settings(dashboard)  # once is enough; again changes nothing

    assert viewer.config_state.state.input_event_bindings.viewer["shift+keyg"] == routes.ACTION
    assert get_session().ai_annotate["binding_registered_for"] is viewer


def test_an_unedited_prompt_is_stored_as_no_override(dashboard, ai_config, ai_volume):
    ai_config()
    mito = next(o for o in dashboard.get(f"{API}/config").get_json()["organelles"] if o["key"] == "mito")

    assert _settings(dashboard, prompt=mito["prompt"])["settings"]["prompt"] is None
    assert _settings(dashboard, prompt="Paint the mitochondria red.")["settings"]["prompt"] == \
        "Paint the mitochondria red."


def test_a_run_needs_settings_first(dashboard, ai_config, ai_volume):
    ai_config()

    response = _run(dashboard)

    assert response.status_code == 400 and "provider" in response.get_json()["error"]


def test_a_run_before_acknowledging_the_destination_is_refused(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()
    assert _settings(dashboard)["acknowledged"] == []

    response = _run(dashboard)

    assert response.status_code == 409
    body = response.get_json()
    assert body["needs_acknowledgement"] is True and body["provider"] == "fake"
    assert body["destination"] == "nowhere (fake provider, runs locally)"
    assert dashboard.get(f"{API}/status").get_json()["status"] == "idle"

    assert _settings(dashboard, acknowledge=True)["acknowledged"] == ["fake"]
    assert dashboard.get(f"{API}/config").get_json()["acknowledged"] == ["fake"]
    assert _run(dashboard).status_code == 200


def test_an_acknowledgement_of_another_destination_than_the_current_one_is_refused(
    dashboard, ai_config, ai_volume, sync_jobs
):
    # The page loaded when the provider sent somewhere else; the config has
    # moved it since, so ticking the box agrees to nothing.
    ai_config()
    response = dashboard.post(f"{API}/settings", json={**SETTINGS, "acknowledge": True,
                                                        "destination": "somewhere it used to send"})
    assert response.status_code == 409
    body = response.get_json()
    assert body["needs_acknowledgement"] is True and body["destination"] == FAKE_DESTINATION
    assert dashboard.get(f"{API}/config").get_json()["acknowledged"] == []


def test_a_dataset_outside_the_allowed_prefixes_is_refused(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config(allowed_dataset_prefixes=["/nowhere/allowed/"])
    _settings(dashboard, acknowledge=True)

    response = _run(dashboard)

    assert response.status_code == 403 and "allowed_dataset_prefixes" in response.get_json()["error"]


def test_the_daily_limit_refuses_the_next_run(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config(daily_call_limit=1)
    status = _ready(dashboard)
    assert dashboard.get(f"{API}/config").get_json()["calls_today"] == 1
    dashboard.post(f"{API}/reject", json={"annotate_id": status["annotate_id"]})

    response = _run(dashboard)

    assert response.status_code == 429 and "daily limit" in response.get_json()["error"]


def test_only_one_job_at_a_time(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()
    _ready(dashboard)

    response = _run(dashboard)

    assert response.status_code == 409 and "Accept or reject" in response.get_json()["error"]


# --- A run, and what is done with it -----------------------------------------------


def _png(data):
    return Image.open(io.BytesIO(base64.b64decode(data)))


def test_a_run_stages_a_preview_and_writes_nothing(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()

    status = _ready(dashboard)

    assert (status["plane"], status["depth_axis"], status["point_nm"]) == ("XY", 0, CENTRE_NM)
    assert status["write_box"] == {"lo": [8, 0, 0], "hi": [9, 16, 16]}
    assert (status["provider"], status["model"], status["label_name"]) == ("fake", "fake-threshold", "mitochondria")
    assert "mitochondria" in status["prompt"] and status["error"] is None
    assert 0 < status["mask_fraction"] < 0.5
    # The polls leave the images out; the page asks for them once.
    assert status["preview"] is None
    preview = dashboard.get(f"{API}/status?preview=1").get_json()["preview"]
    for image in ("input_png", "model_png", "overlay_png"):
        assert _png(preview[image]).size == (16, 16)
    assert not ai_volume[:].any()
    assert staging.staging_dir(_corrections(), status["annotate_id"]).is_dir()
    assert _audit_events(_corrections()) == ["requested", "staged"]


def test_the_view_centre_and_the_layouts_plane_are_the_default(dashboard, ai_config, ai_volume, sync_jobs, viewer):
    ai_config()
    _settings(dashboard, acknowledge=True)
    with viewer.txn() as s:
        s.layout = "yz"  # neuroglancer's yz panel shows cellmap-flow's z, y, x viewer's y-x plane

    assert dashboard.post(f"{API}/run", json={}).status_code == 200

    status = dashboard.get(f"{API}/status").get_json()
    assert (status["plane"], status["point_nm"]) == ("XY", CENTRE_NM)


@pytest.mark.parametrize("overwrite", [False, True])
def test_accept_writes_the_mask_and_undo_takes_it_back(dashboard, ai_config, ai_volume, sync_jobs, overwrite):
    ai_config()
    ai_volume[CENTRE] = 1  # painted background where the model says foreground
    ai_volume[CORNER] = 5  # painted an object where it says background
    ai_volume[0, 0, 0] = 3  # outside the plane
    before = ai_volume[:]
    status = _ready(dashboard)

    body = dashboard.post(f"{API}/accept", json={"annotate_id": status["annotate_id"],
                                                  "overwrite": overwrite}).get_json()

    assert body["success"] and body["reload_viewer"] and body["can_undo"], body
    labels = ai_volume[:]
    assert (labels[PLANE] > 0).all() and labels[0, 0, 0] == 3
    # Overwrite lets the model's object replace the painted background, but
    # its background never erases the object painted where it saw none.
    assert labels[CORNER] == 5
    if overwrite:
        assert labels[CENTRE] >= 2 and body["overwritten"] == 1
    else:
        assert labels[CENTRE] == 1 and body["overwritten"] == 0
    assert body["filled_foreground"] > 0
    assert body["filled_foreground"] + body["filled_background"] == 16 * 16 - (1 if overwrite else 2)
    assert dashboard.get(f"{API}/status").get_json()["status"] == "idle"
    assert not staging.staging_dir(_corrections(), status["annotate_id"]).exists()
    assert _audit_events(_corrections())[-1] == "accepted"

    undo = dashboard.post("/api/finetune/view-labels/undo", json={}).get_json()
    assert undo["success"]
    np.testing.assert_array_equal(ai_volume[:], before)


def test_reject_discards_the_staging(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()
    status = _ready(dashboard)

    response = dashboard.post(f"{API}/reject", json={"annotate_id": status["annotate_id"]})

    assert response.status_code == 200 and response.get_json()["success"]
    assert not staging.staging_dir(_corrections(), status["annotate_id"]).exists()
    assert dashboard.get(f"{API}/status").get_json()["status"] == "idle"
    assert not ai_volume[:].any()
    assert _audit_events(_corrections())[-1] == "rejected"
    # It is gone: deciding on it again is refused.
    assert dashboard.post(f"{API}/accept", json={"annotate_id": status["annotate_id"]}).status_code == 409


def test_resend_asks_again_about_the_same_plane_with_the_new_prompt(dashboard, ai_config, ai_volume, sync_jobs,
                                                                   monkeypatch):
    ai_config()
    asked = []
    real = routes.get_backend

    def recording(provider):
        backend = real(provider)
        return SimpleNamespace(segment=lambda request, model: asked.append(request) or backend.segment(request, model))

    monkeypatch.setattr(routes, "get_backend", recording)
    status = _ready(dashboard)

    response = dashboard.post(f"{API}/resend", json={"annotate_id": status["annotate_id"],
                                                      "prompt": "Paint every dark disk red."})

    assert response.get_json() == {"success": True, "annotate_id": status["annotate_id"], "status": "running"}
    resent = dashboard.get(f"{API}/status").get_json()
    assert resent["status"] == "ready" and resent["annotate_id"] == status["annotate_id"]
    assert resent["prompt"] == "Paint every dark disk red."
    assert len(asked) == 2 and asked[1].image.tobytes() == asked[0].image.tobytes()
    assert "Paint every dark disk red." in asked[1].prompt and "Paint every dark disk red." not in asked[0].prompt
    assert dashboard.get(f"{API}/config").get_json()["calls_today"] == 2
    assert _audit_events(_corrections()) == ["requested", "staged", "resent", "staged"]


@pytest.mark.parametrize("route", ["accept", "reject", "resend"])
@pytest.mark.parametrize("annotate_id", ["../../etc", "A" * 32, "", None, 7])
def test_a_malformed_annotate_id_is_refused(dashboard, ai_config, ai_volume, route, annotate_id):
    ai_config()

    response = dashboard.post(f"{API}/{route}", json={"annotate_id": annotate_id, "prompt": "x"})

    assert response.status_code == 400 and not response.get_json()["success"]


def test_an_id_that_is_not_the_current_job_is_refused(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()
    _ready(dashboard)

    response = dashboard.post(f"{API}/accept", json={"annotate_id": "0" * 32})

    assert response.status_code == 409 and not ai_volume[:].any()


# --- Shift+G -------------------------------------------------------------------


def _press_shift_g(viewer, voxel):
    routes._on_annotate_key(SimpleNamespace(mouse_voxel_coordinates=voxel, viewer_state=viewer.state))


def test_shift_g_runs_at_the_voxel_under_the_mouse(dashboard, ai_config, ai_volume, sync_jobs, viewer):
    ai_config()
    _settings(dashboard, acknowledge=True)
    with viewer.txn() as s:
        s.layout = "yz"

    _press_shift_g(viewer, [8.5, 7.5, 9.0])

    status = dashboard.get(f"{API}/status").get_json()
    assert status["status"] == "ready" and status["point_nm"] == [136.0, 120.0, 144.0]
    assert viewer.config_state.state.status_messages["ai_annotate"] == "AI annotate: ready for review in the Finetune tab."


def test_shift_g_says_why_it_cannot_run(dashboard, ai_config, ai_volume, sync_jobs, viewer):
    ai_config()
    _settings(dashboard)

    _press_shift_g(viewer, [8, 8, 8])

    message = viewer.config_state.state.status_messages["ai_annotate"]
    assert message.startswith("AI annotate: Acknowledge")
    status = dashboard.get(f"{API}/status").get_json()
    assert status["status"] == "failed" and status["error"] in message

    _press_shift_g(viewer, None)
    assert "Hover the mouse" in viewer.config_state.state.status_messages["ai_annotate"]


# --- Secrets ---------------------------------------------------------------------

SECRET = "sk-test-0123456789abcdef"


@pytest.fixture
def fake_secret():
    secrets.register_secret(SECRET)
    yield SECRET
    secrets._clear_registered_secrets()


@pytest.mark.parametrize("failure", [
    pytest.param(RuntimeError(f"401 Unauthorized: key {SECRET} rejected"), id="an-sdk-error"),
    pytest.param(AIAnnotateError("auth", "The provider refused the credentials."), id="a-categorised-error"),
])
def test_no_secret_reaches_an_answer_or_the_log(dashboard, ai_config, ai_volume, sync_jobs, monkeypatch, caplog,
                                                 fake_secret, failure):
    ai_config()
    cellmap_logger = logging.getLogger("cellmap_flow")
    cellmap_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)

    def segment(request, model):
        try:
            raise ValueError(f"upstream said: Bearer {SECRET}")
        except ValueError as cause:
            raise failure from cause

    monkeypatch.setattr(routes, "get_backend", lambda provider: SimpleNamespace(segment=segment))
    answers = [dashboard.get(f"{API}/config"), dashboard.post(f"{API}/settings", json={**SETTINGS, "acknowledge": True,
                                                                         "destination": FAKE_DESTINATION})]
    try:
        answers.append(_run(dashboard))
        status = dashboard.get(f"{API}/status")
        answers.append(status)
    finally:
        cellmap_logger.removeHandler(caplog.handler)

    body = status.get_json()
    assert body["status"] == "failed"
    expected = routes.GENERIC_FAILURE if isinstance(failure, RuntimeError) else failure.user_message
    assert body["error"] == expected
    assert "failed" in caplog.text  # the detail was logged...
    for answer in answers:
        assert SECRET not in answer.get_data(as_text=True)
    assert SECRET not in caplog.text  # ...without the secret
    with open(f"{_corrections()}/ai_annotate_log.jsonl") as f:
        assert SECRET not in f.read()


def test_resend_checks_the_config_again(dashboard, ai_config, ai_volume, sync_jobs):
    # A result staged under one config is not resent once the config no
    # longer allows its dataset.
    ai_config()
    status = _ready(dashboard)
    ai_config(allowed_dataset_prefixes=["/nowhere/allowed/"])

    response = dashboard.post(f"{API}/resend", json={"annotate_id": status["annotate_id"], "prompt": "again"})

    assert response.status_code == 403 and "allowed_dataset_prefixes" in response.get_json()["error"]
    assert dashboard.get(f"{API}/status").get_json()["status"] == "ready"
    assert dashboard.get(f"{API}/config").get_json()["calls_today"] == 1


def test_a_failed_accept_keeps_the_result_to_try_again_or_reject(dashboard, ai_config, ai_volume, sync_jobs,
                                                                monkeypatch):
    ai_config()
    status = _ready(dashboard)

    def broken(*args, **kwargs):
        raise RuntimeError("MinIO went away")

    monkeypatch.setattr(routes, "paint_box", broken)
    response = dashboard.post(f"{API}/accept", json={"annotate_id": status["annotate_id"], "overwrite": False})

    assert response.status_code == 500 and "MinIO" not in response.get_json()["error"]
    assert dashboard.get(f"{API}/status").get_json()["status"] == "ready"
    assert _run(dashboard).status_code == 409  # the result still waits for a decision
    assert dashboard.post(f"{API}/reject", json={"annotate_id": status["annotate_id"]}).get_json()["success"]
    assert not staging.staging_dir(_corrections(), status["annotate_id"]).exists()


def test_running_out_of_ids_says_what_to_do(dashboard, ai_config, ai_volume, sync_jobs, monkeypatch):
    ai_config()
    status = _ready(dashboard)

    def no_room(*args, **kwargs):
        raise ValueError("3 new objects do not fit in the volume's uint8 labels: make a new annotation volume")

    monkeypatch.setattr(routes.pipeline, "labels_for_box", no_room)
    response = dashboard.post(f"{API}/accept", json={"annotate_id": status["annotate_id"], "overwrite": False})

    assert response.status_code == 409 and "make a new annotation volume" in response.get_json()["error"]
    assert dashboard.get(f"{API}/status").get_json()["status"] == "ready"


def test_accept_continues_the_id_of_the_object_in_the_plane_beside(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()
    ai_volume[7, 6:11, 6:11] = 5  # the same disk, painted in the plane before as object 5
    status = _ready(dashboard)

    body = dashboard.post(f"{API}/accept", json={"annotate_id": status["annotate_id"], "overwrite": False}).get_json()

    assert body["success"], body
    assert ai_volume[CENTRE] == 5
    assert ai_volume[7, 8, 8] == 5 and ai_volume[9].max() == 0  # only the plane itself is written


def test_shift_g_replacing_a_failed_resend_removes_its_staged_plane(dashboard, ai_config, ai_volume, sync_jobs,
                                                                   viewer, monkeypatch):
    ai_config()
    status = _ready(dashboard)

    def refusing(request, model):
        raise AIAnnotateError("quota", "The provider's quota is used up; try again later.")

    monkeypatch.setattr(routes, "get_backend", lambda provider: SimpleNamespace(segment=refusing))
    dashboard.post(f"{API}/resend", json={"annotate_id": status["annotate_id"], "prompt": "again"})
    assert dashboard.get(f"{API}/status").get_json()["status"] == "failed"
    assert staging.staging_dir(_corrections(), status["annotate_id"]).is_dir()

    _press_shift_g(viewer, None)  # refused: the mouse is not over the data

    failed = dashboard.get(f"{API}/status").get_json()
    assert failed["status"] == "failed" and failed["annotate_id"] is None
    assert not staging.staging_dir(_corrections(), status["annotate_id"]).exists()



def test_a_call_that_fails_to_sign_in_is_not_counted(dashboard, ai_config, ai_volume, sync_jobs, monkeypatch):
    # Nothing reached the model, so nothing was spent: the day's count stays.
    ai_config()

    def signed_out(request, model):
        raise AIAnnotateError("auth", "Google asks you to sign in again: the saved login has expired.")

    monkeypatch.setattr(routes, "get_backend", lambda provider: SimpleNamespace(segment=signed_out))
    _settings(dashboard, acknowledge=True)
    assert _run(dashboard).status_code == 200

    status = dashboard.get(f"{API}/status").get_json()
    assert status["status"] == "failed" and "sign in again" in status["error"]
    assert dashboard.get(f"{API}/config").get_json()["calls_today"] == 0


def test_unticking_the_acknowledgement_withdraws_it(dashboard, ai_config, ai_volume, sync_jobs):
    ai_config()
    assert _settings(dashboard, acknowledge=True)["acknowledged"] == ["fake"]

    assert _settings(dashboard, acknowledge=False)["acknowledged"] == []

    response = _run(dashboard)
    assert response.status_code == 409 and response.get_json()["needs_acknowledgement"] is True
    # Saving other settings (no acknowledge field) leaves it as it is.
    _settings(dashboard, acknowledge=True)
    assert _settings(dashboard)["acknowledged"] == ["fake"]


def test_a_run_without_an_annotation_volume_offers_to_make_one(dashboard, ai_config):
    ai_config()
    _settings(dashboard)

    response = _run(dashboard)

    body = response.get_json()
    assert response.status_code == 409 and body["needs_volume"] is True
    assert "Create or resume one" in body["error"]


def test_shift_g_without_an_annotation_volume_offers_to_make_one(dashboard, ai_config, viewer, sync_jobs):
    ai_config()
    _settings(dashboard)

    _press_shift_g(viewer, [8.5, 7.5, 9.0])

    status = dashboard.get(f"{API}/status").get_json()
    assert status["status"] == "failed" and status["needs_volume"] is True
