"""Tests for the AI-annotate review routes: status polling, accept, reject."""

import json
import os

import numpy as np
from flask import Flask
from PIL import Image

import cellmap_flow.dashboard.routes.finetune.ai_annotate as ai_annotate
import cellmap_flow.dashboard.routes.finetune.overlay as overlay


def _stage_fake_result(tmp_path, volume_id="vol-test", annotate_id="annotate-1"):
    volume_meta = {"corrections_dir": str(tmp_path)}
    staging_dir = ai_annotate._staging_dir(volume_meta, annotate_id)
    os.makedirs(staging_dir, exist_ok=True)

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4] = 255
    np.save(os.path.join(staging_dir, "mask.npy"), mask)
    Image.new("RGB", (8, 8)).save(os.path.join(staging_dir, "preview.png"))
    meta = {
        "volume_id": volume_id,
        "annotate_id": annotate_id,
        "chunk_indices": [1, 1, 1],
        "z_row_index": 0,
        "label_id": 2,
        "background_label_id": 1,
    }
    with open(os.path.join(staging_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    ai_annotate._set_progress(volume_id, status="ready", annotate_id=annotate_id, error=None)
    return volume_meta, staging_dir


def test_status_response_idle_when_nothing_staged():
    app = Flask(__name__)
    with app.app_context():
        resp = ai_annotate.get_ai_annotate_status_response("vol-nothing-here")
        assert resp.get_json()["status"] == "idle"


def test_status_response_ready_includes_preview(tmp_path, monkeypatch):
    volume_meta, _ = _stage_fake_result(tmp_path)
    monkeypatch.setattr(ai_annotate, "_get_volume_metadata", lambda volume_id: volume_meta)

    app = Flask(__name__)
    with app.app_context():
        resp = ai_annotate.get_ai_annotate_status_response("vol-test")
        body = resp.get_json()
        assert body["status"] == "ready"
        assert "preview_png_base64" in body
    ai_annotate._clear_progress("vol-test")


def test_accept_writes_mask_and_clears_staging(tmp_path, monkeypatch):
    volume_meta, staging_dir = _stage_fake_result(tmp_path)
    monkeypatch.setattr(ai_annotate, "_get_volume_metadata", lambda volume_id: volume_meta)

    write_calls = []
    invalidate_calls = []
    monkeypatch.setattr(
        overlay, "write_ai_mask_to_minio", lambda *a, **kw: write_calls.append((a, kw))
    )
    monkeypatch.setattr(overlay, "_invalidate_annotation_layer", lambda vid: invalidate_calls.append(vid))

    app = Flask(__name__)
    with app.app_context():
        resp = ai_annotate.accept_ai_annotate_response({"volume_id": "vol-test"})
        assert resp.get_json()["success"] is True

    assert len(write_calls) == 1
    assert invalidate_calls == ["vol-test"]
    assert not os.path.exists(staging_dir)
    assert ai_annotate._get_progress("vol-test") is None


def test_reject_clears_staging_without_writing(tmp_path, monkeypatch):
    volume_meta, staging_dir = _stage_fake_result(tmp_path)
    monkeypatch.setattr(ai_annotate, "_get_volume_metadata", lambda volume_id: volume_meta)

    called = []
    monkeypatch.setattr(overlay, "write_ai_mask_to_minio", lambda *a, **kw: called.append(True))

    app = Flask(__name__)
    with app.app_context():
        resp = ai_annotate.reject_ai_annotate_response({"volume_id": "vol-test"})
        assert resp.get_json()["success"] is True

    assert called == []
    assert not os.path.exists(staging_dir)
    assert ai_annotate._get_progress("vol-test") is None


def test_accept_fails_when_nothing_ready():
    app = Flask(__name__)
    with app.app_context():
        resp, status = ai_annotate.accept_ai_annotate_response({"volume_id": "vol-never-staged"})
        assert status == 404
        assert resp.get_json()["success"] is False
