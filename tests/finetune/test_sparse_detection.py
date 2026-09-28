"""Painted (sparse) sessions are recognised from the volume itself.

detect_sparse_annotations looked for per-chunk extracts marked
source == "sparse_volume", which are only written when a session has no
manifest -- and every session has one now. So it was always False: submit's
switch to margin + distillation for scribbles, and its guard against a
distance target on scribbles, never fired, and mask_unannotated was never set.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

from cellmap_flow.dashboard.routes.finetune.common import detect_sparse_annotations

CROP = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}


def _session(tmp_path, arr, crops=()):
    corrections = tmp_path / "corrections"
    vol = corrections / "vol.zarr"
    root = zarr.open_group(str(vol), mode="w")
    root.create_group("annotation").create_dataset(
        "s0", shape=arr.shape, chunks=(8, 8, 8), dtype="uint8", fill_value=0
    )
    root["annotation"]["s0"][:] = arr
    root.attrs["imported_crops"] = list(crops)
    (corrections / "_virtual_sources.json").write_text(json.dumps({
        "kind": "volume_zarr_v1", "volume_zarr_path": str(vol), "raw_dataset_path": "/raw",
    }))
    return corrections


def test_imported_crops_alone_are_dense(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[0:16, 0:16, 0:16] = 1
    arr[4:8, 4:8, 4:8] = 2
    assert detect_sparse_annotations(_session(tmp_path, arr, [CROP])) is False


def test_a_stroke_outside_the_crops_is_sparse(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[0:16, 0:16, 0:16] = 1
    arr[20, 20, 20] = 2
    assert detect_sparse_annotations(_session(tmp_path, arr, [CROP])) is True


def test_a_painted_session_is_sparse(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[3, 3, 3:6] = 1
    assert detect_sparse_annotations(_session(tmp_path, arr)) is True


def test_submit_treats_a_painted_session_as_scribbles(tmp_path, monkeypatch):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import training
    from cellmap_flow.globals import g

    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[3, 3, 3:6] = 2
    corrections = _session(tmp_path, arr)
    captured = {}

    class _Manager:
        jobs = {}

        def submit_finetuning_job(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="j", output_dir=Path(tmp_path), lsf_job=None)

    monkeypatch.setattr(g, "finetune_job_manager", _Manager(), raising=False)
    monkeypatch.setattr(g, "models_config", [SimpleNamespace(name="m")], raising=False)
    with app.test_request_context():
        response = training.submit_finetuning_response({
            "model_name": "m", "corrections_path": str(corrections), "output_type": "distance",
        })
    assert response.get_json()["success"], response.get_json()
    assert captured["mask_unannotated"] is True
    assert captured["output_type"] == "binary", "a distance target cannot be built from scribbles"
    assert captured["loss_type"] == "margin"
