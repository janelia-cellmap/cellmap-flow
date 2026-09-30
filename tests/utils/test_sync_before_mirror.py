"""Local chunks are only pushed to MinIO after MinIO's strokes are pulled.

`mc mirror --overwrite` uploads every local chunk over MinIO's copy. The
browser paints straight into MinIO and the sync runs every 30 s, so a YAML
import into a painted volume, or a resume, overwrote recent strokes with
stale local chunks.
"""

import subprocess
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard import finetune_utils as fu
from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync


class _Alive:
    pid = 1

    def poll(self):
        return None


@pytest.fixture
def order(monkeypatch):
    events = []
    monkeypatch.setattr(fu, "minio_state", {"process": _Alive(), "ip": "127.0.0.1", "port": 9000,
                                            "bucket": "annotations", "output_base": None})
    monkeypatch.setattr(fu, "_require_minio_binaries", lambda: None)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: events.append(("mc", cmd[1])) or SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(
        sync, "sync_volume",
        lambda volume_id, force=False, zarr_path=None, **k: events.append(("pull", volume_id, zarr_path)),
    )
    return events


@pytest.mark.parametrize("in_minio, pulled", [(True, True), (False, False)])
def test_painted_chunks_are_pulled_before_the_mirror(order, monkeypatch, tmp_path, in_minio, pulled):
    asked = []
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: SimpleNamespace(
        exists=lambda path: asked.append(path) or in_minio))
    fu.ensure_minio_serving(str(tmp_path / "vol-1.zarr"), "vol-1")
    assert asked == ["annotations/vol-1.zarr/annotation/s0"]
    pull = [("pull", "vol-1", str(tmp_path / "vol-1.zarr"))] if pulled else []
    assert order == pull + [("mc", "mirror")]


def test_a_yaml_import_pulls_strokes_before_writing_crops(monkeypatch, tmp_path):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import yaml_crops
    from cellmap_flow.globals import g

    events = []
    meta = {
        "zarr_path": str(tmp_path / "vol-1.zarr"), "input_size": [8] * 3, "output_size": [4] * 3,
        "input_voxel_size": [8] * 3, "output_voxel_size": [8] * 3, "minio_url": "http://m/vol-1.zarr",
    }
    crops = SimpleNamespace(crops=[SimpleNamespace(path="/crop.zarr")], patches_per_epoch=None,
                            jitter_voxels=None, seed=0, dense_to_sparse_ratio=None)
    monkeypatch.setattr(yaml_crops, "parse_crops_yaml", lambda text: crops)
    monkeypatch.setattr(yaml_crops, "_get_selected_model_config", lambda name: (object(), None))
    monkeypatch.setattr(yaml_crops, "ensure_corrections_storage", lambda path: (None, str(tmp_path)))
    monkeypatch.setattr(yaml_crops, "_find_session_annotation_volume", lambda d: ("vol-1", meta))
    monkeypatch.setattr(yaml_crops, "_ensure_editable_layer", lambda *a: None)
    monkeypatch.setattr(yaml_crops, "sync_annotation_volume_from_minio",
                        lambda vid, **k: events.append(("pull", vid)))
    monkeypatch.setattr(yaml_crops, "write_crop_into_volume",
                        lambda m, entry, progress_callback=None: events.append(("write", entry.path))
                        or {"n_fg_voxels": 0})
    monkeypatch.setattr(yaml_crops, "ensure_minio_serving", lambda *a, **k: events.append(("mirror",)))
    monkeypatch.setattr(yaml_crops, "write_manifest", lambda *a: None)
    monkeypatch.setattr(yaml_crops, "refresh_annotated_regions_layer", lambda **k: None)
    monkeypatch.setattr(g, "dataset_path", "/data/raw.zarr", raising=False)

    with app.test_request_context():
        response = yaml_crops.load_crops_from_yaml_response({"model_name": "m", "yaml": "crops: []"})

    assert response.get_json()["success"], response.get_json()
    assert events[:3] == [("pull", "vol-1"), ("write", "/crop.zarr"), ("mirror",)]
