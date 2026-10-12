"""Stand-ins for the AI-annotate tests: a config file, an annotation volume over a raw dataset, and jobs run at once.

``ai_config(**settings)`` writes a config with a ``fake`` provider (no
network) and points ``CELLMAP_FLOW_AI_ANNOTATE_CONFIG`` at it; HOME is a
fresh directory, so the daily call count starts at 0 in every test.

``ai_volume()`` is the session's annotation volume: a 16^3 grid of 16 nm
voxels whose labels are in an in-memory "MinIO", as in
tests/utils/test_view_labels.py, over a raw dataset of the same grid that is
bright everywhere but a dark disk of radius 4 voxels around y = x = 8 in
every z. The viewer looks at its centre, voxel (8, 8, 8), 128 nm on each
axis. It returns the MinIO labels.

``sync_jobs`` runs the routes' background jobs in the request that starts
them, so a test sees the job's outcome in the next status call.
"""

import neuroglancer
import numpy as np
import pytest
import yaml
import zarr

from cellmap_flow.dashboard.routes.finetune import ai_annotate as routes
from cellmap_flow.dashboard.routes.finetune import view_labels
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import fill, minio
from tests.utils.test_view_labels import _Bucket, _volume_array

VOXEL_NM = 16
SHAPE = (16, 16, 16)
CENTRE_NM = [128.0, 128.0, 128.0]


@pytest.fixture
def ai_config(tmp_path, monkeypatch):
    """``ai_config(**settings)``: write the config (enabled, a ``fake`` provider) and use it; its path."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def write(**settings):
        config = {
            "enabled": True,
            "crop_size_px": 16,
            "providers": {"fake": {"type": "fake", "models": ["fake-threshold"]}},
            **settings,
        }
        path = tmp_path / "ai_annotate.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setenv("CELLMAP_FLOW_AI_ANNOTATE_CONFIG", str(path))
        return path

    return write


def _raw(tmp_path):
    """The raw dataset: 200 everywhere, 30 in a disk of radius 4 around y = x = 8; its path."""
    rows, cols = np.ogrid[: SHAPE[1], : SHAPE[2]]
    disk = (rows - 8) ** 2 + (cols - 8) ** 2 <= 16
    data = np.broadcast_to(np.where(disk, 30, 200).astype(np.uint8), SHAPE).copy()
    root = zarr.open(str(tmp_path / "raw.zarr"), mode="a")
    array = root.create_dataset("raw", data=data, chunks=SHAPE)
    array.attrs["resolution"] = [VOXEL_NM] * 3
    array.attrs["offset"] = [0, 0, 0]
    return str(tmp_path / "raw.zarr" / "raw")


@pytest.fixture
def ai_volume(tmp_path, monkeypatch, viewer):
    """The session's volume and its raw, served by a fake MinIO; returns its MinIO labels."""
    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[VOXEL_NM] * 3)
        s.position = [8, 8, 8]
    corrections = tmp_path / "session" / "corrections"
    corrections.mkdir(parents=True)
    local = zarr.open_group(str(tmp_path / "vol-1.zarr"), mode="w")
    _volume_array(local, "u1")
    bucket = _Bucket()
    store = bucket.stores["annotations/vol-1.zarr"] = zarr.MemoryStore()
    monkeypatch.setattr(minio, "make_s3_filesystem", lambda state: bucket)
    monkeypatch.setattr(fill.s3fs, "S3Map", lambda root, s3, check: s3.stores[root])
    monkeypatch.setitem(get_session().minio_state, "ip", "m")
    monkeypatch.setitem(get_session().minio_state, "port", 9000)
    get_session().dataset_path = _raw(tmp_path)
    monkeypatch.setattr(get_session(), "annotation_volumes", {"vol-1": {
        "zarr_path": str(tmp_path / "vol-1.zarr"), "corrections_dir": str(corrections), "model_name": "model",
        "output_size": [8] * 3, "input_size": [8] * 3, "output_voxel_size": [VOXEL_NM] * 3,
        "input_voxel_size": [VOXEL_NM] * 3, "dataset_offset_nm": [VOXEL_NM / 2] * 3,
        "dataset_path": get_session().dataset_path, "resample": False,
    }})
    monkeypatch.setattr(view_labels, "sync_annotation_volume_from_minio", lambda volume_id: None)
    return _volume_array(zarr.open_group(store), "u1")


@pytest.fixture
def sync_jobs(monkeypatch):
    """Run the routes' background jobs at once, in the request that starts them."""
    monkeypatch.setattr(routes, "_start_thread", lambda target, *args: target(*args))
