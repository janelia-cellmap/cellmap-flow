"""Instance-correction volumes: where they are seeded, and what they hold."""

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.session.instance import seed_instance_volume

CENTRE = [8.0, 168.0, 8.0]  # voxel 0's centre: corner (0, 160, 0) at 16 nm
OME = {
    "multiscales": [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [16.0] * 3},
                        {"type": "translation", "translation": CENTRE},
                    ],
                }
            ],
        }
    ]
}


def make_instances(path, group_attrs=None, s0_attrs=None):
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(group_attrs or {})
    ids = np.zeros((8, 8, 8), np.uint32)
    ids[2:4, 2:4, 2:4] = 7
    s0 = group.create_dataset("s0", data=ids, chunks=(4, 4, 4))
    s0.attrs.update(s0_attrs or {})
    return str(path)


@pytest.mark.parametrize(
    "group_attrs, s0_attrs",
    [
        (None, {"resolution": [16] * 3, "offset": [0, 160, 0]}),  # offset is the corner
        (OME, None),  # OME only: the translation is already the centre
    ],
    ids=["resolution-offset", "ome"],
)
def test_a_seeded_volume_lies_on_its_segmentation(tmp_path, group_attrs, s0_attrs):
    ok, path = seed_instance_volume(
        str(tmp_path / "roi_annotation.zarr"),
        make_instances(tmp_path / "instances.zarr", group_attrs, s0_attrs),
        "/raw.zarr",
        "model",
        input_size=[12] * 3,
        input_voxel_size=[16] * 3,
        dilation_radius_voxels=1,
    )
    assert ok, path

    volume = zarr.open_group(path, mode="r")
    transforms = volume["annotation"].attrs["multiscales"][0]["datasets"][0]
    assert transforms["coordinateTransformations"][1]["translation"] == CENTRE
    assert volume.attrs["dataset_offset_nm"] == CENTRE
    assert volume.attrs["type"] == "annotation_volume"

    labels = volume["annotation/s0"][:]
    assert labels.dtype == np.uint16
    assert labels[2, 2, 2] == 8  # instance 7 is label 8
    assert labels[1, 2, 2] == 1  # the background shell, one voxel out
    assert labels[6, 6, 6] == 0  # unannotated


CREATE = "/api/viewer/create-instance-correction"
SYNC = "/api/viewer/sync-instance-correction"
CC3D = "/api/viewer/cc3d-relabel-annotation"


class _Unbuildable:
    name = "model"

    @property
    def config(self):
        raise AssertionError("the dashboard built the model")


@pytest.fixture
def client(monkeypatch, tmp_path, viewer, dashboard):
    from cellmap_flow.dashboard.state import get_session

    make_instances(tmp_path / "instances.zarr", s0_attrs={"resolution": [16] * 3, "offset": [0] * 3})
    zarr.open_group(str(tmp_path / "vols" / "roi_annotation.zarr"), mode="w")
    (tmp_path / "vols" / "plain.zarr").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.setattr(get_session(), "models_config", [_Unbuildable()])
    monkeypatch.setattr(get_session(), "dataset_path", "/raw.zarr")
    return dashboard


@pytest.mark.parametrize(
    "url, body",
    [
        (CREATE, {"roi_name": "roi", "instance_zarr_path": "{tmp}/instances.zarr",
                  "model_name": "model", "output_dir": "{tmp}/missing"}),
        (CREATE, {"roi_name": "../roi", "instance_zarr_path": "{tmp}/instances.zarr",
                  "model_name": "model"}),
        (CREATE, {"roi_name": "roi", "instance_zarr_path": "{tmp}/instances.zarr"}),
        (CREATE, {"roi_name": "roi", "reuse_existing": True, "source_zarr_path": "{tmp}/vols"}),
        (SYNC, {"zarr_path": "{tmp}/vols/roi_annotation.zarr", "dst_path": "{tmp}/elsewhere/copy.zarr"}),
        (SYNC, {"zarr_path": "{tmp}/vols/roi_annotation.zarr", "dst_path": "{tmp}/vols/plain.zarr"}),
        (SYNC, {"zarr_path": "{tmp}/vols/roi_annotation.zarr", "dst_path": "{tmp}/vols/copy"}),
        (CC3D, {"zarr_path": "{tmp}/vols/roi_annotation.zarr", "target_label": 5,
                "snapshot_dir": "{tmp}/elsewhere/snapshots"}),
        (CC3D, {"zarr_path": "{tmp}/vols/roi_annotation.zarr", "target_label": "five"}),
    ],
    ids=["output-dir-missing", "roi-name-path", "no-model", "source-not-zarr",
         "dst-elsewhere", "dst-not-a-zarr", "dst-not-dot-zarr", "snapshots-elsewhere", "bad-label"],
)
def test_bad_paths_are_refused_before_anything_is_written(client, tmp_path, url, body):
    body = {k: v.format(tmp=tmp_path) if isinstance(v, str) else v for k, v in body.items()}
    before = sorted(tmp_path.rglob("*"))
    response = client.post(url, json=body)
    assert response.status_code == 400
    assert response.get_json()["success"] is False and response.get_json()["error"]
    assert sorted(tmp_path.rglob("*")) == before


def test_a_fresh_seed_asks_the_server_for_geometry(client, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from cellmap_flow.dashboard.routes.finetune import instance_correction
    from cellmap_flow.dashboard.state import get_session
    from cellmap_flow.models import geometry_cache

    geometry = SimpleNamespace(read_shape=[192] * 3, write_shape=[64] * 3,
                               input_voxel_size=[16] * 3, output_voxel_size=[16] * 3)
    monkeypatch.setattr(geometry_cache, "model_geometry_config", lambda name: geometry)
    served = []
    monkeypatch.setattr(instance_correction, "ensure_minio_serving",
                        lambda *a, **k: served.append((a, k)) or "http://m:9000/annotations/roi_annotation.zarr")

    response = client.post(CREATE, json={"roi_name": "roi", "model_name": "model",
                                         "instance_zarr_path": str(tmp_path / "instances.zarr")})
    assert response.status_code == 200, response.get_json()
    (path, volume_id), kwargs = served[0]
    assert volume_id == "roi_annotation" and kwargs["mc_target_name"] == "roi_annotation.zarr"
    assert get_session().annotation_volumes["roi_annotation"]["zarr_path"] == path
    assert zarr.open_group(path, mode="r").attrs["chunk_size"] == [4, 4, 4]
