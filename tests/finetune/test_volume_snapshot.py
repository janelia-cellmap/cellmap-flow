"""Making an annotation volume through the dashboard's routes, and resuming one.

Over one raw dataset: the layer a new volume is painted in, a painted
session resumed elsewhere with its strokes, and only over the dataset it was
painted on. mc and s3fs are fakes; nothing is served.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import neuroglancer
import numpy as np
import pytest
import s3fs
import zarr

from cellmap_flow.dashboard import finetune_utils as fu

NORM = [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}]
POST = [{"name": "SigmoidPostprocessor"}]
# What a server reports for a model: 12^3 input at 8 nm, 4^3 output at 16 nm.
GEOMETRY = SimpleNamespace(
    read_shape=[96, 96, 96], write_shape=[64, 64, 64],
    input_voxel_size=[8, 8, 8], output_voxel_size=[16, 16, 16], output_channels=1,
)


def _ome(paths, scales, translation):
    return [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [
            {"path": p, "coordinateTransformations": [
                {"type": "scale", "scale": [vs] * 3},
                {"type": "translation", "translation": [t] * 3}]}
            for p, vs, t in zip(paths, scales, translation)
        ],
    }]


class _Alive:
    pid = 1

    def poll(self):
        return None


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A raw pyramid, a crop, a segmentation, a viewer and a MinIO that has nothing yet."""
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.state import get_session
    from cellmap_flow.models import geometry_cache
    from cellmap_flow.process_chain import process_chain

    # Janelia-style: every level's corner is -4 nm. s1 has 21 voxels a side,
    # so the volume (4^3 chunks) is padded to 24.
    raw = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w").create_group("em")
    for name, n in (("s0", 42), ("s1", 21)):
        raw.create_dataset(name, data=np.zeros((n,) * 3, np.uint8), chunks=(8, 8, 8))
    raw.attrs["multiscales"] = _ome(["s0", "s1"], [8.0, 16.0], [0.0, 4.0])

    # A 16 nm crop whose voxel 0 is centred at 68 nm (corner 60), fg id 5.
    labels = np.zeros((6, 6, 6), np.uint8)
    labels[1:4, 1:4, 1:4] = 5
    crop = zarr.open_group(str(tmp_path / "crop.zarr"), mode="w")
    crop.create_dataset("s0", data=labels, chunks=labels.shape)
    crop.attrs["multiscales"] = _ome(["s0"], [16.0], [68.0])
    (tmp_path / "crops.yaml").write_text(
        f"crops:\n  - path: {tmp_path / 'crop.zarr'}\n    name: c1\n    fg_ids: [5]\n    mode: dense\n"
    )

    # Two instances on a 16 nm grid whose corner is (0, 160, 0).
    ids = np.zeros((8, 8, 8), np.uint32)
    ids[1:3, 1:3, 1:3] = 3
    ids[5:7, 5:7, 5:7] = 9
    seg = zarr.open_group(str(tmp_path / "seg.zarr"), mode="w")
    seg.create_dataset("s0", data=ids, chunks=(4, 4, 4)).attrs.update(
        resolution=[16] * 3, offset=[0, 160, 0]
    )

    def fake_run(cmd, **kwargs):
        assert cmd[:3] == ["mc", "mirror", "--overwrite"], cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    state = {"process": _Alive(), "ip": "10.0.0.1", "port": 9000, "bucket": "annotations",
             "output_base": None, "sync_thread": None}
    volumes, sessions = {}, {}
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(s3fs, "S3FileSystem", lambda **kw: SimpleNamespace(exists=lambda path: False))
    monkeypatch.setattr(fu, "_require_minio_binaries", lambda: None)
    monkeypatch.setattr(geometry_cache, "model_geometry_config", lambda name: GEOMETRY)
    for name, value in dict(
        minio_state=state, annotation_volumes=volumes, output_sessions=sessions, viewer=neuroglancer.Viewer(),
        raw=None, dataset_path=str(tmp_path / "raw.zarr" / "em"), models_config=[SimpleNamespace(name="m")],
        # The volumes as made with Resample unticked: at the data's voxel sizes.
        resample=False,
    ).items():
        monkeypatch.setattr(get_session(), name, value)
    for name, value in dict(input_norm_config=NORM, postprocess_config=POST).items():
        monkeypatch.setattr(process_chain(), name, value)
    return SimpleNamespace(tmp=tmp_path, client=app.test_client())


@pytest.mark.parametrize("url, body", [
    pytest.param("/api/finetune/load-crops", {"model_name": "m", "output_path": "{tmp}/c", "yaml": "{tmp}/crops.yaml"},
                 id="load-crops"),
    pytest.param("/api/viewer/create-instance-correction",
                 {"roi_name": "roi", "model_name": "m", "instance_zarr_path": "{tmp}/seg.zarr"},
                 id="instance-correction"),
])
def test_the_layer_a_volume_is_painted_in_comes_selected_with_the_draw_tools_bound(world, url, body):
    """As add-to-viewer's does: these routes built their own layer, with no
    keys bound and left unselected."""
    from cellmap_flow.dashboard.state import get_session

    answer = world.client.post(url, json={k: v.format(tmp=world.tmp) for k, v in body.items()}).get_json()
    layer = answer.get("layer_name") or f"annotation_{answer['volume_id']}"
    state = get_session().viewer.state
    assert (state.selected_layer.layer, state.selected_layer.visible) == (layer, True)
    bindings = state.layers[layer].to_json()["toolBindings"]
    assert {key: tool if isinstance(tool, str) else tool["type"] for key, tool in bindings.items()} == {
        "A": "vox-brush", "F": "vox-flood-fill"}


def test_a_painted_session_resumes_elsewhere_with_its_strokes(world):
    """A stroke on disk, then the session is listed and resumed into another
    output path: the copy holds the stroke, and its manifest points the
    trainer at the copy, over the same raw data."""
    post = lambda url, **body: world.client.post(url, json=body).get_json()  # noqa: E731
    created = post("/api/finetune/create-volume", model_name="m", output_path=str(world.tmp / "a"))
    zarr.open_array(str(Path(created["zarr_path"]) / "annotation" / "s0"), mode="r+")[0:4, 0:4, 0:4] = 2
    (listed,) = post("/api/finetune/list-existing-sessions", output_path=str(world.tmp / "a"))["sessions"]
    assert listed["chunk_count"] == 1

    resumed = post("/api/finetune/load-existing-volume", output_path=str(world.tmp / "b"),
                   source_session_path=listed["session_path"])
    assert resumed["success"] and resumed["painted_chunk_count"] == 1
    copy = zarr.open_array(str(Path(resumed["zarr_path"]) / "annotation" / "s0"), mode="r")[:]
    assert (copy[0:4, 0:4, 0:4] == 2).all() and (copy == 2).sum() == 4 ** 3
    manifest = json.loads((Path(resumed["zarr_path"]).parent / "_virtual_sources.json").read_text())
    assert (manifest["volume_zarr_path"], manifest["raw_dataset_path"]) == (
        resumed["zarr_path"], str(world.tmp / "raw.zarr" / "em"))


def test_a_session_resumes_only_over_the_dataset_it_was_painted_on(world):
    """A jrc_axolotl-heart-1 session resumed in a jrc_amphiuma-means-heart-1
    dashboard drew its strokes over the wrong EM, trained on axolotl, and
    served axolotl over amphiuma's view (2026-10-01)."""
    from cellmap_flow.dashboard.state import get_session

    post = lambda url, **body: world.client.post(url, json=body)  # noqa: E731
    created = post("/api/finetune/create-volume", model_name="m", output_path=str(world.tmp / "a")).get_json()
    get_session().dataset_path = str(world.tmp / "other.zarr" / "em")
    refused = post("/api/finetune/load-existing-volume", output_path=str(world.tmp / "b"),
                   source_session_path=str(Path(created["zarr_path"]).parents[1]))
    assert refused.status_code == 409 and "raw.zarr/em" in refused.get_json()["error"]
    assert not (world.tmp / "b").exists()


def test_a_volume_painted_before_any_model_is_ranked_and_carried_but_not_trained(world, monkeypatch):
    """No model loaded: New Volume offers the raw's levels and plans on the
    raw's grid at the one asked for, a volume for no model and without a
    training manifest. A training job on it is refused (paint_first); the
    rank lists it, the running model ticked and none its model, scores the
    model on its paint, and Use this model carries the paint onto the
    model's 16 nm grid."""
    from cellmap_flow.dashboard.routes.finetune import rank_routes, view_labels
    from cellmap_flow.dashboard.state import get_session

    session = get_session()
    monkeypatch.setattr(session, "models_config", [])
    post = lambda url, **body: world.client.post(url, json=body)  # noqa: E731
    assert world.client.get("/api/finetune/models").get_json()["raw_voxel_sizes"] == [[8.0] * 3, [16.0] * 3]

    created = post("/api/finetune/create-volume", voxel_size=[8, 8, 8], output_path=str(world.tmp / "a")).get_json()
    assert created["success"], created
    # s0's grid: 42 voxels from its corner (-4 nm), voxel 0 centred at 0, in one 42^3 chunk.
    assert created["metadata"]["output_voxel_size"] == [8.0] * 3 and created["metadata"]["dataset_offset_nm"] == [0.0] * 3
    assert created["metadata"]["dataset_shape_voxels"] == [42] * 3
    volume_id, zarr_path = created["volume_id"], Path(created["zarr_path"])
    assert session.annotation_volumes[volume_id]["model_name"] is None
    assert zarr.open_group(str(zarr_path), mode="r").attrs["model_name"] is None
    assert not (zarr_path.parent / "_virtual_sources.json").exists()
    labels = zarr.open_array(str(zarr_path / "annotation" / "s0"), mode="r+")
    labels[4:12, 4:12, 4:12] = 2
    labels[20:28, 20:28, 20:28] = 1

    # A model now running: training it on this volume is refused, the error saying where to go.
    monkeypatch.setattr(session, "models_config", [SimpleNamespace(name="m")])
    monkeypatch.setattr(session, "jobs", [SimpleNamespace(model_name="m", host="http://gpu:8000")])
    refused = post("/api/finetune/submit", model_name="m", corrections_path=str(zarr_path.parent))
    assert refused.status_code == 400 and refused.get_json()["paint_first"] is True

    monkeypatch.setattr("cellmap_flow.models.hf_catalog.list_huggingface_models", lambda: {})
    monkeypatch.setattr(rank_routes, "fetch_model_info", lambda host: {"channels": ["nuc"], "output_voxel_size": [16] * 3})
    listed = world.client.get("/api/finetune/rank/candidates").get_json()
    assert listed["volume"] == {"id": volume_id, "model_name": None, "painted_chunks": 1}
    assert {c["name"] for c in listed["candidates"] if c["ticked"]} == {"m"}
    assert not any(c["volume_model"] for c in listed["candidates"])

    # The model's logits on the volume's grid: +4 over the object painted, -4 elsewhere.
    def read_prediction(host, name, volume, lo, hi, info, postprocess=(), input_norm=None):
        logits = np.full((1, 42, 42, 42), -4.0, "f4")
        logits[:, 4:12, 4:12, 4:12] = 4.0
        return logits[(slice(None),) + tuple(slice(a, b) for a, b in zip(lo, hi))], lo, hi

    monkeypatch.setattr(view_labels, "read_prediction", read_prediction)
    monkeypatch.setattr(rank_routes, "sync_annotation_volume_from_minio", lambda volume_id: None)
    monkeypatch.setattr(rank_routes, "autodetect_output_type", lambda config, output_type, offsets: ("binary", None))
    monkeypatch.setattr(rank_routes, "_raw_window", lambda volume, boxes: None)
    scored = post("/api/finetune/rank/score", name="m").get_json()
    assert scored["success"] and scored["matches"] == 1, scored

    used = post("/api/finetune/rank/use", name="m").get_json()
    assert used["success"], used
    # 8^3 painted 8 nm voxels of each label, 4^3 of the model's 16 nm voxels each.
    assert (used["from_volume"], used["source_voxels"], used["painted_voxels"]) == (volume_id, 2 * 8 ** 3, 2 * 4 ** 3)
    carried = session.annotation_volumes[used["volume_id"]]
    assert (carried["model_name"], carried["output_voxel_size"]) == ("m", [16.0] * 3)
    assert (Path(carried["zarr_path"]).parent / "_virtual_sources.json").exists(), "the model's volume is trained on"


@pytest.mark.parametrize("zoom, voxel", [(None, 8.0), (2.5, 16.0)], ids=["no zoom: the finest", "zoomed out: s1"])
def test_new_volume_without_a_model_or_a_size_is_at_the_level_the_viewer_shows(world, monkeypatch, zoom, voxel):
    from cellmap_flow.dashboard.state import get_session

    session = get_session()
    monkeypatch.setattr(session, "models_config", [])
    if zoom is not None:
        # 2.5 of the viewer's 8 nm voxels a screen pixel: 20 nm, where Neuroglancer draws s1.
        with session.viewer.txn() as s:
            s.dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[8, 8, 8])
            s.position = [21, 21, 21]
            s.cross_section_scale = zoom
    created = world.client.post("/api/finetune/create-volume", json={"output_path": str(world.tmp / "a")}).get_json()
    assert created["success"], created
    assert created["metadata"]["output_voxel_size"] == [voxel] * 3
