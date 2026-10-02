"""Labelling the view in one click: a seed from the model's prediction, or all background.

The volume is a 16^3 grid of 16 nm voxels, its labels in an in-memory
"MinIO"; the view is one 8-voxel output patch, voxels 4..12 on each axis.
The model's server is faked at the HTTP level, serving a prediction over
the same grid, so the reading of chunks and the placing of voxels are
tested along with the labels.
"""

import json
from types import SimpleNamespace

import neuroglancer
import numpy as np
import pytest
import zarr

from cellmap_flow.dashboard.routes.finetune import view_labels
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import fill, minio
from cellmap_flow.serving import virtual_zarr
from cellmap_flow.serving.protocol import ARGS_KEY
from cellmap_flow.serving.probe import SIGNED_UNIT, UNBOUNDED, UNIT

SEED, BACKGROUND, SPLIT = (f"/api/finetune/view-labels/{what}" for what in ("seed", "background", "split"))
BOX = (slice(4, 12),) * 3


class _Bucket:
    """MinIO's bucket as s3fs sees it: an in-memory zarr per key."""

    def __init__(self):
        self.stores = {}

    def _split(self, path):
        root, _, rest = path.partition(".zarr/")
        return self.stores.get(root + ".zarr", {}), rest

    def exists(self, path):
        store, rest = self._split(path)
        return rest in store

    def put(self, local, path):
        store, rest = self._split(path)
        store[rest] = open(local, "rb").read()


def _volume_array(group, dtype):
    return group.require_group("annotation").create_dataset(
        "s0", shape=(16,) * 3, chunks=(8,) * 3, dtype=dtype, fill_value=0, overwrite=True
    )


@pytest.fixture
def served(tmp_path, monkeypatch, viewer):
    """The session's volume, served by a fake MinIO; returns its MinIO labels."""
    def make(dtype="u1", position=(8, 8, 8)):
        with viewer.txn() as s:
            s.dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[16] * 3)
            s.position = list(position)
        local = zarr.open_group(str(tmp_path / "vol-1.zarr"), mode="w")
        _volume_array(local, dtype)
        bucket = _Bucket()
        store = bucket.stores["annotations/vol-1.zarr"] = zarr.MemoryStore()
        monkeypatch.setattr(minio, "make_s3_filesystem", lambda state: bucket)
        monkeypatch.setattr(fill.s3fs, "S3Map", lambda root, s3, check: s3.stores[root])
        monkeypatch.setitem(get_session().minio_state, "ip", "m")
        monkeypatch.setitem(get_session().minio_state, "port", 9000)
        monkeypatch.setattr(get_session(), "annotation_volumes", {"vol-1": {
            "zarr_path": str(tmp_path / "vol-1.zarr"), "corrections_dir": str(tmp_path), "model_name": "model",
            "output_size": [8] * 3, "output_voxel_size": [16] * 3, "dataset_offset_nm": [8.0] * 3,
        }})
        monkeypatch.setattr(view_labels, "sync_annotation_volume_from_minio", lambda volume_id: synced.append(volume_id))
        return _volume_array(zarr.open_group(store), dtype)

    synced = []
    make.synced = synced
    return make


@pytest.fixture
def server(monkeypatch):
    """A running server for "model" that predicts ``server.prediction`` (z, y, x, c) over the volume."""
    fake = SimpleNamespace(prediction=None, output_class=UNIT)

    def get(url, timeout):
        path = url.split(ARGS_KEY)[2]
        data = fake.prediction
        if path == "/.zattrs":
            body = json.dumps(virtual_zarr.zattrs("zyx", [16] * 3, [0] * 3, True, "model")).encode()
        else:
            served = zarr.open_array(zarr.MemoryStore(), mode="w", shape=data.shape, chunks=(4, 4, 4, data.shape[3]),
                                     dtype="f4", compressor=None)
            served[:] = data
            body = served.store[path[len("/s0/"):]]
        return SimpleNamespace(status_code=200, content=body, json=lambda: json.loads(body))

    monkeypatch.setattr(view_labels.requests, "get", get)
    monkeypatch.setattr(view_labels, "fetch_model_info", lambda host: {
        "available": True, "output_class": fake.output_class,
        "output_voxel_size": [16] * 3, "effective_output_voxel_size": [16] * 3,
    })
    get_session().jobs = [SimpleNamespace(model_name="model", host="http://gpu:8000")]
    return fake


def _predicting(foreground, low=0.1, high=0.9, channels=1):
    return np.where(foreground[..., None], high, low).repeat(channels, axis=-1).astype("f4")


@pytest.mark.parametrize("output_class, low, high", [
    pytest.param(UNIT, 0.4, 0.6, id="probabilities-at-0.5"),
    pytest.param(SIGNED_UNIT, -0.2, 0.2, id="tanh-at-0"),
    pytest.param(UNBOUNDED, -1.0, 1.0, id="logits-or-distance-at-0"),
])
def test_a_seed_is_the_thresholded_prediction_in_the_unpainted_voxels_of_the_view(
        dashboard, served, server, output_class, low, high):
    labels = served()
    labels[6, 6, 6] = 1  # painted background where the model says foreground
    labels[10, 10, 10] = 2  # painted foreground where it says background
    foreground = np.zeros((16,) * 3, bool)
    foreground[5:9, 5:9, 5:9] = True
    server.prediction, server.output_class = _predicting(foreground, low, high), output_class

    body = dashboard.post(SEED, json={}).get_json()

    assert body["success"] and body["model"] == "model" and body["reload_viewer"], body
    expected = np.zeros((16,) * 3, "u1")
    # The object's id is the lowest the box does not hold: 2 is the painted voxel's.
    expected[BOX] = np.where(foreground[BOX], 3, 1)
    expected[6, 6, 6], expected[10, 10, 10] = 1, 2
    np.testing.assert_array_equal(labels[:], expected)
    assert (body["filled_foreground"], body["filled_background"]) == (4**3 - 1, 8**3 - 4**3 - 1)
    assert served.synced == ["vol-1"]


@pytest.mark.parametrize("dtype, ids", [
    pytest.param("u2", (9, 10), id="instance-volume-counts-up"),
    pytest.param("u1", (2, 3), id="uint8-volume-reuses-free-ids"),
])
def test_an_affinity_seed_labels_each_object(dashboard, served, server, monkeypatch, dtype, ids):
    """A painted object keeps its id over the rest of it; a new one gets the
    next id (an instance volume) or the lowest free one (uint8)."""
    labels = served(dtype)
    labels[5, 5, 5] = 9 if dtype == "u2" else 2
    foreground = np.zeros((16,) * 3, bool)
    foreground[4:7, 4:7, 4:7] = foreground[9:12, 9:12, 9:12] = True
    # Three affinity channels and an LSD one, which the seed must not read.
    server.prediction = _predicting(foreground, low=0.4, channels=4)
    server.prediction[..., 3] = 0.9
    monkeypatch.setattr(view_labels, "find_model_config", lambda name: object())
    monkeypatch.setattr(view_labels, "autodetect_output_type",
                        lambda config, output_type, offsets: ("affinities", "[[1,0,0],[0,1,0],[0,0,1]]"))

    assert dashboard.post(SEED, json={}).get_json()["success"]
    assert (labels[4:7, 4:7, 4:7] == ids[0]).all() and (labels[9:12, 9:12, 9:12] == ids[1]).all()
    assert labels[8, 8, 8] == 1


@pytest.mark.parametrize("position, box", [
    pytest.param((8, 8, 8), BOX, id="the-view"),
    pytest.param((1, 8, 8), (slice(0, 5), BOX[1], BOX[2]), id="clipped-to-the-volume"),
])
def test_all_background_fills_only_the_unpainted_voxels_of_the_view(dashboard, served, position, box):
    labels = served(position=position)
    labels[box[0].start, 6, 6] = 2

    body = dashboard.post(BACKGROUND, json={}).get_json()

    expected = np.zeros((16,) * 3, "u1")
    expected[box] = 1
    expected[box[0].start, 6, 6] = 2
    np.testing.assert_array_equal(labels[:], expected)
    assert body["offset_voxels"] == [s.start for s in box]


def test_a_chunk_only_on_disk_goes_up_before_the_box_is_written(dashboard, served, tmp_path):
    """Else the box rewrites the chunk from zeros and the pull copies that over disk's."""
    labels = served()
    _volume_array(zarr.open_group(str(tmp_path / "vol-1.zarr")), "u1")[0, 0, 0] = 2

    dashboard.post(BACKGROUND, json={})

    assert labels[0, 0, 0] == 2 and labels[4, 4, 4] == 1


@pytest.mark.parametrize("url, setup, status", [
    pytest.param(SEED, lambda: get_session().annotation_volumes.clear(), 409, id="no-volume"),
    pytest.param(BACKGROUND, lambda: get_session().minio_state.update(ip=None), 409, id="minio-not-running"),
    pytest.param(SEED, lambda: get_session().jobs.clear(), 409, id="no-server"),
])
def test_refusals_write_nothing(dashboard, served, server, url, setup, status):
    labels = served()
    setup()
    response = dashboard.post(url, json={})
    assert response.status_code == status and response.get_json()["error"]
    assert not labels[:].any()


def test_a_large_box_is_labelled_only_once_confirmed(dashboard, served, monkeypatch):
    labels = served()
    monkeypatch.setattr(view_labels, "LARGE_BOX_VOXELS", 8**3)
    asked = dashboard.post(BACKGROUND, json={"size_nm": 16 * 16})
    assert asked.status_code == 409 and asked.get_json()["needs_confirmation"]
    assert not labels[:].any()
    assert dashboard.post(BACKGROUND, json={"size_nm": 16 * 16, "confirm": True}).get_json()["success"]
    assert labels[:].all()


def _wall(labels, through_every_slice):
    """One object, id 2, across the view, with a background wall at y = 8 in every slice or all but one."""
    labels[BOX] = 2
    labels[4:12 if through_every_slice else 11, 8, 4:12] = 1


@pytest.mark.parametrize("paint, expected, counts", [
    pytest.param(lambda l: _wall(l, True), lambda e: e.__setitem__((slice(4, 12), slice(9, 12), slice(4, 12)), 3),
                 {"objects": 2, "split": 1, "merged": 0}, id="a-wall-through-every-slice-splits"),
    pytest.param(lambda l: _wall(l, False), lambda e: None, {"objects": 1, "split": 0, "merged": 0},
                 id="a-wall-missing-a-slice-splits-nothing"),
])
def test_split_relabels_the_views_objects_by_connected_component(dashboard, served, paint, expected, counts):
    """The larger side keeps the id, the smaller gets the lowest free one; the
    brush paints one slice, and an object cut in one slice is still one object."""
    labels = served()
    paint(labels)
    before = labels[:]

    body = dashboard.post(SPLIT, json={}).get_json()

    assert body["success"] and {k: body[k] for k in counts} == counts, body
    want = before.copy()
    expected(want)
    np.testing.assert_array_equal(labels[:], want)
    assert body["reload_viewer"] is bool(counts["split"])


def test_split_merges_objects_a_stroke_joins_and_keeps_the_rest(dashboard, served):
    labels = served("u2")
    labels[4:12, 4:7, 4:12] = 9
    labels[4:12, 8:12, 4:12] = 10
    labels[4:12, 7, 4:12] = 10  # a stroke joining them: one object, mostly 10
    labels[0:3, 0:3, 0:3] = 20  # outside the view: untouched

    body = dashboard.post(SPLIT, json={}).get_json()

    assert (body["objects"], body["split"], body["merged"]) == (1, 0, 1)
    assert (labels[4:12, 4:12, 4:12] == 10).all() and (labels[0:3, 0:3, 0:3] == 20).all()


def test_the_seed_settings_raise_the_threshold_and_drop_specks(dashboard, served, server):
    """A probability of 0.6 is foreground at the model's boundary and background
    at 0.7; a lone voxel is a speck once min_size is 2."""
    labels = served()
    foreground = np.zeros((16,) * 3, bool)
    foreground[5:9, 5:9, 5:9] = True
    foreground[10, 10, 10] = True  # a one-voxel speck
    server.prediction = _predicting(foreground, low=0.1, high=0.6)

    body = dashboard.post(SEED, json={"threshold": 0.7, "min_size": 2}).get_json()
    assert body["success"] and body["filled_foreground"] == 0 and body["threshold"] == 0.7, body
    assert (labels[BOX] == 1).all()

    labels[BOX] = 0
    body = dashboard.post(SEED, json={"min_size": 2}).get_json()
    assert body["filled_foreground"] == 4 ** 3 and labels[10, 10, 10] == 1 and (labels[5:9, 5:9, 5:9] == 2).all()
    assert dashboard.post(SEED, json={"threshold": 1.5}).status_code == 400


def test_a_seed_reads_the_model_chosen_else_the_latest_finetune(dashboard, served, server, monkeypatch):
    served()
    get_session().jobs = [SimpleNamespace(model_name=name, host=f"http://{name}:8000")
                          for name in ("model", "model_finetuned_1", "model_finetuned_2", "other")]
    sources = dashboard.get("/api/finetune/view-labels/sources").get_json()
    assert (sources["models"], sources["default"]) == (
        ["model", "model_finetuned_1", "model_finetuned_2", "other"], "model_finetuned_2")
    read = []
    monkeypatch.setattr(view_labels, "read_prediction",
                        lambda host, name, *a: read.append(name) or (np.zeros((1, 8, 8, 8), "f4"), a[1], a[2]))
    dashboard.post(SEED, json={})
    dashboard.post(SEED, json={"model": "model"})
    dashboard.post(SEED, json={"model": "other"})
    assert read == ["model_finetuned_2", "model", "other"]
    assert dashboard.post(SEED, json={"model": "gone"}).status_code == 409


def test_the_picker_lists_a_lone_model_before_any_volume_exists(dashboard):
    """It was empty: it only listed the volume's model, and there was none."""
    get_session().jobs = [SimpleNamespace(model_name="mito_aff", host="http://gpu:8000")]
    get_session().annotation_volumes.clear()
    assert dashboard.get("/api/finetune/view-labels/sources").get_json() == {
        "success": True, "models": ["mito_aff"], "default": "mito_aff"}


def test_a_label_change_re_reads_only_the_paint_layer_under_a_new_url(dashboard, served, viewer):
    """Neuroglancer keeps a source's chunks per URL, so the same URL re-read
    nothing, and reloading the viewer had every server recompute the view.
    A port with one more leading zero is the same server to a browser."""
    served()
    with viewer.txn() as s:
        s.layers["other"] = neuroglancer.ImageLayer(source="zarr://http://x/raw")
        # What the user set in the Draw tab, as the browser syncs it back.
        s.layers["annotation_vol-1"] = neuroglancer.viewer_state.make_layer({
            "type": "segmentation", "tab": "Draw", "paintValue": "7", "brushSize": 12,
            "source": {"url": "s3+http://m:9000/annotations/vol-1.zarr/annotation",
                       "subsources": {"default": {"writingEnabled": True}}}})

    first = dashboard.post(BACKGROUND, json={}).get_json()
    assert first["reload_viewer"] and first["layer_refreshed"]
    layers = viewer.state.layers
    state = layers["annotation_vol-1"].layer.to_json()
    assert state["source"]["url"] == "s3+http://m:09000/annotations/vol-1.zarr/annotation"
    # Only the URL changed: rebuilding the layer reset the paint value and brush size.
    assert (state["paintValue"], state["brushSize"], state["tab"]) == ("7", 12, "Draw")
    assert state["source"]["subsources"] == {"default": {"writingEnabled": True}}
    assert [layer.name for layer in layers] == ["other", "annotation_vol-1"]

    dashboard.post(SPLIT, json={})  # nothing to relabel: the URL stays
    assert viewer.state.layers["annotation_vol-1"].layer.to_json()["source"]["url"].endswith(":09000/annotations/vol-1.zarr/annotation")
