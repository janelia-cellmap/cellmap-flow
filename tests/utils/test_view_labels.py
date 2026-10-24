"""Labelling the view in one click: a seed from the model's prediction, or all background.

The volume is a 16^3 grid of 16 nm voxels, its labels in an in-memory
"MinIO"; the view is one 8-voxel output patch, voxels 4..12 on each axis.
The model's server is a local HTTP server of the zarr a test predicts over
the same grid, read as any server is, so the reading of chunks and the
placing of voxels are tested along with the labels; and one test reads a
running chunkmirage server.
"""

import functools
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import neuroglancer
import numpy as np
import pytest
import worldarray as wa
import zarr
from starlette.testclient import TestClient

from cellmap_flow.dashboard.routes.finetune import view_labels
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import fill, minio
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
from cellmap_flow.serving.client import fetch_model_info
from cellmap_flow.serving.probe import UNBOUNDED, UNIT
from tests.utils.serving_helpers import COARSER_MODEL, layer, read_chunk, write_raw, write_script

SEED, BACKGROUND, SPLIT, UNDO = (f"/api/finetune/view-labels/{what}" for what in ("seed", "background", "split", "undo"))
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


class _ServedFiles(SimpleHTTPRequestHandler):
    """The files of one served zarr at any layer's URL, ``/<layer name>/zarr/<key>``."""

    def translate_path(self, path):
        return super().translate_path("/" + path.partition("/zarr/")[2])

    def log_message(self, *args):
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A running server for "model", over HTTP on this machine: ``server.serve(prediction)``
    (z, y, x, c over the volume) serves it as the chunkmirage server does, an OME group under
    the layer's name followed by ``/zarr``, channels first, "/" keys."""
    root = tmp_path / "served"
    files = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_ServedFiles, directory=str(root)))
    # Polled often, so shutting it down takes no half second.
    threading.Thread(target=files.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    fake = SimpleNamespace(output_class=UNIT, host=f"http://127.0.0.1:{files.server_port}")

    def serve(prediction, output_class=UNIT):
        data = np.moveaxis(prediction, -1, 0)
        level = wa.create(str(root / "s0"), data.shape, data.dtype, voxel_size=(16,) * 3, translation=(8.0,) * 3,
                           axes=("c", "z", "y", "x"), chunks=(data.shape[0], 4, 4, 4), format="zarr2", separator="/")
        level.write_index((0,) * data.ndim, data.shape, data)
        fake.output_class = output_class

    fake.serve = serve
    monkeypatch.setattr(view_labels, "fetch_model_info", lambda host: {
        "available": True, "output_class": fake.output_class, "engine": "chunkmirage",
        "output_voxel_size": [16] * 3, "effective_output_voxel_size": [16] * 3,
    })
    get_session().jobs = [SimpleNamespace(model_name="model", host=fake.host)]
    yield fake
    files.shutdown()
    files.server_close()


def _predicting(foreground, low=0.1, high=0.9, channels=1):
    return np.where(foreground[..., None], high, low).repeat(channels, axis=-1).astype("f4")


@pytest.mark.parametrize("output_class, low, high", [
    pytest.param(UNIT, 0.4, 0.6, id="probabilities-at-0.5"),
    pytest.param(UNBOUNDED, -1.0, 1.0, id="logits-or-distance-at-0"),
])
def test_a_seed_is_the_thresholded_prediction_in_the_unpainted_voxels_of_the_view(
        dashboard, served, server, output_class, low, high):
    labels = served()
    labels[6, 6, 6] = 1  # painted background where the model says foreground
    labels[10, 10, 10] = 2  # painted foreground where it says background
    foreground = np.zeros((16,) * 3, bool)
    foreground[5:9, 5:9, 5:9] = True
    server.serve(_predicting(foreground, low, high), output_class)

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
    prediction = _predicting(foreground, low=0.4, channels=4)
    prediction[..., 3] = 0.9
    server.serve(prediction)
    monkeypatch.setattr(view_labels, "find_model_config", lambda name: object())
    monkeypatch.setattr(view_labels, "autodetect_output_type",
                        lambda config, output_type, offsets: ("affinities", "[[1,0,0],[0,1,0],[0,0,1]]"))

    assert dashboard.post(SEED, json={}).get_json()["success"]
    assert (labels[4:7, 4:7, 4:7] == ids[0]).all() and (labels[9:12, 9:12, 9:12] == ids[1]).all()
    assert labels[8, 8, 8] == 1


def test_the_prediction_read_from_a_running_chunkmirage_server_is_its_chunks_on_the_annotation_grid(tmp_path):
    """tensorstore reads the server over HTTP as neuroglancer does, each chunk
    computed on asking. Its output (2 channels, 16 nm voxels from the raw's
    corner at 0 nm, in 4^3 chunks) under 8 nm annotation voxels: each takes
    the output voxel its centre lies in, and the box ends where the output does."""
    data = (np.arange(12**3) % 251).astype(np.uint8).reshape((12,) * 3)
    chunkmirage = ChunkmirageServer(write_raw(tmp_path, data), ScriptModelConfig(script_path=write_script(tmp_path, COARSER_MODEL)))
    thread = threading.Thread(target=chunkmirage.run, kwargs={"port": 0}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while getattr(chunkmirage, "port", None) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        host = f"http://127.0.0.1:{chunkmirage.port}"
        volume = {"output_voxel_size": [8] * 3, "dataset_offset_nm": [4.0] * 3}  # voxel 0's centre
        prediction, lo, hi = view_labels.read_prediction(
            host, "m", volume, np.array([2, 3, 1]), np.array([14, 12, 9]), fetch_model_info(host))
        # Every chunk of the served 6^3, asked for one by one.
        client, served = TestClient(chunkmirage.app), np.zeros((2, 6, 6, 6), np.float32)
        for index in np.ndindex(2, 2, 2):
            chunk = read_chunk(client, layer(), index)
            served[(slice(None), *(slice(4 * i, 4 * i + n) for i, n in zip(index, chunk.shape[1:])))] = chunk
    finally:
        chunkmirage.stop()
        thread.join(10)

    assert (lo.tolist(), hi.tolist()) == ([2, 3, 1], [12, 12, 9])
    on_annotation_grid = served.repeat(2, axis=1).repeat(2, axis=2).repeat(2, axis=3)
    np.testing.assert_array_equal(prediction, on_annotation_grid[:, 2:12, 3:12, 1:9])


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


def test_a_seed_reads_the_model_chosen_else_the_latest_finetune(dashboard, served, server, monkeypatch):
    served()
    get_session().jobs = [SimpleNamespace(model_name=name, host=server.host)
                          for name in ("model", "model_finetuned_1", "model_finetuned_2", "other")]
    sources = dashboard.get("/api/finetune/view-labels/sources").get_json()
    assert (sources["models"], sources["default"]) == (
        ["model", "model_finetuned_1", "model_finetuned_2", "other"], "model_finetuned_2")
    read = []
    monkeypatch.setattr(view_labels, "read_prediction",
                        lambda host, name, *a, **kw: read.append(name) or (np.zeros((1, 8, 8, 8), "f4"), a[1], a[2]))
    dashboard.post(SEED, json={})
    dashboard.post(SEED, json={"model": "model"})
    dashboard.post(SEED, json={"model": "other"})
    assert read == ["model_finetuned_2", "model", "other"]
    assert dashboard.post(SEED, json={"model": "gone"}).status_code == 409


def test_a_volume_painted_before_any_model_seeds_from_the_model_picked_or_the_only_one_served(
        dashboard, served, server, monkeypatch):
    """A volume made before a model was chosen (model_name None) names no
    model to seed from: the one picked under Prediction from does, else the
    only server running; with several running and none picked, the seed is
    refused, not the volume."""
    served()
    get_session().annotation_volumes["vol-1"]["model_name"] = None
    read = []
    monkeypatch.setattr(view_labels, "read_prediction",
                        lambda host, name, *a, **kw: read.append(name) or (np.zeros((1, 8, 8, 8), "f4"), a[1], a[2]))
    get_session().jobs = [SimpleNamespace(model_name=name, host=server.host) for name in ("model", "other")]
    assert dashboard.get("/api/finetune/view-labels/sources").get_json()["default"] is None
    assert dashboard.post(SEED, json={}).status_code == 409
    assert dashboard.post(SEED, json={"model": "other"}).status_code == 200
    get_session().jobs = get_session().jobs[:1]
    assert dashboard.post(SEED, json={}).status_code == 200
    assert read == ["other", "model"]


def test_undo_takes_back_the_last_actions_and_keeps_strokes_painted_since(dashboard, served):
    """All Background over objects the user had not seen: Undo puts the box
    back, except where they painted after it."""
    labels = served()
    labels[5, 5, 5] = 2  # painted before
    assert dashboard.post(UNDO, json={}).status_code == 409, "nothing to undo yet"

    assert dashboard.post(BACKGROUND, json={}).get_json()["can_undo"]
    labels[6, 6, 6] = 3  # painted after the fill, over a voxel it set to 1
    labels[BOX[0].start, 4, 4] = 2  # and another
    body = dashboard.post(UNDO, json={}).get_json()

    assert body["success"] and not body["can_undo"]
    expected = np.zeros((16,) * 3, "u1")
    expected[5, 5, 5], expected[6, 6, 6], expected[BOX[0].start, 4, 4] = 2, 3, 2
    np.testing.assert_array_equal(labels[:], expected)
    assert body["restored"] == 8 ** 3 - 3
    assert dashboard.post(UNDO, json={}).status_code == 409


@pytest.mark.parametrize("output_type, dtype", [("flows", "uint16"), ("affinities", "uint16"), ("binary", "uint8"),
                                                ("distance", "uint8")])
def test_a_new_volume_for_an_instance_model_holds_uint16_ids(monkeypatch, output_type, dtype):
    """282 objects of a Cellpose model's masks in one box did not fit a uint8 volume."""
    from cellmap_flow.dashboard.routes.finetune import common

    monkeypatch.setattr(common, "autodetect_output_type", lambda config, output_type_, offsets: (output_type, None))
    assert common.annotation_dtype_for(object()) == dtype
