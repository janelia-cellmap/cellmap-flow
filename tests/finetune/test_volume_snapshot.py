"""What each way of making an annotation volume leaves on disk, pinned.

Recorded before the volume code moved into cellmap_flow.finetune.session.
Four things make volumes over one raw dataset: create-volume, a YAML crop
import into a fresh session, build_corrections and the instance-correction
seeder. For each, every .zgroup/.zattrs/.zarray, the chunk files and
_virtual_sources.json are kept, with the volume's registry record. MinIO and
neuroglancer write chunks straight into these arrays, and the trainer reads
them and the manifest, so they are a format (see PHASE3 R3).

Resuming a session, listing sessions and one round of syncing a painted
chunk back from MinIO are recorded too. mc and s3fs are fakes that keep the
bucket in a dict; nothing is served.
"""

import hashlib
import json
import os
import re
import subprocess
from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace

import neuroglancer
import numpy as np
import pytest
import s3fs
import zarr

from cellmap_flow.dashboard import finetune_utils as fu

BUCKET = "annotations"
NORM = [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}]
POST = [{"name": "SigmoidPostprocessor"}]
# What a server reports for a model: 12^3 input at 8 nm, 4^3 output at 16 nm.
GEOMETRY = SimpleNamespace(
    read_shape=[96, 96, 96], write_shape=[64, 64, 64],
    input_voxel_size=[8, 8, 8], output_voxel_size=[16, 16, 16], output_channels=1,
)


def _ome(path, scales, translation):
    return [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [
            {"path": p, "coordinateTransformations": [
                {"type": "scale", "scale": [vs] * 3},
                {"type": "translation", "translation": [t] * 3}]}
            for p, vs, t in zip(path, scales, translation)
        ],
    }]


class FakeBucket:
    """The bucket as s3fs sees it: objects by key, each with an ETag."""

    def __init__(self):
        self.objects = {}

    def put(self, key, data):
        self.objects[key] = (bytes(data), hashlib.md5(bytes(data)).hexdigest())

    def exists(self, path):
        path = path.rstrip("/")
        return any(k == path or k.startswith(path + "/") for k in self.objects)

    def ls(self, path, detail=False):
        prefix = path.rstrip("/") + "/"
        names = sorted({prefix + k[len(prefix):].split("/")[0] for k in self.objects if k.startswith(prefix)})
        if not names:
            raise FileNotFoundError(path)
        if not detail:
            return names
        return [{"name": n, "ETag": f'"{self.objects[n][1]}"', "size": len(self.objects[n][0])}
                if n in self.objects else {"name": n, "type": "directory"} for n in names]

    def get(self, src, dst):
        Path(dst).write_bytes(self.objects[src][0])

    def cat(self, path):
        return self.objects[path][0]


class _Store(MutableMapping):
    """s3fs.S3Map over the fake bucket."""

    def __init__(self, bucket, root):
        self.bucket, self.root = bucket, root.rstrip("/")

    def __getitem__(self, key):
        try:
            return self.bucket.objects[f"{self.root}/{key}"][0]
        except KeyError:
            raise KeyError(key) from None

    def __setitem__(self, key, value):
        self.bucket.put(f"{self.root}/{key}", value)

    def __delitem__(self, key):
        del self.bucket.objects[f"{self.root}/{key}"]

    def __iter__(self):
        prefix = self.root + "/"
        return (k[len(prefix):] for k in list(self.bucket.objects) if k.startswith(prefix))

    def __len__(self):
        return sum(1 for _ in self)


class _Alive:
    pid = 1

    def poll(self):
        return None


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A raw pyramid, a crop, a segmentation, a viewer and a fake MinIO."""
    from cellmap_flow.dashboard.routes.finetune import common
    from cellmap_flow.globals import g
    from cellmap_flow.utils import model_geometry

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

    bucket = FakeBucket()

    def fake_run(cmd, **kwargs):
        # `mc mirror --overwrite <zarr> myserver/<bucket>/<key>`: upload it all.
        assert cmd[:3] == ["mc", "mirror", "--overwrite"], cmd
        src, key = cmd[3], cmd[4].split("/", 1)[1]
        for path in Path(src).rglob("*"):
            if path.is_file():
                bucket.put(f"{key}/{path.relative_to(src)}", path.read_bytes())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    state = {"process": _Alive(), "ip": "10.0.0.1", "port": 9000, "bucket": BUCKET,
             "output_base": None, "sync_thread": None}
    volumes, sessions = {}, {}
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(s3fs, "S3FileSystem", lambda **kw: bucket)
    monkeypatch.setattr(s3fs, "S3Map", lambda root, s3, check=True: _Store(bucket, root))
    monkeypatch.setattr(fu, "_require_minio_binaries", lambda: None)
    monkeypatch.setattr(fu, "minio_state", state)
    monkeypatch.setattr(fu, "annotation_volumes", volumes)
    monkeypatch.setattr(fu, "output_sessions", sessions)
    monkeypatch.setattr(common, "output_sessions", sessions)
    monkeypatch.setattr(model_geometry, "model_geometry_config", lambda name: GEOMETRY)
    for name, value in dict(
        annotation_volumes=volumes, output_sessions=sessions, viewer=neuroglancer.Viewer(),
        raw=None, dataset_path=str(tmp_path / "raw.zarr" / "em"),
        models_config=[SimpleNamespace(name="m")], input_norm_config=NORM, postprocess_config=POST,
    ).items():
        monkeypatch.setattr(g, name, value, raising=False)

    from cellmap_flow.dashboard.app import app

    return SimpleNamespace(tmp=tmp_path, client=app.test_client(), bucket=bucket, state=state,
                           volumes=volumes, viewer=g.viewer)


class _Names:
    """Replaces what changes from run to run: paths, ids, clocks."""

    def __init__(self, tmp):
        self.tmp, self.ids = str(tmp), {}

    def __call__(self, value):
        if isinstance(value, dict):
            return {self(k): ("<time>" if k in ("created_at",) else self(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        if not isinstance(value, str):
            return value
        value = value.replace(self.tmp, "<tmp>")
        value = re.sub(r"vol-[0-9a-f]{8}-\d{8}-\d{6}",
                       lambda m: self.ids.setdefault(m.group(), f"vol-{len(self.ids) + 1}"), value)
        return re.sub(r"\d{8}_\d{6}", "<session>", value)


def _tree(root, names):
    """The metadata files, the manifest and the chunk keys under ``root``."""
    tree = {}
    for path in sorted(Path(root).rglob("*")):
        rel = names(str(path.relative_to(root)))
        if path.name in (".zgroup", ".zattrs", ".zarray", "_virtual_sources.json"):
            tree[rel] = names(json.loads(path.read_text()))
        elif re.fullmatch(r"\d+\.\d+\.\d+", path.name):
            tree.setdefault(names(str(path.parent.relative_to(root))) + "/<chunks>", []).append(path.name)
    return tree


def _copy(record):
    """A registry record as it is now, with the chunk ETags (blosc output,
    which differs between library versions) replaced."""
    record = json.loads(json.dumps(record))
    record["chunk_sync_state"] = {k: "<etag>" for k in record.get("chunk_sync_state", {})}
    return record


def _canonical(value):
    # json, so 16 and 16.0 differ: a writer that turned ints into floats
    # changes the files even though == would not notice.
    return json.dumps(value, sort_keys=True, indent=1)


def _record(world):
    """Everything this test pins, by creator."""
    names = _Names(world.tmp)
    post = lambda url, **body: world.client.post(url, json=body).get_json()  # noqa: E731
    out = {}

    session_a = world.tmp / "a"
    created = post("/api/finetune/create-volume", model_name="m", output_path=str(session_a))
    vid = created["volume_id"]
    out["create_volume"] = {
        "response": created, "record": _copy(world.volumes[vid]), "tree": _tree(session_a, names),
    }

    session_c = world.tmp / "c"
    loaded = post("/api/finetune/load-crops", model_name="m", output_path=str(session_c),
                  yaml=str(world.tmp / "crops.yaml"))
    out["load_crops"] = {
        "response": loaded, "record": _copy(world.volumes[loaded["volume_id"]]),
        "tree": _tree(session_c, names),
    }

    from cellmap_flow.finetune.build_corrections import build_corrections

    built = world.tmp / "built"
    record = build_corrections(
        raw_dataset_path=str(world.tmp / "raw.zarr" / "em"), crops_yaml=str(world.tmp / "crops.yaml"),
        output_dir=str(built), input_shape=(12, 12, 12), output_shape=(4, 4, 4),
        input_voxel_size=(8, 8, 8), output_voxel_size=(16, 16, 16), model_name="m",
        input_norm=NORM, postprocess=POST,
    )
    out["build_corrections"] = {
        "tree": _tree(built, names),
        "record": {k: record[k] for k in ("geometry", "crops", "total_fg_voxels")},
    }

    seeded = post("/api/viewer/create-instance-correction", roi_name="roi", model_name="m",
                  instance_zarr_path=str(world.tmp / "seg.zarr"), dilation_radius_voxels=1)
    out["instance_correction"] = {
        "response": seeded, "record": _copy(world.volumes[seeded["volume_id"]]),
        "tree": _tree(world.tmp / "instance_corrections", names),
    }

    # A stroke painted into the created volume's first chunk, in MinIO.
    painted = zarr.open_array(_Store(world.bucket, f"{BUCKET}/{vid}.zarr/annotation/s0"), mode="r+")
    painted[0:4, 0:4, 0:4] = 2
    synced = fu.sync_all_annotations_from_minio(force=False)
    state = world.volumes[vid]["chunk_sync_state"]
    out["sync"] = {
        "synced": synced,
        "chunk_sync_state": {k: v == world.bucket.objects[f"{BUCKET}/{vid}.zarr/annotation/s0/{k}"][1]
                             for k, v in state.items()},
        "local_chunks": sorted(os.listdir(Path(created["zarr_path"]) / "annotation" / "s0")),
    }

    out["list_existing_sessions"] = post("/api/finetune/list-existing-sessions", output_path=str(session_a))
    session_path = out["list_existing_sessions"]["sessions"][0]["session_path"]
    resumed = post("/api/finetune/load-existing-volume", source_session_path=session_path,
                   output_path=str(world.tmp / "b"))
    out["load_existing_volume"] = {
        "response": resumed, "record": _copy(world.volumes[resumed["volume_id"]]),
        "tree": _tree(world.tmp / "b", names),
    }
    out["viewer_layers"] = [layer.name for layer in world.viewer.state.layers]
    return names(out)


ZGROUP = {"zarr_format": 2}
BLOSC = {"blocksize": 0, "clevel": 3, "cname": "zstd", "id": "blosc", "shuffle": 1}
RAW = "<tmp>/raw.zarr/em"
# The root attrs a volume over RAW gets; each creator differs from these a little.
ATTRS = {
    "chunk_size": [4, 4, 4], "claimed_input_voxel_size": [8, 8, 8],
    "claimed_output_voxel_size": [16, 16, 16], "created_at": "<time>",
    "dataset_offset_nm": [4.0, 4.0, 4.0], "dataset_path": RAW, "dataset_shape_voxels": [24, 24, 24],
    "input_norm": NORM, "input_size": [12, 12, 12], "input_voxel_size": [8.0, 8.0, 8.0],
    "model_name": "m", "output_voxel_size": [16.0, 16.0, 16.0], "postprocess": POST,
    "type": "annotation_volume",
}
CROP = {"annotation_offset_voxels": [4, 4, 4], "annotation_shape_voxels": [6, 6, 6],
        "n_fg_voxels": 27, "name": "c1", "path": "<tmp>/crop.zarr"}
CROP_CHUNKS = ["1.1.1", "1.1.2", "1.2.1", "1.2.2", "2.1.1", "2.1.2", "2.2.1", "2.2.2"]
RECORD = {
    "chunk_sync_state": {}, "claimed_input_voxel_size": [8, 8, 8],
    "claimed_output_voxel_size": [16, 16, 16], "dataset_offset_nm": [4.0, 4.0, 4.0],
    "dataset_path": RAW, "input_size": [12, 12, 12], "input_voxel_size": [8.0, 8.0, 8.0],
    "model_name": "m", "output_size": [4, 4, 4], "output_voxel_size": [16.0, 16.0, 16.0],
}


def _volume(at, attrs, *, translation=(4.0, 4.0, 4.0), shape=(24, 24, 24), dtype="|u1",
            chunks=None, synced=False):
    files = {
        f"{at}/.zgroup": ZGROUP,
        f"{at}/.zattrs": attrs,
        f"{at}/annotation/.zgroup": ZGROUP,
        f"{at}/annotation/.zattrs": {"multiscales": [{
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [{"coordinateTransformations": [
                {"scale": [16.0, 16.0, 16.0], "type": "scale"},
                {"translation": list(translation), "type": "translation"}], "path": "s0"}],
            "name": "annotation", "version": "0.4"}]},
        f"{at}/annotation/s0/.zarray": {
            "chunks": [4, 4, 4], "compressor": BLOSC, "dtype": dtype, "fill_value": 0,
            "filters": None, "order": "C", "shape": list(shape), "zarr_format": 2},
    }
    if chunks:
        files[f"{at}/annotation/s0/<chunks>"] = chunks
    if synced:  # the sync from MinIO copies s0's (empty) attrs
        files[f"{at}/annotation/s0/.zattrs"] = {}
    return files


def _manifest(volume, input_voxel_size=(8.0, 8.0, 8.0)):
    return {
        "dense_to_sparse_ratio": None, "input_norm": NORM, "input_size_voxels": [12, 12, 12],
        "input_voxel_size_nm": list(input_voxel_size), "jitter_voxels": None,
        "kind": "volume_zarr_v1", "output_size_voxels": [4, 4, 4],
        "output_voxel_size_nm": [16.0, 16.0, 16.0], "patches_per_epoch": None,
        "postprocess": POST, "raw_dataset_path": RAW, "seed": 0, "volume_zarr_path": volume,
    }


def _session(base, volume, attrs, **kw):
    return {
        "<session>/corrections/.zgroup": ZGROUP,
        "<session>/corrections/_virtual_sources.json": _manifest(
            f"<tmp>/{base}/<session>/corrections/{volume}.zarr"),
        **_volume(f"<session>/corrections/{volume}.zarr", attrs, **kw),
    }


def _url(key):
    return f"http://10.0.0.1:9000/annotations/{key}.zarr"


EXPECTED = {
    "create_volume": {
        "response": {
            "metadata": {"chunk_size": [4, 4, 4], "claimed_output_voxel_size": [16, 16, 16],
                         "dataset_offset_nm": [4.0, 4.0, 4.0], "dataset_shape_voxels": [24, 24, 24],
                         "output_voxel_size": [16.0, 16.0, 16.0]},
            "minio_url": _url("vol-1"), "neuroglancer_url": _url("vol-1") + "/annotation",
            "success": True, "volume_id": "vol-1",
            "zarr_path": "<tmp>/a/<session>/corrections/vol-1.zarr",
        },
        "record": {**RECORD, "corrections_dir": "<tmp>/a/<session>/corrections",
                   "zarr_path": "<tmp>/a/<session>/corrections/vol-1.zarr"},
        "tree": _session("a", "vol-1", ATTRS),
    },
    "load_crops": {
        "response": {"created_new_volume": True, "errors": [], "fg_voxels_written": 27,
                     "n_crops_imported": 1, "n_crops_requested": 1, "n_errors": 0,
                     "success": True, "volume_id": "vol-2"},
        "record": {**RECORD, "corrections_dir": "<tmp>/c/<session>/corrections",
                   "minio_url": _url("vol-2"), "zarr_path": "<tmp>/c/<session>/corrections/vol-2.zarr"},
        "tree": _session("c", "vol-2", {**ATTRS, "imported_crops": [CROP]},
                         chunks=CROP_CHUNKS, synced=True),
    },
    "build_corrections": {
        "tree": {
            "_virtual_sources.json": _manifest("<tmp>/built/vol-3.zarr"),
            **_volume("vol-3.zarr", {**ATTRS, "claimed_input_voxel_size": [8.0, 8.0, 8.0],
                                     "claimed_output_voxel_size": [16.0, 16.0, 16.0],
                                     "imported_crops": [CROP]}, chunks=CROP_CHUNKS),
        },
        "record": {
            "crops": [{"connected_components": False, "fg_ids": [5], "mode": "dense",
                       "n_fg_voxels": 27, "name": "c1", "path": "<tmp>/crop.zarr"}],
            "geometry": {
                "claimed_input_voxel_size_nm": [8.0, 8.0, 8.0],
                "claimed_output_voxel_size_nm": [16.0, 16.0, 16.0],
                "dataset_offset_nm": [4.0, 4.0, 4.0], "dataset_shape_voxels": [24, 24, 24],
                "effective_input_voxel_size_nm": [8.0, 8.0, 8.0],
                "effective_output_voxel_size_nm": [16.0, 16.0, 16.0],
                "input_shape": [12, 12, 12], "output_shape": [4, 4, 4],
            },
            "total_fg_voxels": 27,
        },
    },
    "instance_correction": {
        "response": {
            "layer_name": "roi_annotation", "minio_url": _url("roi_annotation"),
            "mode": "fresh_seed", "neuroglancer_url": _url("roi_annotation") + "/annotation",
            "reload_page": True, "success": True, "volume_id": "roi_annotation",
            "zarr_path": "<tmp>/instance_corrections/roi_annotation.zarr",
        },
        "record": {
            **{k: v for k, v in RECORD.items() if not k.startswith("claimed")},
            "corrections_dir": "<tmp>/instance_corrections", "dataset_offset_nm": [8.0, 168.0, 8.0],
            "minio_url": _url("roi_annotation"),
            "zarr_path": "<tmp>/instance_corrections/roi_annotation.zarr",
        },
        # On the segmentation's grid, uint16, seeded chunks only.
        "tree": _volume(
            "roi_annotation.zarr",
            {**ATTRS, "claimed_input_voxel_size": [8.0, 8.0, 8.0],
             "claimed_output_voxel_size": [16.0, 16.0, 16.0], "dataset_offset_nm": [8.0, 168.0, 8.0],
             "dataset_shape_voxels": [8, 8, 8], "seed_dilation_radius_voxels": 1,
             "seed_n_instances": 9, "seed_source_instance_zarr": "<tmp>/seg.zarr"},
            translation=(8.0, 168.0, 8.0), shape=(8, 8, 8), dtype="<u2", chunks=["0.0.0", "1.1.1"],
        ),
    },
    # One round of the periodic sync: the stroke reaches vol-1's own zarr and
    # its ETag is recorded. The other two had their mirrored chunks pulled back.
    "sync": {"chunk_sync_state": {"0.0.0": True}, "local_chunks": [".zarray", ".zattrs", "0.0.0"],
             "synced": 3},
    "list_existing_sessions": {"sessions": [{
        "chunk_count": 1, "session_id": "<session>", "session_path": "<tmp>/a/<session>",
        "volumes": [{"path": "<tmp>/a/<session>/corrections/vol-1.zarr", "volume_id": "vol-1"}],
    }], "success": True},
    "load_existing_volume": {
        "response": {
            "copied_count": 1, "copied_minio": False,
            "metadata": ATTRS, "minio_url": _url("vol-1"),
            "neuroglancer_url": _url("vol-1") + "/annotation", "new_session_path": "<tmp>/b/<session>",
            "painted_chunk_count": 1, "skipped_chunk_extracts": 0, "success": True,
            "volume_id": "vol-1", "zarr_path": "<tmp>/b/<session>/corrections/vol-1.zarr",
        },
        "record": {**{k: v for k, v in RECORD.items() if not k.startswith("claimed")},
                   "corrections_dir": "<tmp>/b/<session>/corrections",
                   "zarr_path": "<tmp>/b/<session>/corrections/vol-1.zarr"},
        "tree": _session("b", "vol-1", ATTRS, chunks=["0.0.0"], synced=True),
    },
    "viewer_layers": ["annotation_vol-2", "annotated_regions", "roi_annotation"],
}


def test_what_each_volume_creator_writes(world):
    actual = _record(world)
    assert {k: _canonical(v) for k, v in actual.items()} == {
        k: _canonical(v) for k, v in EXPECTED.items()
    }
