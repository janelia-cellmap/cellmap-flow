"""What each way of making an annotation volume leaves on disk, pinned.

Four things make volumes over one raw dataset: create-volume, a YAML crop
import into a fresh session, build_corrections and the instance-correction
seeder. For each, every .zgroup/.zattrs/.zarray, the chunk keys and
_virtual_sources.json are kept, and the route's response. MinIO and
neuroglancer write chunks straight into these arrays, and the trainer reads
them and the manifest, so they are a format (PHASE3 R3). Listing and
resuming a session are recorded too. mc and s3fs are fakes; nothing is served.
"""

import json
import re
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

    mirrored = []

    def fake_run(cmd, **kwargs):
        assert cmd[:3] == ["mc", "mirror", "--overwrite"], cmd
        mirrored.append(cmd[4])
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
    ).items():
        monkeypatch.setattr(get_session(), name, value)
    for name, value in dict(input_norm_config=NORM, postprocess_config=POST).items():
        monkeypatch.setattr(process_chain(), name, value)
    return SimpleNamespace(tmp=tmp_path, client=app.test_client(), mirrored=mirrored)


class _Names:
    """Replaces what changes from run to run: paths, ids, clocks."""

    def __init__(self, tmp):
        self.tmp, self.ids = str(tmp), {}

    def __call__(self, value):
        if isinstance(value, dict):
            return {self(k): ("<time>" if k == "created_at" else self(v)) for k, v in value.items()}
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


def _record(world):
    names = _Names(world.tmp)
    post = lambda url, **body: world.client.post(url, json=body).get_json()  # noqa: E731
    out = {}

    created = post("/api/finetune/create-volume", model_name="m", output_path=str(world.tmp / "a"))
    out["create_volume"] = {"response": created, "tree": _tree(world.tmp / "a", names)}

    loaded = post("/api/finetune/load-crops", model_name="m", output_path=str(world.tmp / "c"),
                  yaml=str(world.tmp / "crops.yaml"))
    out["load_crops"] = {"response": loaded, "tree": _tree(world.tmp / "c", names)}

    from cellmap_flow.finetune.build_corrections import build_corrections

    record = build_corrections(
        raw_dataset_path=str(world.tmp / "raw.zarr" / "em"), crops_yaml=str(world.tmp / "crops.yaml"),
        output_dir=str(world.tmp / "built"), input_shape=(12, 12, 12), output_shape=(4, 4, 4),
        input_voxel_size=(8, 8, 8), output_voxel_size=(16, 16, 16), model_name="m",
        input_norm=NORM, postprocess=POST,
    )
    out["build_corrections"] = {"tree": _tree(world.tmp / "built", names),
                                "record": {k: record[k] for k in ("geometry", "crops")}}

    seeded = post("/api/viewer/create-instance-correction", roi_name="roi", model_name="m",
                  instance_zarr_path=str(world.tmp / "seg.zarr"), dilation_radius_voxels=1)
    out["instance_correction"] = {"response": seeded,
                                  "tree": _tree(world.tmp / "instance_corrections", names)}

    # A stroke on disk, then the session is listed and resumed elsewhere.
    zarr.open_array(str(Path(created["zarr_path"]) / "annotation" / "s0"), mode="r+")[0:4, 0:4, 0:4] = 2
    out["list_existing_sessions"] = post("/api/finetune/list-existing-sessions",
                                         output_path=str(world.tmp / "a"))
    resumed = post("/api/finetune/load-existing-volume", output_path=str(world.tmp / "b"),
                   source_session_path=out["list_existing_sessions"]["sessions"][0]["session_path"])
    out["load_existing_volume"] = {"response": resumed, "tree": _tree(world.tmp / "b", names)}
    out["mirrored"] = world.mirrored
    return names(out)


ZGROUP = {"zarr_format": 2}
RAW = "<tmp>/raw.zarr/em"
# The root attrs of a volume over RAW; each creator differs from these a little.
ATTRS = {
    "chunk_size": [4, 4, 4], "claimed_input_voxel_size": [8, 8, 8],
    "claimed_output_voxel_size": [16, 16, 16], "created_at": "<time>",
    "dataset_offset_nm": [4.0, 4.0, 4.0], "dataset_path": RAW, "dataset_shape_voxels": [24, 24, 24],
    "input_norm": NORM, "input_size": [12, 12, 12], "input_voxel_size": [8.0, 8.0, 8.0],
    "model_name": "m", "output_voxel_size": [16.0, 16.0, 16.0], "postprocess": POST,
    "type": "annotation_volume",
}
FLOAT_CLAIMS = {"claimed_input_voxel_size": [8.0, 8.0, 8.0], "claimed_output_voxel_size": [16.0, 16.0, 16.0]}
CROP = {"annotation_offset_voxels": [4, 4, 4], "annotation_shape_voxels": [6, 6, 6],
        "n_fg_voxels": 27, "name": "c1", "path": "<tmp>/crop.zarr"}
CROP_CHUNKS = ["1.1.1", "1.1.2", "1.2.1", "1.2.2", "2.1.1", "2.1.2", "2.2.1", "2.2.2"]


def _volume(at, attrs, *, translation=(4.0, 4.0, 4.0), shape=(24, 24, 24), dtype="|u1", chunks=None):
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
            "chunks": [4, 4, 4], "dtype": dtype, "fill_value": 0, "filters": None, "order": "C",
            "compressor": {"blocksize": 0, "clevel": 3, "cname": "zstd", "id": "blosc", "shuffle": 1},
            "shape": list(shape), "zarr_format": 2},
    }
    if chunks:
        files[f"{at}/annotation/s0/<chunks>"] = chunks
    return files


def _manifest(volume):
    return {
        "dense_to_sparse_ratio": None, "input_norm": NORM, "input_size_voxels": [12, 12, 12],
        "input_voxel_size_nm": [8.0, 8.0, 8.0], "jitter_voxels": None, "kind": "volume_zarr_v1",
        "output_size_voxels": [4, 4, 4], "output_voxel_size_nm": [16.0, 16.0, 16.0],
        "patches_per_epoch": None, "postprocess": POST, "raw_dataset_path": RAW, "seed": 0,
        "volume_zarr_path": volume,
    }


def _session(base, volume, attrs, **kw):
    return {
        "<session>/corrections/.zgroup": ZGROUP,
        "<session>/corrections/_virtual_sources.json": _manifest(
            f"<tmp>/{base}/<session>/corrections/{volume}.zarr"),
        **_volume(f"<session>/corrections/{volume}.zarr", attrs, **kw),
    }


def _served(key):
    url = f"http://10.0.0.1:9000/annotations/{key}.zarr"
    return {"minio_url": url, "neuroglancer_url": url + "/annotation", "success": True, "volume_id": key}


EXPECTED = {
    "create_volume": {
        "response": {
            **_served("vol-1"), "zarr_path": "<tmp>/a/<session>/corrections/vol-1.zarr",
            "metadata": {"chunk_size": [4, 4, 4], "claimed_output_voxel_size": [16, 16, 16],
                         "dataset_offset_nm": [4.0, 4.0, 4.0], "dataset_shape_voxels": [24, 24, 24],
                         "output_voxel_size": [16.0, 16.0, 16.0]},
        },
        "tree": _session("a", "vol-1", ATTRS),
    },
    "load_crops": {
        "response": {"created_new_volume": True, "errors": [], "fg_voxels_written": 27,
                     "n_crops_imported": 1, "n_crops_requested": 1, "n_errors": 0,
                     "success": True, "volume_id": "vol-2"},
        "tree": _session("c", "vol-2", {**ATTRS, "imported_crops": [CROP]}, chunks=CROP_CHUNKS),
    },
    "build_corrections": {
        "tree": {
            "_virtual_sources.json": _manifest("<tmp>/built/vol-3.zarr"),
            **_volume("vol-3.zarr", {**ATTRS, **FLOAT_CLAIMS, "imported_crops": [CROP]}, chunks=CROP_CHUNKS),
        },
        # build_record.json
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
        },
    },
    "instance_correction": {
        "response": {**_served("roi_annotation"), "layer_name": "roi_annotation",
                     "mode": "fresh_seed", "reload_page": True,
                     "zarr_path": "<tmp>/instance_corrections/roi_annotation.zarr"},
        # On the segmentation's grid, uint16, seeded chunks only.
        "tree": _volume(
            "roi_annotation.zarr",
            {**ATTRS, **FLOAT_CLAIMS, "dataset_offset_nm": [8.0, 168.0, 8.0],
             "dataset_shape_voxels": [8, 8, 8], "seed_dilation_radius_voxels": 1,
             "seed_n_instances": 9, "seed_source_instance_zarr": "<tmp>/seg.zarr"},
            translation=(8.0, 168.0, 8.0), shape=(8, 8, 8), dtype="<u2", chunks=["0.0.0", "1.1.1"],
        ),
    },
    "list_existing_sessions": {"success": True, "sessions": [{
        "chunk_count": 1, "session_id": "<session>", "session_path": "<tmp>/a/<session>",
        "volumes": [{"path": "<tmp>/a/<session>/corrections/vol-1.zarr", "volume_id": "vol-1"}],
    }]},
    "load_existing_volume": {
        "response": {
            **_served("vol-1"), "copied_count": 1, "copied_minio": False, "metadata": ATTRS,
            "new_session_path": "<tmp>/b/<session>", "painted_chunk_count": 1,
            "skipped_chunk_extracts": 0, "zarr_path": "<tmp>/b/<session>/corrections/vol-1.zarr",
        },
        "tree": _session("b", "vol-1", ATTRS, chunks=["0.0.0"]),
    },
    # Each volume is mirrored when served; a crop import mirrors again after writing.
    "mirrored": [f"myserver/annotations/{key}.zarr"
                 for key in ("vol-1", "vol-2", "vol-2", "roi_annotation", "vol-1")],
}


def test_what_each_volume_creator_writes(world):
    # As JSON, so an int written as a float (16 -> 16.0) counts as a change.
    canonical = lambda d: {k: json.dumps(v, sort_keys=True, indent=1) for k, v in d.items()}  # noqa: E731
    assert canonical(_record(world)) == canonical(EXPECTED)
