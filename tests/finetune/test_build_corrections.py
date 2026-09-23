"""build_corrections must produce, from a raw multiscale group and a crops
manifest, the same three artefacts the dashboard's YAML import produces:
an annotation volume at the model's output voxel size with the crop written
at its physical offset, a volume_zarr_v1 manifest the trainer accepts, and
a build record with the provenance."""

import json
import os
import tempfile

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.build_corrections import build_corrections, DEFAULT_INPUT_NORM


def _ome_group(path, arrays_by_scale, translation_nm):
    """Write a v2 OME-NGFF multiscale group {s0: (data, scale_nm), ...}."""
    grp = zarr.open_group(path, mode="w")
    datasets = []
    for name, (data, scale) in arrays_by_scale.items():
        grp.create_dataset(name, data=data, chunks=data.shape)
        datasets.append({
            "path": name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(scale)] * 3},
                {"type": "translation", "translation": [float(t) for t in translation_nm]},
            ],
        })
    grp.attrs["multiscales"] = [{
        "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in "zyx"],
        "datasets": datasets,
        "version": "0.4",
    }]
    return path


def test_build_from_synthetic_raw_and_crop():
    with tempfile.TemporaryDirectory() as tmp:
        # raw: 8 nm s0 (64^3) and 16 nm s1 (32^3), origin 0
        raw = _ome_group(
            os.path.join(tmp, "raw.zarr"),
            {"s0": (np.zeros((64, 64, 64), np.uint8), 8.0),
             "s1": (np.zeros((32, 32, 32), np.uint8), 16.0)},
            translation_nm=(0, 0, 0),
        )
        # crop: 16 nm labels, 8^3, sitting at voxel (8, 8, 8) of the 16 nm grid,
        # class id 50 in the middle, 0 elsewhere (dense -> background)
        labels = np.zeros((8, 8, 8), np.uint8)
        labels[2:6, 2:6, 2:6] = 50
        crop = _ome_group(os.path.join(tmp, "crop.zarr"),
                          {"s0": (labels, 16.0)}, translation_nm=(128, 128, 128))
        crops_yaml = os.path.join(tmp, "crops.yaml")
        with open(crops_yaml, "w") as f:
            f.write(f"crops:\n  - path: {crop}\n    name: c1\n    fg_ids: [50]\n    mode: dense\n")

        out = os.path.join(tmp, "corrections")
        record = build_corrections(
            raw_dataset_path=raw, crops_yaml=crops_yaml, output_dir=out,
            input_shape=(16, 16, 16), output_shape=(8, 8, 8),
            input_voxel_size=(16, 16, 16), output_voxel_size=(16, 16, 16),
            model_name="toy", patches_per_epoch=5,
            extra_record={"dataset": "toy_ds", "class": "mito_group"},
        )

        # manifest the trainer reads
        manifest = json.load(open(os.path.join(out, "_virtual_sources.json")))
        assert manifest["kind"] == "volume_zarr_v1"
        assert manifest["raw_dataset_path"] == raw
        assert manifest["output_size_voxels"] == [8, 8, 8]
        assert manifest["input_voxel_size_nm"] == [16.0, 16.0, 16.0]
        assert manifest["patches_per_epoch"] == 5
        assert manifest["input_norm"] == DEFAULT_INPUT_NORM
        assert manifest["volume_zarr_path"] == record["volume_zarr_path"]

        # the volume holds the crop at its physical place, remapped to 1=bg, 2=fg
        vol = zarr.open(record["volume_zarr_path"], mode="r")
        arr = vol[[k for k in vol.array_keys()][0]] if hasattr(vol, "array_keys") and list(vol.array_keys()) else None
        if arr is None:
            # annotation volume is a group with an s0 array
            arr = zarr.open(record["volume_zarr_path"], mode="r")
            arr = arr[list(arr.group_keys())[0]]["s0"] if list(arr.group_keys()) else arr["s0"]
        written = arr[8:16, 8:16, 8:16]
        assert written.shape == (8, 8, 8)
        assert set(np.unique(written).tolist()) == {1, 2}
        assert (written[2:6, 2:6, 2:6] == 2).all()
        assert written[0, 0, 0] == 1
        # outside the crop nothing is annotated
        assert arr[0:8, 0:8, 0:8].max() == 0

        # provenance
        assert record["total_fg_voxels"] == 4 ** 3
        assert record["crops"][0]["n_fg_voxels"] == 4 ** 3
        assert record["dataset"] == "toy_ds" and record["class"] == "mito_group"
        assert "fg_ids: [50]" in record["crops_yaml"]
        rec_on_disk = json.load(open(os.path.join(out, "build_record.json")))
        assert rec_on_disk["geometry"]["effective_output_voxel_size_nm"] == [16.0, 16.0, 16.0]


def test_refuses_to_overwrite_an_existing_build():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "corrections")
        os.makedirs(out)
        open(os.path.join(out, "_virtual_sources.json"), "w").write("{}")
        with pytest.raises(FileExistsError):
            build_corrections(raw_dataset_path="x", crops_yaml="y", output_dir=out,
                              input_shape=(1, 1, 1), output_shape=(1, 1, 1),
                              input_voxel_size=(1, 1, 1), output_voxel_size=(1, 1, 1))


def test_geometry_must_be_supplied_somehow():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            build_corrections(raw_dataset_path="x", crops_yaml="y",
                              output_dir=os.path.join(tmp, "c"))
