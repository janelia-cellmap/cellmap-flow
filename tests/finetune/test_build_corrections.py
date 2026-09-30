"""build_corrections: from a raw pyramid and a crops manifest, the volume, the
manifest the trainer reads and a build record, as the dashboard's YAML import
makes them (their files are pinned by test_volume_snapshot)."""

import json

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.build_corrections import DEFAULT_INPUT_NORM, build_corrections


def test_a_build_writes_the_crop_at_its_place_and_records_where_it_came_from(tmp_path, ome_zarr):
    raw = ome_zarr(tmp_path / "raw.zarr", ("s0", np.zeros((64,) * 3, np.uint8), 8.0, 0.0),
                   ("s1", np.zeros((32,) * 3, np.uint8), 16.0, 0.0))
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[2:6, 2:6, 2:6] = 50  # dense: the zeros around it are background
    crop = ome_zarr(tmp_path / "crop.zarr", ("s0", labels, 16.0, 128.0))  # voxel 0's corner is 120 nm
    (tmp_path / "crops.yaml").write_text(f"crops:\n  - path: {crop}\n    name: c1\n    fg_ids: [50]\n    mode: dense\n")
    out = tmp_path / "corrections"
    record = build_corrections(
        raw_dataset_path=raw, crops_yaml=str(tmp_path / "crops.yaml"), output_dir=str(out),
        input_shape=(16, 16, 16), output_shape=(8, 8, 8), input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16), model_name="toy", patches_per_epoch=5,
        extra_record={"dataset": "toy_ds", "class": "mito_group"},
    )

    manifest = json.loads((out / "_virtual_sources.json").read_text())
    assert (manifest["kind"], manifest["raw_dataset_path"], manifest["volume_zarr_path"]) == (
        "volume_zarr_v1", raw, record["volume_zarr_path"])
    assert (manifest["patches_per_epoch"], manifest["input_norm"]) == (5, DEFAULT_INPUT_NORM)
    # At its physical place (the volume's corner is -8 nm), remapped to 1 background, 2 foreground.
    volume = zarr.open(record["volume_zarr_path"], mode="r")["annotation/s0"]
    written = volume[8:16, 8:16, 8:16]
    assert (written[2:6, 2:6, 2:6] == 2).all() and written[0, 0, 0] == 1 and set(np.unique(written)) == {1, 2}
    assert volume[0:8, 0:8, 0:8].max() == 0, "nothing annotated outside the crop"
    assert record["total_fg_voxels"] == record["crops"][0]["n_fg_voxels"] == 4 ** 3
    assert (record["dataset"], record["class"]) == ("toy_ds", "mito_group") and "fg_ids: [50]" in record["crops_yaml"]
    assert json.loads((out / "build_record.json").read_text())["geometry"]["effective_output_voxel_size_nm"] == [16.0] * 3


@pytest.mark.parametrize("built, geometry, error", [
    (True, True, FileExistsError),  # a build is never overwritten
    (False, False, ValueError),  # the model's geometry has to come from somewhere
])
def test_a_build_refuses(tmp_path, built, geometry, error):
    out = tmp_path / "corrections"
    if built:
        out.mkdir()
        (out / "_virtual_sources.json").write_text("{}")
    shapes = dict(input_shape=(1,) * 3, output_shape=(1,) * 3, input_voxel_size=(1,) * 3,
                  output_voxel_size=(1,) * 3) if geometry else {}
    with pytest.raises(error):
        build_corrections(raw_dataset_path="x", crops_yaml="y", output_dir=str(out), **shapes)
