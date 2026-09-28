"""Non-integer voxel sizes, and reading a scale at a voxel size it is not."""

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.utils.ds import find_closest_scale, get_ds_info


def _pyramid(path, scales, translations=None, shapes=None, fill_index=True):
    """An OME-Zarr v2 group; each level's voxels hold their z index."""
    root = zarr.open(path, mode="w")
    datasets = []
    for level, scale in enumerate(scales):
        shape = shapes[level] if shapes else (8, 4, 4)
        data = np.zeros(shape, dtype=np.uint16)
        if fill_index:
            data += np.arange(shape[0], dtype=np.uint16)[:, None, None]
        root.create_dataset(f"s{level}", data=data)
        transforms = [{"type": "scale", "scale": list(scale)}]
        if translations:
            transforms.append({"type": "translation", "translation": list(translations[level])})
        datasets.append({"path": f"s{level}", "coordinateTransformations": transforms})
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": datasets,
        }
    ]
    return path


def test_non_integer_voxel_sizes_are_not_truncated(tmp_path):
    path = _pyramid(str(tmp_path / "f.zarr"), [(5.24, 4, 4)], shapes=[(200, 4, 4)])
    voxel_size, _, _, roi, _, _ = get_ds_info(path + "/s0")
    assert tuple(voxel_size) == pytest.approx((5.24, 4.0, 4.0))
    assert tuple(roi.shape) == (1048, 16, 16)  # 200 * 5.24, not 200 * 5

    idi = ImageDataInterface(path + "/s0")
    # z = 524 nm is voxel 100 at 5.24 nm (it was voxel 104 at 5 nm).
    got = idi.to_ndarray_ts(Roi((524, 0, 0), (52, 4, 4)))
    assert got[:, 0, 0].tolist() == list(range(100, 109))


def test_scale_matching_compares_real_voxel_sizes(tmp_path):
    path = _pyramid(str(tmp_path / "p.zarr"), [(5.24, 4, 4), (10.48, 8, 8)])
    assert find_closest_scale(path, (10.48, 8, 8))[0] == "s1"
    # (10, 8, 8) is not (10.48, 8, 8); s1 is coarser in z, so s0 it is.
    assert find_closest_scale(path, (10, 8, 8))[0] == "s0"


def test_float_noise_from_unit_conversion_is_still_a_whole_number():
    from cellmap_flow.utils.zarr_v3 import coordinate_or_floats

    assert coordinate_or_floats([8.999999999999998, 4.0, 4.0]) == Coordinate(9, 4, 4)
    assert coordinate_or_floats([5.24, 4, 4]) == (5.24, 4.0, 4.0)


def test_relabelled_scale_is_read_on_its_own_grid(tmp_path):
    # 6 and 12 nm levels, both starting at 120 nm; a 16 nm model gets s1.
    path = _pyramid(
        str(tmp_path / "rl.zarr"),
        [(6, 6, 6), (12, 12, 12)],
        translations=[(120, 120, 120)] * 2,
    )
    idi = ImageDataInterface(path, voxel_size=(16, 16, 16))
    assert idi.path.endswith("s1")
    assert tuple(idi.voxel_size) == (16, 16, 16)
    assert tuple(idi.actual_voxel_size) == (12, 12, 12)
    assert tuple(idi.requested_voxel_size) == (16, 16, 16)
    # Voxel i stays voxel i: s1's first voxel is at 120 / 12 * 16 = 160.
    assert tuple(idi.roi.offset) == (160, 160, 160)
    assert tuple(idi.roi.shape) == (8 * 16, 4 * 16, 4 * 16)

    first = idi.to_ndarray_ts(Roi((160, 160, 160), (48, 16, 16)))
    assert first[:, 0, 0].tolist() == [0, 1, 2]


def test_relabelling_warns_and_can_be_refused(tmp_path, caplog):
    path = _pyramid(str(tmp_path / "rl.zarr"), [(6, 6, 6), (12, 12, 12)])
    with caplog.at_level("WARNING"):
        ImageDataInterface(path, voxel_size=(16, 16, 16))
    assert any("requested" in r.getMessage() for r in caplog.records)

    with pytest.raises(ValueError, match="requested"):
        ImageDataInterface(path, voxel_size=(16, 16, 16), on_voxel_size_mismatch="error")


def test_an_exact_scale_is_unchanged(tmp_path):
    path = _pyramid(
        str(tmp_path / "ex.zarr"),
        [(6, 6, 6), (12, 12, 12)],
        translations=[(120, 120, 120)] * 2,
    )
    idi = ImageDataInterface(path, voxel_size=(12, 12, 12))
    assert tuple(idi.roi.offset) == (120, 120, 120)
    assert isinstance(idi.voxel_size, Coordinate)
    assert idi.to_ndarray_ts(Roi((120, 120, 120), (12, 12, 12))).ravel().tolist() == [0]


def test_model_geometry_cache_keeps_fractional_voxel_sizes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cellmap_flow.utils import model_geometry

    monkeypatch.setattr(model_geometry, "CACHE_PATH", str(tmp_path / "cache.json"))
    script = tmp_path / "model.py"
    script.write_text("")
    model_config = SimpleNamespace(script_path=str(script))
    config = SimpleNamespace(
        read_shape=(52.4, 40, 40),
        write_shape=(52.4, 40, 40),
        input_voxel_size=(5.24, 4, 4),
        output_voxel_size=(5.24, 4, 4),
        output_channels=1,
    )
    model_geometry.store_geometry(model_config, config)
    cached = model_geometry.load_cached_geometry(model_config)
    assert cached.input_voxel_size == [5.24, 4, 4]
