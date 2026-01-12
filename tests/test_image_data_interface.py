import pytest
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi, Coordinate
import os


@pytest.fixture
def test_zarr_path():
    """Path to test zarr dataset."""
    return "tests/tmp.zarr/s0"


def test_image_interface_default_voxel_size(test_zarr_path):
    """Test ImageDataInterface with default voxel size from dataset."""
    if not os.path.exists(test_zarr_path):
        pytest.skip(f"Test data not found at {test_zarr_path}")

    inf = ImageDataInterface(test_zarr_path)

    # Test reading a small ROI
    roi = Roi((0, 0, 0), (8, 8, 8))
    raw_input = inf.to_ndarray_ts(roi)

    assert raw_input.shape == (
        1,
        1,
        1,
    ), f"Expected shape (1, 1, 1), but got {raw_input.shape}"


def test_image_interface_custom_voxel_size(test_zarr_path):
    """Test ImageDataInterface with custom voxel size override."""
    if not os.path.exists(test_zarr_path):
        pytest.skip(f"Test data not found at {test_zarr_path}")

    inf = ImageDataInterface(test_zarr_path, voxel_size=Coordinate(30, 8, 8))

    # Test reading a ROI with the custom voxel size
    roi = Roi((0, 0, 0), (30, 8, 8))
    raw_input = inf.to_ndarray_ts(roi)

    # Should read 1x1x1 voxels since ROI matches voxel size
    assert raw_input.shape == (
        1,
        1,
        1,
    ), f"Expected shape (1, 1, 1), but got {raw_input.shape}"


def test_image_interface_axes_names(test_zarr_path):
    """Test that ImageDataInterface correctly reads axes names from dataset."""
    if not os.path.exists(test_zarr_path):
        pytest.skip(f"Test data not found at {test_zarr_path}")

    inf = ImageDataInterface(test_zarr_path)

    # Axes names should be read from the dataset metadata
    assert hasattr(
        inf, "axes_names"
    ), "ImageDataInterface should have axes_names attribute"
    assert isinstance(inf.axes_names, list), "axes_names should be a list"


def test_image_interface_voxel_size_property(test_zarr_path):
    """Test that voxel_size property works correctly."""
    if not os.path.exists(test_zarr_path):
        pytest.skip(f"Test data not found at {test_zarr_path}")

    # Test with default voxel size
    inf_default = ImageDataInterface(test_zarr_path)
    assert hasattr(inf_default, "voxel_size"), "Should have voxel_size attribute"

    # Test with custom voxel size
    custom_voxel_size = Coordinate(30, 8, 8)
    inf_custom = ImageDataInterface(test_zarr_path, voxel_size=custom_voxel_size)
    assert (
        inf_custom.voxel_size == custom_voxel_size
    ), f"Expected voxel_size {custom_voxel_size}, got {inf_custom.voxel_size}"
