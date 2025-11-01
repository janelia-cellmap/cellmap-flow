"""
Pytest configuration and shared fixtures for cellmap-flow tests.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
from funlib.geometry.coordinate import Coordinate
from funlib.geometry.roi import Roi
import torch
import zarr


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_3d_array():
    """Create a sample 3D numpy array for testing."""
    return np.random.random((64, 64, 64)).astype(np.float32)


@pytest.fixture
def sample_4d_array():
    """Create a sample 4D numpy array for testing (batch, channel, z, y, x)."""
    return np.random.random((1, 1, 32, 32, 32)).astype(np.float32)


@pytest.fixture
def sample_roi():
    """Create a sample ROI for testing."""
    return Roi(Coordinate([0, 0, 0]), Coordinate([64, 64, 64]))


@pytest.fixture
def mock_zarr_dataset(temp_dir):
    """Create a mock zarr dataset for testing."""
    zarr_path = os.path.join(temp_dir, "test.zarr")

    # Create zarr group with multiscales metadata
    group = zarr.open_group(zarr_path, mode="w")

    # Add multiscales metadata
    group.attrs["multiscales"] = [
        {
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"scale": [1.0, 1.0, 1.0], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"},
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"scale": [2.0, 2.0, 2.0], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"},
                    ],
                },
            ]
        }
    ]

    # Create datasets
    data = np.random.random((64, 64, 64)).astype(np.uint8)
    group.create_dataset("s0", data=data)
    group.create_dataset("s1", data=data[::2, ::2, ::2])

    return zarr_path


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model for testing."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 1, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    return MockModel()


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    config = Mock()
    config.model_name = "test_model"
    config.model_type = "torch"
    config.read_shape = [32, 32, 32]
    config.write_shape = [16, 16, 16]
    config.output_dtype = np.float32
    config.model = Mock()
    return config


@pytest.fixture
def mock_flow_instance():
    """Create a mock Flow instance for testing."""
    with patch("cellmap_flow.globals.Flow") as mock_flow:
        instance = Mock()
        instance.jobs = []
        instance.models_config = []
        instance.servers = []
        instance.raw = None
        instance.input_norms = []
        instance.postprocess = []
        instance.viewer = None
        instance.dataset_path = None
        instance.model_catalog = {}
        instance.queue = "gpu_h100"
        instance.charge_group = "cellmap"
        instance.neuroglancer_thread = None
        mock_flow.return_value = instance
        yield instance


@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration for testing."""
    return {
        "data_path": "/test/path",
        "charge_group": "test_group",
        "queue": "gpu_h100",
        "models": [{"type": "dacapo", "run_name": "test_run", "iteration": 100}],
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging

    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_neuroglancer():
    """Mock neuroglancer for testing."""
    with (
        patch("neuroglancer.set_server_bind_address"),
        patch("neuroglancer.Viewer") as mock_viewer,
    ):
        yield mock_viewer


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Auto-mark GPU tests."""
    for item in items:
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
