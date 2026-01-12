"""
Test data utilities and model configurations.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from funlib.geometry.coordinate import Coordinate
from funlib.geometry.roi import Roi

from cellmap_flow.utils.data import (
    ModelConfig,
    DaCapoModelConfig,
    BioModelConfig,
    ScriptModelConfig,
    CellMapModelConfig,
    check_config,
)


class TestModelConfig:
    """Test the base ModelConfig class."""

    def test_config_property_caching(self):
        """Test that config property caches result."""

        class TestConfig(ModelConfig):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def _get_config(self):
                self.call_count += 1
                mock_config = Mock()
                # Add all required attributes for config validation
                mock_config.model = Mock()
                mock_config.read_shape = [32, 32, 32]
                mock_config.write_shape = [16, 16, 16]
                mock_config.input_voxel_size = [1, 1, 1]
                mock_config.output_voxel_size = [1, 1, 1]
                mock_config.output_channels = 1
                mock_config.block_shape = [32, 32, 32]
                return mock_config

        config = TestConfig()

        # First access should call _get_config
        config1 = config.config
        assert config.call_count == 1

        # Second access should use cached value
        config2 = config.config
        assert config.call_count == 1
        assert config1 is config2

    @patch("cellmap_flow.utils.data.check_config")
    def test_config_property_validation(self, mock_check_config):
        """Test that config property validates configuration."""

        class TestConfig(ModelConfig):
            def _get_config(self):
                return Mock()

        config = TestConfig()
        _ = config.config
        mock_check_config.assert_called_once()

    def test_output_dtype_default(self):
        """Test default output dtype."""

        class TestConfig(ModelConfig):
            def __init__(self):
                super().__init__()
                self.name = "test_model"

            def _get_config(self):
                # Create a config that doesn't have output_dtype attribute
                class ConfigMock:
                    def __init__(self):
                        self.model = Mock()
                        self.read_shape = [32, 32, 32]
                        self.write_shape = [16, 16, 16]
                        self.input_voxel_size = [1, 1, 1]
                        self.output_voxel_size = [1, 1, 1]
                        self.output_channels = 1
                        self.block_shape = [32, 32, 32]
                        # Explicitly NOT setting output_dtype

                return ConfigMock()

        config = TestConfig()
        assert config.output_dtype == np.float32

    def test_output_dtype_custom(self):
        """Test custom output dtype."""

        class TestConfig(ModelConfig):
            def _get_config(self):
                mock_config = Mock()
                # Add all required attributes for config validation
                mock_config.model = Mock()
                mock_config.read_shape = [32, 32, 32]
                mock_config.write_shape = [16, 16, 16]
                mock_config.input_voxel_size = [1, 1, 1]
                mock_config.output_voxel_size = [1, 1, 1]
                mock_config.output_channels = 1
                mock_config.block_shape = [32, 32, 32]
                mock_config.output_dtype = np.uint8
                return mock_config

        config = TestConfig()
        assert config.output_dtype == np.uint8


class TestDaCapoModelConfig:
    """Test DaCapoModelConfig functionality."""

    @patch("cellmap_flow.utils.data.get_dacapo_channels")
    @patch("cellmap_flow.utils.data.get_dacapo_run_model")
    def test_get_config(self, mock_get_dacapo_run, mock_get_dacapo_channels):
        """Test configuration loading."""
        mock_run = Mock()
        mock_run.model = Mock()
        mock_run.model.eval_input_shape = [32, 32, 32]
        mock_run.model.compute_output_shape.return_value = (None, [16, 16, 16])
        mock_run.model.scale.return_value = [4, 4, 4]

        # Fix the datasplit mock to be properly indexable
        mock_train_data = Mock()
        mock_train_data.raw.voxel_size = [1, 1, 1]
        mock_run.datasplit.train = [mock_train_data]  # Make it a list
        mock_run.task = Mock()

        mock_get_dacapo_run.return_value = mock_run
        # Mock get_dacapo_channels to return a list with length
        mock_get_dacapo_channels.return_value = ["channel1", "channel2", "channel3"]

        # Mock torch.cuda.is_available to avoid CUDA operations
        with patch("torch.cuda.is_available", return_value=False):
            config = DaCapoModelConfig(run_name="test_run", iteration=1000)

            result = config._get_config()
            assert hasattr(result, "model")
            assert hasattr(result, "output_channels")
            mock_get_dacapo_run.assert_called_once_with("test_run", 1000)


class TestBioModelConfig:
    """Test BioModelConfig functionality."""

    def test_init_with_edge_length(self):
        """Test initialization with edge length processing."""
        config = BioModelConfig(
            model_name="test_model", voxel_size=[1, 1, 1], edge_length_to_process=64
        )
        assert config.voxels_to_process == 64**3


class TestScriptModelConfig:
    """Test ScriptModelConfig functionality."""

    @patch("cellmap_flow.utils.load_py.load_safe_config")
    def test_get_config(self, mock_load_config):
        """Test configuration loading."""
        mock_config = Mock()
        mock_config.write_shape = [16, 16, 16]
        mock_config.output_channels = 2
        mock_load_config.return_value = mock_config

        config = ScriptModelConfig(script_path="/path/to/script.py")

        result = config._get_config()
        assert result is mock_config
        mock_load_config.assert_called_once_with("/path/to/script.py")


class TestCheckConfig:
    """Test configuration validation."""

    def test_check_config_valid(self):
        """Test validation of valid configuration."""
        config = Mock()
        config.model = Mock()
        config.read_shape = [32, 32, 32]
        config.write_shape = [16, 16, 16]
        config.input_voxel_size = [1, 1, 1]
        config.output_voxel_size = [1, 1, 1]
        config.output_channels = 2
        config.block_shape = [16, 16, 16, 2]

        # Should not raise any exception
        check_config(config)

    def test_check_config_missing_model(self):
        """Test validation with missing model."""
        config = Mock(
            spec=[
                "read_shape",
                "write_shape",
                "input_voxel_size",
                "output_voxel_size",
                "output_channels",
                "block_shape",
            ]
        )
        config.read_shape = [32, 32, 32]
        config.write_shape = [16, 16, 16]
        config.input_voxel_size = [1, 1, 1]
        config.output_voxel_size = [1, 1, 1]
        config.output_channels = 2
        config.block_shape = [16, 16, 16, 2]

        with pytest.raises(AssertionError):
            check_config(config)

    def test_check_config_missing_read_shape(self):
        """Test validation with missing read_shape."""
        config = Mock(
            spec=[
                "model",
                "write_shape",
                "input_voxel_size",
                "output_voxel_size",
                "output_channels",
                "block_shape",
            ]
        )
        config.model = Mock()
        config.write_shape = [16, 16, 16]
        config.input_voxel_size = [1, 1, 1]
        config.output_voxel_size = [1, 1, 1]
        config.output_channels = 2
        config.block_shape = [16, 16, 16, 2]

        with pytest.raises(AssertionError):
            check_config(config)
