"""
Test processing and inference functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from funlib.geometry.coordinate import Coordinate
from funlib.geometry.roi import Roi

from cellmap_flow.inferencer import Inferencer, predict, apply_postprocess
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.utils.data import ModelConfig


@pytest.fixture
def mock_cuda_device_patch():
    """Fixture to patch torch.device and torch.cuda.is_available for CUDA."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.device") as mock_device,
    ):
        mock_cuda_device = Mock()
        mock_cuda_device.type = "cuda"
        mock_device.return_value = mock_cuda_device
        yield mock_cuda_device


class TestInferencer:
    """Test the Inferencer class."""

    def test_init_with_gpu(self, mock_model_config, mock_cuda_device_patch):
        """Test Inferencer initialization with GPU available."""
        # Fix mock config to have proper numeric attributes
        mock_model_config.config.read_shape = [32, 32, 32]
        mock_model_config.config.write_shape = [16, 16, 16]
        mock_model_config.config.output_dtype = np.float32
        mock_model_config.config.model = Mock()

        inferencer = Inferencer(mock_model_config)
        assert inferencer.device.type == "cuda"
        assert inferencer.use_half_prediction is True

    @pytest.fixture
    def mock_cpu_device_patch(self):
        """Fixture to patch torch.device and torch.cuda.is_available for CPU."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.device") as mock_device,
        ):
            mock_cpu_device = Mock()
            mock_cpu_device.type = "cpu"
            mock_device.return_value = mock_cpu_device
            yield mock_cpu_device

    def test_init_with_no_gpu(self, mock_model_config, mock_cpu_device_patch):
        """Test Inferencer initialization with no GPU."""
        # Fix mock config to have proper numeric attributes
        mock_model_config.config.read_shape = [32, 32, 32]
        mock_model_config.config.write_shape = [16, 16, 16]
        mock_model_config.config.output_dtype = np.float32
        mock_model_config.config.model = Mock()

        inferencer = Inferencer(mock_model_config)
        assert inferencer.device.type == "cpu"

    def test_context_calculation(self, mock_model_config, mock_cuda_device_patch):
        """Test context calculation from read/write shapes."""
        mock_model_config.config.read_shape = [32, 32, 32]
        mock_model_config.config.write_shape = [16, 16, 16]
        mock_model_config.config.output_dtype = np.float32
        mock_model_config.config.model = Mock()

        inferencer = Inferencer(mock_model_config)
        expected_context = Coordinate([8, 8, 8])
        assert inferencer.context == expected_context

    def test_optimize_model_torch(self, mock_torch_model, mock_cuda_device_patch):
        """Test model optimization for PyTorch models."""
        mock_config = Mock()
        mock_config.config = Mock()
        mock_config.config.model = mock_torch_model
        mock_config.config.read_shape = [32, 32, 32]
        mock_config.config.write_shape = [16, 16, 16]
        mock_config.config.output_dtype = np.float32

        # Mock the model's to() method to avoid CUDA operations
        mock_torch_model.to = Mock(return_value=mock_torch_model)
        mock_torch_model.half = Mock(return_value=mock_torch_model)

        inferencer = Inferencer(mock_config)
        inferencer.optimize_model()

        # Model should be moved to device and set to eval mode
        assert mock_torch_model.training is False
        mock_torch_model.to.assert_called()
        mock_torch_model.half.assert_called()

    def test_optimize_model_non_torch(self, mock_cuda_device_patch):
        """Test model optimization with non-PyTorch model."""
        mock_config = Mock()
        mock_config.config = Mock()
        mock_config.config.model = "not_a_torch_model"
        mock_config.config.read_shape = [32, 32, 32]
        mock_config.config.write_shape = [16, 16, 16]
        mock_config.config.output_dtype = np.float32

        inferencer = Inferencer(mock_config)
        # Should not raise an error, just log warning
        inferencer.optimize_model()

    def test_process_chunk_basic(self, mock_model_config, mock_cuda_device_patch):
        """Test basic chunk processing."""
        mock_idi = Mock()
        mock_roi = Roi(Coordinate([0, 0, 0]), Coordinate([32, 32, 32]))

        mock_model_config.config.predict = Mock(return_value=np.ones((16, 16, 16)))
        mock_model_config.config.read_shape = [32, 32, 32]
        mock_model_config.config.write_shape = [16, 16, 16]
        mock_model_config.config.output_dtype = np.float32
        mock_model_config.config.model = Mock()

        inferencer = Inferencer(mock_model_config)
        result = inferencer.process_chunk_basic(mock_idi, mock_roi)

        assert result.shape == (16, 16, 16)
        # Verify that predict was called
        mock_model_config.config.predict.assert_called_once()


class TestPredictFunction:
    """Test the predict function."""

    def test_predict_basic(self, sample_3d_array, mock_cuda_device_patch):
        """Test basic prediction functionality."""
        mock_idi = Mock()
        mock_idi.to_ndarray_ts.return_value = sample_3d_array

        mock_config = Mock()
        mock_model = Mock()

        # Mock the torch tensor and its methods to avoid CUDA operations
        mock_output_tensor = Mock()
        mock_output_tensor.detach.return_value = mock_output_tensor
        mock_output_tensor.cpu.return_value = mock_output_tensor
        # The final numpy() call should return an array where [0] gives our result
        batch_result = np.array([np.ones((16, 16, 16)), np.zeros((16, 16, 16))])
        mock_output_tensor.numpy.return_value = batch_result
        mock_model.forward.return_value = mock_output_tensor
        mock_config.model = mock_model

        read_roi = Roi(Coordinate([0, 0, 0]), Coordinate([64, 64, 64]))
        write_roi = Roi(Coordinate([8, 8, 8]), Coordinate([16, 16, 16]))

        # Mock torch operations completely
        with patch("torch.from_numpy") as mock_from_numpy:
            # Mock the complete tensor chain
            mock_input_tensor = Mock()
            mock_input_tensor.float.return_value = mock_input_tensor
            mock_input_tensor.half.return_value = mock_input_tensor
            mock_input_tensor.to.return_value = mock_input_tensor
            mock_from_numpy.return_value = mock_input_tensor

            # Use the fixture's device
            device = mock_cuda_device_patch

            result = predict(
                read_roi,
                write_roi,
                mock_config,
                idi=mock_idi,
                device=device,
            )

            assert result.shape == (16, 16, 16)
            mock_model.forward.assert_called_once()

    def test_predict_missing_idi(self):
        """Test predict function with missing idi parameter."""
        read_roi = Roi(Coordinate([0, 0, 0]), Coordinate([64, 64, 64]))
        write_roi = Roi(Coordinate([8, 8, 8]), Coordinate([16, 16, 16]))

        with pytest.raises(ValueError, match="idi must be provided"):
            predict(read_roi, write_roi, Mock())

    def test_predict_missing_device(self):
        """Test predict function with missing device parameter."""
        read_roi = Roi(Coordinate([0, 0, 0]), Coordinate([64, 64, 64]))
        write_roi = Roi(Coordinate([8, 8, 8]), Coordinate([16, 16, 16]))

        with pytest.raises(ValueError, match="device must be provided"):
            predict(read_roi, write_roi, Mock(), idi=Mock())


class TestApplyPostprocess:
    """Test postprocessing functionality."""

    @patch("cellmap_flow.globals.g")
    def test_apply_postprocess_empty(self, mock_g, sample_3d_array):
        """Test postprocessing with no postprocessors."""
        mock_g.postprocess = []

        result = apply_postprocess(sample_3d_array)
        np.testing.assert_array_equal(result, sample_3d_array)

    @patch("cellmap_flow.inferencer.g")
    def test_apply_postprocess_single(self, mock_g, sample_3d_array):
        """Test postprocessing with single postprocessor."""
        mock_postprocessor = Mock(return_value=sample_3d_array * 2)
        mock_g.postprocess = [mock_postprocessor]

        result = apply_postprocess(sample_3d_array)

        mock_postprocessor.assert_called_once_with(sample_3d_array)
        np.testing.assert_array_equal(result, sample_3d_array * 2)

    @patch("cellmap_flow.inferencer.g")
    def test_apply_postprocess_multiple(self, mock_g, sample_3d_array):
        """Test postprocessing with multiple postprocessors."""
        mock_postprocessor1 = Mock(side_effect=lambda x, **kwargs: x * 2)
        mock_postprocessor2 = Mock(side_effect=lambda x, **kwargs: x + 1)
        mock_g.postprocess = [mock_postprocessor1, mock_postprocessor2]

        result = apply_postprocess(sample_3d_array)

        mock_postprocessor1.assert_called_once()
        mock_postprocessor2.assert_called_once()
        # Result should be ((sample_3d_array * 2) + 1)
        expected = sample_3d_array * 2 + 1
        np.testing.assert_array_equal(result, expected)
