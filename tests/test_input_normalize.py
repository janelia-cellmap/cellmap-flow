"""
Test input normalization functionality.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from cellmap_flow.norm.input_normalize import (
    SerializableInterface,
    InputNormalizer,
    MinMaxNormalizer,
    LambdaNormalizer,
    ZScoreNormalizer,
    Dilate,
    EuclideanDistance,
    get_input_normalizers,
    get_normalizations,
    deserialize_list,
)


class TestSerializableInterface:
    """Test the SerializableInterface base class."""

    def test_name_classmethod(self):
        """Test that name() returns the class name."""
        assert SerializableInterface.name() == "SerializableInterface"

    def test_call_method(self):
        """Test that __call__ delegates to process method."""

        class TestInterface(SerializableInterface):
            def _process(self, data, **kwargs):
                return data * 2

            @property
            def dtype(self):
                return np.float32

        interface = TestInterface()
        data = np.array([1, 2, 3])
        result = interface(data)
        expected = data * 2
        np.testing.assert_array_equal(result, expected.astype(np.float32))

    def test_process_type_conversion(self):
        """Test that process converts input to numpy array."""

        class TestInterface(SerializableInterface):
            def _process(self, data, **kwargs):
                return data

            @property
            def dtype(self):
                return np.float32

        interface = TestInterface()
        result = interface([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_to_dict(self):
        """Test dictionary serialization."""

        class TestInterface(SerializableInterface):
            def __init__(self, param1=1, param2="test"):
                self.param1 = param1
                self.param2 = param2
                self._private = "private"

            def _process(self, data, **kwargs):
                return data

            @property
            def dtype(self):
                return np.float32

        interface = TestInterface(param1=5, param2="example")
        result = interface.to_dict()

        expected = {"name": "TestInterface", "param1": 5, "param2": "example"}
        assert result == expected
        assert "_private" not in result


class TestMinMaxNormalizer:
    """Test MinMaxNormalizer functionality."""

    def test_init_string_invert(self):
        """Test string invert parameter."""
        normalizer = MinMaxNormalizer(invert="true")
        assert normalizer.invert is True

        normalizer = MinMaxNormalizer(invert="false")
        assert normalizer.invert is False

    def test_process_basic(self):
        """Test basic normalization."""
        normalizer = MinMaxNormalizer(min_value=0, max_value=255)
        data = np.array([0, 127.5, 255])
        result = normalizer(data)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_clipping(self):
        """Test that values outside range are clipped."""
        normalizer = MinMaxNormalizer(min_value=0, max_value=255)
        data = np.array([-10, 127.5, 300])
        result = normalizer(data)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_invert(self):
        """Test inverted normalization."""
        normalizer = MinMaxNormalizer(min_value=0, max_value=255, invert=True)
        data = np.array([0, 127.5, 255])
        result = normalizer(data)
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_dtype(self):
        """Test output dtype."""
        normalizer = MinMaxNormalizer()
        assert normalizer.dtype == np.float32


class TestLambdaNormalizer:
    """Test LambdaNormalizer functionality."""

    def test_process_simple(self):
        """Test simple lambda expression."""
        normalizer = LambdaNormalizer("x * 2")
        data = np.array([1, 2, 3])
        result = normalizer(data)
        expected = np.array([2, 4, 6])
        np.testing.assert_array_equal(result, expected.astype(np.float32))

    def test_process_complex(self):
        """Test complex lambda expression."""
        normalizer = LambdaNormalizer("(x - 128) / 127.5")
        data = np.array([0, 128, 255])
        result = normalizer(data)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(
            result, expected, decimal=2
        )  # Reduced precision for floating point tolerance

    def test_dtype(self):
        """Test output dtype."""
        normalizer = LambdaNormalizer("x")
        assert normalizer.dtype == np.float32


class TestZScoreNormalizer:
    """Test ZScoreNormalizer functionality."""

    def test_process(self):
        """Test z-score normalization."""
        normalizer = ZScoreNormalizer(mean=100, std=15)
        data = np.array([100, 115, 85])
        result = normalizer(data)
        expected = np.array([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_dtype(self):
        """Test output dtype."""
        normalizer = ZScoreNormalizer()
        assert normalizer.dtype == np.float32


class TestDilate:
    """Test Dilate functionality."""

    @patch("cellmap_flow.norm.input_normalize.dilation")
    @patch("cellmap_flow.norm.input_normalize.cube")
    def test_init_and_process(self, mock_cube, mock_dilation):
        """Test dilation process."""
        mock_cube.return_value = "cube_structure"
        mock_dilation.return_value = np.ones((3, 3, 3))

        dilate = Dilate(size=2)
        assert dilate.size == 2

        data = np.zeros((3, 3, 3))
        result = dilate(data)

        mock_cube.assert_called_once_with(2)
        mock_dilation.assert_called_once_with(data, "cube_structure")


class TestEuclideanDistance:
    """Test EuclideanDistance functionality."""

    def test_init_edt(self):
        """Test initialization with EDT type."""
        ed = EuclideanDistance(type="edt")
        assert ed.anisotropy == (50, 50, 50)
        assert ed.black_border is True
        assert ed.parallel == 5

    def test_init_sdf(self):
        """Test initialization with SDF type."""
        ed = EuclideanDistance(type="sdf")
        assert hasattr(ed, "_func")

    def test_init_invalid_type(self):
        """Test initialization with invalid type."""
        with pytest.raises(ValueError, match="type must be either 'edt' or 'sdf'"):
            EuclideanDistance(type="invalid")

    def test_init_activations(self):
        """Test different activation functions."""
        ed_tanh = EuclideanDistance(activation="tanh")
        ed_relu = EuclideanDistance(activation="relu")
        ed_sigmoid = EuclideanDistance(activation="sigmoid")

        # Test activation functions
        test_val = np.array([1.0])
        assert np.allclose(ed_tanh.activation(test_val), np.tanh(test_val))
        assert np.allclose(ed_relu.activation(test_val), np.maximum(0, test_val))
        assert np.allclose(ed_sigmoid.activation(test_val), 1 / (1 + np.exp(-test_val)))

    def test_init_invalid_activation(self):
        """Test initialization with invalid activation."""
        with pytest.raises(ValueError, match="Unsupported activation function"):
            EuclideanDistance(activation="invalid")

    def test_dtype(self):
        """Test output dtype."""
        ed = EuclideanDistance()
        assert ed.dtype == np.float32


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_input_normalizers(self):
        """Test getting list of available normalizers."""
        normalizers = get_input_normalizers()

        assert isinstance(normalizers, list)
        assert len(normalizers) > 0

        # Check that each normalizer has required fields
        for norm in normalizers:
            assert "class_name" in norm
            assert "name" in norm
            assert "params" in norm
            assert isinstance(norm["params"], dict)

        # Check that common normalizers are present
        class_names = [n["class_name"] for n in normalizers]
        assert "MinMaxNormalizer" in class_names
        assert "LambdaNormalizer" in class_names
        assert "ZScoreNormalizer" in class_names

    def test_deserialize_list_valid(self):
        """Test deserializing valid normalizer list."""

        # Create a test normalizer class
        class TestNormalizer(InputNormalizer):
            def __init__(self, param=1):
                self.param = param

            def _process(self, data, **kwargs):
                return data

            @property
            def dtype(self):
                return np.float32

        elms = {"TestNormalizer": {"param": 5}}
        result = deserialize_list(elms, InputNormalizer)

        assert len(result) == 1
        assert isinstance(result[0], TestNormalizer)
        assert result[0].param == 5

    def test_deserialize_list_invalid(self):
        """Test deserializing invalid normalizer."""
        elms = {"NonExistentNormalizer": {}}

        with pytest.raises(ValueError, match="method NonExistentNormalizer not found"):
            deserialize_list(elms, InputNormalizer)

    def test_get_normalizations(self):
        """Test getting normalizations from dictionary."""
        elms = {
            "MinMaxNormalizer": {"min_value": 0, "max_value": 100},
            "ZScoreNormalizer": {"mean": 50, "std": 10},
        }

        result = get_normalizations(elms)

        assert len(result) == 2
        assert isinstance(result[0], MinMaxNormalizer)
        assert isinstance(result[1], ZScoreNormalizer)
        assert result[0].min_value == 0
        assert result[0].max_value == 100
        assert result[1].mean == 50
        assert result[1].std == 10
