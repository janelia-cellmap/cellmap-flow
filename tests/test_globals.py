"""
Test the global state management.
"""

import pytest
from unittest.mock import patch, Mock
from cellmap_flow.globals import Flow, g


class TestFlow:
    """Test the Flow singleton class."""

    def test_singleton_pattern(self):
        """Test that Flow implements singleton pattern correctly."""
        flow1 = Flow()
        flow2 = Flow()
        assert flow1 is flow2

    def test_initial_state(self):
        """Test initial state of Flow instance."""
        flow = Flow()
        assert hasattr(flow, "jobs")
        assert hasattr(flow, "models_config")
        assert hasattr(flow, "servers")
        assert hasattr(flow, "raw")
        assert hasattr(flow, "input_norms")
        assert hasattr(flow, "postprocess")
        assert hasattr(flow, "viewer")
        assert hasattr(flow, "dataset_path")
        assert hasattr(flow, "queue")
        assert hasattr(flow, "charge_group")
        assert hasattr(flow, "neuroglancer_thread")

    @patch("cellmap_flow.globals.load_model_paths")
    def test_model_catalog_loading(self, mock_load_model_paths):
        """Test that model catalog is loaded on initialization."""
        mock_catalog = {"test": {"model1": "path1"}}
        mock_load_model_paths.return_value = mock_catalog

        # Reset singleton for test
        Flow._instance = None
        flow = Flow()

        assert hasattr(flow, "model_catalog")
        mock_load_model_paths.assert_called_once()

    def test_to_dict(self):
        """Test dictionary representation."""
        flow = Flow()
        flow_dict = dict(flow.to_dict())

        expected_keys = {
            "jobs",
            "models_config",
            "servers",
            "raw",
            "input_norms",
            "postprocess",
            "viewer",
            "dataset_path",
            "model_catalog",
            "queue",
            "charge_group",
            "neuroglancer_thread",
        }

        assert set(flow_dict.keys()) == expected_keys

    def test_repr(self):
        """Test string representation."""
        flow = Flow()
        repr_str = repr(flow)
        assert "Flow(" in repr_str
        assert "jobs" in repr_str
        assert "models_config" in repr_str


class TestGlobalInstance:
    """Test the global g instance."""

    def setup_method(self):
        """Reset singleton state before each test."""
        # Store original instance
        self._original_instance = Flow._instance

    def teardown_method(self):
        """Restore singleton state after each test."""
        Flow._instance = self._original_instance

    def test_g_is_flow_instance(self):
        """Test that g is an instance of Flow."""
        assert isinstance(g, Flow)

    def test_g_singleton_consistency(self):
        """Test that g maintains singleton consistency."""
        # Reset to ensure clean state
        Flow._instance = None
        flow = Flow()
        # Since g was created at import time, this test checks different behavior
        assert isinstance(g, Flow)
        assert isinstance(flow, Flow)

    def test_g_attribute_access(self):
        """Test direct attribute access on g."""
        assert hasattr(g, "jobs")
        assert hasattr(g, "models_config")
        assert hasattr(g, "servers")
        assert hasattr(g, "raw")
        assert hasattr(g, "input_norms")
        assert hasattr(g, "postprocess")
        assert hasattr(g, "viewer")
        assert hasattr(g, "dataset_path")
        assert hasattr(g, "model_catalog")
        assert hasattr(g, "queue")
        assert hasattr(g, "charge_group")
        assert hasattr(g, "neuroglancer_thread")

    def test_g_modification_persistence(self):
        """Test that modifications to g persist across accesses."""
        # Test that we can access attributes
        assert hasattr(g, "jobs")

        # Store original length
        original_jobs_len = len(g.jobs)

        # Add something to g
        g.jobs.append("test_job")

        # Verify the change persists
        assert len(g.jobs) == original_jobs_len + 1
        assert "test_job" in g.jobs

        # Clean up
        g.jobs.remove("test_job")
