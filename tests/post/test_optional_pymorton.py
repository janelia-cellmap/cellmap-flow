import builtins
import importlib
import sys

import numpy as np
import pytest


def test_postprocessors_import_without_pymorton(monkeypatch):
    sys.modules.pop("cellmap_flow.post.postprocessors", None)
    sys.modules.pop("pymorton", None)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pymorton" or name.startswith("pymorton."):
            raise ModuleNotFoundError("No module named 'pymorton'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("cellmap_flow.post.postprocessors")

    assert module.DefaultPostprocessor().is_segmentation is False


def test_morton_postprocessor_reports_missing_optional_dependency(monkeypatch):
    sys.modules.pop("cellmap_flow.post.postprocessors", None)
    sys.modules.pop("pymorton", None)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pymorton" or name.startswith("pymorton."):
            raise ModuleNotFoundError("No module named 'pymorton'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("cellmap_flow.post.postprocessors")

    postprocessor = module.MortonSegmentationRelabeling()
    data = np.ones((1, 1, 1, 1), dtype=np.uint8)

    with pytest.raises(ImportError, match="pymorton is required"):
        postprocessor._process(data, chunk_corner=(0, 0, 0), chunk_num_voxels=1)
