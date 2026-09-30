"""A task YAML and a two-channel script model for the blockwise tests; the
datasets come from the root conftest (raw_zarr, ome_pyramid)."""

import pytest
import yaml

from tests.utils.serving_helpers import POOLING_MODEL


@pytest.fixture
def pooling_model(model_script):
    """``pooling_model(in_vs=8, out_vs=8)``: POOLING_MODEL (channels "a" and "b") as a script; its path."""
    return lambda in_vs=8, out_vs=8: model_script(POOLING_MODEL, name=f"model_{in_vs}_{out_vs}.py",
                                                  in_vs=in_vs, out_vs=out_vs)


@pytest.fixture
def task_yaml(tmp_path):
    """``task_yaml(data_path, script_path, model_extra=None, **overrides)``: a
    blockwise task YAML; its path."""

    def make(data_path, script_path, model_extra=None, **overrides):
        config = {
            "data_path": data_path,
            "output_path": str(tmp_path / "out.zarr"),
            "task_name": "t",
            "workers": 1,
            "charge_group": "grp",
            "queue": "gpu_h100",
            "models": [{"type": "script", "name": "m", "script_path": script_path, **(model_extra or {})}],
            **overrides,
        }
        path = tmp_path / "task.yaml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    return make
