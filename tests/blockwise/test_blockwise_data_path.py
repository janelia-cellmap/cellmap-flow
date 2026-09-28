"""Blockwise applies a model's ``scale`` by the same rule as the other launchers.

It ignored ``scale`` and let the reader pick the level closest to the model's
input voxel size, so the same YAML read different data here than under
cellmap_flow_yaml.
"""

from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor


def test_scale_selects_the_level_of_a_multiscale_group(multiscale_group, model_script, task_yaml):
    path = task_yaml(multiscale_group, model_script(8, 8), model_extra={"scale": "s1"})
    processor = CellMapFlowBlockwiseProcessor(path, create=True)
    assert processor.input_path.rstrip("/").endswith("ms.zarr/s1")
    assert processor.idi_raw.path.rstrip("/").endswith("s1")


def test_an_array_data_path_is_used_as_is(raw_array, model_script, task_yaml):
    raw = raw_array()
    path = task_yaml(raw, model_script(8, 8), model_extra={"scale": "raw"})
    processor = CellMapFlowBlockwiseProcessor(path, create=True)
    assert processor.input_path == raw
