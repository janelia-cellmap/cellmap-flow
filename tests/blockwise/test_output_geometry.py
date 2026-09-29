"""The blockwise output sits where the raw data is.

Outside separate-bbox mode the output array was created at offset (0, 0, 0)
while the task covered the raw ROI with its real offset, so a dataset that
does not start at 0 had its prediction written shifted and clipped, and task
blocks straddled output chunks. Offsets are corners here, as before.
"""

import pytest
import zarr
from funlib.geometry import Coordinate

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor


class _Stop(Exception):
    pass


def _task_rois(monkeypatch, processor):
    """The total_roi of each task run() would schedule."""
    recorded = []

    def fake_task(name, **kwargs):
        recorded.append(kwargs["total_roi"])
        raise _Stop

    monkeypatch.setattr(blockwise_processor.daisy, "Task", fake_task)
    with pytest.raises(_Stop):
        processor.run()
    return recorded


def test_the_output_grid_starts_at_the_raw_corner(
    raw_array, model_script, task_yaml, monkeypatch
):
    # 16 voxels of 8 nm from (8, 16, 24); the model writes 16 nm voxels. The
    # output grid is anchored at the raw corner, so every output voxel covers
    # exactly two raw voxels (snapping to multiples of 16 nm instead put it
    # half an output voxel off the data it was computed from).
    raw = raw_array(shape=(16, 16, 16), voxel_size=(8, 8, 8), offset=(8, 16, 24))
    processor = CellMapFlowBlockwiseProcessor(task_yaml(raw, model_script(8, 16)), create=True)

    out = processor.output_arrays[0]
    assert out.roi.offset == Coordinate(8, 16, 24)
    assert out.roi.shape == Coordinate(128, 128, 128)
    assert out.voxel_size == Coordinate(16, 16, 16)

    # OME translation is voxel 0's centre: the corner plus half a voxel.
    channel = processor.output_channels[0]
    attrs = zarr.open_group(str(processor.output_path / channel), mode="r").attrs
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    translation = next(t["translation"] for t in transforms if t["type"] == "translation")
    assert translation == [16.0, 24.0, 32.0]

    # The whole-volume task covers exactly that ROI (the model has no
    # context), so its blocks start on the output chunk grid.
    assert _task_rois(monkeypatch, processor) == [out.roi]


def test_a_raw_dataset_at_the_origin_is_unchanged(raw_array, model_script, task_yaml):
    raw = raw_array(shape=(16, 16, 16), voxel_size=(8, 8, 8), offset=(0, 0, 0))
    processor = CellMapFlowBlockwiseProcessor(task_yaml(raw, model_script(8, 16)), create=True)
    out = processor.output_arrays[0]
    assert out.roi.offset == Coordinate(0, 0, 0)
    assert out.roi.shape == Coordinate(128, 128, 128)
