"""track_progress markers identify one block of one task.

Markers were named by ``block.block_id[1]`` alone, in one directory per YAML
task name. daisy numbers blocks from each task's own total_roi and every
bounding box is its own task, so once ROI 1 was done, ROI 2's blocks with the
same numbers were skipped as done. A changed block size reused the numbers
for different regions too.
"""

import contextlib

import daisy
import pytest

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor, check_block


def _block(task_id, offset, size=32):
    total = daisy.Roi(offset, (size * 4,) * 3)
    roi = daisy.Roi(offset, (size,) * 3)
    return daisy.Block(total, roi, roi, task_id=task_id)


class FakeClient:
    """daisy.Client handing out a fixed list of blocks."""

    blocks = []

    def __init__(self):
        self._blocks = list(self.blocks)

    @contextlib.contextmanager
    def acquire_block(self):
        yield self._blocks.pop(0) if self._blocks else None


@pytest.fixture
def worker(raw_array, model_script, task_yaml, tmp_path, monkeypatch):
    path = task_yaml(
        raw_array(), model_script(), track_progress=True, tmp_dir=str(tmp_path / "progress")
    )
    processor = CellMapFlowBlockwiseProcessor(path, create=True)
    monkeypatch.setattr(processor, "process_fn", lambda block: None)
    monkeypatch.setattr(blockwise_processor.daisy, "Client", FakeClient)
    return processor


def test_a_block_done_in_one_bounding_box_is_not_done_in_another(worker):
    roi1 = _block("predict_m_roi1", (0, 0, 0))
    roi2 = _block("predict_m_roi2", (640, 640, 640))
    assert roi1.block_id[1] == roi2.block_id[1], "same index, different task"

    FakeClient.blocks = [roi1]
    worker.client()

    assert check_block(worker.tmp_dir, roi1)
    assert not check_block(worker.tmp_dir, roi2)


def test_a_marker_does_not_survive_a_change_of_block_size(worker):
    small = _block("predict_m", (0, 0, 0), size=32)
    FakeClient.blocks = [small]
    worker.client()

    large = _block("predict_m", (0, 0, 0), size=64)
    assert large.block_id == small.block_id
    assert check_block(worker.tmp_dir, small)
    assert not check_block(worker.tmp_dir, large)


def test_a_failed_block_leaves_no_marker(worker, monkeypatch):
    def boom(block):
        raise RuntimeError("model fell over")

    monkeypatch.setattr(worker, "process_fn", boom)
    block = _block("predict_m", (0, 0, 0))
    FakeClient.blocks = [block]
    worker.client()
    assert block.status == daisy.BlockStatus.FAILED
    assert not check_block(worker.tmp_dir, block)
