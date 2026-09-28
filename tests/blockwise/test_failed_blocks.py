"""Failed blocks make the blockwise master fail, and workers log why.

daisy.run_blockwise returns all(state.is_done()), and is_done() counts failed
blocks, so the master logged the state and exited 0 however many blocks had
failed. Workers logged a failure as one line with no traceback.
"""

import contextlib
import logging

import daisy
import pytest
from click.testing import CliRunner

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import (
    CellMapFlowBlockwiseProcessor,
    _blocks_not_done,
    _run_blockwise,
)


def _task(process_function, name="t"):
    return daisy.Task(
        name,
        total_roi=daisy.Roi((0, 0, 0), (40, 40, 40)),
        read_roi=daisy.Roi((0, 0, 0), (10, 10, 10)),
        write_roi=daisy.Roi((0, 0, 0), (10, 10, 10)),
        process_function=process_function,
        read_write_conflict=False,
        fit="overhang",
        max_retries=0,
        num_workers=1,
    )


def _serial(stop_event):
    # Same scheduler and bookkeeping as daisy.Server, without worker
    # processes or sockets.
    return daisy.SerialServer()


def test_the_task_states_come_back_with_their_failures():
    def process(block):
        if block.write_roi.offset[0] == 0:
            block.status = daisy.BlockStatus.FAILED

    states = _run_blockwise([_task(process)], server_factory=_serial)

    assert states["t"].total_block_count == 64
    assert states["t"].failed_count == 16
    # What daisy.run_blockwise reports for the same run:
    assert states["t"].is_done()
    assert _blocks_not_done(states["t"]) == 16


def test_a_clean_run_has_nothing_left():
    states = _run_blockwise([_task(lambda block: None)], server_factory=_serial)
    assert _blocks_not_done(states["t"]) == 0


def _state(total, completed, failed=0):
    state = daisy.task_state.TaskState()
    state.total_block_count = total
    state.completed_count = completed
    state.failed_count = failed
    return state


@pytest.fixture
def master(raw_array, model_script, task_yaml):
    return CellMapFlowBlockwiseProcessor(task_yaml(raw_array(), model_script()), create=True)


def test_run_reports_failure_when_blocks_failed(master, monkeypatch, caplog):
    monkeypatch.setattr(
        blockwise_processor,
        "_run_blockwise",
        lambda tasks: {tasks[0].task_id: _state(8, 6, failed=2)},
    )
    with caplog.at_level(logging.ERROR, logger=blockwise_processor.logger.name):
        assert master.run() is False
    assert "predict_mt: 2" in caplog.text


def test_run_reports_success_when_every_block_ran(master, monkeypatch):
    monkeypatch.setattr(
        blockwise_processor, "_run_blockwise", lambda tasks: {tasks[0].task_id: _state(8, 8)}
    )
    assert master.run() is True


def test_cellmap_flow_blockwise_exits_non_zero(master, monkeypatch, task_yaml, raw_array, model_script):
    from cellmap_flow.blockwise.cli import cli

    monkeypatch.setattr(CellMapFlowBlockwiseProcessor, "run", lambda self: False)
    result = CliRunner().invoke(cli, [task_yaml(raw_array(), model_script())])
    assert result.exit_code == 1
    assert "some blocks were not processed" in result.output


def test_cellmap_flow_blockwise_multiple_runs_the_rest_then_exits_non_zero(
    monkeypatch, task_yaml, raw_array, model_script
):
    from cellmap_flow.blockwise.multiple_cli import cli

    outcomes = [False, True]
    ran = []

    def run(self):
        ran.append(self.yaml_config)
        return outcomes.pop(0)

    monkeypatch.setattr(CellMapFlowBlockwiseProcessor, "run", run)
    first = task_yaml(raw_array(), model_script())
    result = CliRunner().invoke(cli, [first, first])
    assert len(ran) == 2
    assert result.exit_code == 1


def test_a_worker_logs_the_traceback_of_a_failed_block(master, monkeypatch, caplog):
    def boom(block):
        raise RuntimeError("CUDA out of memory")

    class Client:
        blocks = [daisy.Block(daisy.Roi((0,) * 3, (32,) * 3), daisy.Roi((0,) * 3, (32,) * 3), daisy.Roi((0,) * 3, (32,) * 3), task_id="t")]

        @contextlib.contextmanager
        def acquire_block(self):
            yield self.blocks.pop() if self.blocks else None

    monkeypatch.setattr(master, "process_fn", boom)
    monkeypatch.setattr(blockwise_processor.daisy, "Client", Client)
    with caplog.at_level(logging.ERROR, logger=blockwise_processor.logger.name):
        master.client()

    (record,) = [r for r in caplog.records if "Error processing block" in r.getMessage()]
    assert record.exc_info is not None
    assert "CUDA out of memory" in caplog.text
