"""`cellmap_flow blockwise` exits non-zero when anything failed, for one YAML or several."""

import pytest
from click.testing import CliRunner

from cellmap_flow.blockwise.cli import cli
from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor


@pytest.mark.parametrize("runs, exit_code", [
    pytest.param([False], 1, id="blocks-failed"),
    pytest.param([True], 0, id="every-block-ran"),
    # One YAML failing does not stop the others.
    pytest.param([False, True], 1, id="several-run-the-rest"),
])
def test_the_exit_code_says_whether_every_block_ran(raw_zarr, pooling_model, task_yaml, monkeypatch, runs,
                                                   exit_code):
    outcomes, ran = list(runs), []
    monkeypatch.setattr(CellMapFlowBlockwiseProcessor, "run", lambda self: ran.append(1) or outcomes.pop(0))
    task = task_yaml(raw_zarr(), pooling_model())
    result = CliRunner().invoke(cli, [task] * len(runs))
    assert (result.exit_code, len(ran)) == (exit_code, len(runs)), result.output
    if exit_code:
        assert f"{task}: some blocks were not processed" in result.output


def test_a_config_error_is_reported_and_exits_non_zero(tmp_path):
    """The YAML loader called sys.exit(1), which the dashboard's precheck could not catch."""
    (tmp_path / "c.yaml").write_text("charge_group: g\nmodels: {}\n")
    result = CliRunner().invoke(cli, [str(tmp_path / "c.yaml")])
    assert result.exit_code == 1 and "data_path" in result.output


def test_a_worker_takes_one_yaml(tmp_path):
    """--client is one worker of one YAML's run, which starts it."""
    (tmp_path / "c.yaml").write_text("")
    result = CliRunner().invoke(cli, [str(tmp_path / "c.yaml")] * 2 + ["--client"])
    assert result.exit_code == 2 and "--client runs one worker" in result.output
