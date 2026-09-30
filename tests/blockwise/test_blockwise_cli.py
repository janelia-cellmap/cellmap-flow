"""cellmap_flow_blockwise and its multi-YAML runner exit non-zero when anything failed."""

import pytest
from click.testing import CliRunner

from cellmap_flow.blockwise import cli as single, multiple_cli
from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor


@pytest.mark.parametrize("cli, runs, exit_code", [
    pytest.param(single.cli, [False], 1, id="blocks-failed"),
    pytest.param(single.cli, [True], 0, id="every-block-ran"),
    # One YAML failing does not stop the others.
    pytest.param(multiple_cli.cli, [False, True], 1, id="multiple-runs-the-rest"),
])
def test_the_exit_code_says_whether_every_block_ran(raw_zarr, pooling_model, task_yaml, monkeypatch, cli, runs,
                                                   exit_code):
    outcomes, ran = list(runs), []
    monkeypatch.setattr(CellMapFlowBlockwiseProcessor, "run", lambda self: ran.append(1) or outcomes.pop(0))
    task = task_yaml(raw_zarr(), pooling_model())
    result = CliRunner().invoke(cli, [task] * len(runs))
    assert (result.exit_code, len(ran)) == (exit_code, len(runs)), result.output
    if cli is single.cli and exit_code:
        assert "some blocks were not processed" in result.output


def test_a_config_error_is_reported_and_exits_non_zero(tmp_path):
    """The YAML loader called sys.exit(1), which the dashboard's precheck could not catch."""
    (tmp_path / "c.yaml").write_text("charge_group: g\nmodels: {}\n")
    result = CliRunner().invoke(single.cli, [str(tmp_path / "c.yaml")])
    assert result.exit_code == 1 and "data_path" in result.output
