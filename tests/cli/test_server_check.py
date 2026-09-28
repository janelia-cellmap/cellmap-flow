"""`cellmap_flow <type> --server-check` runs one chunk through the model.

It called ``_chunk_impl`` with six arguments against a five-argument
signature, so the check crashed with a TypeError before touching the model,
for every model type, from both the per-type commands and ``run``.
"""

import os

import pytest
from click.testing import CliRunner

from cellmap_flow.cli.cli import cli

HERE = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(HERE, "script_test", "fake_model_script.py")
RAW = os.path.join(HERE, "script_test", "dummy.zarr", "raw")


@pytest.mark.parametrize(
    "argv",
    [
        ["script", "--script-path", SCRIPT, "-d", RAW, "--server-check"],
        ["run", "-m", "script", "-c", f"script_path={SCRIPT}", "-d", RAW, "--server-check"],
    ],
    ids=["per-type", "run"],
)
def test_server_check_processes_a_chunk(argv):
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Server check passed" in result.output
