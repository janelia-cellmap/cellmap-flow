"""``cellmap_flow envs``: list, install and check the environments models run in.

pixi and the environments' pythons are never run: subprocess.run is
replaced, and what it was asked to run is checked.
"""

import gc
import re
import sys
import textwrap
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from cellmap_flow.cli import envs_cli
from cellmap_flow.cli.main import cli
from cellmap_flow.models.configs.base import ModelConfig

MANIFEST = """
[pypi-dependencies]
cellmap-flow = { path = ".", extras = ["finetune"] }

[environments]
cellpose4 = { features = [], no-default-feature = true }
dacapo = { features = [] }
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A pixi.toml (dacapo installed, cellpose4 not), pixi at /opt/pixi, an
    aliases file, and subprocess.run recording what it is asked to run."""
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(textwrap.dedent(MANIFEST))
    (tmp_path / ".pixi" / "envs" / "dacapo" / "bin").mkdir(parents=True)
    (tmp_path / ".pixi" / "envs" / "dacapo" / "bin" / "python").write_text("")
    (tmp_path / "conda" / "bin").mkdir(parents=True)
    (tmp_path / "conda" / "bin" / "python").write_text("")
    aliases = tmp_path / "envs.yaml"
    aliases.write_text(f"mine: {tmp_path}/conda\n")
    monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(manifest))
    monkeypatch.setenv("CELLMAP_FLOW_ENVS_FILE", str(aliases))
    monkeypatch.setenv("PIXI_EXE", "/opt/pixi")
    ran = []
    monkeypatch.setattr(envs_cli.subprocess, "run", lambda argv: ran.append(argv) or SimpleNamespace(returncode=0))
    gc.collect()  # no type an earlier test defined
    return SimpleNamespace(root=tmp_path, manifest=manifest, ran=ran)


def _invoke(*args):
    return CliRunner().invoke(cli, ["envs", *args])


def test_list_says_where_each_is_and_which_types_default_to_it(setup):
    class CellposeTestModelConfig(ModelConfig):
        cli_name = "cellpose-test"
        default_env = "cellpose4"

    class NoSuchEnvModelConfig(ModelConfig):
        cli_name = "nowhere-test"
        default_env = "nowhere"

    class PerModelConfig(ModelConfig):
        cli_name = "per-model-test"

        @property
        def default_env(self):
            """``fly`` for a raw checkpoint. Else none."""

    result = _invoke()
    assert result.exit_code == 0, result.output
    assert _invoke("list").output == result.output
    rows = {line.split()[0]: line.split() for line in result.output.splitlines()[4:] if line.strip()}
    root = setup.root
    assert rows["NAME"] == ["NAME", "KIND", "INSTALLED", "FINETUNE", "DEFAULT", "FOR", "WHERE"]
    # The real cellpose type defaults to cellpose4 too.
    assert rows["cellpose4"] == ["cellpose4", "pixi", "no", "no", "cellpose,", "cellpose-test",
                                 f"{root}/.pixi/envs/cellpose4"]
    assert rows["dacapo"] == ["dacapo", "pixi", "yes", "yes", "-", f"{root}/.pixi/envs/dacapo"]
    assert rows["mine"] == ["mine", "alias", "yes", "unchecked", "-", f"{root}/conda"]
    assert rows["nowhere"] == ["nowhere", "missing", "-", "-", "nowhere-test", "-"]
    assert "Decided per model:\n" in result.output and "  per-model-test: fly for a raw checkpoint\n" in result.output


def test_install_runs_pixi_on_the_lock_as_it_is(setup):
    result = _invoke("install", "cellpose4")
    assert result.exit_code == 0, result.output
    assert setup.ran == [["/opt/pixi", "install", "--frozen", "--manifest-path", str(setup.manifest), "-e", "cellpose4"]]


@pytest.mark.parametrize("name, error", [
    ("mine", "mine is an alias of .*/conda .in .*envs.yaml.: there is nothing to install"),
    ("cellpose5", "has no environment 'cellpose5'; it has cellpose4, dacapo, default"),
    ("/some/env", "is a path: install it with conda or pip"),
])
def test_install_refuses_what_pixi_cannot_install(setup, name, error):
    result = _invoke("install", name)
    assert result.exit_code == 1 and not setup.ran
    assert re.search(error, result.output.replace("\n", " ")), result.output


def test_install_without_pixi_says_how_to_get_it(setup, monkeypatch):
    monkeypatch.delenv("PIXI_EXE")
    monkeypatch.setattr(envs_cli.envs.shutil, "which", lambda program: None)
    result = _invoke("install", "cellpose4")
    assert result.exit_code == 1 and "https://pixi.sh" in result.output and not setup.ran


@pytest.mark.parametrize("name, python", [
    ("cellpose4", ["/opt/pixi", "run", "--frozen", "--manifest-path", "MANIFEST", "-e", "cellpose4", "python"]),
    ("mine", ["ROOT/conda/bin/python"]),
    ("current", [sys.executable]),
])
def test_check_imports_cellmap_flow_in_the_env(setup, name, python):
    result = _invoke("check", name)
    assert result.exit_code == 0 and result.output.endswith(f"{name}: ok\n"), result.output
    python = [a.replace("MANIFEST", str(setup.manifest)).replace("ROOT", str(setup.root)) for a in python]
    assert setup.ran == [[*python, "-P", "-c", "import cellmap_flow; print(cellmap_flow.__file__)"]]
    assert ("not installed" in result.output) is (name == "cellpose4")


def test_a_check_that_fails_fails(setup, monkeypatch):
    monkeypatch.setattr(envs_cli.subprocess, "run", lambda argv: SimpleNamespace(returncode=1))
    result = _invoke("check", "mine")
    assert result.exit_code == 1 and "mine cannot import cellmap_flow (exit status 1)" in result.output
