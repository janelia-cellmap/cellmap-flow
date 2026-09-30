"""Validating a config, or importing the package, writes nothing to ~/.cellmap_flow.

``cellmap_flow_yaml --validate-only`` saved the YAML's queue and charge group
into ~/.cellmap_flow/server_config.yaml before returning, so checking a file
changed the defaults of the next dashboard. And every ``import cellmap_flow``
created ~/.cellmap_flow/plugins, including inside each LSF job.

Each check runs in a fresh interpreter with HOME pointed at an empty
directory, so anything written anywhere under it is caught.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tests", "script_test", "fake_model_script.py")


def _run(home, *argv):
    env = {**os.environ, "HOME": str(home), "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, env=env, timeout=600
    )


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home


def test_validate_only_leaves_the_saved_server_config_alone(home, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "data_path: /d.zarr\n"
        "charge_group: someone_elses_group\n"
        "queue: gpu_h200\n"
        f"models:\n  m: {{type: script, script_path: {SCRIPT}}}\n"
    )

    result = _run(home, "-m", "cellmap_flow.cli.yaml_cli", str(config), "--validate-only")

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Configuration is valid" in result.stdout
    assert list(home.iterdir()) == [], [str(p) for p in home.rglob("*")]
