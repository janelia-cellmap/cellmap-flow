"""The checks the macOS CI job runs in each pixi environment, run here too.

Each script runs in a subprocess, as the job runs it, because what it checks
(a second OpenMP runtime) aborts the process rather than raising.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parents[1] / "smoke"


def _run(script):
    return subprocess.run([sys.executable, str(SMOKE / script)], capture_output=True, text=True, timeout=300)


def test_numpy_and_torch_share_one_openmp_runtime():
    pytest.importorskip("torch")
    result = _run("one_openmp.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OMP: Error" not in result.stderr


def test_a_zoo_model_description_loads():
    pytest.importorskip("bioimageio.spec")
    result = _run("zoo_description.py")
    assert result.returncode == 0, result.stdout + result.stderr
