"""FORCE_SAFE_CONFIG is read as a boolean, when a script is loaded.

It was ``os.getenv("FORCE_SAFE_CONFIG", False)`` as a default argument: any
non-empty string counted as true -- including the "False" that the error
message tells users to set -- and the value was fixed at import.
"""

import os
import subprocess
import sys

import pytest

from cellmap_flow.utils.load_py import load_safe_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def unsafe_script(tmp_path):
    path = tmp_path / "unsafe.py"
    path.write_text("import subprocess\nvalue = 3\n")
    return str(path)


@pytest.mark.parametrize("setting", ["False", "false", "0", "no", ""])
def test_a_false_setting_allows_the_script(monkeypatch, unsafe_script, setting):
    monkeypatch.setenv("FORCE_SAFE_CONFIG", setting)
    assert load_safe_config(unsafe_script).value == 3


@pytest.mark.parametrize("setting", ["True", "1", "yes"])
def test_a_true_setting_made_after_import_refuses_the_script(monkeypatch, unsafe_script, setting):
    monkeypatch.setenv("FORCE_SAFE_CONFIG", setting)
    with pytest.raises(ValueError, match="Unsafe script"):
        load_safe_config(unsafe_script)


def test_unset_allows_the_script(monkeypatch, unsafe_script):
    monkeypatch.delenv("FORCE_SAFE_CONFIG", raising=False)
    assert load_safe_config(unsafe_script).value == 3


def test_an_explicit_argument_wins(monkeypatch, unsafe_script):
    monkeypatch.setenv("FORCE_SAFE_CONFIG", "1")
    assert load_safe_config(unsafe_script, force_safe=False).value == 3


def test_false_in_the_environment_at_import_time_allows_the_script(tmp_path, unsafe_script):
    code = (
        "from cellmap_flow.utils.load_py import load_safe_config\n"
        f"print(load_safe_config({unsafe_script!r}).value)\n"
    )
    env = {**os.environ, "FORCE_SAFE_CONFIG": "False", "HOME": str(tmp_path), "PYTHONPATH": ROOT}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("3")
