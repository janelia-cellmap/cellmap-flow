"""FORCE_SAFE_CONFIG is read as a boolean, when a script is loaded.

It was ``os.getenv("FORCE_SAFE_CONFIG", False)`` as a default argument: any
non-empty string counted as true -- including the "False" that the error
message tells users to set -- and the value was fixed at import. Every row
sets it after import.
"""

import pytest

from cellmap_flow.utils.load_py import load_safe_config


@pytest.mark.parametrize("setting, argument, allowed", [
    pytest.param("False", None, True, id="False"),  # what the error message tells users to set
    pytest.param("false", None, True, id="false"),
    pytest.param("0", None, True, id="0"),
    pytest.param("no", None, True, id="no"),
    pytest.param("", None, True, id="empty"),
    pytest.param(None, None, True, id="unset"),
    pytest.param("True", None, False, id="True"),
    pytest.param("1", None, False, id="1"),
    pytest.param("yes", None, False, id="yes"),
    pytest.param("1", False, True, id="an-explicit-argument-wins"),
])
def test_an_unsafe_script_is_refused_only_when_asked(tmp_path, monkeypatch, setting, argument, allowed):
    script = tmp_path / "unsafe.py"
    script.write_text("import subprocess\nvalue = 3\n")
    if setting is None:
        monkeypatch.delenv("FORCE_SAFE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("FORCE_SAFE_CONFIG", setting)
    if allowed:
        assert load_safe_config(str(script), force_safe=argument).value == 3
    else:
        with pytest.raises(ValueError, match="Unsafe script"):
            load_safe_config(str(script), force_safe=argument)
