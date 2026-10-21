"""The two chunkmirage pins move together: pyproject.toml's, the commit the
Python inference server is built on, and web/chunkmirage.lock's, the commit
the browser engine is cloned from. One repository, so one commit, or the
browser serves a chunk format or protocol the server does not."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PIN = re.compile(r"chunkmirage @ git\+https://github\.com/JaneliaSciComp/chunkmirage@([0-9a-f]{40})")
LOCK_PIN = re.compile(r"^sha=([0-9a-f]{40})$", re.MULTILINE)


def _pin(path, pattern):
    found = pattern.findall(path.read_text())
    assert len(found) == 1, f"{path.relative_to(ROOT)} should pin chunkmirage once, found {found}"
    return found[0]


def test_the_python_and_browser_chunkmirage_pins_agree():
    assert _pin(ROOT / "pyproject.toml", PYPROJECT_PIN) == _pin(ROOT / "web" / "chunkmirage.lock", LOCK_PIN)
