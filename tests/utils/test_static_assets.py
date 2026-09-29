"""The dashboard's static files: what the package ships.

The pages load their scripts as they are, with no bundler, so a file the
package data leaves out only shows up in a browser, as a page whose scripts
silently never run.
"""

import glob
import tomllib
from pathlib import Path

import pytest

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).parent
REPO = DASHBOARD.parents[1]


def _matches(patterns, root):
    return {Path(p) for pat in patterns for p in glob.glob(str(root / pat), recursive=True)}


def test_the_wheel_and_the_sdist_ship_every_dashboard_file():
    pyproject = REPO / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not running from a source tree")
    config = tomllib.loads(pyproject.read_text())
    package_data = config["tool"]["setuptools"]["package-data"]["cellmap_flow.dashboard"]
    manifest = set()
    for line in (REPO / "MANIFEST.in").read_text().splitlines():
        words = line.split()
        if words[:1] == ["include"]:
            manifest |= _matches(words[1:], REPO)
        elif words[:1] == ["recursive-include"]:
            manifest |= _matches([f"{words[1]}/**/{pat}" for pat in words[2:]], REPO)

    files = [p for d in ("templates", "static") for p in (DASHBOARD / d).rglob("*") if p.is_file()]
    assert files
    assert [str(f) for f in files if f not in _matches(package_data, DASHBOARD)] == []
    assert [str(f) for f in files if f not in manifest] == []
