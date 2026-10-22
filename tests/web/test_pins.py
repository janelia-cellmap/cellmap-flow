"""Pins that move together: chunkmirage's two, and the neuroglancer fork's two.

chunkmirage is pinned twice. pyproject.toml's is a version range, the release the Python inference server
is installed from: PyPI refuses an upload whose dependencies name a URL, so
a git pin there would keep cellmap-flow from being released.
web/chunkmirage.lock's is a commit, the one the browser engine is cloned
from (the wheel carries no web/). One repository, so the lock's commit is
the release's own commit or a later one, or the browser serves a chunk
format or protocol the server does not. That order is checked by hand on a
bump (pyproject.toml's comment names the release's commit, for
``git merge-base --is-ancestor <release> <lock>`` in a full clone):
web/chunkmirage/ is a shallow clone, with no history to check it in."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# A requirement on chunkmirage by version: its name, any extras, then
# comparisons only (no "@ <url>").
BY_VERSION = re.compile(r"chunkmirage\s*(\[[^\]]*\])?\s*[<>=!~][^@;]*")
LOCK_PIN = re.compile(r"^sha=([0-9a-f]{40})$", re.MULTILINE)


def test_the_server_takes_a_chunkmirage_release_and_the_browser_a_commit():
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    chunkmirage = [d for d in dependencies if re.match(r"chunkmirage\b", d)]
    assert len(chunkmirage) == 1, f"pyproject.toml should depend on chunkmirage once, found {chunkmirage}"
    assert BY_VERSION.fullmatch(chunkmirage[0].strip()), (
        f"pyproject.toml should take chunkmirage from PyPI by version, not {chunkmirage[0]!r}: "
        "PyPI refuses a release whose dependencies name a URL"
    )
    commits = LOCK_PIN.findall((ROOT / "web" / "chunkmirage.lock").read_text())
    assert len(commits) == 1, f"web/chunkmirage.lock should pin one full commit, found {commits}"


def test_the_doctors_neuroglancer_fork_install_names_the_commit_pixi_installs():
    """The paint fork is installed by pixi (pixi.toml's finetune feature) and,
    for a pip install, by the line `cellmap_flow doctor` prints: the two name
    one commit, or a pip user who follows the doctor gets a different client
    from the one that deploys. pyproject.toml's comment names it too."""
    from cellmap_flow.cli.doctor import NEUROGLANCER_FORK_COMMIT, NEUROGLANCER_FORK_INSTALL

    pixi = tomllib.loads((ROOT / "pixi.toml").read_text())
    pinned = pixi["feature"]["finetune"]["pypi-dependencies"]["neuroglancer"]["rev"]
    assert NEUROGLANCER_FORK_COMMIT == pinned
    assert NEUROGLANCER_FORK_INSTALL.endswith(f"@{pinned}")
    assert pinned in (ROOT / "pyproject.toml").read_text()
