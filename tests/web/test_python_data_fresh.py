"""web/src/generated/python-data.json describes the Python beside it.

web/build/export_data.py writes the constants and catalogs the browser app
shares with Python (the op registry and which ops the browser runs, the
default chain, the organelle catalog and the prompt's sentences, the shaders,
the ranking's constants, the finetune log's patterns, the unit table), and
the app imports that file in place of copies of its own. When the Python
changes and the file is not regenerated, the browser runs on the old values;
this test catches that.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "web" / "build" / "export_data.py"


def test_python_data_is_fresh():
    # In a process of its own: other tests define ops of their own (a
    # PostProcessor subclass), which an in-process export would list. The
    # current interpreter, with this checkout first on its path.
    pythonpath = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "web/src/generated/python-data.json is stale: a constant or catalog the browser app "
        "reads from Python changed since it was exported. Regenerate with\n"
        "    python web/build/export_data.py\n"
        "(or npm run render in web/, with the pixi default environment's python) and commit "
        "the result with the change.\n\n"
        f"{result.stdout}{result.stderr}"
    )
