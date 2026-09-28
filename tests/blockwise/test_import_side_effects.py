"""Importing the blockwise modules leaves the process's logging alone.

blockwise_processor called basicConfig and set the root logger to INFO, twice,
at import time, and the dashboard imports it lazily for the blockwise
precheck, so a dashboard running at WARNING started logging at INFO.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_importing_blockwise_keeps_the_log_level(tmp_path):
    code = (
        "import logging\n"
        "import cellmap_flow.globals\n"  # what a running dashboard has already done
        "root = logging.getLogger()\n"
        "root.setLevel(logging.WARNING)\n"
        "handlers = list(root.handlers)\n"
        "import cellmap_flow.blockwise.blockwise_processor\n"
        "import cellmap_flow.blockwise.cli\n"
        "import cellmap_flow.blockwise.multiple_cli\n"
        "assert root.level == logging.WARNING, logging.getLevelName(root.level)\n"
        "assert root.handlers == handlers, root.handlers\n"
        "print('ok')\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
