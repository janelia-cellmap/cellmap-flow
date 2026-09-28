"""Shared test setup.

HOME is redirected before anything imports cellmap_flow: importing the package
executes ~/.cellmap_flow/plugins/*.py and the Flow singleton reads
~/.cellmap_flow/server_config.yaml, so otherwise the suite depends on (and can
write to) the developer's real config.
"""

import importlib
import os
import shutil
import tempfile
from collections import deque

os.environ["HOME"] = tempfile.mkdtemp(prefix="cellmap_flow_test_home_")

import pytest  # noqa: E402


def _can_import(module):
    # peft raises ImportError subclasses (and occasionally other errors) when
    # its transformers/huggingface-hub pins don't match, not only when absent.
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _has_cuda():
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


_REQUIREMENTS = {
    "finetune": (lambda: _can_import("peft"), "peft is not importable"),
    "gpu": (_has_cuda, "no CUDA device"),
    "lsf": (lambda: shutil.which("bsub") is not None, "bsub not found"),
    "minio": (
        lambda: shutil.which("minio") is not None and shutil.which("mc") is not None,
        "minio/mc binaries not found",
    ),
    "network": (
        lambda: os.environ.get("CELLMAP_FLOW_NETWORK_TESTS") == "1",
        "set CELLMAP_FLOW_NETWORK_TESTS=1 to run",
    ),
}


def pytest_collection_modifyitems(config, items):
    available = {}
    for item in items:
        for marker, (check, reason) in _REQUIREMENTS.items():
            # Not `marker in item.keywords`: keywords include parent package
            # names, so every test under tests/finetune/ would match "finetune".
            if item.get_closest_marker(marker) is None:
                continue
            if marker not in available:
                available[marker] = check()
            if not available[marker]:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _restore_flow_state():
    """Undo whatever a test does to the process-wide Flow singleton."""
    from cellmap_flow.globals import g

    saved = {
        key: value.copy() if isinstance(value, (list, dict, set, deque)) else value
        for key, value in vars(g).items()
    }
    yield
    vars(g).clear()
    vars(g).update(saved)
