"""Job cleanup on Ctrl+C/SIGTERM is installed by entry points, not on import.

Importing bsub_utils set SIGINT and SIGTERM handlers for every importer and
raised ValueError when that first import happened off the main thread; the
handler then exited 0 on SIGTERM, as if the run had succeeded.
"""

import os
import signal
import subprocess
import sys
import threading

import pytest
from click.testing import CliRunner

from cellmap_flow.globals import g
from cellmap_flow.utils import bsub_utils

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_importing_leaves_signal_handlers_alone_even_off_the_main_thread(tmp_path):
    code = (
        "import signal, threading\n"
        "before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))\n"
        "errors = []\n"
        "def load():\n"
        "    try:\n"
        "        import cellmap_flow.utils.bsub_utils\n"
        "    except Exception as e:\n"
        "        errors.append(repr(e))\n"
        "t = threading.Thread(target=load); t.start(); t.join()\n"
        "assert not errors, errors\n"
        "after = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))\n"
        "assert after == before, (before, after)\n"
        "print('ok')\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": ROOT}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_installing_off_the_main_thread_is_a_no_op():
    outcome = []
    t = threading.Thread(target=lambda: outcome.append(bsub_utils.install_cleanup_handlers()))
    t.start()
    t.join()
    assert outcome == [False]


def test_the_handler_kills_jobs_and_exits_with_the_signal_status():
    class Job:
        model_name = "m"
        killed = False

        def kill(self):
            self.killed = True

    job = Job()
    g.jobs = [job]
    with pytest.raises(SystemExit) as exc:
        bsub_utils.cleanup_handler(signal.SIGTERM, None)
    assert job.killed
    assert exc.value.code == 128 + signal.SIGTERM


@pytest.fixture
def installs(monkeypatch):
    """Record install_cleanup_handlers calls instead of touching this process's handlers."""
    calls = []
    fake = lambda: calls.append(True) or True  # noqa: E731
    monkeypatch.setattr(bsub_utils, "install_cleanup_handlers", fake)
    return calls


def test_cellmap_flow_installs_them(monkeypatch, installs):
    from cellmap_flow.cli import cli as cli_module

    monkeypatch.setattr(cli_module, "install_cleanup_handlers", bsub_utils.install_cleanup_handlers)
    monkeypatch.setattr(cli_module, "cli", lambda: None)
    cli_module.main()
    assert installs == [True]


def test_cellmap_flow_yaml_installs_them_before_starting_jobs(monkeypatch, tmp_path, installs):
    from cellmap_flow.cli import yaml_cli

    order = []
    monkeypatch.setattr(yaml_cli, "install_cleanup_handlers", lambda: order.append("install"))
    monkeypatch.setattr(yaml_cli, "run_multiple", lambda *a, **k: order.append("run"))
    config = tmp_path / "c.yaml"
    config.write_text("data_path: /d.zarr\ncharge_group: grp\nqueue: gpu_h100\nmodels: {}\n")

    result = CliRunner().invoke(yaml_cli.main, [str(config)])

    assert result.exit_code == 0, result.output
    assert order == ["install", "run"]


def test_cellmap_flow_view_installs_them(monkeypatch, installs):
    import neuroglancer

    from cellmap_flow.cli import viewer_cli
    from cellmap_flow.dashboard import app
    from cellmap_flow.utils import scale_pyramid

    class FakeViewer:
        def txn(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        layers = {}
        dimensions = None

    monkeypatch.setattr(neuroglancer, "Viewer", FakeViewer)
    monkeypatch.setattr(neuroglancer, "set_server_bind_address", lambda *a: None)
    monkeypatch.setattr(scale_pyramid, "get_raw_layer", lambda *a, **k: "raw")
    monkeypatch.setattr(app, "create_and_run_app", lambda **k: None)

    result = CliRunner().invoke(viewer_cli.main, ["-d", "/d.zarr"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert installs == [True]
