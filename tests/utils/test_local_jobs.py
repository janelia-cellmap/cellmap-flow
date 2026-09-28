"""Jobs run on this machine when there is no LSF.

``run_locally`` gave the child stdout/stderr PIPEs that nothing read once the
host marker had been found, so a server (or a finetune run) froze as soon as
it had logged about 64 KB. Killing it terminated only the direct child, so a
``bash -c`` wrapper died while the python under it kept the GPU. And the
marker was looked for with select() plus readline() on buffered pipes: a
marker that arrived in the same write as an earlier line stayed in Python's
buffer, invisible to select, until the wait timed out.
"""

import os
import sys
import time

import pytest

from cellmap_flow.utils import bsub_utils
from cellmap_flow.utils.bsub_utils import run_locally
from cellmap_flow.utils.web_utils import IP_PATTERN

MARKER = f"{IP_PATTERN[0]}http://10.0.0.1:1234{IP_PATTERN[1]}"


@pytest.fixture(autouse=True)
def _log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bsub_utils, "SERVER_LOG_DIR", tmp_path / "server_logs")


def _python(code):
    return [sys.executable, "-c", code]


def _wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(pid, 0); it is dead for our purposes.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def test_a_chatty_server_keeps_running_after_its_host_is_found(tmp_path):
    done = tmp_path / "done"
    code = (
        f"print({MARKER!r}, flush=True)\n"
        "import sys\n"
        "for _ in range(4000):\n"
        "    sys.stdout.write('x' * 99 + '\\n')\n"
        "    sys.stderr.write('y' * 99 + '\\n')\n"
        "sys.stdout.flush()\n"
        f"open({str(done)!r}, 'w').close()\n"
        "import time; time.sleep(60)\n"
    )
    job = run_locally(_python(code), "chatty")
    try:
        assert job.wait_for_host(timeout=15) == "http://10.0.0.1:1234"
        # 800 KB of output: far past a pipe buffer. It must not block.
        assert _wait_for(done.exists), "the child blocked writing its own logs"
        assert job.log_file.stat().st_size > 700_000
        assert "y" * 99 in job.peek()
    finally:
        job.kill()


def test_a_marker_in_the_same_write_as_other_output_is_found():
    code = (
        "import sys, time\n"
        f"sys.stdout.write('starting up\\n' + {MARKER!r} + '\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    job = run_locally(_python(code), "burst")
    try:
        started = time.monotonic()
        assert job.wait_for_host(timeout=10) == "http://10.0.0.1:1234"
        assert time.monotonic() - started < 5
    finally:
        job.kill()


def test_kill_takes_down_the_whole_process_tree(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    job = run_locally(["bash", "-c", f"sleep 1000 & echo $! > {pidfile}; wait"], "tree")
    assert _wait_for(lambda: pidfile.exists() and pidfile.read_text().strip())
    grandchild = int(pidfile.read_text())
    assert _alive(grandchild)

    job.kill()

    assert _wait_for(lambda: not _alive(grandchild), timeout=10), (
        "the process under the bash wrapper survived kill()"
    )


def test_a_server_that_dies_is_reported_with_its_output():
    job = run_locally(_python("import sys; print('boom: no model'); sys.exit(3)"), "dies")
    assert job.wait_for_host(timeout=10) is None
    assert job.get_status() == bsub_utils.JobStatus.FAILED
    assert "boom: no model" in job.peek()
