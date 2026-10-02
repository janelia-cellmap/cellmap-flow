"""Jobs run on this machine when there is no LSF, with real child processes.

The child's stdout/stderr went to PIPEs nothing read once the host marker was
found, so a server (or a finetune run) froze after logging about 64 KB;
killing it ended only the direct child, so a ``bash -c`` wrapper died while
the python under it kept the GPU; and a marker that came in the same write as
an earlier line stayed invisible in Python's buffer until the wait timed out.
"""

import os
import sys
import time

import pytest

from cellmap_flow.jobs import launch, local
from cellmap_flow.jobs.spec import IP_PATTERN, JobStatus

MARKER = f"{IP_PATTERN[0]}http://10.0.0.1:1234{IP_PATTERN[1]}"


@pytest.fixture(autouse=True)
def _log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "SERVER_LOG_DIR", tmp_path / "server_logs")


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
        with open(f"/proc/{pid}/stat") as f:  # a zombie still answers kill(pid, 0)
            return f.read().split()[2] != "Z"
    except (ProcessLookupError, OSError):
        return False


def test_a_chatty_server_is_found_at_once_and_keeps_running(tmp_path):
    done = tmp_path / "done"
    code = (
        "import sys, time\n"
        f"sys.stdout.write('starting up\\n' + {MARKER!r} + '\\n'); sys.stdout.flush()\n"
        "for _ in range(4000):\n"
        "    sys.stdout.write('x' * 99 + '\\n'); sys.stderr.write('y' * 99 + '\\n')\n"
        f"sys.stdout.flush(); open({str(done)!r}, 'w').close(); time.sleep(60)\n"
    )
    job = local.run([sys.executable, "-c", code], "chatty")
    try:
        started = time.monotonic()
        assert job.wait_for_host(timeout=15) == "http://10.0.0.1:1234"
        assert time.monotonic() - started < 5, "the marker shared a write with the line before it"
        # 800 KB of output, far past a pipe buffer: it must not block.
        assert _wait_for(done.exists), "the child blocked writing its own logs"
        assert job.log_file.stat().st_size > 700_000 and "y" * 99 in job.peek()
    finally:
        job.kill()


def test_kill_takes_down_the_whole_process_tree(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    job = local.run(["bash", "-c", f"sleep 1000 & echo $! > {pidfile}; wait"], "tree")
    assert _wait_for(lambda: pidfile.exists() and pidfile.read_text().strip())
    grandchild = int(pidfile.read_text())
    assert _alive(grandchild)
    job.kill()
    assert _wait_for(lambda: not _alive(grandchild), timeout=10), "the process under the bash wrapper survived"


def test_a_server_that_dies_is_reported_with_its_output():
    job = local.run([sys.executable, "-c", "import sys; print('boom: no model'); sys.exit(3)"], "dies")
    assert job.wait_for_host(timeout=10) is None
    assert job.get_status() == JobStatus.FAILED and "boom: no model" in job.peek()
