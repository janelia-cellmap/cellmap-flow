"""jobs/synced_tee.py: tee that syncs the log file, so another host reads it live."""

import os
import subprocess
import sys
import threading
import time

from cellmap_flow.jobs import synced_tee


def test_a_line_is_synced_within_the_interval_without_waiting_for_more(tmp_path, monkeypatch):
    """The last line before a pause (a checkpoint being saved) must not wait for
    the next one: the node's NFS client held unsynced lines for ~30 s."""
    synced = []
    monkeypatch.setattr(synced_tee.os, "fsync", lambda fd: synced.append(os.path.getsize(tmp_path / "log")))
    read_end, write_end = os.pipe()
    source, sink = os.fdopen(read_end, "rb"), open(os.devnull, "wb")
    runner = threading.Thread(target=synced_tee.tee, args=(source, sink, tmp_path / "log", 0.05))
    runner.start()
    os.write(write_end, b"Epoch 1/20 - Loss: 0.1\n")
    time.sleep(0.3)
    assert synced and synced[-1] == len(b"Epoch 1/20 - Loss: 0.1\n"), "synced while the writer pauses"
    os.write(write_end, b"Epoch 2/20 - Loss: 0.05\n")
    os.close(write_end)
    runner.join(timeout=5)
    assert (tmp_path / "log").read_bytes() == b"Epoch 1/20 - Loss: 0.1\nEpoch 2/20 - Loss: 0.05\n"
    assert synced[-1] == (tmp_path / "log").stat().st_size, "and once more at the end"


def test_as_a_script_it_copies_stdin_to_stdout_and_the_file(tmp_path):
    """Run with no cellmap_flow on the path: it needs only the standard library."""
    log = tmp_path / "training_log.txt"
    log.write_bytes(b"an older run\n")
    done = subprocess.run([sys.executable, "-P", "-I", synced_tee.__file__, str(log)],
                          input=b"a\nb\n", capture_output=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert done.stdout == b"a\nb\n" and log.read_bytes() == b"a\nb\n", "truncated first, as tee does"
