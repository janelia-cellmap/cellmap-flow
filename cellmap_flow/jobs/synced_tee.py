"""``tee`` for a log another host is reading: the file is synced as it grows.

A finetune job writes its log on its own node and the dashboard reads it
from another. ``tee`` writes each line into the node's NFS page cache, and
the client sends it on only when the page is old enough (about 30 s), so
the dashboard's log and loss plot got ten epochs at once, each stamped with
the right time. This copies stdin to stdout and to the file, and a thread
syncs the file (``os.fsync``) at most every SYNC_INTERVAL_S seconds while
there is something unsynced, so a line reaches the server within a second,
including the last one before a pause. Syncing on every line would make a
burst of output (a traceback) wait on the file server line by line.

Standard library only, and run as a script (``python -P <this file> LOG``),
so it works with any interpreter and needs nothing of cellmap-flow's.
"""

import os
import sys
import threading

SYNC_INTERVAL_S = 1.0


def tee(source, sink, path, interval=SYNC_INTERVAL_S):
    """Copy ``source``'s lines (binary) to ``sink`` and to ``path``, truncated first, syncing it."""
    unsynced = threading.Event()
    done = threading.Event()
    lock = threading.Lock()

    with open(path, "wb", buffering=0) as log:

        def sync_now():
            with lock:
                if unsynced.is_set():
                    unsynced.clear()
                    os.fsync(log.fileno())

        def syncer():
            while not done.wait(interval):
                sync_now()

        thread = threading.Thread(target=syncer, daemon=True)
        thread.start()
        try:
            for line in iter(source.readline, b""):
                sink.write(line)
                sink.flush()
                with lock:
                    log.write(line)
                    unsynced.set()
        finally:
            done.set()
            thread.join()
            sync_now()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} LOG_FILE")
    tee(sys.stdin.buffer, sys.stdout.buffer, sys.argv[1])
