"""Where a served chunk's time goes.

The server starts a record when a chunk request arrives (``start``), the
stages it passes through add their time to it (``stage``, ``add``), and the
server logs it when the chunk is sent (``finish``). The record is per thread,
and the server answers each request on its own thread, so concurrent chunks
keep separate records. Outside a request (blockwise, scripts, tests) no
record is started, and ``stage`` and ``add`` do nothing.
"""

import contextlib
import threading
import time

_local = threading.local()


def start():
    """Begin a record for the chunk this thread is serving."""
    _local.stages = {}


def add(name, seconds):
    """Add ``seconds`` to stage ``name`` of this thread's record, if one is started."""
    stages = getattr(_local, "stages", None)
    if stages is not None:
        stages[name] = stages.get(name, 0.0) + seconds


@contextlib.contextmanager
def stage(name):
    """Time the block as stage ``name`` of this thread's record."""
    begin = time.perf_counter()
    try:
        yield
    finally:
        add(name, time.perf_counter() - begin)


def finish():
    """This thread's record, ``{stage: seconds}`` in the order first timed; ends it."""
    stages = getattr(_local, "stages", None) or {}
    _local.stages = None
    return stages
