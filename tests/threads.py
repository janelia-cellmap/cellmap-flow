"""Patching ``time.sleep`` for one test without the other threads in its process."""

import threading
import time


def patch_sleep(monkeypatch, module, on_sleep):
    """``module``'s ``time.sleep`` calls ``on_sleep(seconds)`` instead, on this
    thread only; any other thread sleeps as it would.

    ``module.time`` is the time module itself, so patching its ``sleep``
    patches it for every thread in the process. A thread an earlier test left
    running (a job's monitor, a publish) that slept meanwhile ran ``on_sleep``
    between the test's own steps: a log grew by two chunks between two reads,
    and the monitor read the server's line and the next iteration's together.
    """
    real, here = time.sleep, threading.get_ident()
    monkeypatch.setattr(module.time, "sleep",
                        lambda seconds: on_sleep(seconds) if threading.get_ident() == here else real(seconds))
