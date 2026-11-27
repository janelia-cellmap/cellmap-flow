"""Postprocessing steps that hold Python's lock, run in processes of their own.

A model server computes the chunks of a view on threads. A step that keeps
Python's lock while it works (the mutex watershed of ``AffinityPostprocessor``)
runs one chunk at a time on them however many cores the job has: four chunks
of a ViT's 128-voxel affinities took 19.8 s on four threads, 5 s each as one
alone does, and 6.7 s in four processes. So a step that says
``separate_process`` is run here, in a pool of worker processes, one per core
the job may use; the thread that asked waits for it without the lock.

Only a step with no state of its own can: the worker builds its own copy from
the step as the Processing tab sent it (``{"name": ..., **params}``), so what
a step learns from one chunk (a blockwise merger's equivalences) would stay
in the worker. The chunk and its result are copied to and from the worker,
tens of ms against the seconds such a step takes.

``CELLMAP_FLOW_POSTPROCESS_WORKERS`` sets the number of workers; 0 runs every
step in the server's own process, as with one core.
"""

import functools
import json
import logging
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

logger = logging.getLogger(__name__)

WORKERS_ENV = "CELLMAP_FLOW_POSTPROCESS_WORKERS"

_lock = threading.Lock()
_executor = None
_broken = False


def worker_count() -> int:
    """How many worker processes to run: ``WORKERS_ENV``, else the cores this
    job may use (the smaller of the cores the process is allowed on and the
    slots LSF gave it), and none when that is one."""
    asked = os.environ.get(WORKERS_ENV)
    if asked is not None:
        try:
            return max(0, int(asked))
        except ValueError:
            logger.warning(f"{WORKERS_ENV}={asked!r} is not a number; using the job's cores")
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        cores = os.cpu_count() or 1
    slots = os.environ.get("LSB_DJOB_NUMPROC", "")
    if slots.isdigit() and int(slots) > 0:
        cores = min(cores, int(slots))
    return cores if cores > 1 else 0


def _pool():
    """The pool, started on first use; None when steps run in this process
    (no workers, or the pool broke before)."""
    global _executor
    with _lock:
        if _executor is None and not _broken:
            workers = worker_count()
            if workers:
                # spawn: the server holds a GPU context and many threads,
                # neither of which survives a fork.
                _executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
                logger.info(f"Postprocessing steps that hold Python's lock run in {workers} worker processes")
        return _executor


@functools.lru_cache(maxsize=64)
def _built(step_json):
    """A worker's own copy of a step, built once."""
    from cellmap_flow.pipeline_spec import PipelineSpec

    return PipelineSpec((), (json.loads(step_json),)).build()[1][0]


def _work(step_json, block, kwargs):
    return np.asarray(_built(step_json)(block, **kwargs))


def warm(step: dict):
    """Start the workers and have them build ``step``, without waiting: the
    first chunks of a layer then do not wait for Python to start in each."""
    pool = _pool()
    if pool is None:
        return
    step_json = json.dumps(step, sort_keys=True)
    try:
        for _ in range(worker_count()):
            pool.submit(_built, step_json)
    except (BrokenProcessPool, RuntimeError) as e:
        logger.warning(f"Could not start the postprocessing workers: {e}")


def run(step: dict, post, block, **kwargs):
    """``post(block, **kwargs)``, computed in a worker by its own copy of
    ``step`` (which ``post`` was built from). In this process when there are
    no workers, or once the pool has broken (a worker killed): the chunk is
    still computed, as before there were workers. What the step raises is
    raised here."""
    global _executor, _broken
    pool = _pool()
    if pool is None:
        return post(block, **kwargs)
    try:
        return pool.submit(_work, json.dumps(step, sort_keys=True), np.asarray(block), kwargs).result()
    except BrokenProcessPool as e:
        with _lock:
            if not _broken:
                logger.warning(f"The postprocessing workers stopped ({e}); steps run in the server's own process "
                               f"from now on, one chunk at a time for those that hold Python's lock")
            _broken, _executor = True, None
        return post(block, **kwargs)
