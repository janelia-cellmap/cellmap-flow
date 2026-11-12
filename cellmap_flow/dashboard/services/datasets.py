"""Showing the running models on the dataset the person opens.

An inference server serves one model, and can hold more than one dataset:
the one it serves and the one before (serving.protocol.DATASET_CONTROL_PATH).
So opening another dataset (POST /api/set-data) asks each running server to
open it there, rather than start a new job or load its model again, and
draws each model's layer on the new viewer as its server answers.

- ``switch_job(job, path)``: have ``job``'s server serve ``path``; why it
  could not, or None.
- ``to_open_dataset(job)``: the same for the session's dataset, when the
  job's server serves another: a model that came up after an Open
  (services.launch), a finetune's next iteration (finetune_layers), a model
  the ranking reads (routes.finetune.rank_routes).
- ``switch_all(path, viewer)``: every running server, in parallel and in the
  background, each model's layer drawn into ``viewer`` as its server
  answers.

A job keeps where it is (``dataset_path``), the switch under way
(``switching_to``) and why its last one failed (``switch_error``), as
jobs.spec.Job lists them. Session.on_current_dataset reads the first: only a
job on the open dataset gets a layer (routes.pipeline's redraw,
services.launch._show); the Models tab's Loaded models list says why another
is not shown, and its Submit starts that one again
(services.launch.update_run_models).
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.serving import client

logger = logging.getLogger(__name__)

# Servers asked at once. Each answer waits on its server reading the new
# dataset's metadata, over NFS or the network, so thirty asked one after
# another took as long as thirty reads; a few at a time is that divided by
# this, without thirty threads.
MAX_PARALLEL_SWITCHES = 8

_locks_guard = threading.Lock()


def _lock_of(job) -> threading.RLock:
    """The lock that puts ``job``'s switches one after another: a second Open
    while the first's switch is under way asks only once that one is
    answered, so the server ends on the dataset opened last."""
    with _locks_guard:
        lock = getattr(job, "_switch_lock", None)
        if lock is None:
            lock = job._switch_lock = threading.RLock()
        return lock


def _switch(job, path):
    """``(error, model_info)``: ``switch_job``'s answer, and the model_info
    its server answered with (None when not asked, or it gave none), which
    a layer is drawn from without asking again."""
    with _lock_of(job):
        if getattr(job, "dataset_path", None) == path:
            return None, None
        job.switching_to = path
        try:
            # The address the dashboard reaches the server by: a finetune
            # job's host is the viewers' (finetune_layers).
            answer = client.switch_dataset(getattr(job, "control_url", None) or job.host, path,
                                           getattr(job, "control_token", None))
        finally:
            job.switching_to = None
        if answer.get("success"):
            job.dataset_path, job.switch_error = path, None
            info = answer.get("model_info")
            return None, info if isinstance(info, dict) else None
        job.switch_error = answer.get("error") or "its server did not say why"
        return job.switch_error, None


def switch_job(job, path) -> Optional[str]:
    """Have ``job``'s server open ``path`` and serve it from now on, unless
    it serves it already. Returns why it could not, or None.

    On success the job's ``dataset_path`` is ``path`` and its
    ``switch_error`` None; on failure ``switch_error`` says why, and the job
    is where it was. Never raises (client.switch_dataset does not); one
    job's switches run one at a time.
    """
    return _switch(job, path)[0]


def to_open_dataset(job) -> Optional[str]:
    """Have ``job``'s server serve the session's dataset when it serves
    another (Session.on_current_dataset). Returns why it could not, or None:
    it does now, it did already, or there is nothing open to serve."""
    session = get_session()
    if session.on_current_dataset(job) or not session.dataset_path:
        return None
    return switch_job(job, session.dataset_path)


# What a job's switch comes to when a newer Open came first: not asked,
# not counted.
_SUPERSEDED = object()


def switch_all(path, viewer) -> threading.Thread:
    """Switch every running server to ``path``, in the background, and draw
    each model's layer into ``viewer`` as soon as its own server has
    switched: as Submit draws one (services.launch.draw_layer), from the
    model_info the switch answered with. Returns the thread.

    Up to MAX_PARALLEL_SWITCHES servers are asked at once, each on its
    own: one that is slow, cannot read the dataset or cannot switch at all
    holds up no other. A layer is drawn only while ``path`` is still the
    session's dataset and ``viewer`` its viewer: an Open after this one
    wins, and its own switch_all asks each server again. A job not yet up
    (no host) is not asked; services.launch._show switches it as it comes
    up. It ends with one log line, which says which could not and why.
    """
    session = get_session()
    jobs = [job for job in list(session.jobs) if getattr(job, "host", None)]
    thread = threading.Thread(target=_switch_all, args=(session, path, viewer, jobs), daemon=True,
                              name="switch-dataset")
    thread.start()
    return thread


def _switch_all(session, path, viewer, jobs):
    from cellmap_flow.dashboard.services.launch import draw_layer

    def still_open():
        return session.dataset_path == path and session.viewer is viewer

    def one(job):
        with _lock_of(job):
            # Checked once the job's earlier switch is answered: a newer
            # Open's is asked instead of this one.
            if not still_open():
                return _SUPERSEDED
            error, info = _switch(job, path)
        # Not for a model stopped while its server switched (its ✕).
        if error is None and still_open() and any(j is job for j in session.jobs):
            try:
                draw_layer(job, viewer, info=info)
            except Exception as e:
                logger.error(f"Could not draw {job.model_name}'s layer on {path}: {e}", exc_info=True)
                return f"its layer could not be drawn ({e})"
        return error

    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SWITCHES, thread_name_prefix="switch-dataset") as pool:
        outcomes = list(pool.map(one, jobs))
    asked = [(job, outcome) for job, outcome in zip(jobs, outcomes) if outcome is not _SUPERSEDED]
    if not asked:
        return
    failed = [(job, error) for job, error in asked if error]
    line = f"Showing {len(asked) - len(failed)} of {len(asked)} models on {path}"
    line += "".join(f"; {job.model_name} could not: {error}" for job, error in failed)
    (logger.warning if failed else logger.info)(line)
