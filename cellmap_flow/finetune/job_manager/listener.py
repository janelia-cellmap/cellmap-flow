"""What the job manager tells about its jobs, and whom.

The manager knows nothing of the viewer. The dashboard registers a listener
(dashboard.finetune_layers.FinetuneLayerListener) that shows each finished
iteration's model once the job's inference server is up.
"""

import logging

from cellmap_flow.finetune.job_manager.state import FinetuneJob

logger = logging.getLogger(__name__)


class FinetuneJobListener:
    """What the job manager tells its listeners (FinetuneJobManager.add_listener).

    Both are called on the job's monitor thread. A listener need not define
    both; one that raises is logged and does not stop the others.

    While listeners run, ``job.finetuned_model_name`` is still the name the
    job's model had before the event (None before the first), so one that
    replaces a viewer layer can find the old one. The manager sets it to
    ``model_name`` once they have all run.
    """

    def on_server_ready(self, job: FinetuneJob, url: str, model_name: str) -> None:
        """The job's inference server is up at ``url``, serving ``model_name``."""

    def on_iteration_complete(self, job: FinetuneJob, model_name: str) -> None:
        """The job finished a training iteration and named its model ``model_name``."""


class Listeners:
    """A manager's listeners, each told once per event, in the order they were added."""

    def __init__(self):
        self._listeners = []

    def add(self, listener) -> None:
        """Tell ``listener`` about job events. Adding one already added does nothing."""
        if not any(other is listener for other in self._listeners):
            self._listeners.append(listener)

    def remove(self, listener) -> None:
        """Stop telling ``listener``."""
        self._listeners = [other for other in self._listeners if other is not listener]

    def notify(self, event: str, *args) -> None:
        """Call each listener's ``event`` handler (a FinetuneJobListener method) with ``args``."""
        for listener in list(self._listeners):
            handler = getattr(listener, event, None)
            if handler is None:
                continue
            try:
                handler(*args)
            except Exception as e:
                logger.error(f"Finetune listener {listener!r} failed in {event}: {e}", exc_info=True)
