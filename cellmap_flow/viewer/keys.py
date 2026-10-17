"""Keys bound in a neuroglancer viewer, and the status-bar messages they leave: each
feature's under a status key of its own, so that they do not overwrite one another."""

import logging
import threading
import weakref

logger = logging.getLogger(__name__)

# {viewer: its (action, key) pairs bound}; weak, so a replaced viewer goes with it.
_bound = weakref.WeakKeyDictionary()


def bind_key(viewer, action, key, handler):
    """Bind ``key`` ("shift+keyg") in ``viewer`` to ``action``, run by ``handler``, once per viewer.

    Two requests racing both bind it, which is harmless: neuroglancer keeps a
    set of handlers per action.
    """
    if viewer is None or (action, key) in _bound.get(viewer, ()):
        return
    viewer.actions.add(action, handler)
    with viewer.config_state.txn() as cs:
        cs.input_event_bindings.viewer[key] = action
    _bound.setdefault(viewer, set()).add((action, key))


def say(viewer, status_key, message):
    """Show ``message`` in the status bar under ``status_key`` (None clears it); never raises."""
    if viewer is None:
        return
    try:
        with viewer.config_state.txn() as s:
            if message:
                s.status_messages[status_key] = message
            else:
                s.status_messages.pop(status_key, None)
    except Exception as e:
        logger.debug(f"Could not set the viewer's status message: {e}")


def later(seconds, function, *args):
    """Run ``function(*args)`` in ``seconds``, on a daemon timer. Tests replace this."""
    timer = threading.Timer(seconds, function, args)
    timer.daemon = True
    timer.start()


def say_for(viewer, status_key, message, seconds):
    """``say`` ``message`` for ``seconds``."""
    say(viewer, status_key, message)
    if viewer is not None:
        later(seconds, _clear, viewer, status_key, message)


def _clear(viewer, status_key, message):
    """Clear ``message``, unless another has replaced it since; never raises.

    The config state is the dashboard's alone (the page never changes it),
    so a transaction that leaves it as it was costs one push of it.
    """
    try:
        with viewer.config_state.txn() as s:
            if s.status_messages.get(status_key) == message:
                s.status_messages.pop(status_key, None)
    except Exception as e:
        logger.debug(f"Could not clear the viewer's status message: {e}")
