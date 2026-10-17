"""Keys bound in a neuroglancer viewer, and the messages they leave in its status bar.

The dashboard's keys over the viewer (the Finetune tab's Shift+G, which asks
a hosted model for a first guess, and Shift+R, which restarts a finetune
job; the Review tab's t, which picks a segment) each bind an action to a key
and tell the user how it went in neuroglancer's status bar:

- ``bind_key(viewer, action, key, handler)``: bind ``key`` to ``action``,
  run by ``handler``, once per viewer.
- ``say(viewer, status_key, message)``: show ``message`` in the status bar,
  or clear it (None).
- ``say_for(viewer, status_key, message, seconds)``: show it for
  ``seconds``, then clear it unless another has replaced it since.

The status bar shows one message per status key, so each feature says its
own under a key of its own, and one's message does not overwrite another's.

Nothing here reads the dashboard's state: callers pass the viewer.
"""

import logging
import threading
import weakref

logger = logging.getLogger(__name__)

# {viewer: the (action, key) pairs bound in it}. A new dataset makes a new
# viewer, so the viewers bound are remembered rather than a flag. Weakly:
# a viewer replaced is forgotten with it, and the one that replaced it is
# bound the first time a caller asks.
_bound = weakref.WeakKeyDictionary()
_bound_lock = threading.Lock()


def bind_key(viewer, action, key, handler):
    """Bind ``key`` in ``viewer`` to the action ``action``, which runs ``handler``; once per viewer.

    ``key`` is in neuroglancer's spelling ("shift+keyg", "keyt"). Callers ask
    on every request after which there may be a new viewer, and only a
    viewer not bound yet is bound; one bound since it was made is left as it
    is. None for ``viewer`` (no dataset opened yet) does nothing.

    Two requests binding the same viewer at once both bind it, which does no
    harm: neuroglancer keeps a set of handlers per action, so adding the one
    module-level handler again adds nothing, and the key is set to the same
    action twice.
    """
    if viewer is None:
        return
    with _bound_lock:
        if (action, key) in _bound.get(viewer, ()):
            return
    viewer.actions.add(action, handler)
    with viewer.config_state.txn() as cs:
        cs.input_event_bindings.viewer[key] = action
    with _bound_lock:
        _bound.setdefault(viewer, set()).add((action, key))


def say(viewer, status_key, message):
    """Show ``message`` in ``viewer``'s status bar under ``status_key`` (None clears it); never raises.

    None for ``viewer`` does nothing.
    """
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


def _later(seconds, function, *args):
    """Run ``function(*args)`` in ``seconds``, on a daemon timer. Tests replace this."""
    timer = threading.Timer(seconds, function, args)
    timer.daemon = True
    timer.start()


def say_for(viewer, status_key, message, seconds):
    """Show ``message`` under ``status_key`` for ``seconds``, then clear it, unless another has replaced it since.

    None for ``viewer`` does nothing.
    """
    say(viewer, status_key, message)
    if viewer is not None:
        _later(seconds, _clear, viewer, status_key, message)


def _clear(viewer, status_key, message):
    """Clear ``message`` from under ``status_key``, unless another has replaced it since; never raises.

    The config state is the dashboard's alone (the page never changes it),
    so a transaction that leaves it as it was costs one push of it.
    """
    try:
        with viewer.config_state.txn() as s:
            if s.status_messages.get(status_key) == message:
                s.status_messages.pop(status_key, None)
    except Exception as e:
        logger.debug(f"Could not clear the viewer's status message: {e}")
