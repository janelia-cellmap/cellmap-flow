"""How far a long request has got, for the page to poll while it waits.

Importing crops from a YAML and resuming a session each take one request
that can run for minutes. The page sends a ``load_id`` with it and polls a
progress route with the same id meanwhile; the request records each step
under that id as it goes.
"""

import threading
import time

from flask import jsonify


class Progress:
    """The latest progress of each request in flight, by its ``load_id``.

    A request's entry is a dict of what it last reported (its ``phase``,
    ``done`` and whatever counts it keeps), plus ``created_at`` and
    ``updated_at`` in epoch seconds. An entry not updated for
    ``ttl_seconds`` is dropped, so the entries of loads nobody polls any
    more do not pile up. Request threads update and read it concurrently.
    """

    def __init__(self, ttl_seconds=300):
        self._entries = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def update(self, load_id, **fields):
        """Merge ``fields`` into ``load_id``'s entry; without an id, nothing."""
        if not load_id:
            return
        with self._lock:
            now = time.time()
            entry = self._entries.setdefault(load_id, {"created_at": now})
            entry.update(fields)
            entry["updated_at"] = now
            for stale in [key for key, e in self._entries.items() if now - e["updated_at"] > self._ttl_seconds]:
                del self._entries[stale]

    def response(self, load_id):
        """A progress route's answer: ``{"success": True, "progress": <entry>}``;
        a 400 without a ``load_id``, and a 404 for one it has no entry for."""
        if not load_id:
            return jsonify({"success": False, "error": "Missing 'load_id' query param"}), 400
        with self._lock:
            entry = self._entries.get(load_id)
            entry = dict(entry) if entry else None
        if entry is None:
            return jsonify({"success": False, "error": f"Unknown load_id {load_id}"}), 404
        return jsonify({"success": True, "progress": entry})
