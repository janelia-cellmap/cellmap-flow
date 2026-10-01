"""Which session a base output path means, and the volumes registered for painting."""

import logging
import os
import re
from datetime import datetime
from typing import Optional

from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME

logger = logging.getLogger(__name__)

# A session directory: <base>/<YYYYmmdd_HHMMSS>.
SESSION_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


class SessionStore:
    """Sessions and the volume registry, kept in the dicts it is given.

    ``sessions`` maps a base output path to the session this process made
    under it (the dashboard's ``g.output_sessions``); ``volumes`` maps a
    volume id -- its MinIO bucket key without ".zarr" -- to its record
    (``g.annotation_volumes``). The store holds nothing else, so any number
    of them over the same dicts agree. Both dicts live in memory only;
    ``latest_on_disk`` is what finds a session again after a restart.
    """

    def __init__(self, sessions: dict, volumes: Optional[dict] = None):
        self._sessions = sessions
        self._volumes = {} if volumes is None else volumes

    def get_or_create(self, base: str) -> str:
        """The session this process uses under ``base``: the one it made
        before, else a new ``<base>/<timestamp>`` (not created on disk)."""
        base = os.path.expanduser(base)
        if base not in self._sessions:
            session_path = os.path.join(base, datetime.now().strftime("%Y%m%d_%H%M%S"))
            self._sessions[base] = session_path
            logger.info(f"Created new session path: {session_path}")
        return self._sessions[base]

    def latest_on_disk(self, base: str) -> Optional[str]:
        """The newest session under ``base`` that has something to train on.

        That is the latest ``<base>/<YYYYmmdd_HHMMSS>/`` whose corrections/
        holds a virtual-sources manifest. The in-memory map dies with the
        dashboard, so after a restart get_or_create hands out a new, empty
        session, and submitting training for the base path failed with
        "Corrections path does not exist" although the session painted
        before the restart was right there. None when there is none.
        """
        base = os.path.expanduser(str(base))
        try:
            entries = sorted(os.listdir(base), reverse=True)
        except OSError:
            return None
        for entry in entries:
            session = os.path.join(base, entry)
            if SESSION_DIR_RE.match(entry) and os.path.isfile(
                os.path.join(session, "corrections", VIRTUAL_MANIFEST_FILENAME)
            ):
                return session
        return None

    def register_volume(self, volume_id: str, *, keep_sync_state: bool = False, **data) -> dict:
        """Record a volume being served for painting; returns its record.

        The record starts with no chunk sync state, so the next sync compares
        MinIO against nothing and pulls every chunk. ``keep_sync_state``
        updates an existing record instead, keeping the state a pull just
        recorded (an instance correction reattached after its pre-mirror pull).
        """
        if keep_sync_state:
            record = self._volumes.setdefault(volume_id, {"chunk_sync_state": {}})
            record.update(data)
        else:
            record = self._volumes[volume_id] = {**data, "chunk_sync_state": {}}
        return record

    def volumes(self) -> dict:
        """The registry itself: volume id -> record."""
        return self._volumes

    def session_volume(self, corrections_dir: Optional[str] = None):
        """``(volume_id, record)`` of the volume a session is painting, or ``(None, None)``.

        That is the last one registered for ``corrections_dir``, or for any
        corrections dir when it is None: a session can hold several volumes,
        and the latest is the one the user just made or reattached. Every
        route asks this, so a crop import, the good regions and training all
        mean the same volume; load-crops used to take the first instead, and
        imported into one volume while the user painted another.
        """
        for volume_id, record in reversed(list(self._volumes.items())):
            registered_dir = record.get("corrections_dir")
            if registered_dir and (corrections_dir is None or str(registered_dir) == str(corrections_dir)):
                return volume_id, record
        return None, None
