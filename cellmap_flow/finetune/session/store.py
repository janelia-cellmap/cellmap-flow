"""Which session a base output path means, the volumes registered for
painting, and the session's current volume among them."""

import logging
import os
import re
from datetime import datetime
from typing import Optional

from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME, trained_volume_name

logger = logging.getLogger(__name__)

# A session directory: <base>/<YYYYmmdd_HHMMSS>.
SESSION_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

# Where a session's retired volumes go: <corrections>/superseded/<id>.zarr.
# Everything that looks for a session's volumes lists the corrections dir's
# top level only (the list and resume of sessions, the annotated regions, the
# trainer through the manifest), so a volume here is kept and never seen.
SUPERSEDED_DIRNAME = "superseded"


class SessionStore:
    """Sessions and the volume registry, kept in the dicts it is given.

    ``sessions`` maps a base output path to the session this process made
    under it (the dashboard's ``get_session().output_sessions``); ``volumes``
    maps a volume id -- its MinIO bucket key without ".zarr" -- to its
    record (``get_session().annotation_volumes``). The store holds nothing else, so any number
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
        """``(volume_id, record)`` of the session's current volume, or ``(None, None)``.

        That is the one the session's manifest trains on (trained_volume_name)
        when it is registered, else the last one registered: for
        ``corrections_dir``, or, when it is None, for the corrections dir of
        the last volume registered for any. Every route asks this, so Rank,
        Use this model, Seed, a crop import, the good regions and training
        all mean the volume training reads; load-crops used to take the first
        instead, and imported into one volume while the user painted another.

        The manifest first, not the last registered alone: a resumed session
        had every volume of its source copied and the periodic sync
        registered each one it found in MinIO, so "the last registered" was
        whichever the sync listed last -- the original 4 nm paint, while the
        manifest pointed training at the newest volume, and Rank scored one
        volume, training read another and Use this model carried from a
        third. The manifest is read on every call (one small file), as a
        dashboard restart forgets the registry but not it.
        """
        registered = [
            (volume_id, record)
            for volume_id, record in self._volumes.items()
            if record.get("corrections_dir")
            and (corrections_dir is None or str(record["corrections_dir"]) == str(corrections_dir))
        ]
        if not registered:
            return None, None
        corrections = str(registered[-1][1]["corrections_dir"])
        trained = trained_volume_name(corrections)
        for volume_id, record in reversed(registered):
            if (
                trained
                and str(record["corrections_dir"]) == corrections
                and os.path.basename(str(record.get("zarr_path", "")).rstrip("/")) == trained
            ):
                return volume_id, record
        return registered[-1]

    def retire_volume(self, volume_id: str) -> Optional[str]:
        """Take a volume out of the session for good, keeping it on disk.

        Its record leaves the registry, so nothing syncs it from MinIO or
        ranks, seeds or trains it, and its zarr moves to
        ``<corrections>/superseded/<id>.zarr`` (SUPERSEDED_DIRNAME), where
        listing and resuming sessions and the annotated regions do not look.
        Never deleted: it holds what was painted on it. Returns where it
        went, or None when it could not be moved (logged): it is out of the
        registry all the same, and the manifest names the volume that
        replaced it, so the session's volume is that one either way.

        Use this model retires the volume it carried the paint from: kept
        beside the new one, every volume a session ever had was listed,
        resumed, drawn and synced, four copies of one paint in one session.
        """
        record = self._volumes.pop(volume_id, None)
        zarr_path = str((record or {}).get("zarr_path") or "").rstrip("/")
        corrections_dir = (record or {}).get("corrections_dir")
        if not zarr_path or not corrections_dir or not os.path.isdir(zarr_path):
            return None
        target = os.path.join(str(corrections_dir), SUPERSEDED_DIRNAME, os.path.basename(zarr_path))
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):
                raise FileExistsError(f"{target} exists already")
            # One rename within the corrections dir: the volume is wholly at
            # one place or the other, never half copied.
            os.rename(zarr_path, target)
        except OSError as e:
            logger.warning(f"Could not move the retired volume {zarr_path} to {target}: {e}; "
                           "it stays where it is, out of the session.")
            return None
        logger.info(f"Retired {volume_id}: moved to {target}")
        return target
