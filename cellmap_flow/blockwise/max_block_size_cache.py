"""Remembering a model's probed max block size for a GPU queue.

Mirrors ``models.geometry_cache``'s pattern (a self-invalidating key, an
atomically-rewritten JSON file), but keyed by the model *and* the GPU queue:
a bigger GPU fits a bigger block, so the same model gets a different answer
on ``gpu_h100`` than on ``gpu_l4``. ``models.geometry_cache.cache_key`` only
covers script and HuggingFace models (the only ones it was written for);
this one also covers a checkpoint-file model (Fly) and a folder-based export
(CellMap), since a GPU-block-size probe is exactly as useful for those.
"""

import json
import logging
import os
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

CACHE_PATH = os.path.expanduser("~/.cellmap_flow/max_block_size_cache.json")

# Enough for any realistic number of model+queue combinations a pipeline
# builder session cycles through, small enough the file stays trivial to
# read and rewrite.
MAX_ENTRIES = 200

_lock = threading.Lock()

# Attributes that name a file whose mtime invalidates the key when it
# changes, checked in this order (the first a model config has wins).
_PATH_ATTRS = ("script_path", "checkpoint_path", "folder_path")


def _path_key(model_config) -> str:
    for attr in _PATH_ATTRS:
        path = getattr(model_config, attr, None)
        if path:
            try:
                mtime = os.stat(path).st_mtime_ns
            except OSError:
                return None
            return f"{attr}:{os.path.abspath(path)}:{mtime}"
    return None


def cache_key(model_config, queue: str) -> str:
    """A key that changes when the model or the queue does, or None if we
    cannot make one (no identifying path, and no repo+revision)."""
    if model_config is None or not queue:
        return None

    key = _path_key(model_config)
    if key is None:
        repo = getattr(model_config, "repo", None)
        revision = getattr(model_config, "revision", None)
        if repo and revision:
            key = f"hf:{repo}:{revision}"
    if key is None:
        return None
    return f"{key}@{queue}"


def _read_cache() -> dict:
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(data: dict) -> None:
    """Replace the cache file atomically, as geometry_cache does: several
    dashboards may share this path, so a half-written file must never be
    read by another process."""
    directory = os.path.dirname(CACHE_PATH)
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".max-block-size-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, CACHE_PATH)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.debug(f"Could not write the max block size cache: {e}")


def load_cached(model_config, queue: str) -> dict:
    """The cached probe result for this exact model on this queue, as a
    plain dict (``MaxBlockSizeResult.from_dict`` reads it), or None."""
    key = cache_key(model_config, queue)
    if not key:
        return None
    entry = _read_cache().get(key)
    if entry is None:
        return None
    result = {k: v for k, v in entry.items() if k != "stored_at"}
    # A result probed before a bigger tile's answer was held against the
    # native tile's (its fitting candidates carry no "agreement") may call a
    # tile fitting that the GPU answers wrong at: probe again.
    if any(c.get("fits") and c.get("multiple", 1) > 1 and "agreement" not in c for c in result.get("candidates", [])):
        return None
    return result


def stored_at(model_config, queue: str):
    """When the cached result for this model and queue was stored (epoch
    seconds), or None: no entry, or one from before entries said."""
    key = cache_key(model_config, queue)
    if not key:
        return None
    return (_read_cache().get(key) or {}).get("stored_at")


def store(model_config, queue: str, result) -> None:
    """Remember a probe ``result`` (a MaxBlockSizeResult) we just paid for,
    and when: a dashboard polling a probe job takes an entry stored since
    it submitted the job as that job's answer (routes.blockwise)."""
    key = cache_key(model_config, queue)
    if not key:
        return
    with _lock:
        data = _read_cache()
        data[key] = {**result.to_dict(), "stored_at": time.time()}
        if len(data) > MAX_ENTRIES:
            for stale in list(data)[: len(data) - MAX_ENTRIES]:
                data.pop(stale, None)
        _write_cache(data)
