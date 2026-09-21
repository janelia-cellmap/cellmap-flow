"""Get a model's geometry without building the model, when possible.

Sizing an annotation volume needs five numbers: ``read_shape``,
``write_shape``, the two voxel sizes, and the output channel count. Reading
them from ``ModelConfig.config`` costs a full model load -- for a script model
that executes the config file, which downloads weights, loads adapters, and
runs a validation forward pass. One measured session spent 43s of a 47s
"create annotation volume" there, on the dashboard's CPU, for numbers the
inference server already had.

Three sources, cheapest first:

1. a running inference server, which already holds the model;
2. this cache, keyed so that it invalidates itself when the model changes;
3. building the model, which then populates the cache.

Reading the declarations statically is deliberately *not* among them. The
script contract allows geometry to depend on the model -- for instance
``example/dacapo_run_retrieve.py`` has ``output_voxel_size =
Coordinate(model.scale(voxel_size))`` -- so parsing literals would silently
produce wrong numbers for those scripts rather than failing.
"""

import json
import logging
import os
import tempfile
import threading
from types import SimpleNamespace

from cellmap_flow.utils.server_info import (
    GEOMETRY_FIELDS,
    OPTIONAL_FIELDS,
    model_geometry_config,
)

logger = logging.getLogger(__name__)

CACHE_PATH = os.path.expanduser("~/.cellmap_flow/model_geometry_cache.json")

# Enough for any realistic number of models a person cycles through, small
# enough that the file stays trivial to read and rewrite.
MAX_ENTRIES = 200

_lock = threading.Lock()


def cache_key(model_config):
    """A key that changes when the model does, or None if we cannot make one.

    Only keys that invalidate themselves are allowed. A script is keyed by
    path and mtime, so editing it misses the cache. A HuggingFace repo is
    keyed by an explicit revision, which is immutable; without one the
    revision is whatever ``main`` points at today, so those are not cached
    rather than risk serving geometry for a model that has since moved.
    """
    if model_config is None:
        return None

    script_path = getattr(model_config, "script_path", None)
    if script_path:
        try:
            mtime = os.stat(script_path).st_mtime_ns
        except OSError:
            return None
        return f"script:{os.path.abspath(script_path)}:{mtime}"

    repo = getattr(model_config, "repo", None)
    revision = getattr(model_config, "revision", None)
    if repo and revision:
        return f"hf:{repo}:{revision}"

    return None


def _read_cache():
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(data):
    """Replace the cache file atomically.

    Several dashboards share this path, so a half-written file would be read
    by another process. Write a sibling temp file and rename over the target.
    """
    directory = os.path.dirname(CACHE_PATH)
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".geometry-", suffix=".json")
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
        logger.debug(f"Could not write the model geometry cache: {e}")


def load_cached_geometry(model_config):
    """Geometry remembered for this exact model, or None."""
    key = cache_key(model_config)
    if not key:
        return None
    entry = _read_cache().get(key)
    if not isinstance(entry, dict):
        return None
    if any(entry.get(f) is None for f in GEOMETRY_FIELDS):
        return None
    logger.info(f"Model geometry from cache for {key.split(':')[0]} model")
    fields = {f: entry[f] for f in GEOMETRY_FIELDS}
    for f in OPTIONAL_FIELDS:
        if entry.get(f) is not None:
            fields[f] = entry[f]
    return SimpleNamespace(**fields)


def store_geometry(model_config, config):
    """Remember the geometry of a model we just paid to build."""
    key = cache_key(model_config)
    if not key or config is None:
        return
    try:
        entry = {}
        for field in GEOMETRY_FIELDS:
            value = getattr(config, field)
            # Coordinate and ndarray are both common here and neither is
            # JSON-serializable; output_channels is a plain int.
            entry[field] = (
                [int(v) for v in value] if hasattr(value, "__iter__") else int(value)
            )
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"Not caching geometry for {key}: {e}")
        return

    # Channel names are optional but worth keeping: the finetune tab reads
    # them to tell an affinity model from a binary one, and a cache hit that
    # dropped them would silently downgrade that to "binary".
    channels = (
        getattr(config, "channels", None)
        or getattr(config, "channels_names", None)
        or getattr(config, "classes", None)
    )
    if channels:
        try:
            entry["channels"] = [str(c) for c in channels]
        except TypeError:
            pass

    with _lock:
        data = _read_cache()
        data[key] = entry
        if len(data) > MAX_ENTRIES:
            for stale in list(data)[: len(data) - MAX_ENTRIES]:
                data.pop(stale, None)
        _write_cache(data)


def resolve_model_geometry(model_name, model_config):
    """The five geometry fields, by the cheapest route that can supply them.

    Returns an object exposing ``read_shape``, ``write_shape``,
    ``input_voxel_size``, ``output_voxel_size`` and ``output_channels`` --
    either a stand-in or, on the build path, the real ``ModelConfig.config``.
    """
    geometry = model_geometry_config(model_name)
    if geometry is not None:
        return geometry

    geometry = load_cached_geometry(model_config)
    if geometry is not None:
        return geometry

    if model_config is None:
        return None

    config = model_config.config
    store_geometry(model_config, config)
    return config
