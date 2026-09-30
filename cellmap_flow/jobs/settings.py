"""The launcher's settings: which queue, who is billed, how long a job may run.

One ``LauncherSettings`` per process, from ``launcher_settings()``. It holds
one attribute per key of ``SERVER_CONFIG_DEFAULTS``, the one place a key is
added, and ``cached``: whether the values came from, or were saved to, the
file.

The settings cross processes through ~/.cellmap_flow/server_config.yaml,
whose keys and format are what this module reads and writes:

- ``launcher_settings()`` loads the file over the defaults the first time any
  code in the process asks, so a script, a blockwise master and the dashboard
  all get the saved values without asking for them.
- The CLIs apply their flags (``-q``, ``-P``) and a YAML's values to it and
  ``save()`` it; the dashboard's /api/server-config does the same.
- ``start_hosts``, the finetune submit and blockwise read it for their
  defaults. ``config.yaml.load_config`` reads the file itself, not this.

Nothing here is read at import: the file is opened on the first call, and
``SERVER_CONFIG_PATH`` is looked up when the file is read or written, so a
test can point it elsewhere.
"""

import os
import threading
from typing import Any, Dict, Optional

import yaml

from cellmap_flow.jobs.site import current_site

SERVER_CONFIG_PATH = os.path.expanduser("~/.cellmap_flow/server_config.yaml")

# The dashboard's job settings until the user saves their own. The queue,
# run limit and core counts are the site's (jobs/site.py says why each is
# what it is); no charge group is assumed, so a YAML without one is refused
# rather than billed to someone.
_SITE = current_site()
SERVER_CONFIG_DEFAULTS = {
    "queue": _SITE.default_queue,
    "charge_group": "",
    # LSF's own default on the GPU queues is 120 minutes, which killed
    # inference servers two hours into a session. See default_walltime in
    # jobs/site.py for why this matches the Fileglancer app's own 8 hours.
    "walltime": _SITE.default_walltime,
    # Try other GPU queues when the requested one is busy or closed. On by
    # default because a job that starts elsewhere beats one that never
    # starts; turn it off when the queue itself matters (a benchmark pinned
    # to one GPU model, a charge group valid on only one queue).
    "cycle_gpu_queues": True,
    "nb_cores_master": _SITE.server_cpus,
    "nb_cores_worker": _SITE.worker_cpus,
    "nb_workers": 14,
}

SERVER_CONFIG_KEYS = list(SERVER_CONFIG_DEFAULTS.keys())


def load_server_config_cache() -> Optional[Dict[str, Any]]:
    """Load server config from cache file. Returns None if not found."""
    if os.path.exists(SERVER_CONFIG_PATH):
        with open(SERVER_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return None


def save_server_config_cache(config: Dict[str, Any]) -> None:
    """Save server config to cache file."""
    os.makedirs(os.path.dirname(SERVER_CONFIG_PATH), exist_ok=True)
    with open(SERVER_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class LauncherSettings:
    """The saved settings, one attribute per key, and ``cached``.

    The slots come from the defaults, so a misspelt setting raises instead
    of being stored beside the real one, and a key added to the defaults is
    loaded and saved without being named anywhere else. (Assigning each key
    by hand is how adding "walltime" once gave save() a key no instance had,
    and every cellmap_flow_yaml run died at startup.)
    """

    __slots__ = (*SERVER_CONFIG_KEYS, "cached")

    def __init__(self, values: Optional[Dict[str, Any]] = None, cached: bool = False):
        """``values`` over the defaults, key by key; other keys are ignored."""
        values = values or {}
        for key, default in SERVER_CONFIG_DEFAULTS.items():
            setattr(self, key, values.get(key, default))
        self.cached = cached

    @classmethod
    def load(cls) -> "LauncherSettings":
        """The saved file over the defaults. A key the file lacks (it was
        saved before the key existed) gets its default."""
        saved = load_server_config_cache() or {}
        return cls(saved, cached=bool(saved))

    def as_dict(self) -> Dict[str, Any]:
        """{key: value}, every key, as the file holds them."""
        return {key: getattr(self, key) for key in SERVER_CONFIG_KEYS}

    def save(self) -> None:
        """Write every key to the file, for this and later processes."""
        save_server_config_cache(self.as_dict())
        self.cached = True

    def __repr__(self):
        return f"LauncherSettings({self.as_dict()}, cached={self.cached})"


_current: Optional[LauncherSettings] = None
# yaml_cli starts its servers on a thread pool, and each start_hosts may be
# the first to ask: without the lock two threads could each load a copy and
# one's changes would land on the copy nobody else sees.
_lock = threading.Lock()


def launcher_settings() -> LauncherSettings:
    """This process's settings, loaded from the file on the first call.

    Read it at call time, never into a module global: tests swap in fresh
    settings per test, and a copy taken at import would outlive them.
    """
    global _current
    if _current is None:
        with _lock:
            if _current is None:
                _current = LauncherSettings.load()
    return _current
