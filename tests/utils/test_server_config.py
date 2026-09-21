"""Every declared server-config key must survive a full round trip.

SERVER_CONFIG_KEYS is derived from SERVER_CONFIG_DEFAULTS, and both the
dashboard routes and Flow.save_server_config() iterate it with a bare
getattr. When Flow.__new__ assigned each attribute by hand instead, adding
"walltime" to the defaults gave save_server_config() a key no instance
carried, and every `cellmap_flow_yaml` run died on AttributeError at
startup. These tests fail on that shape rather than on the specific key.
"""

import importlib

import pytest
import yaml


@pytest.fixture
def globals_module(tmp_path, monkeypatch):
    """A freshly imported globals with its config cache pointed at tmp_path.

    Flow is a singleton, so the tests must not touch the developer's real
    ~/.cellmap_flow/server_config.yaml, nor inherit an instance built by an
    earlier test.
    """
    import cellmap_flow.globals as G

    G = importlib.reload(G)
    monkeypatch.setattr(G, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    G.Flow._instance = None
    yield G
    G.Flow._instance = None


def test_every_declared_key_exists_on_a_fresh_instance(globals_module):
    G = globals_module
    flow = G.Flow()
    missing = [k for k in G.SERVER_CONFIG_KEYS if not hasattr(flow, k)]
    assert not missing, f"declared in SERVER_CONFIG_DEFAULTS but never assigned: {missing}"


def test_defaults_apply_when_there_is_no_cache(globals_module):
    G = globals_module
    flow = G.Flow()
    for key, default in G.SERVER_CONFIG_DEFAULTS.items():
        assert getattr(flow, key) == default
    assert flow._server_config_cached is False


def test_a_cache_written_before_a_key_existed_still_loads(globals_module, tmp_path):
    """The case that actually broke: an on-disk cache predating a new key.

    Cached values win, and the key the file has never heard of falls back to
    its default instead of going unset.
    """
    G = globals_module
    new_key = "walltime"
    assert new_key in G.SERVER_CONFIG_DEFAULTS
    old_cache = {k: v for k, v in G.SERVER_CONFIG_DEFAULTS.items() if k != new_key}
    old_cache["queue"] = "gpu_a100"
    (tmp_path / "server_config.yaml").write_text(yaml.safe_dump(old_cache))

    flow = G.Flow()
    assert flow.queue == "gpu_a100"
    assert getattr(flow, new_key) == G.SERVER_CONFIG_DEFAULTS[new_key]
    assert flow._server_config_cached is True

    # The crash site.
    flow.save_server_config()
    saved = yaml.safe_load((tmp_path / "server_config.yaml").read_text())
    assert set(saved) == set(G.SERVER_CONFIG_KEYS)
    assert saved["queue"] == "gpu_a100"


def test_save_round_trips_an_edited_value(globals_module, tmp_path):
    G = globals_module
    flow = G.Flow()
    flow.walltime = "06:30"
    flow.save_server_config()

    G.Flow._instance = None
    assert G.Flow().walltime == "06:30"
