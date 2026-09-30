"""The plugins directory is created by registering a plugin, and nothing else
(every LSF job imports the package, and with it the plugin loader)."""

import pytest

from cellmap_flow.utils import plugin_manager


def test_registering_creates_the_directory_and_the_plugin_round_trips(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    monkeypatch.setattr(plugin_manager, "PLUGINS_DIR", plugins)
    assert plugin_manager.list_plugins() == []
    with pytest.raises(FileNotFoundError):
        plugin_manager.unregister_plugin("missing")
    assert not plugins.exists(), "listing and unregistering create nothing"

    (tmp_path / "my_norm.py").write_text("X = 1\n")
    dest = plugin_manager.register_plugin(str(tmp_path / "my_norm.py"))
    assert dest == plugins / "my_norm.py" and plugin_manager.list_plugins() == [dest]
    plugin_manager.unregister_plugin("my_norm")
    assert plugin_manager.list_plugins() == []
