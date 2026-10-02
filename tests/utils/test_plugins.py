"""The plugins directory is created by registering a plugin, and nothing else
(every LSF job imports the package, and with it the plugin loader)."""

import pytest

from cellmap_flow import plugins


def test_registering_creates_the_directory_and_the_plugin_round_trips(tmp_path, monkeypatch):
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(plugins, "PLUGINS_DIR", plugins_dir)
    assert plugins.list_plugins() == []
    with pytest.raises(FileNotFoundError):
        plugins.unregister_plugin("missing")
    assert not plugins_dir.exists(), "listing and unregistering create nothing"

    (tmp_path / "my_norm.py").write_text("X = 1\n")
    dest = plugins.register_plugin(str(tmp_path / "my_norm.py"))
    assert dest == plugins_dir / "my_norm.py" and plugins.list_plugins() == [dest]
    plugins.unregister_plugin("my_norm")
    assert plugins.list_plugins() == []
