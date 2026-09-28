"""The plugins directory is created by registering a plugin, and nothing else."""

import pytest

from cellmap_flow.utils import plugin_manager


@pytest.fixture
def plugins_dir(tmp_path, monkeypatch):
    target = tmp_path / "plugins"
    monkeypatch.setattr(plugin_manager, "PLUGINS_DIR", target)
    return target


def test_listing_and_unregistering_do_not_create_it(plugins_dir):
    assert plugin_manager.list_plugins() == []
    with pytest.raises(FileNotFoundError):
        plugin_manager.unregister_plugin("missing")
    assert not plugins_dir.exists()


def test_registering_creates_it_and_the_plugin_round_trips(plugins_dir, tmp_path):
    source = tmp_path / "my_norm.py"
    source.write_text("X = 1\n")

    dest = plugin_manager.register_plugin(str(source))

    assert dest == plugins_dir / "my_norm.py"
    assert plugin_manager.list_plugins() == [dest]
    plugin_manager.unregister_plugin("my_norm")
    assert plugin_manager.list_plugins() == []
