"""The launcher's settings and ~/.cellmap_flow/server_config.yaml (jobs.settings).

The file is how the CLIs, the dashboard, blockwise masters and scripts share
the queue, billing and run limit, so its keys and format are pinned here. A
file saved before a key existed must still load, and save every key: when
the settings assigned each key by hand, adding "walltime" to the defaults
gave save() a key no instance had, and every cellmap_flow_yaml run died on
AttributeError at startup.
"""

import pytest
import yaml

from cellmap_flow.jobs import settings
from cellmap_flow.jobs.settings import SERVER_CONFIG_DEFAULTS, LauncherSettings, launcher_settings

WITHOUT_WALLTIME = {k: v for k, v in SERVER_CONFIG_DEFAULTS.items() if k != "walltime"}


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "server_config.yaml"
    monkeypatch.setattr(settings, "SERVER_CONFIG_PATH", str(path))
    return path


@pytest.mark.parametrize("saved, expected, cached", [
    pytest.param(None, {}, False, id="no-file-every-default"),
    pytest.param("", {}, False, id="an-empty-file-every-default"),
    pytest.param(yaml.safe_dump({**WITHOUT_WALLTIME, "queue": "gpu_a100"}), {"queue": "gpu_a100"}, True,
                 id="saved-before-walltime-existed"),
    pytest.param("queue: gpu_l4\nretired_key: 3\n", {"queue": "gpu_l4"}, True, id="an-unknown-key-is-ignored"),
])
def test_the_settings_are_the_saved_file_over_the_defaults(config_file, saved, expected, cached):
    if saved is not None:
        config_file.write_text(saved)
    loaded = LauncherSettings.load()
    assert (loaded.as_dict(), loaded.cached) == ({**SERVER_CONFIG_DEFAULTS, **expected}, cached)


def test_save_writes_every_key_in_the_files_format(config_file):
    config_file.write_text(yaml.safe_dump(WITHOUT_WALLTIME))
    loaded = LauncherSettings.load()
    loaded.charge_group, loaded.walltime = "my_lab", "12:00"
    loaded.save()  # where it crashed
    assert config_file.read_text() == (
        "charge_group: my_lab\ncycle_gpu_queues: true\nnb_cores_master: 4\nnb_cores_worker: 12\n"
        "nb_workers: 14\nqueue: gpu_h100\nwalltime: '12:00'\n"
    )
    assert loaded.cached and LauncherSettings.load().walltime == "12:00"


def test_the_process_loads_its_settings_from_the_file_on_first_use(config_file, monkeypatch):
    config_file.write_text("queue: gpu_a100\n")
    monkeypatch.setattr(settings, "_current", None)
    first = launcher_settings()
    assert (first.queue, launcher_settings()) == ("gpu_a100", first)
    with pytest.raises(AttributeError):
        first.wall_time = "06:30"  # misspelt: refused, not stored beside walltime
