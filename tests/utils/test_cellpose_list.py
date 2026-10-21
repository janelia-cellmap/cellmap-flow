"""The Models tab's Cellpose panel: Submit starting what is ticked as a
``cellpose`` model in the cellpose4 environment, a blank voxel size refused,
unticked ones stopped, changed ones restarted, and the running ones ticked
again with their settings when the page is reloaded. No job is started:
start_hosts records the commands.

``models_tab`` and its helpers serve test_bioimage_zoo's Submit tests too."""

import json
import re
import shlex
import textwrap

import pytest

from cellmap_flow.dashboard.services import launch
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models.models_config import CellposeModelConfig
from cellmap_flow.process_chain import process_chain

MANIFEST = """
[pypi-dependencies]
cellmap-flow = { path = ".", editable = true }

[feature.cellpose4.pypi-dependencies]
cellpose = "==4.0.6"

[environments]
cellpose4 = { features = ["cellpose4"], no-default-feature = true }
"""


class Job:
    """A started job, as the session lists it."""

    def __init__(self, name):
        self.model_name, self.host, self.killed = name, f"http://{name}:1", False

    def kill(self):
        self.killed = True


class _InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def models_tab(dashboard, viewer, monkeypatch, tmp_path):
    """``models_tab(**body)`` POSTs /api/models; the commands started are ``models_tab.commands``."""
    commands = []
    monkeypatch.setattr(launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    monkeypatch.setenv("CELLMAP_FLOW_ENVS_FILE", str(tmp_path / "no-aliases.yaml"))
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.dataset_path, session.jobs = "/data/raw.zarr", []

    def post(**body):
        return dashboard.post("/api/models", json=body)

    post.commands = commands
    return post


def pixi_manifest(tmp_path, monkeypatch, text):
    """A checkout's pixi.toml holding ``text``, and pixi at /opt/pixi."""
    manifest = tmp_path / "checkout" / "pixi.toml"
    manifest.parent.mkdir()
    manifest.write_text(textwrap.dedent(text))
    monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(manifest))
    monkeypatch.setenv("PIXI_EXE", "/opt/pixi")
    return manifest


def served_entry(command):
    """(the server command's argv, the model entry it serves)."""
    argv = shlex.split(command)
    return argv, json.loads(argv[argv.index("--model") + 1])


@pytest.fixture
def submit(models_tab):
    """POST /api/models with these Cellpose rows ticked; the commands started are ``submit.commands``."""
    session = get_session()
    session.models_config = [mc for mc in session.models_config if not isinstance(mc, CellposeModelConfig)]

    def post(*selections):
        return models_tab(selected_cellpose_models=list(selections))

    post.commands = models_tab.commands
    return post


def _panel(dashboard):
    html = dashboard.get("/").get_data(as_text=True)
    return json.loads(re.search(r'<script type="application/json" id="cellpose-data">(.*?)</script>', html, re.S)
                      .group(1))


def test_submit_serves_a_ticked_model_from_the_cellpose4_environment(submit, tmp_path, monkeypatch):
    manifest = pixi_manifest(tmp_path, monkeypatch, MANIFEST)

    answer = submit({"model": "cpsam_v2", "voxel_size": "64", "output": "flows"})
    assert answer.status_code == 200
    assert answer.get_json()["cellpose_models"][0]["model"] == "cpsam_v2"
    (command,) = submit.commands
    argv, entry = served_entry(command)
    assert argv[:7] == ["/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest), "-e", "cellpose4"]
    assert entry == {"type": "cellpose", "pretrained_model": "cpsam_v2", "voxel_size": [64, 64, 64],
                     "output": "flows", "name": "cellpose_sam_v2"}
    assert [mc.name for mc in get_session().models_config if isinstance(mc, CellposeModelConfig)] == [
        "cellpose_sam_v2"]


def test_a_3d_row_runs_its_own_job_in_3d_and_is_ticked_again_as_3d(submit, dashboard):
    """Link slices means nothing in 3D, so it is left out rather than refused."""
    submit({"model": "cpsam_v2", "voxel_size": "8", "output": "masks", "do_3d": True, "stitch_threshold": 0.3},
           {"model": "cpsam_v2", "voxel_size": "8", "output": "masks"})
    (_, entry3d), (_, entry2d) = (served_entry(c) for c in submit.commands)
    assert entry3d["do_3d"] is True and "stitch_threshold" not in entry3d and "do_3d" not in entry2d
    assert (entry3d["name"], entry2d["name"]) == ("cellpose_sam_v2_masks_3d", "cellpose_sam_v2_masks")
    get_session().jobs = [Job("cellpose_sam_v2_masks_3d"), Job("cellpose_sam_v2_masks")]
    running = _panel(dashboard)["running"]
    assert [(r["output"], r["do_3d"]) for r in running] == [("masks", True), ("masks", False)]


@pytest.mark.parametrize("selection, keyword", [
    ({"model": "cpsam_v2", "voxel_size": ""}, "voxel size"),
    ({"model": "cpsam_v2", "voxel_size": "64", "output": "masks", "stitch_threshold": 2}, "stitch_threshold"),
    ({"model": "cpsam_v2", "voxel_size": "64", "output": "labels"}, "output"),
    ({"model": "cpdino", "voxel_size": "64"}, "cpdino"),
], ids=["blank-voxel-size", "stitch-above-one", "bad-output", "dino"])
def test_a_ticked_model_without_its_settings_refuses_the_whole_submit(submit, selection, keyword):
    running = Job("mito")
    get_session().jobs = [running]
    answer = submit(selection)
    assert answer.status_code == 400 and keyword in answer.get_json()["error"]
    assert submit.commands == [] and not running.killed


def test_unticking_a_model_stops_it_and_a_running_one_is_not_started_again(submit):
    submit({"model": "cpsam_v2", "voxel_size": "64"})
    job = Job("cellpose_sam_v2")
    get_session().jobs = [job]
    submit({"model": "cpsam_v2", "voxel_size": "64"})
    assert len(submit.commands) == 1 and not job.killed
    submit()
    assert job.killed and get_session().jobs == []


def test_a_running_model_given_another_voxel_size_is_restarted_with_it(submit):
    """Its name does not say its voxel size: kept running, the change would
    have been silently ignored."""
    submit({"model": "cpsam_v2", "voxel_size": "64"})
    job = Job("cellpose_sam_v2")
    get_session().jobs = [job]
    assert submit({"model": "cpsam_v2", "voxel_size": "32"}).status_code == 200
    assert job.killed and len(submit.commands) == 2
    assert served_entry(submit.commands[-1])[1]["voxel_size"] == [32, 32, 32]
    (config,) = [mc for mc in get_session().models_config if isinstance(mc, CellposeModelConfig)]
    assert config.voxel_size == (32, 32, 32)


def test_the_page_lists_the_models_and_ticks_the_running_ones_with_their_settings(submit, dashboard):
    submit({"model": "cpsam_v2", "voxel_size": "16,8,8", "output": "masks", "stitch_threshold": 0.3},
           {"model": "cpsam_v2", "voxel_size": "64"}, {"model": "cpsam", "voxel_size": "32"})
    get_session().jobs = [Job("cellpose_sam_v2_masks"), Job("cellpose_sam_v2")]  # cellpose_sam's has gone
    panel = _panel(dashboard)
    assert [m["model"] for m in panel["models"]] == ["cpsam_v2", "cpsam"]
    assert panel["running"] == [
        {"model": "cpsam_v2", "output": "masks", "do_3d": False, "voxel_size": [16, 8, 8], "stitch_threshold": 0.3},
        {"model": "cpsam_v2", "output": "flows", "do_3d": False, "voxel_size": [64, 64, 64], "stitch_threshold": 0.0},
    ]
