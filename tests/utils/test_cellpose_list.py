"""The Models tab's Cellpose panel: Submit starting what is ticked as a
``cellpose`` model in the cellpose4 environment, a blank voxel size refused,
unticked ones stopped, changed ones restarted, and the running ones ticked
again with their settings when the page is reloaded. No job is started:
start_hosts records the commands."""

import json
import re
import shlex
import textwrap

import pytest

from cellmap_flow.dashboard.requests import CellposeSelection
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


class _Job:
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
def submit(dashboard, viewer, monkeypatch, tmp_path):
    """POST /api/models with these Cellpose rows ticked; the commands started are ``submit.commands``."""
    commands = []
    monkeypatch.setattr(launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    monkeypatch.setenv("CELLMAP_FLOW_ENVS_FILE", str(tmp_path / "no-aliases.yaml"))
    process_chain().input_norms, process_chain().postprocess = [], []
    session = get_session()
    session.dataset_path, session.jobs = "/data/raw.zarr", []
    session.models_config = [mc for mc in session.models_config if not isinstance(mc, CellposeModelConfig)]

    def post(*selections):
        return dashboard.post("/api/models", json={"selected_cellpose_models": list(selections)})

    post.commands = commands
    return post


def _served_entry(command):
    argv = shlex.split(command)
    return argv, json.loads(argv[argv.index("--model") + 1])


def _panel(dashboard):
    html = dashboard.get("/").get_data(as_text=True)
    return json.loads(re.search(r'<script type="application/json" id="cellpose-data">(.*?)</script>', html, re.S)
                      .group(1))


@pytest.mark.parametrize("model, output, name", [
    ("cpsam_v2", "flows", "cellpose_sam_v2"),
    ("cpsam_v2", "probability", "cellpose_sam_v2_probability"),
    ("cpsam_v2", "masks", "cellpose_sam_v2_masks"),
    ("cpsam", "flows", "cellpose_sam"),
])
def test_a_job_is_named_by_its_model_and_any_output_but_the_default(model, output, name):
    assert launch.cellpose_job_name(model, output) == name


@pytest.mark.parametrize("typed, sizes", [("64", [64, 64, 64]), ("16, 8,8", [16, 8, 8]), ("", None), (None, None)])
def test_a_typed_voxel_size_is_read_as_z_y_x(typed, sizes):
    assert CellposeSelection(model="cpsam_v2", voxel_size=typed).voxel_size == sizes


def test_submit_serves_a_ticked_model_from_the_cellpose4_environment(submit, tmp_path, monkeypatch):
    manifest = tmp_path / "checkout" / "pixi.toml"
    manifest.parent.mkdir()
    manifest.write_text(textwrap.dedent(MANIFEST))
    monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(manifest))
    monkeypatch.setenv("PIXI_EXE", "/opt/pixi")

    answer = submit({"model": "cpsam_v2", "voxel_size": "64", "output": "flows"})
    assert answer.status_code == 200
    assert answer.get_json()["cellpose_models"][0]["model"] == "cpsam_v2"
    (command,) = submit.commands
    argv, entry = _served_entry(command)
    assert argv[:7] == ["/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest), "-e", "cellpose4"]
    assert entry == {"type": "cellpose", "pretrained_model": "cpsam_v2", "voxel_size": [64, 64, 64],
                     "output": "flows", "name": "cellpose_sam_v2"}
    assert [mc.name for mc in get_session().models_config if isinstance(mc, CellposeModelConfig)] == [
        "cellpose_sam_v2"]


def test_one_models_outputs_run_side_by_side_and_masks_take_the_slice_linking(submit):
    answer = submit({"model": "cpsam_v2", "voxel_size": "64", "output": "flows", "stitch_threshold": 0.5},
                    {"model": "cpsam_v2", "voxel_size": "64", "output": "masks", "stitch_threshold": "0.5"})
    assert answer.status_code == 200
    probability, masks = (_served_entry(c)[1] for c in submit.commands)
    # Read for masks only: the flows' is dropped, as the tab hides it.
    assert (probability["name"], "stitch_threshold" in probability) == ("cellpose_sam_v2", False)
    assert (masks["name"], masks["output"], masks["stitch_threshold"]) == ("cellpose_sam_v2_masks", "masks", 0.5)


@pytest.mark.parametrize("selection, message", [
    ({"model": "cpsam_v2", "voxel_size": ""}, r"Cellpose-SAM v2 \(cpsam_v2\) needs a voxel size: enter one \(nm\)"),
    ({"model": "cpsam_v2", "voxel_size": "64", "output": "masks", "stitch_threshold": 2},
     "stitch_threshold is an IoU, from 0 to 1"),
    ({"model": "cpsam_v2", "voxel_size": "64", "output": "labels"}, "output must be one of"),
    ({"model": "cpdino", "voxel_size": "64"}, "not one of the Models tab's Cellpose models"),
], ids=["blank-voxel-size", "stitch-above-one", "bad-output", "dino"])
def test_a_ticked_model_without_its_settings_refuses_the_whole_submit(submit, selection, message):
    running = _Job("mito")
    get_session().jobs = [running]
    answer = submit(selection)
    assert answer.status_code == 400 and re.search(message, answer.get_json()["error"])
    assert submit.commands == [] and not running.killed


def test_the_same_output_ticked_twice_is_refused(submit):
    row = {"model": "cpsam", "voxel_size": "32", "output": "masks"}
    answer = submit(row, row)
    assert answer.status_code == 400 and "ticked twice" in answer.get_json()["error"] and submit.commands == []


def test_unticking_a_model_stops_it_and_a_running_one_is_not_started_again(submit):
    submit({"model": "cpsam_v2", "voxel_size": "64"})
    job = _Job("cellpose_sam_v2")
    get_session().jobs = [job]
    submit({"model": "cpsam_v2", "voxel_size": "64"})
    assert len(submit.commands) == 1 and not job.killed
    submit()
    assert job.killed and get_session().jobs == []


def test_a_running_model_given_another_voxel_size_is_restarted_with_it(submit):
    """Its name does not say its voxel size: kept running, the change would
    have been silently ignored."""
    submit({"model": "cpsam_v2", "voxel_size": "64"})
    job = _Job("cellpose_sam_v2")
    get_session().jobs = [job]
    assert submit({"model": "cpsam_v2", "voxel_size": "32"}).status_code == 200
    assert job.killed and len(submit.commands) == 2
    assert _served_entry(submit.commands[-1])[1]["voxel_size"] == [32, 32, 32]
    (config,) = [mc for mc in get_session().models_config if isinstance(mc, CellposeModelConfig)]
    assert config.voxel_size == (32, 32, 32)


def test_a_model_of_the_same_name_ticked_in_the_list_does_not_keep_other_settings_running(submit, dashboard):
    """A YAML's "cellpose_sam" serves probability; the panel's cpsam with flows
    has the same name. Both ticked, the YAML's tick kept the probability job
    running and the flows were never served."""
    get_session().models_config.append(CellposeModelConfig(
        pretrained_model="cpsam", output="probability", voxel_size=64, name="cellpose_sam"))
    job = _Job("cellpose_sam")
    get_session().jobs = [job]

    response = dashboard.post("/api/models", json={
        "selected_models": ["cellpose_sam"],
        "selected_cellpose_models": [{"model": "cpsam", "voxel_size": "64", "output": "flows"}],
    })

    assert response.status_code == 200 and job.killed and len(submit.commands) == 1
    assert _served_entry(submit.commands[0])[1]["output"] == "flows"


def test_a_model_on_the_panel_is_not_ticked_in_the_model_list_too(submit, dashboard):
    """Listed twice, unticking it on the panel left its tick in the list, which kept it running."""
    submit({"model": "cpsam_v2", "voxel_size": "64"}, {"model": "cpsam_v2", "voxel_size": "64", "output": "probability"})
    get_session().models_config.append(CellposeModelConfig(pretrained_model="/my/weights", voxel_size=64, name="mine"))
    get_session().jobs = [_Job("cellpose_sam_v2"), _Job("cellpose_sam_v2_probability"), _Job("mine")]
    html = dashboard.get("/").get_data(as_text=True)
    listed = re.findall(r'class="form-check-input model-checkbox"[^>]*value="([^"]+)"', html, re.S)
    assert "mine" in listed  # its own weights: no row on the panel
    assert "cellpose_sam_v2" not in listed and "cellpose_sam_v2_probability" not in listed


def test_the_page_lists_the_models_and_ticks_the_running_ones_with_their_settings(submit, dashboard):
    submit({"model": "cpsam_v2", "voxel_size": "16,8,8", "output": "masks", "stitch_threshold": 0.3},
           {"model": "cpsam_v2", "voxel_size": "64"}, {"model": "cpsam", "voxel_size": "32"})
    get_session().jobs = [_Job("cellpose_sam_v2_masks"), _Job("cellpose_sam_v2")]  # cellpose_sam's has gone
    panel = _panel(dashboard)
    assert [m["model"] for m in panel["models"]] == ["cpsam_v2", "cpsam"]
    assert panel["running"] == [
        {"model": "cpsam_v2", "output": "masks", "voxel_size": [16, 8, 8], "stitch_threshold": 0.3},
        {"model": "cpsam_v2", "output": "flows", "voxel_size": [64, 64, 64], "stitch_threshold": 0.0},
    ]
