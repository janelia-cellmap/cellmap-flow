"""The dashboard's blockwise routes, which the pipeline builder calls in turn:
validate, generate (which writes the task YAML the blockwise CLI runs),
precheck and submit.

Each route answers 200 whatever happens: {"valid": ...} from validate,
{"success": ...} from the others, with an "error" the builder shows when it
is false. LSF here is conftest's ``fake_lsf``."""

import copy
import os
import re

import pytest
import yaml

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.process_chain import process_chain

SETTINGS = {"charge_group": "grp", "queue": "gpu_h100", "nb_workers": 2, "nb_cores_worker": 12,
            "nb_cores_master": 4, "tmp_dir": "/scratch/progress"}
PIPELINE = {
    "inputs": [{"params": {"dataset_path": "/data/raw.zarr/raw"}}],
    "outputs": [{"params": {"dataset_path": "/out/pred"}}],
    "models": [{"name": "m", "params": {"type": "script", "script_path": "/s.py"}}],
    "blockwise_config": [{"params": SETTINGS}],
}
ACCEPTED = "Your request was accepted.\nJob <5150> is submitted to default queue <local>.\n"
# A fly model as its entry says it all (nothing is read beside /c.ts): the
# one type whose tile the entry sets, so the one a block_size can resize.
FLY = {"type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito"], "input_voxel_size": [8, 8, 8],
       "output_voxel_size": [8, 8, 8], "input_size": [178, 178, 178], "output_size": [56, 56, 56]}
# The same network at a 336 tile (its context, 61 a side, kept), as a probe
# pick sets it in the builder.
FLY336 = {**FLY, "input_size": [458, 458, 458], "output_size": [336, 336, 336]}


def pipeline(**changes):
    """PIPELINE with these lists (or model_mode) in place of its own."""
    return {**copy.deepcopy(PIPELINE), **copy.deepcopy(changes)}


@pytest.fixture
def tasks(tmp_path):
    session = get_session()
    session.blockwise_tasks_dir, session.walltime = str(tmp_path / "tasks"), "12:00"
    return tmp_path / "tasks"


BOX = {"offset": [0, 0, 0], "shape": [64, 64, 64]}


def test_generate_writes_the_task_yaml_the_blockwise_cli_runs(dashboard, tasks, monkeypatch):
    """Every part of the builder's pipeline reaches the task YAML. json_data
    was a dict keyed by step name, so a chain using a step twice kept only
    the last one, and blockwise computed something else than the dashboard
    showed; a step without a name is left out. The builder's text fields send
    a list as it was typed. A task without resample resamples, so the
    dashboard's Resample unticked is written. The typed job name names the
    task (and so its master's LSF job) as a file name may be named."""
    monkeypatch.setattr(get_session(), "resample", False)
    sent = pipeline(
        inputs=[{"params": {"dataset_path": "/data/raw.zarr/raw", "bounding_boxes": [BOX]}}],
        outputs=[{"params": {"dataset_path": "/out/pred", "output_channels": ["mito", "er"]}}],
        models=[PIPELINE["models"][0], {"name": "t", "params": {"script_path": "/t.py", "channels": "[mito, er]"}}],
        model_mode="OR",
        normalizers=[{"name": "LambdaNormalizer", "params": {"expression": "x*2"}},
                     {"name": "LambdaNormalizer", "params": {"expression": "x-1"}},
                     {"params": {"expression": "x"}}],
        postprocessors=[{"name": "ThresholdPostprocessor", "params": {"threshold": 0.5}}],
    )
    answer = dashboard.post("/api/blockwise/generate", json={"pipeline": sent, "job_name": "  a/b\\c:d  "}).get_json()
    assert answer["success"], answer
    assert re.fullmatch(r"a_b_c_d_\d{8}_\d{6}", answer["task_name"])

    (path,) = answer["task_paths"]
    assert os.listdir(tasks) == [os.path.basename(path)] == [f"{answer['task_name']}.yaml"]
    task = yaml.safe_load(open(path))
    assert task == answer["task_config"]
    assert (task["data_path"], task["output_path"], task["task_name"]) == (
        "/data/raw.zarr/raw", "/out/pred.zarr", answer["task_name"])
    assert (task["workers"], task["walltime"], task["model_mode"], task["resample"]) == (2, "12:00", "OR", False)
    assert [m["name"] for m in task["models"]] == ["m", "t"] and task["models"][1]["channels"] == ["mito", "er"]
    assert task["bounding_boxes"] == [BOX] and task["output_channels"] == ["mito", "er"]
    assert [s.get("expression") for s in task["json_data"]["input_norm"]] == ["x*2", "x-1"]
    assert task["json_data"]["postprocess"] == [{"name": "ThresholdPostprocessor", "threshold": 0.5}]


@pytest.mark.parametrize("sent, expected", [
    ("[84, 84, 84]", [84, 84, 84]),
    ([84, 84, 84], [84, 84, 84]),
    ("", None),
    (None, None),
])
def test_generate_writes_block_size_when_given(dashboard, tasks, sent, expected):
    """block_size, the output's chunk shape, is optional (blockwise_processor.py
    chunks by the model's output tile without one); given, it reaches the
    task YAML as a list whether the builder's text field sent it bracketed
    or as a real array, and the model's tile is left as it is: a 336 tile
    written in 84 chunks, for readers that want them small."""
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(
            models=[{"name": "f", "params": FLY336}],
            blockwise_config=[{"params": {**SETTINGS, "block_size": sent}}],
        ), "job_name": "j"},
    ).get_json()
    assert answer["success"], answer
    assert answer["task_config"].get("block_size") == expected
    (entry,) = answer["task_config"]["models"]
    assert (entry["input_size"], entry["output_size"]) == ([458, 458, 458], [336, 336, 336])


def test_a_block_size_the_tile_is_not_a_multiple_of_is_refused_by_validate_and_generate(dashboard, tasks):
    """A block is one model tile, laid on the output's chunk grid: a tile
    that is not a whole number of chunks has two blocks writing one chunk
    at once. (Generate once rewrote the model's tile to the block size
    instead, which corrupted a 336 tile meant to be written in 84s.)"""
    for block in ([100, 100, 100], [392, 392, 392]):
        body = {"pipeline": pipeline(
            models=[{"name": "f", "params": FLY336}],
            blockwise_config=[{"params": {**SETTINGS, "block_size": block}}],
        ), "job_name": "j"}
        validated = dashboard.post("/api/blockwise/validate", json=body).get_json()
        generated = dashboard.post("/api/blockwise/generate", json=body).get_json()
        assert validated["valid"] is False and generated["success"] is False, block
        for error in (validated["error"], generated["error"]):
            assert f"Block Size {block} must divide f's output_size [336, 336, 336]" in error
    assert dashboard.post("/api/blockwise/validate", json={"pipeline": pipeline(
        models=[{"name": "f", "params": FLY336}],
        blockwise_config=[{"params": {**SETTINGS, "block_size": [336, 336, 336]}}],
    )}).get_json()["valid"] is True


def test_a_model_that_does_not_say_its_tile_is_left_to_the_run_to_check(dashboard, tasks):
    """A script's tile is read off the script, which the dashboard does not
    run: the block size is written, and CellMapFlowBlockwiseProcessor
    checks it against the tile when the task starts."""
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(blockwise_config=[{"params": {**SETTINGS, "block_size": [84, 84, 84]}}]),
              "job_name": "j"},
    ).get_json()
    assert answer["success"], answer
    assert answer["task_config"]["block_size"] == [84, 84, 84]


@pytest.mark.parametrize("sent, expected", [
    ("uint16", "uint16"),
    ("float32", "float32"),
    ("", None),
    (None, None),
])
def test_generate_writes_the_output_nodes_dtype(dashboard, tasks, sent, expected):
    """Blank leaves the dtype to the task (the chain's last declared, else
    the model's); a pick is written as the numpy name."""
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(outputs=[{"params": {"dataset_path": "/out/pred", "dtype": sent}}]),
              "job_name": "j"},
    ).get_json()
    assert answer["success"], answer
    assert answer["task_config"].get("dtype") == expected


def test_generate_refuses_a_dtype_numpy_does_not_know(dashboard, tasks):
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(outputs=[{"params": {"dataset_path": "/out/pred", "dtype": "bytes8"}}]),
              "job_name": "j"},
    ).get_json()
    assert answer["success"] is False
    assert "bytes8" in answer["error"] and "numpy dtype" in answer["error"]


def test_generate_takes_the_nodes_walltime_and_tasks_dir(dashboard, tasks, tmp_path):
    """The node's Walltime is the task's -W, over the dashboard's; its Tasks
    Directory is where the YAML goes (it was shown but never read, so every
    task landed in the dashboard's directory)."""
    mine = tmp_path / "mine" / "tasks"
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(blockwise_config=[{"params": {
            **SETTINGS, "walltime": "02:30", "blockwise_tasks_dir": str(mine)}}]), "job_name": "j"},
    ).get_json()
    assert answer["success"], answer
    assert answer["task_config"]["walltime"] == "02:30"
    (path,) = answer["task_paths"]
    assert os.path.dirname(path) == str(mine) and os.path.isfile(path)
    assert not (tasks / os.path.basename(path)).exists()


def test_generate_without_them_keeps_the_dashboards(dashboard, tasks):
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(blockwise_config=[{"params": {**SETTINGS, "walltime": "", "blockwise_tasks_dir": ""}}]),
              "job_name": "j"},
    ).get_json()
    assert answer["success"], answer
    assert answer["task_config"]["walltime"] == "12:00"  # the tasks fixture's session walltime
    assert os.path.dirname(answer["task_paths"][0]) == str(tasks)


@pytest.mark.parametrize("walltime", ["8h", "08:60", "eight", "8:0"])
def test_generate_refuses_a_walltime_bsub_would_not(dashboard, tasks, walltime):
    answer = dashboard.post(
        "/api/blockwise/generate",
        json={"pipeline": pipeline(blockwise_config=[{"params": {**SETTINGS, "walltime": walltime}}]), "job_name": "j"},
    ).get_json()
    assert answer["success"] is False
    assert "Walltime" in answer["error"] and "HH:MM" in answer["error"]


def test_submit_gives_the_master_the_nodes_walltime_and_logs_in_its_tasks_dir(dashboard, tasks, fake_lsf, tmp_path):
    fake_lsf.answers["bsub"] = [ACCEPTED]
    mine = tmp_path / "mine"
    answer = dashboard.post(
        "/api/blockwise/submit",
        json={"pipeline": pipeline(blockwise_config=[{"params": {
            **SETTINGS, "walltime": "02:30", "blockwise_tasks_dir": str(mine)}}]), "job_name": "j"},
    ).get_json()
    assert answer["success"], answer
    (bsub_argv,) = fake_lsf.commands("bsub")
    assert bsub_argv[bsub_argv.index("-W") + 1] == "02:30"
    assert any(arg.startswith(str(mine)) for arg in bsub_argv), bsub_argv


@pytest.mark.parametrize("steps, expected", [
    pytest.param([], {"dtype": None, "source": "model"}, id="nothing-but-the-model"),
    pytest.param([{"name": "ThresholdPostprocessor", "params": {"threshold": 0.5}}],
                 {"dtype": "uint8", "source": "ThresholdPostprocessor"}, id="one-step"),
    # The last step that declares one decides, as blockwise_processor takes it.
    pytest.param([{"name": "ThresholdPostprocessor", "params": {"threshold": 0.5}},
                  {"name": "SigmoidPostprocessor", "params": {}}],
                 {"dtype": "float32", "source": "SigmoidPostprocessor"}, id="the-last-that-declares"),
])
def test_output_dtype_is_what_the_task_would_write(dashboard, steps, expected):
    answer = dashboard.post(
        "/api/blockwise/output-dtype",
        json={"pipeline": {"models": PIPELINE["models"], "postprocessors": steps}},
    ).get_json()
    assert answer == {"success": True, **expected}


@pytest.mark.parametrize("route, sent, field", [
    pytest.param("validate", {"json": {"pipeline": pipeline(models=[])}}, "models", id="validate/no-models"),
    # A body the builder would not send: what is wrong, and where.
    pytest.param("validate", {"data": "pipeline", "content_type": "text/plain"}, "JSON", id="validate/not-json"),
    pytest.param("validate", {"json": {"pipeline": pipeline(inputs=["/data/raw.zarr/raw"])}}, "pipeline.inputs.0",
                 id="validate/a-node-not-an-object"),
    # A number was opened as a file descriptor: [1] closed the dashboard's stdout.
    pytest.param("precheck", {"json": {"yaml_paths": [None]}}, "yaml_paths.0", id="precheck/a-path-not-text"),
    # It was found missing only here, after generate and precheck had passed.
    pytest.param("submit", {"json": {"pipeline": pipeline(blockwise_config=[{"params": {
        k: v for k, v in SETTINGS.items() if k != "nb_cores_master"}}])}}, "nb_cores_master",
                 id="submit/no-master-cores"),
    # Refused as precheck refuses them; submit generated a task anew and ran that.
    pytest.param("submit", {"json": {"pipeline": PIPELINE, "yaml_paths": "/tasks/t.yaml"}}, "yaml_paths",
                 id="submit/a-path-not-in-a-list"),
])
def test_a_bad_request_is_answered_with_what_is_wrong(dashboard, tasks, fake_lsf, route, sent, field):
    response = dashboard.post(f"/api/blockwise/{route}", **sent)
    body = response.get_json()
    assert response.status_code == 200 and body["valid" if route == "validate" else "success"] is False
    assert field in body["error"], body
    assert fake_lsf.commands("bsub") == []


def test_the_precheck_passes_a_task_without_side_effects(dashboard, raw_zarr, pooling_model, task_yaml, tmp_path,
                                                        monkeypatch):
    """It built the processor (create=True): it created the outputs, loaded each
    model into the dashboard, and replaced the dashboard's own chain."""
    def refuse(self):
        raise AssertionError("the precheck loaded a model")

    monkeypatch.setattr(ScriptModelConfig, "_get_config", refuse)
    process_chain().input_norms, process_chain().postprocess = ["the dashboard's own"], ["chain"]
    body = dashboard.post("/api/blockwise/precheck", json={"yaml_paths": [task_yaml(raw_zarr(), pooling_model())]})
    assert body.get_json()["success"] is True
    assert not os.path.exists(tmp_path / "out.zarr")
    assert (get_session().input_norms, get_session().postprocess) == (["the dashboard's own"], ["chain"])


def test_the_precheck_answers_a_config_error(dashboard, tmp_path):
    (tmp_path / "bad.yaml").write_text("charge_group: g\nmodels: {}\n")  # no data_path
    body = dashboard.post("/api/blockwise/precheck", json={"yaml_paths": [str(tmp_path / "bad.yaml")]}).get_json()
    assert body["success"] is False and "data_path" in body["error"]


@pytest.mark.parametrize("yaml_paths, as_given", [
    pytest.param("prechecked", True, id="prechecked-yamls-as-they-are"),
    pytest.param(["/no/such/task.yaml"], False, id="missing"),
])
def test_submit_runs_the_prechecked_yamls_or_else_generates_them(dashboard, tasks, fake_lsf, yaml_paths, as_given):
    fake_lsf.answers["bsub"] = [ACCEPTED]
    payload = {"pipeline": PIPELINE}
    if yaml_paths == "prechecked":
        (generated,) = dashboard.post("/api/blockwise/generate", json={"pipeline": PIPELINE}).get_json()["task_paths"]
        # Under a name generate never uses, so a regenerated file cannot pass for it.
        yaml_paths = [os.path.join(os.path.dirname(generated), "prechecked_task.yaml")]
        os.rename(generated, yaml_paths[0])
    if yaml_paths is not None:
        payload["yaml_paths"] = yaml_paths
    before = sorted(tasks.glob("*.yaml"))

    body = dashboard.post("/api/blockwise/submit", json=payload).get_json()

    assert body["success"], body
    (argv,) = fake_lsf.commands("bsub")
    if as_given:
        assert body["task_paths"] == yaml_paths and argv[-1] == yaml_paths[0]
        assert sorted(tasks.glob("*.yaml")) == before, "nothing is regenerated"
    else:
        (task,) = body["task_paths"]
        assert os.path.dirname(task) == str(tasks) and argv[-1] == task


@pytest.mark.parametrize("answer, success, job_id", [
    pytest.param(ACCEPTED, True, "5150", id="accepted"),
    pytest.param((255, "", "Project grp is not valid\n"), False, None, id="refused"),
    # bsub said yes, so the task is taken to be queued.
    pytest.param("Your request was accepted.\n", True, "unknown", id="accepted-without-an-id"),
])
def test_what_bsub_answered_is_what_submit_reports(dashboard, tasks, fake_lsf, answer, success, job_id):
    fake_lsf.answers["bsub"] = [answer]
    body = dashboard.post("/api/blockwise/submit", json={"pipeline": PIPELINE, "job_name": "my run"}).get_json()
    assert (body["success"], body.get("job_id")) == (success, job_id) and len(fake_lsf.commands("bsub")) == 1
    if not success:  # LSF's own reason, for the builder to show
        assert "Project grp is not valid" in body["error"]
    if success:  # the master is the task's LSF job
        (argv,) = fake_lsf.commands("bsub")
        assert argv[argv.index("-J") + 1] == body["task_name"]
    if job_id == "5150":  # the master's log, where its %J became the id
        assert body["log_path"] == os.path.join(str(tasks), f"{body['task_name']}_5150.log")
