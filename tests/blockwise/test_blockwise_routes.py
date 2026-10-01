"""The dashboard's blockwise routes, which the pipeline builder calls in turn:
validate, generate (which writes the task YAML the blockwise CLI runs),
precheck and submit.

The task YAML is pinned as text, byte for byte but for the timestamp: it is
what the blockwise CLI runs, and a file the user may keep and edit. Each route
answers 200 whatever happens: {"valid": ...} from validate, {"success": ...}
from the others, with an "error" the builder shows when it is false. The
master's bsub argv is pinned in test_bsub_argv_snapshot; LSF here is
conftest's ``fake_lsf``."""

import copy
import json
import os
import re
import textwrap
from pathlib import Path

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


def pipeline(**changes):
    """PIPELINE with these lists (or model_mode) in place of its own."""
    return {**copy.deepcopy(PIPELINE), **copy.deepcopy(changes)}


def as_the_builder_sends(body):
    """``body`` as JSON text, keys in their order, as JSON.stringify sends it.
    (The test client's ``json=`` sorts them, and the task YAML keeps the order
    a model's params came in.)"""
    return {"data": json.dumps(body), "content_type": "application/json"}


def unstamped(text):
    """``text`` with generate's timestamps (the task names') as <ts>."""
    return re.sub(r"\d{8}_\d{6}", "<ts>", text)


@pytest.fixture
def tasks(tmp_path):
    session = get_session()
    session.blockwise_tasks_dir, session.walltime = str(tmp_path / "tasks"), "12:00"
    return tmp_path / "tasks"


# --- the task YAML --------------------------------------------------------------


def task(tail="", output_path="/out/pred.zarr", task_name="run_<ts>"):
    """The task YAML generate writes for PIPELINE, for that output path and task
    name, with ``tail`` after its models."""
    return f"""\
data_path: /data/raw.zarr/raw
output_path: {output_path}
task_name: {task_name}
charge_group: grp
queue: gpu_h100
workers: 2
cpu_workers: 12
tmp_dir: /scratch/progress
walltime: '12:00'
models:
- name: m
  type: script
  script_path: /s.py
""" + textwrap.dedent(tail)


MODEL_M = PIPELINE["models"][0]
BOX_1 = {"offset": [0, 0, 0], "shape": [64, 64, 64]}
BOX_2 = {"offset": [64, 0, 0], "shape": [32, 32, 32]}
BOX_1_YAML = "- offset:\n  - 0\n  - 0\n  - 0\n  shape:\n  - 64\n  - 64\n  - 64\n"
BOX_2_YAML = "- offset:\n  - 64\n  - 0\n  - 0\n  shape:\n  - 32\n  - 32\n  - 32\n"


def with_input(**params):
    return [{"params": {"dataset_path": "/data/raw.zarr/raw", **params}}]


@pytest.mark.parametrize("sent, written", [
    pytest.param(pipeline(), {"run_<ts>.yaml": task()}, id="one-model"),
    pytest.param(pipeline(outputs=[{"params": {"dataset_path": "/out/pred.zarr/labels/"}}]),
                 {"run_<ts>.yaml": task(output_path="/out/pred.zarr/labels")}, id="an-output-inside-a-zarr"),
    pytest.param(pipeline(models=[MODEL_M, {"name": "n", "params": {"type": "script", "script_path": "/n.py"}}],
                          model_mode="OR"),
                 {"run_<ts>.yaml": task("""\
                     - name: n
                       type: script
                       script_path: /n.py
                     model_mode: OR
                     """)}, id="several-models-and-a-merge-mode"),
    pytest.param(pipeline(model_mode="OR"), {"run_<ts>.yaml": task()}, id="a-merge-mode-for-one-model"),
    # Without params a model node's config is its settings; params, when it
    # has them, even if it also has a config.
    pytest.param(pipeline(models=[MODEL_M, {"name": "c", "config": {"type": "script", "script_path": "/c.py"}},
                                  {"name": "p", "params": {"script_path": "/p.py"}, "config": {"script_path": "/x.py"}}]),
                 {"run_<ts>.yaml": task("""\
                     - name: c
                       type: script
                       script_path: /c.py
                     - name: p
                       script_path: /p.py
                     """)}, id="a-model-by-its-config"),
    # The builder's text fields send a list as it was typed.
    pytest.param(pipeline(models=[MODEL_M, {"name": "t", "params": {
        "channels": "[mito, er]", "input_size": "(1, 2)", "output_size": "'[4]'", "input_voxel_size": [8],
        "output_voxel_size": "[8,, 8]", "other": "[a]"}}]),
                 {"run_<ts>.yaml": task("""\
                     - name: t
                       channels:
                       - mito
                       - er
                       input_size:
                       - 1
                       - 2
                       output_size:
                       - 4
                       input_voxel_size:
                       - 8
                       output_voxel_size: '[8,, 8]'
                       other: '[a]'
                     """)}, id="list-fields-typed-as-text"),
    pytest.param(pipeline(inputs=with_input(bounding_boxes=[BOX_1])),
                 {"run_<ts>.yaml": task("bounding_boxes:\n" + BOX_1_YAML)}, id="bounding-boxes"),
    pytest.param(pipeline(inputs=with_input(bounding_boxes=[BOX_1, BOX_2], separate_bounding_boxes_zarrs=True)), {
        "run_<ts>_box1.yaml": task("bounding_boxes:\n" + BOX_1_YAML + "separate_bounding_boxes_zarrs: true\n",
                                   output_path="/out/pred.zarr/box_1", task_name="run_<ts>_box1"),
        "run_<ts>_box2.yaml": task("bounding_boxes:\n" + BOX_2_YAML + "separate_bounding_boxes_zarrs: true\n",
                                   output_path="/out/pred.zarr/box_2", task_name="run_<ts>_box2"),
    }, id="a-zarr-per-box"),
    pytest.param(pipeline(inputs=with_input(separate_bounding_boxes_zarrs=True)),
                 {"run_<ts>.yaml": task("separate_bounding_boxes_zarrs: true\n")}, id="a-zarr-per-box-without-boxes"),
    # json_data was a dict keyed by step name, so a chain using a step twice
    # kept only the last one, and blockwise computed something else than the
    # dashboard showed. A step without a name is left out.
    pytest.param(pipeline(normalizers=[{"name": "LambdaNormalizer", "params": {"expression": "x*2"}},
                                       {"name": "LambdaNormalizer", "params": {"expression": "x-1"}},
                                       {"params": {"expression": "x"}}],
                          postprocessors=[{"name": "ThresholdPostprocessor", "params": {"threshold": 0.5}},
                                          {"name": "SigmoidPostprocessor", "params": None}]),
                 {"run_<ts>.yaml": task("""\
                     json_data:
                       input_norm:
                       - name: LambdaNormalizer
                         expression: x*2
                       - name: LambdaNormalizer
                         expression: x-1
                       postprocess:
                       - name: ThresholdPostprocessor
                         threshold: 0.5
                       - name: SigmoidPostprocessor
                     """)}, id="a-chain"),
    pytest.param(pipeline(outputs=[{"params": {"dataset_path": "/out/pred", "output_channels": ["mito", "er"]}}]),
                 {"run_<ts>.yaml": task("output_channels:\n- mito\n- er\n")}, id="output-channels"),
    pytest.param(pipeline(inputs=with_input(bounding_boxes=[BOX_1]),
                          models=[MODEL_M, {"name": "n", "params": {"type": "script", "script_path": "/n.py"}}],
                          model_mode="OR", normalizers=[{"name": "LambdaNormalizer", "params": {"expression": "x"}}],
                          outputs=[{"params": {"dataset_path": "/out/pred", "output_channels": ["mito"]}}]),
                 {"run_<ts>.yaml": task(textwrap.dedent("""\
                     - name: n
                       type: script
                       script_path: /n.py
                     bounding_boxes:
                     """) + BOX_1_YAML + textwrap.dedent("""\
                     model_mode: OR
                     json_data:
                       input_norm:
                       - name: LambdaNormalizer
                         expression: x
                       postprocess: []
                     output_channels:
                     - mito
                     """))}, id="every-optional-key-in-order"),
])
def test_the_task_yamls_generate_writes(dashboard, tasks, sent, written):
    answer = dashboard.post("/api/blockwise/generate",
                            **as_the_builder_sends({"pipeline": sent, "job_name": "run"})).get_json()
    assert answer["success"], answer
    files = [(unstamped(os.path.basename(path)), unstamped(Path(path).read_text())) for path in answer["task_paths"]]
    assert files == list(written.items())
    assert sorted(os.listdir(tasks)) == sorted(os.path.basename(path) for path in answer["task_paths"])


def test_generate_answers_with_the_task_it_wrote(dashboard, tasks):
    answer = dashboard.post("/api/blockwise/generate",
                            **as_the_builder_sends({"pipeline": PIPELINE, "job_name": "run"})).get_json()
    assert json.loads(unstamped(json.dumps(answer)).replace(str(tasks), "<tasks>")) == {
        "success": True,
        "message": "Blockwise task generated successfully",
        "task_name": "run_<ts>",
        "task_paths": ["<tasks>/run_<ts>.yaml"],
        "task_yaml": task(),
        "task_config": yaml.safe_load(task()),
    }


# --- what each route answers ------------------------------------------------------


NOT_JSON = {"data": "pipeline", "content_type": "text/plain"}
NO_MASTER_CORES = {key: value for key, value in SETTINGS.items() if key != "nb_cores_master"}


def invalid(error):
    return {"valid": False, "error": error}


def failed(error):
    return {"success": False, "error": error}


@pytest.mark.parametrize("route, sent, answer", [
    pytest.param("validate", {"json": {"pipeline": PIPELINE}},
                 {"valid": True, "message": "Pipeline is ready for blockwise processing"}, id="validate/ready"),
    # The builder shows these as they are.
    pytest.param("validate", {"json": {}}, invalid("No input nodes defined"), id="validate/no-pipeline"),
    pytest.param("validate", {"json": {"pipeline": pipeline(inputs=[])}}, invalid("No input nodes defined"),
                 id="validate/no-inputs"),
    pytest.param("validate", {"json": {"pipeline": pipeline(outputs=[])}}, invalid("No output nodes defined"),
                 id="validate/no-outputs"),
    pytest.param("validate", {"json": {"pipeline": pipeline(models=[])}}, invalid("No models defined"),
                 id="validate/no-models"),
    pytest.param("validate", {"json": {"pipeline": pipeline(blockwise_config=[])}},
                 invalid("No blockwise configuration defined"), id="validate/no-settings"),
    pytest.param("validate", {"json": {"pipeline": pipeline(inputs=[{"params": {}}])}},
                 invalid("Input node missing dataset_path"), id="validate/an-input-without-a-path"),
    pytest.param("validate", {"json": {"pipeline": pipeline(outputs=[{"params": {"dataset_path": ""}}])}},
                 invalid("Output node missing dataset_path"), id="validate/an-output-without-a-path"),
    # A body the builder would not send: what is wrong, and where.
    pytest.param("validate", NOT_JSON, invalid("expected a JSON object"), id="validate/not-json"),
    pytest.param("validate", {"json": [PIPELINE]}, invalid("expected a JSON object"),
                 id="validate/not-an-object"),
    pytest.param("validate", {"json": {"pipeline": pipeline(inputs=["/data/raw.zarr/raw"])}},
                 invalid("pipeline.inputs.0: Input should be a valid dictionary or instance of PipelineNode"),
                 id="validate/a-node-not-an-object"),
    pytest.param("validate", {"json": {"pipeline": pipeline(inputs=[{"params": None}])}},
                 invalid("pipeline.inputs.0.params: Input should be a valid dictionary"),
                 id="validate/a-node-without-params"),
    pytest.param("validate", {"json": {"pipeline": pipeline(models={"m": MODEL_M["params"]})}},
                 invalid("pipeline.models: Input should be a valid list"), id="validate/models-not-a-list"),
    pytest.param("validate", {"json": {"pipeline": pipeline(blockwise_config=[{"params": {"queue": "gpu_h100"}}])}},
                 invalid("pipeline.blockwise_config.0.params.charge_group: Field required"),
                 id="validate/settings-missing"),

    pytest.param("generate", {"json": {"pipeline": pipeline(models=[])}}, failed("No models defined"),
                 id="generate/invalid"),
    pytest.param("generate", {"json": {"pipeline": pipeline(models=[{"name": "m", "params": None}])}},
                 failed("pipeline.models.0.params: Input should be a valid dictionary"),
                 id="generate/model-params-not-an-object"),
    pytest.param("generate", {"json": {"pipeline": pipeline(normalizers="LambdaNormalizer")}},
                 failed("pipeline.normalizers: Input should be a valid list"), id="generate/a-chain-not-a-list"),

    pytest.param("precheck", {"json": {}}, failed("No YAML paths provided. Please generate task first."),
                 id="precheck/none-given"),
    pytest.param("precheck", {"json": {"yaml_paths": "/tasks/t.yaml"}},
                 failed("yaml_paths: Input should be a valid list"), id="precheck/a-path-not-in-a-list"),
    # A number was opened as a file descriptor: [1] closed the dashboard's stdout.
    pytest.param("precheck", {"json": {"yaml_paths": [None]}},
                 failed("yaml_paths.0: Input should be a valid string"), id="precheck/a-path-not-text"),

    pytest.param("submit", {"json": {"pipeline": pipeline(outputs=[])}}, failed("No output nodes defined"),
                 id="submit/invalid"),
    # It was found missing only here, after generate and precheck had passed.
    pytest.param("submit", {"json": {"pipeline": pipeline(blockwise_config=[{"params": NO_MASTER_CORES}])}},
                 failed("pipeline.blockwise_config.0.params.nb_cores_master: Field required"),
                 id="submit/no-master-cores"),
    # Refused as precheck refuses them; submit generated a task anew and ran that.
    pytest.param("submit", {"json": {"pipeline": PIPELINE, "yaml_paths": "/tasks/t.yaml"}},
                 failed("yaml_paths: Input should be a valid list"), id="submit/a-path-not-in-a-list"),
    pytest.param("submit", {"json": {"pipeline": PIPELINE, "yaml_paths": [None]}},
                 failed("yaml_paths.0: Input should be a valid string"), id="submit/a-path-not-text"),
])
def test_what_each_route_answers(dashboard, tasks, fake_lsf, route, sent, answer):
    response = dashboard.post(f"/api/blockwise/{route}", **sent)
    assert (response.status_code, response.get_json()) == (200, answer)
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
    assert body.get_json() == {"success": True, "message": "success"}
    assert not os.path.exists(tmp_path / "out.zarr")
    assert (get_session().input_norms, get_session().postprocess) == (["the dashboard's own"], ["chain"])


def test_the_precheck_answers_a_config_error(dashboard, tmp_path):
    (tmp_path / "bad.yaml").write_text("charge_group: g\nmodels: {}\n")  # no data_path
    body = dashboard.post("/api/blockwise/precheck", json={"yaml_paths": [str(tmp_path / "bad.yaml")]}).get_json()
    assert body["success"] is False and "data_path" in body["error"]


@pytest.mark.parametrize("yaml_paths, as_given", [
    pytest.param("prechecked", True, id="prechecked-yamls-as-they-are"),
    pytest.param(None, False, id="none-given"),
    pytest.param([], False, id="empty"),
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


@pytest.mark.parametrize("typed, stem", [
    pytest.param("nuc cerebellum", "nuc_cerebellum", id="spaces"),
    pytest.param("  a/b\\c:d  ", "a_b_c_d", id="unsafe-characters"),  # a file name and an LSF job name
    pytest.param("", "cellmap_flow", id="none-typed"),
])
def test_the_typed_job_name_names_the_task_and_its_master(dashboard, tasks, fake_lsf, typed, stem):
    fake_lsf.answers["bsub"] = [ACCEPTED]
    generated = dashboard.post("/api/blockwise/generate", json={"pipeline": PIPELINE, "job_name": typed}).get_json()
    task = generated["task_name"]
    (path,) = generated["task_paths"]
    assert re.fullmatch(rf"{stem}_\d{{8}}_\d{{6}}", task) and os.path.basename(path) == f"{task}.yaml"
    assert yaml.safe_load(open(path))["task_name"] == task
    body = dashboard.post("/api/blockwise/submit", json={"pipeline": PIPELINE, "yaml_paths": [path],
                                                         "task_name": task}).get_json()
    (argv,) = fake_lsf.commands("bsub")
    assert argv[argv.index("-J") + 1] == body["task_name"] == task


@pytest.mark.parametrize("answer, expected", [
    pytest.param(ACCEPTED, {"success": True, "job_id": "5150"}, id="accepted"),
    pytest.param((255, "", "Project grp is not valid\n"),
                 {"success": False, "error": "LSF error: Project grp is not valid\n"}, id="refused"),
    # bsub said yes, so the task is taken to be queued.
    pytest.param("Your request was accepted.\n", {"success": True, "job_id": "unknown"}, id="accepted-without-an-id"),
])
def test_what_bsub_answered_is_what_submit_reports(dashboard, tasks, fake_lsf, answer, expected):
    fake_lsf.answers["bsub"] = [answer]
    body = dashboard.post("/api/blockwise/submit", json={"pipeline": PIPELINE, "job_name": "my run"}).get_json()
    assert {k: body.get(k) for k in expected} == expected and len(fake_lsf.commands("bsub")) == 1
    if expected.get("job_id") == "5150":  # the master's log, where its %J became the id
        assert body["log_path"] == os.path.join(str(tasks), f"{body['task_name']}_5150.log")
