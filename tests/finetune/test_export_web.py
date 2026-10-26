"""A model finetuned on the server exports to be published (export_web): a
cellmap-models export a server opens and ONNX files in a folder the catalog
reads, each computing what the server serves; a model too big for the browser
the first alone; and the dashboard follows the export's job until it says so."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.finetune import export_web
from cellmap_flow.jobs.spec import JobStatus

# A funlib-style UNet, as small as onnx_web's rewrites allow: valid
# convolutions, a 2x max pool, a 2x transposed convolution back up, and the
# skip connection cropped to it and concatenated. 20^3 in, 12^3 out.
SCRIPT = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(nn.Conv3d(1, 4, 3), nn.ReLU())
        self.bottom = nn.Sequential(
            nn.MaxPool3d(2), nn.Conv3d(4, 8, 3), nn.ReLU(), nn.ConvTranspose3d(8, 4, 2, stride=2)
        )
        self.up = nn.Sequential(nn.Conv3d(8, 4, 3), nn.ReLU(), nn.Conv3d(4, 2, 1))

    def forward(self, x):
        skip = self.down(x)
        below = self.bottom(skip)
        c = (skip.shape[2] - below.shape[2]) // 2
        skip = skip[:, :, c:c + below.shape[2], c:c + below.shape[3], c:c + below.shape[4]]
        return self.up(torch.cat([skip, below], 1))


input_voxel_size = output_voxel_size = Coordinate(16, 16, 16)
read_shape = Coordinate(20, 20, 20) * input_voxel_size
write_shape = Coordinate(12, 12, 12) * output_voxel_size
output_channels = 2
channels = ["mito_aff_y", "mito_aff_x"]
torch.manual_seed(0)
model = TinyUNet()
"""

# The same UNet in 2D, on one z-slice: depth-1 kernels, pooling and
# upsampling in y and x. 1x20x20 in, 1x12x12 out.
SCRIPT_2D = (SCRIPT.replace("nn.Conv3d(1, 4, 3)", "nn.Conv3d(1, 4, (1, 3, 3))")
             .replace("nn.MaxPool3d(2)", "nn.MaxPool3d((1, 2, 2))")
             .replace("nn.Conv3d(4, 8, 3)", "nn.Conv3d(4, 8, (1, 3, 3))")
             .replace("nn.ConvTranspose3d(8, 4, 2, stride=2)", "nn.ConvTranspose3d(8, 4, (1, 2, 2), stride=(1, 2, 2))")
             .replace("nn.Conv3d(8, 4, 3)", "nn.Conv3d(8, 4, (1, 3, 3))")
             .replace("c = (skip.shape[2] - below.shape[2]) // 2", "c = (skip.shape[3] - below.shape[3]) // 2")
             .replace("skip = skip[:, :, c:c + below.shape[2], c:c + below.shape[3], c:c + below.shape[4]]",
                      "skip = skip[:, :, :, c:c + below.shape[3], c:c + below.shape[4]]")
             .replace("Coordinate(20, 20, 20)", "Coordinate(1, 20, 20)")
             .replace("Coordinate(12, 12, 12)", "Coordinate(1, 12, 12)"))

INPUT_NORM = {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}, "LambdaNormalizer": {"expression": "x*2-1"}}


def _cellmap_folder(path, model):
    """``model`` exported as cellmap_models exports one, at 20^3: TorchScript to
    serve, and the torch.export program the trainer unflattens and trains."""
    import torch

    x = torch.rand(1, 1, 20, 20, 20)
    path.mkdir()
    with torch.no_grad():
        torch.jit.trace(model.eval(), x).save(str(path / "model.ts"))
    torch.export.save(torch.export.export(model, (x,)), str(path / "model.pt2"))
    (path / "metadata.json").write_text(json.dumps({
        "model_name": "tiny", "model_type": "UNet", "framework": "torch", "spatial_dims": 3,
        "in_channels": 1, "out_channels": 2, "iteration": 1, "channels_names": ["mito_aff_y", "mito_aff_x"],
        "input_voxel_size": [16] * 3, "output_voxel_size": [16] * 3,
        "input_shape": [20] * 3, "output_shape": [12] * 3,
    }))
    return {"type": "cellmap", "folder_path": str(path), "name": "tiny"}


@pytest.fixture(params=["script", "cellmap"])
def finetuned(request, tmp_path, annotation_volume):
    """A LoRA finetune of the tiny UNet, as a run leaves it: its adapter, the
    YAML that serves it, and the session's corrections it was trained on.

    The base is the UNet as a script model, which trains as it is, or as a
    cellmap folder, which the trainer trains as the torch.export program
    unflattened, where every layer is an FX interpreter over the weights and
    the transposed convolution stays one; "2d" is the 2D UNet as a script
    model.
    """
    pytest.importorskip("peft")
    import torch

    from cellmap_flow.finetune.finetuned_model_templates import generate_finetuned_model_yaml
    from cellmap_flow.finetune.lora_wrapper import save_lora_adapter, wrap_model_with_lora
    from cellmap_flow.finetune.model_loading import load_trainable_model, model_config_from_entry
    from cellmap_flow.finetune.session.manifest import write_manifest

    script = tmp_path / "tiny_unet.py"
    script.write_text(SCRIPT_2D if request.param == "2d" else SCRIPT)
    block = ([1, 20, 20], [1, 12, 12]) if request.param == "2d" else ([20] * 3, [12] * 3)
    base = {"type": "script", "script_path": str(script), "name": "tiny"}
    if request.param == "cellmap":
        base = _cellmap_folder(tmp_path / "tiny", model_config_from_entry(base).config.model)
    # As the trainer adapts it.
    peft = wrap_model_with_lora(load_trainable_model(model_config_from_entry(base)),
                                lora_r=2, lora_alpha=4, lora_dropout=0.0)
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.fill_(0.1)
    save_lora_adapter(peft, str(tmp_path / "run" / "lora_adapter"))

    labels = np.zeros((32, 32, 32), np.uint8)
    labels[12:20, 12:20, 12:20] = 2
    labels[12:20, 12:20, 4:8] = 1
    raw = np.random.default_rng(0).integers(0, 256, labels.shape, dtype=np.uint8)
    volume = annotation_volume(labels, raw=raw)
    corrections = tmp_path / "session" / "corrections"
    write_manifest(str(corrections), {
        "kind": "volume_zarr_v1", "volume_zarr_path": volume.path, "raw_dataset_path": volume.raw,
        "input_size_voxels": block[0], "output_size_voxels": block[1],
        "input_voxel_size_nm": [16] * 3, "output_voxel_size_nm": [16] * 3, "input_norm": INPUT_NORM,
    })
    yaml_path = generate_finetuned_model_yaml(
        lora_adapter_path=str(tmp_path / "run" / "lora_adapter"), base_model_dict=base,
        model_name="tiny_finetuned_20261026_120000", output_path=tmp_path / "session" / "models" / "tiny.yaml",
        data_path=volume.raw, json_data={"input_norm": INPUT_NORM, "postprocess": {}},
    )
    return SimpleNamespace(yaml=yaml_path, corrections=corrections, run=tmp_path / "session" / "runs" / "run")


class FakeJob:
    """A launched job whose status the test sets."""

    def __init__(self):
        self.status = JobStatus.RUNNING

    def get_status(self):
        return self.status


@pytest.fixture
def launched(monkeypatch):
    """The jobs start_export launches, each a FakeJob, in turn; nothing is started."""
    from cellmap_flow.finetune.job_manager import submit

    jobs = []

    def launch(name, command, **settings):
        jobs.append(FakeJob())
        return jobs[-1]

    monkeypatch.setattr(submit, "launch", launch)
    return jobs


def _served(yaml_path, x):
    """What the server serves for ``x``: the YAML's finetuned model, the base with its adapter on."""
    import torch

    from cellmap_flow.finetune.model_loading import model_config_from_entry

    with torch.no_grad():
        return model_config_from_entry(export_web.serving_entry(yaml_path)[1]).config.model(torch.from_numpy(x)).numpy()


def _check_server_form(directory, x, served):
    """The export's server form in ``directory`` is a cellmap-models export:
    served as a folder of them is (type: cellmap), it predicts ``served`` for
    ``x``; it says its input normalization; and a finetune of it can attach
    LoRA adapters, as to cellmap's own models."""
    import torch

    from cellmap_flow.finetune.model_loading import load_trainable_model
    from cellmap_flow.finetune.trainable import finetune_modes
    from cellmap_flow.models.registry import build_model
    from cellmap_flow.pipeline_spec import normalize_steps

    config = build_model({"type": "cellmap", "folder_path": str(directory)}, "published")
    model = config.config.model
    with torch.no_grad():
        exported = model(torch.from_numpy(x).to(next(model.parameters()).device)).cpu().numpy()
    assert np.abs(exported - served).max() <= 1e-5 * np.abs(served).max()
    assert json.loads((directory / "metadata.json").read_text())["input_norm"] == list(normalize_steps(INPUT_NORM))
    assert "lora" in finetune_modes(load_trainable_model(config))


@pytest.mark.finetune
def test_a_finetuned_model_exports_as_the_server_serves_it(finetuned, launched):
    pytest.importorskip("cellmap_models.model_export.onnx_web")
    ort = pytest.importorskip("onnxruntime")

    from cellmap_flow.models.configs.onnx import parse_manifest
    from cellmap_flow.pipeline_spec import normalize_steps

    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert export_web.export_status(export)["status"] == "running"

    # The job, run here: the module with the flags start_export gave it.
    assert export_web.main(["--yaml", str(finetuned.yaml), "--corrections", str(finetuned.corrections),
                            "--output", str(export.output_dir)]) == 0
    status = export_web.export_status(export)
    assert status["status"] == "done", status["error"]
    assert status["no_browser"] is None
    assert set(status["server"]["files"]) == {"metadata.json", "model.ts", "model.pt2", "export_record.json"}
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, 20, 20, 20)).astype(np.float32)
    served = _served(finetuned.yaml, x)
    _check_server_form(export.output_dir, x, served)

    # The folder is a model of the catalog's: a manifest the server's reader
    # takes, with the model's geometry and normalization, and every variant's file.
    folder = export.output_dir / status["folder"]
    manifest = parse_manifest(json.loads((folder / "manifest.json").read_text()))
    assert manifest == parse_manifest(status["manifest"])
    assert manifest["channels"] == ["mito_aff_y", "mito_aff_x"]
    assert manifest["input_voxel_size"] == manifest["output_voxel_size"] == [16, 16, 16]
    assert manifest["input_norm"] == list(normalize_steps(INPUT_NORM))
    assert {v["variant"] for v in manifest["variants"]} == {"fp32", "fp16", "int8w", "int8w_fp16"}
    assert all((folder / v["file"]).is_file() for v in manifest["variants"])
    # None has a 3D transposed convolution, which onnxruntime-web's WebGPU cannot run.
    import onnx

    assert not any(node.op_type == "ConvTranspose" for v in manifest["variants"]
                   for node in onnx.load(str(folder / v["file"])).graph.node)

    # Its fp32 file computes what the server serves: the base with its
    # adapter on, on a block it was not validated on.
    variant = next(v for v in manifest["variants"] if v["variant"] == "fp32")
    assert variant["input_size"] == [20] * 3 and variant["output_size"] == [12] * 3
    session = ort.InferenceSession(str(folder / variant["file"]), providers=["CPUExecutionProvider"])
    exported = session.run([manifest["io"]["output"]], {manifest["io"]["input"]: x})[0]
    assert np.abs(exported - served).max() <= 1e-5 * np.abs(served).max()


@pytest.mark.finetune
@pytest.mark.parametrize("finetuned", ["cellmap"], indirect=True)
def test_a_model_too_big_for_the_browser_exports_for_servers_alone(finetuned, launched, monkeypatch):
    pytest.importorskip("cellmap_models.model_export.onnx_web")
    # Over protobuf's 2 GB as an 800M-parameter UNet is, cheaply: any
    # network is over a cap of no bytes.
    monkeypatch.setattr(export_web, "ONNX_FILE_LIMIT", 0)
    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert export_web.main(["--yaml", str(finetuned.yaml), "--corrections", str(finetuned.corrections),
                            "--output", str(export.output_dir)]) == 0
    status = export_web.export_status(export)

    assert status["status"] == "done", status["error"]
    assert "2 GB" in status["no_browser"]
    assert status["manifest"]["variants"] == [] and not (export.output_dir / status["folder"]).exists()
    assert (status["server"]["input_size"], status["server"]["output_size"]) == ([20] * 3, [12] * 3)
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, 20, 20, 20)).astype(np.float32)
    _check_server_form(export.output_dir, x, _served(finetuned.yaml, x))


@pytest.mark.finetune
@pytest.mark.parametrize("finetuned", ["2d"], indirect=True)
def test_a_2d_model_exports_for_the_browser_on_slices(finetuned, launched):
    pytest.importorskip("cellmap_models.model_export.student")  # export_onnx_web with blocks
    ort = pytest.importorskip("onnxruntime")

    from cellmap_flow.models.configs.onnx import parse_manifest

    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert export_web.main(["--yaml", str(finetuned.yaml), "--corrections", str(finetuned.corrections),
                            "--output", str(export.output_dir), "--variants", "fp32"]) == 0
    status = export_web.export_status(export)
    assert status["status"] == "done", status["error"]
    assert status["no_browser"] is None

    # One slice in, one slice out, named and sized as the catalog's 2D models.
    manifest = parse_manifest(status["manifest"])
    (variant,) = manifest["variants"]
    assert variant["input_size"] == [1, 20, 20] and variant["output_size"] == [1, 12, 12]
    assert "_2d_in20_" in variant["file"]
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, 1, 20, 20)).astype(np.float32)
    session = ort.InferenceSession(str(export.output_dir / status["folder"] / variant["file"]),
                                   providers=["CPUExecutionProvider"])
    exported = session.run([manifest["io"]["output"]], {manifest["io"]["input"]: x})[0]
    served = _served(finetuned.yaml, x)
    assert np.abs(exported - served).max() <= 1e-5 * np.abs(served).max()


@pytest.mark.parametrize("finetuned", ["script"], indirect=True)
def test_an_export_fails_when_it_says_so_or_its_job_ends_without_a_result(finetuned, launched):
    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    launched[-1].status = JobStatus.FAILED
    (export.output_dir / export_web.LOG_NAME).write_text("Killed\n")
    status = export_web.export_status(export)
    assert status["status"] == "failed" and status["error"]

    # Started again, the earlier result is not read as the new export's.
    export_web._write_result(export.output_dir, {"error": "no can do"})
    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert export_web.export_status(export)["status"] == "running"
    export_web._write_result(export.output_dir, {"error": "no can do"})
    status = export_web.export_status(export)
    assert status["status"] == "failed" and status["error"] == "no can do"
