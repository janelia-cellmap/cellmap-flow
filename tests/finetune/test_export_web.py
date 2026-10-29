"""A model finetuned on the server exports to be published (export_web): a
cellmap-models export a server opens and ONNX files in a folder the catalog
reads, each computing what the server serves; a model too big for the browser
the first alone; and the dashboard follows the export's job until it says so.

A finetuned model is distilled into a small copy for the browser (distill):
the copy, pruned from a funlib UNet or a random StandardUnet for any other
network, computes nearer the finetuned model than it started, exports for
the browser as export_web exports a model, and a model a small copy cannot
be made of is refused before any training."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.finetune import distill, export_web
from cellmap_flow.jobs.spec import JobStatus
from cellmap_flow.portable.publish import SMALL_SUFFIX

# funlib.learn.torch's UNet as its state dict lays it out, in a StandardUnet
# as fly_organelles wraps one: two levels of two valid convolutions, a max
# pool, a transposed convolution back up, the skip cropped to it and
# concatenated first, and a final 1x1 convolution: what onnx_web rewrites,
# and what distill prunes. In 3D (kernel 3, factor 2) 28^3 in, 12^3 out; in
# 2D, on one z-slice, 1x28x28 in, 1x12x12 out.
FUNLIB = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

KERNEL, FACTOR = {kernel}, {factor}


class ConvPass(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv_pass = nn.Sequential(nn.Conv3d(inputs, outputs, KERNEL), nn.ReLU(),
                                       nn.Conv3d(outputs, outputs, KERNEL), nn.ReLU())

    def forward(self, x):
        return self.conv_pass(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(channels, channels, FACTOR, stride=FACTOR)

    def forward(self, left, below):
        up = self.up(below)
        z, y, x = (slice((n - m) // 2, (n - m) // 2 + m) for n, m in zip(left.shape[2:], up.shape[2:]))
        return torch.cat([left[:, :, z, y, x], up], 1)


class UNet(nn.Module):
    def __init__(self, num_fmaps, fmap_inc_factor):
        super().__init__()
        top, below = num_fmaps, num_fmaps * fmap_inc_factor
        self.l_conv = nn.ModuleList([ConvPass(1, top), ConvPass(top, below)])
        self.l_down = nn.ModuleList([nn.MaxPool3d(FACTOR)])
        self.r_up = nn.ModuleList([nn.ModuleList([Upsample(below)])])
        self.r_conv = nn.ModuleList([nn.ModuleList([ConvPass(top + below, top)])])

    def forward(self, x):
        left = self.l_conv[0](x)
        return self.r_conv[0][0](self.r_up[0][0](left, self.l_conv[1](self.l_down[0](left))))


class StandardUnet(nn.Module):
    def __init__(self, out_channels, num_fmaps=8, fmap_inc_factor=4):
        super().__init__()
        self.unet_backbone = UNet(num_fmaps, fmap_inc_factor)
        self.final_conv = nn.Conv3d(num_fmaps, out_channels, 1)

    def forward(self, x):
        return self.final_conv(self.unet_backbone(x))


input_voxel_size = output_voxel_size = Coordinate(16, 16, 16)
read_shape = Coordinate({read}) * input_voxel_size
write_shape = Coordinate({write}) * output_voxel_size
output_channels = 2
channels = ["mito_aff_y", "mito_aff_x"]
torch.manual_seed(0)
model = StandardUnet(2)
# He's initialization and no biases, so that the output varies with the
# input, as a trained network's does, rather than being its biases.
for layer in model.modules():
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
"""

# Not a funlib UNet, in 2D, on one z-slice: 1x28x28 in, 1x24x24 out.
PLAIN_2D = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

input_voxel_size = output_voxel_size = Coordinate(16, 16, 16)
read_shape = Coordinate(1, 28, 28) * input_voxel_size
write_shape = Coordinate(1, 24, 24) * output_voxel_size
output_channels = 2
channels = ["mito_aff_y", "mito_aff_x"]
torch.manual_seed(0)
model = nn.Sequential(nn.Conv3d(1, 4, (1, 3, 3)), nn.ReLU(), nn.Conv3d(4, 2, (1, 3, 3)))
"""

# Writes 32 nm voxels from 16 nm ones, which a small copy cannot.
COARSE = """
from funlib.geometry import Coordinate
import torch.nn as nn

input_voxel_size, output_voxel_size = Coordinate(16, 16, 16), Coordinate(32, 32, 32)
read_shape = Coordinate(28, 28, 28) * input_voxel_size
write_shape = Coordinate(14, 14, 14) * output_voxel_size
output_channels = 2
channels = ["mito_aff_y", "mito_aff_x"]
model = nn.Sequential(nn.Conv3d(1, 2, 2, stride=2))
"""

UNET_3D = FUNLIB.format(kernel=3, factor=2, read="28, 28, 28", write="12, 12, 12")
UNET_2D = FUNLIB.format(kernel="(1, 3, 3)", factor="(1, 2, 2)", read="1, 28, 28", write="1, 12, 12")

# Each base's script, its block in voxels in and out, and the side of the
# volume it is finetuned on. A random StandardUnet reads 1x146x146 for the
# plain network's 1x24x24: a volume about as large, so that its blocks are
# mostly raw, not edge.
BASES = {"script": (UNET_3D, (28,) * 3, (12,) * 3, 48), "cellmap": (UNET_3D, (28,) * 3, (12,) * 3, 48),
         "2d": (UNET_2D, (1, 28, 28), (1, 12, 12), 48), "plain2d": (PLAIN_2D, (1, 28, 28), (1, 24, 24), 128),
         "coarse": (COARSE, (28,) * 3, (14,) * 3, 48)}

NAME = "unet_finetuned_20261026_120000"
INPUT_NORM = {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}, "LambdaNormalizer": {"expression": "x*2-1"}}


def _cellmap_folder(path, model, size, out):
    """``model`` exported as cellmap_models exports one, at ``size`` in and
    ``out`` out: TorchScript to serve, and the torch.export program the
    trainer unflattens and trains."""
    import torch

    x = torch.rand(1, 1, *size)
    path.mkdir()
    with torch.no_grad():
        torch.jit.trace(model.eval(), x).save(str(path / "model.ts"))
    torch.export.save(torch.export.export(model, (x,)), str(path / "model.pt2"))
    (path / "metadata.json").write_text(json.dumps({
        "model_name": "unet", "model_type": "UNet", "framework": "torch", "spatial_dims": 3,
        "in_channels": 1, "out_channels": 2, "iteration": 1, "channels_names": ["mito_aff_y", "mito_aff_x"],
        "input_voxel_size": [16] * 3, "output_voxel_size": [16] * 3,
        "input_shape": list(size), "output_shape": list(out),
    }))
    return {"type": "cellmap", "folder_path": str(path), "name": "unet"}


@pytest.fixture(params=["script", "cellmap"])
def finetuned(request, tmp_path, annotation_volume):
    """A LoRA finetune of a network, as a run leaves it: its adapter, the
    YAML that serves it, and the session's corrections it was trained on.

    "script" and "cellmap" are the funlib UNet as a script model, which
    trains as it is, and as a cellmap folder, which the trainer trains as
    the torch.export program unflattened, as the large cellmap UNets are:
    every layer an FX interpreter over the weights, and the transposed
    convolution one too. "2d" is the 2D UNet as a script model; "plain2d" a
    2D network that is not a funlib UNet (2D, as a 3D StandardUnet student
    of one takes a minute on the CPU); "coarse" one that writes at another
    voxel size than it reads.
    """
    pytest.importorskip("peft")
    import torch

    from cellmap_flow.finetune.finetuned_model_templates import generate_finetuned_model_yaml
    from cellmap_flow.finetune.lora_wrapper import save_lora_adapter, wrap_model_with_lora
    from cellmap_flow.finetune.model_loading import load_trainable_model, model_config_from_entry
    from cellmap_flow.finetune.session.manifest import write_manifest

    text, size, out, side = BASES[request.param]
    script = tmp_path / "unet.py"
    script.write_text(text)
    base = {"type": "script", "script_path": str(script), "name": "unet"}
    if request.param == "cellmap":
        base = _cellmap_folder(tmp_path / "unet", model_config_from_entry(base).config.model, size, out)
    # As the trainer adapts it.
    peft = wrap_model_with_lora(load_trainable_model(model_config_from_entry(base)),
                                lora_r=2, lora_alpha=4, lora_dropout=0.0)
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.fill_(0.1)
    save_lora_adapter(peft, str(tmp_path / "run" / "lora_adapter"))

    labels = np.zeros((side,) * 3, np.uint8)
    c = side // 2
    labels[c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] = 2
    labels[c - 4:c + 4, c - 4:c + 4, c - 14:c - 10] = 1
    raw = np.random.default_rng(0).integers(0, 256, labels.shape, dtype=np.uint8)
    volume = annotation_volume(labels, raw=raw)
    corrections = tmp_path / "session" / "corrections"
    write_manifest(str(corrections), {
        "kind": "volume_zarr_v1", "volume_zarr_path": volume.path, "raw_dataset_path": volume.raw,
        "input_size_voxels": list(size), "output_size_voxels": list(out),
        "input_voxel_size_nm": [16] * 3, "output_voxel_size_nm": [16] * 3, "input_norm": INPUT_NORM,
    })
    yaml_path = generate_finetuned_model_yaml(
        lora_adapter_path=str(tmp_path / "run" / "lora_adapter"), base_model_dict=base,
        model_name=NAME, output_path=tmp_path / "session" / "models" / "unet.yaml",
        data_path=volume.raw, json_data={"input_norm": INPUT_NORM, "postprocess": {}},
    )
    return SimpleNamespace(yaml=yaml_path, corrections=corrections, run=tmp_path / "session" / "runs" / "run")


class FakeJob:
    """A launched job whose status the test sets."""

    def __init__(self):
        self.status = JobStatus.RUNNING
        self.job_id = "123"

    def get_status(self):
        return self.status


@pytest.fixture
def launched(monkeypatch):
    """The jobs start_export and start_distill launch, each a FakeJob, in turn; nothing is started."""
    from cellmap_flow.finetune.job_manager import submit

    jobs = []

    def launch(name, command, **settings):
        jobs.append(FakeJob())
        return jobs[-1]

    monkeypatch.setattr(submit, "launch", launch)
    return jobs


def _run(module, finetuned, job, *flags):
    """The job start_export or start_distill started, run here: ``module``'s
    main with the flags the job was given, and ``flags``."""
    return module.main(["--yaml", str(finetuned.yaml), "--corrections", str(finetuned.corrections),
                        "--output", str(job.output_dir), *flags])


def _distilled(finetuned, run, *flags):
    """start_distill's job, run here, briefly, on the CPU."""
    return _run(distill, finetuned, run, "--workers", "0", "--variants", "fp32", *flags)


def _served(yaml_path, x):
    """What the server serves for ``x``: the YAML's finetuned model, the base with its adapter on."""
    import torch

    from cellmap_flow.finetune.model_loading import model_config_from_entry

    with torch.no_grad():
        return model_config_from_entry(export_web.serving_entry(yaml_path)[1]).config.model(torch.from_numpy(x)).numpy()


def _onnx(path, manifest, x):
    """What the ONNX file at ``path`` of the model ``manifest`` describes computes for ``x``."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([manifest["io"]["output"]], {manifest["io"]["input"]: x})[0]


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
    pytest.importorskip("onnxruntime")
    import onnx

    from cellmap_flow.models.configs.onnx import parse_manifest
    from cellmap_flow.pipeline_spec import normalize_steps

    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert export_web.export_status(export)["status"] == "running"

    assert _run(export_web, finetuned, export) == 0
    status = export_web.export_status(export)
    assert status["status"] == "done", status["error"]
    assert status["no_browser"] is None
    assert set(status["server"]["files"]) == {"metadata.json", "model.ts", "model.pt2", "export_record.json"}
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, 28, 28, 28)).astype(np.float32)
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
    assert not any(node.op_type == "ConvTranspose" for v in manifest["variants"]
                   for node in onnx.load(str(folder / v["file"])).graph.node)

    # Its fp32 file computes what the server serves: the base with its
    # adapter on, on a block it was not validated on.
    variant = next(v for v in manifest["variants"] if v["variant"] == "fp32")
    assert variant["input_size"] == [28] * 3 and variant["output_size"] == [12] * 3
    exported = _onnx(folder / variant["file"], manifest, x)
    assert np.abs(exported - served).max() <= 1e-5 * np.abs(served).max()


@pytest.mark.finetune
@pytest.mark.parametrize("finetuned", ["cellmap"], indirect=True)
def test_a_model_too_big_for_the_browser_exports_for_servers_alone(finetuned, launched, monkeypatch):
    pytest.importorskip("cellmap_models.model_export.onnx_web")
    # Over protobuf's 2 GB as an 800M-parameter UNet is, cheaply: any
    # network is over a cap of no bytes.
    monkeypatch.setattr(export_web, "ONNX_FILE_LIMIT", 0)
    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert _run(export_web, finetuned, export) == 0
    status = export_web.export_status(export)

    assert status["status"] == "done", status["error"]
    assert "2 GB" in status["no_browser"]
    assert status["manifest"]["variants"] == [] and not (export.output_dir / status["folder"]).exists()
    assert (status["server"]["input_size"], status["server"]["output_size"]) == ([28] * 3, [12] * 3)
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, 28, 28, 28)).astype(np.float32)
    _check_server_form(export.output_dir, x, _served(finetuned.yaml, x))


@pytest.mark.finetune
@pytest.mark.parametrize("finetuned", ["2d"], indirect=True)
def test_a_2d_model_exports_for_the_browser_on_slices(finetuned, launched):
    pytest.importorskip("cellmap_models.model_export.student")  # export_onnx_web with blocks
    pytest.importorskip("onnxruntime")

    from cellmap_flow.models.configs.onnx import parse_manifest

    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert _run(export_web, finetuned, export, "--variants", "fp32") == 0
    status = export_web.export_status(export)
    assert status["status"] == "done", status["error"]
    assert status["no_browser"] is None

    # One slice in, one slice out, named and sized as the catalog's 2D models.
    manifest = parse_manifest(status["manifest"])
    (variant,) = manifest["variants"]
    assert variant["input_size"] == [1, 28, 28] and variant["output_size"] == [1, 12, 12]
    assert "_2d_in28_" in variant["file"]
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, 1, 28, 28)).astype(np.float32)
    exported = _onnx(export.output_dir / status["folder"] / variant["file"], manifest, x)
    served = _served(finetuned.yaml, x)
    assert np.abs(exported - served).max() <= 1e-5 * np.abs(served).max()


@pytest.mark.parametrize("finetuned", ["script"], indirect=True)
def test_a_running_export_says_its_job_and_a_trainings_epoch(finetuned, launched):
    export = export_web.start_export(finetuned.yaml, finetuned.corrections, finetuned.run)
    launched[-1].status = JobStatus.PENDING
    assert export_web.export_status(export)["job"]["queued"] is True

    # A training (distill's) prints the trainer's epoch lines.
    launched[-1].status = JobStatus.RUNNING
    (export.output_dir / export_web.LOG_NAME).write_text(
        "Starting epoch 1 of 100...\nEpoch 1/100 - Loss: 0.5\nStarting epoch 2 of 100...\n")
    job = export_web.export_status(export)["job"]
    assert (job["id"], job["queued"], job["epoch"], job["epochs"]) == ("123", False, 2, 100)
    assert job["log"] == str(export.log_file)


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


def _middle(a, shape):
    """The middle ``shape`` of ``a``'s last three axes."""
    o = [(n - s) // 2 for n, s in zip(a.shape[-3:], shape)]
    return a[..., o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]]


def _student_outputs(finetuned, run, status, parameters=None):
    """On a block the student was neither trained nor measured on: the
    exported fp32 file's output, the kept student's, a student as it starts
    (the pruned network, or a new random one), and what the server serves
    there, the middle the student writes of the finetuned model's output."""
    import torch

    from cellmap_flow.models.configs.onnx import parse_manifest

    manifest = parse_manifest(status["manifest"])
    # The block it trained at; the others are checked as the export checks them.
    variant = manifest["variants"][0]
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, *variant["input_size"])).astype(np.float32)
    exported = _onnx(run.output_dir / status["folder"] / variant["file"], manifest, x)
    network, model, _ = export_web.finetuned_network(finetuned.yaml)
    geometry = model.base_model_config.geometry
    student = distill.make_student(network, geometry, 2, "unbounded", parameters)
    with torch.no_grad():
        start = student.network(torch.from_numpy(x)).numpy()
        student.network.load_state_dict(torch.load(run.output_dir / distill.CHECKPOINT_NAME)["state_dict"])
        kept = student.network(torch.from_numpy(x)).numpy()
    served = _served(finetuned.yaml, _middle(x, geometry.input_shape))
    return exported, kept, start, _middle(served, exported.shape[2:])


@pytest.mark.finetune
def test_a_funlib_unet_distills_from_its_pruned_copy_into_one_for_the_browser(finetuned, launched):
    pytest.importorskip("cellmap_models.model_export.student")
    pytest.importorskip("onnxruntime")

    run = distill.start_distill(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert export_web.export_status(run)["status"] == "running"
    assert _distilled(finetuned, run, "--epochs", "4", "--steps-per-epoch", "15", "--learning-rate", "1e-3") == 0
    status = export_web.export_status(run)
    assert status["status"] == "done", status["error"]

    # A tenth of the teacher or less, in a folder of the catalog's beside the
    # finetuned model's own, and closer to the teacher on the held-out
    # blocks than the pruned network it started from.
    assert status["folder"] == export_web.folder_name(NAME) + SMALL_SUFFIX
    distillation = status["manifest"]["distillation"]
    assert distillation["init"] == "pruned"
    assert status["manifest"]["params"] <= distillation["teacher_params"] / 10
    assert distillation["distilled"]["mse"] < 0.7 * distillation["start"]["mse"]

    # Its fp32 file computes what the student it kept computes, and nearer
    # what the server serves than the pruned network does.
    exported, kept, start, served = _student_outputs(finetuned, run, status)
    assert np.abs(exported - kept).max() <= 1e-5 * np.abs(kept).max()
    assert np.mean((exported - served) ** 2) < np.mean((start - served) ** 2)

    # Started again, it carries on from the student it kept.
    run = distill.start_distill(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert _distilled(finetuned, run, "--epochs", "1", "--steps-per-epoch", "2") == 0
    again = export_web.export_status(run)["manifest"]["distillation"]
    assert again["steps"] >= distillation["steps"] > 2
    assert again["start"] == distillation["start"]


@pytest.mark.finetune
@pytest.mark.parametrize("finetuned", ["plain2d"], indirect=True)
def test_any_other_network_distills_from_a_random_standard_unet_for_the_browser(finetuned, launched):
    pytest.importorskip("cellmap_models.model_export.student")
    pytest.importorskip("onnxruntime")

    from cellmap_flow.models.configs.onnx import parse_manifest

    run = distill.start_distill(finetuned.yaml, finetuned.corrections, finetuned.run, parameters=20_000)
    assert _distilled(finetuned, run, "--parameters", "20000", "--epochs", "3", "--steps-per-epoch", "10",
                      "--learning-rate", "3e-3") == 0
    status = export_web.export_status(run)
    assert status["status"] == "done", status["error"]
    distillation = status["manifest"]["distillation"]
    assert distillation["init"] == "random"
    assert distillation["distilled"]["mse"] < 0.7 * distillation["start"]["mse"]

    # A 2D StandardUnet whose output, the middle of the teacher's, is the
    # teacher's 1x24x24, read from more raw around it than the teacher reads;
    # exported at larger slices too.
    exported, kept, start, served = _student_outputs(finetuned, run, status, parameters=20_000)
    assert exported.shape == served.shape == (1, 2, 1, 24, 24)
    assert np.abs(exported - kept).max() <= 1e-5 * np.abs(kept).max()
    assert np.mean((exported - served) ** 2) < np.mean((start - served) ** 2)
    sizes = [v["input_size"] for v in parse_manifest(status["manifest"])["variants"]]
    assert len(sizes) > 1 and all(size[0] == 1 for size in sizes)


@pytest.mark.parametrize("finetuned", ["coarse"], indirect=True)
def test_a_model_a_small_copy_cannot_be_made_of_is_refused_before_training(finetuned, launched):
    pytest.importorskip("cellmap_models.model_export.student")

    run = distill.start_distill(finetuned.yaml, finetuned.corrections, finetuned.run)
    assert _distilled(finetuned, run) == 1
    status = export_web.export_status(run)
    assert status["status"] == "failed" and status["error"]
    # The student is kept before the first step: there is none.
    assert not (run.output_dir / distill.CHECKPOINT_NAME).exists()
