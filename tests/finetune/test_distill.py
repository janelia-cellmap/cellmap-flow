"""A finetuned model is distilled into a small copy for the browser
(distill): the copy, pruned from a funlib UNet or a random StandardUnet for
any other network, computes nearer the finetuned model than it started,
exports for the browser as export_web exports a model, and a model a small
copy cannot be made of is refused before any training."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.finetune import distill, export_web
from cellmap_flow.jobs.spec import JobStatus

# funlib.learn.torch's UNet as its state dict lays it out, in a StandardUnet
# as fly_organelles wraps one: two levels of two valid convolutions, a 2x max
# pool, a 2x transposed convolution back up, the skip cropped to it and
# concatenated first, and a final 1x1 convolution. 28^3 in, 12^3 out.
FUNLIB = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn


class ConvPass(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv_pass = nn.Sequential(nn.Conv3d(inputs, outputs, 3), nn.ReLU(),
                                       nn.Conv3d(outputs, outputs, 3), nn.ReLU())

    def forward(self, x):
        return self.conv_pass(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(channels, channels, 2, stride=2)

    def forward(self, left, below):
        up = self.up(below)
        c = (left.shape[2] - up.shape[2]) // 2
        return torch.cat([left[:, :, c:c + up.shape[2], c:c + up.shape[3], c:c + up.shape[4]], up], 1)


class UNet(nn.Module):
    def __init__(self, num_fmaps, fmap_inc_factor):
        super().__init__()
        top, below = num_fmaps, num_fmaps * fmap_inc_factor
        self.l_conv = nn.ModuleList([ConvPass(1, top), ConvPass(top, below)])
        self.l_down = nn.ModuleList([nn.MaxPool3d(2)])
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
read_shape = Coordinate(28, 28, 28) * input_voxel_size
write_shape = Coordinate(12, 12, 12) * output_voxel_size
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

# Not a funlib UNet. 28^3 in, 24^3 out; PLAIN_2D below is the test's.
PLAIN = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

input_voxel_size = output_voxel_size = Coordinate(16, 16, 16)
read_shape = Coordinate(28, 28, 28) * input_voxel_size
write_shape = Coordinate(24, 24, 24) * output_voxel_size
output_channels = 2
channels = ["mito_aff_y", "mito_aff_x"]
torch.manual_seed(0)
model = nn.Sequential(nn.Conv3d(1, 4, 3), nn.ReLU(), nn.Conv3d(4, 2, 3))
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

# The plain network in 2D, on one z-slice. 1x28x28 in, 1x24x24 out.
PLAIN_2D = (PLAIN.replace("Coordinate(28, 28, 28)", "Coordinate(1, 28, 28)")
            .replace("Coordinate(24, 24, 24)", "Coordinate(1, 24, 24)")
            .replace("nn.Conv3d(1, 4, 3)", "nn.Conv3d(1, 4, (1, 3, 3))")
            .replace("nn.Conv3d(4, 2, 3)", "nn.Conv3d(4, 2, (1, 3, 3))"))

# Each teacher's script and its block, in voxels in and out.
SCRIPTS = {"script": (FUNLIB, (28,) * 3, (12,) * 3), "cellmap": (FUNLIB, (28,) * 3, (12,) * 3),
           "plain2d": (PLAIN_2D, (1, 28, 28), (1, 24, 24)),
           "coarse": (COARSE, (28,) * 3, (14,) * 3)}

INPUT_NORM = {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}, "LambdaNormalizer": {"expression": "x*2-1"}}


def _cellmap_folder(path, model, size, out):
    """``model`` exported as cellmap_models exports one: TorchScript to serve,
    and the torch.export program the trainer unflattens and trains."""
    import torch

    x = torch.rand(1, 1, size, size, size)
    path.mkdir()
    with torch.no_grad():
        torch.jit.trace(model.eval(), x).save(str(path / "model.ts"))
    torch.export.save(torch.export.export(model, (x,)), str(path / "model.pt2"))
    (path / "metadata.json").write_text(json.dumps({
        "model_name": "big", "model_type": "UNet", "framework": "torch", "spatial_dims": 3,
        "in_channels": 1, "out_channels": 2, "iteration": 1, "channels_names": ["mito_aff_y", "mito_aff_x"],
        "input_voxel_size": [16] * 3, "output_voxel_size": [16] * 3,
        "input_shape": [size] * 3, "output_shape": [out] * 3,
    }))
    return {"type": "cellmap", "folder_path": str(path), "name": "big"}


@pytest.fixture(params=["script", "cellmap"])
def teacher(request, tmp_path, annotation_volume):
    """A LoRA finetune of a network, as a run leaves it: its adapter, the
    YAML that serves it, and the session's corrections it was trained on.

    "script" and "cellmap" are the funlib UNet as a script model and as a
    cellmap folder (trained as the unflattened torch.export program, as the
    large cellmap UNets are); "plain2d" is a 2D network that is not a funlib
    UNet (2D, as a 3D StandardUnet student of one takes a minute on the CPU);
    "coarse" one that writes at another voxel size than it reads.
    """
    pytest.importorskip("peft")
    import torch

    from cellmap_flow.finetune.finetuned_model_templates import generate_finetuned_model_yaml
    from cellmap_flow.finetune.lora_wrapper import save_lora_adapter, wrap_model_with_lora
    from cellmap_flow.finetune.model_loading import load_trainable_model, model_config_from_entry
    from cellmap_flow.finetune.session.manifest import write_manifest

    text, size, out = SCRIPTS[request.param]
    script = tmp_path / "big_unet.py"
    script.write_text(text)
    base = {"type": "script", "script_path": str(script), "name": "big"}
    if request.param == "cellmap":
        base = _cellmap_folder(tmp_path / "big", model_config_from_entry(base).config.model, size[0], out[0])
    peft = wrap_model_with_lora(load_trainable_model(model_config_from_entry(base)),
                                lora_r=2, lora_alpha=4, lora_dropout=0.0)
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.fill_(0.1)
    save_lora_adapter(peft, str(tmp_path / "run" / "lora_adapter"))

    # A random StandardUnet reads 1x146x146 for the plain teacher's 1x24x24:
    # a volume about as large, so that its blocks are mostly raw, not edge.
    side = 128 if request.param.startswith("plain") else 48
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
        model_name="big_finetuned_20261026_120000", output_path=tmp_path / "session" / "models" / "big.yaml",
        data_path=volume.raw, json_data={"input_norm": INPUT_NORM, "postprocess": {}},
    )
    return SimpleNamespace(yaml=yaml_path, corrections=corrections, run=tmp_path / "session" / "runs" / "run")


class FakeJob:
    """A launched job, running until the test says otherwise."""

    def get_status(self):
        return JobStatus.RUNNING


@pytest.fixture
def launched(monkeypatch):
    """start_distill launches nothing: each job is a FakeJob."""
    from cellmap_flow.finetune.job_manager import submit

    monkeypatch.setattr(submit, "launch", lambda name, command, **settings: FakeJob())


def _run(teacher, run, *flags):
    """The job start_distill started, run here, briefly, on the CPU."""
    return distill.main(["--yaml", str(teacher.yaml), "--corrections", str(teacher.corrections),
                         "--output", str(run.output_dir), "--workers", "0", "--variants", "fp32", *flags])


def _middle(a, shape):
    """The middle ``shape`` of ``a``'s last three axes."""
    o = [(n - s) // 2 for n, s in zip(a.shape[-3:], shape)]
    return a[..., o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]]


def _outputs(teacher, run, status, parameters=None):
    """On a block the student was neither trained nor measured on: the
    exported fp32 file's output, the kept student's, a student as it starts
    (the pruned network, or a new random one), and what the server serves
    there, the middle the student writes of the finetuned model's output."""
    import onnxruntime as ort
    import torch

    from cellmap_flow.finetune.model_loading import model_config_from_entry
    from cellmap_flow.models.configs.onnx import parse_manifest

    manifest = parse_manifest(status["manifest"])
    # The block it trained at; the others are checked as the export checks them.
    variant = manifest["variants"][0]
    x = np.random.default_rng(1).uniform(-1, 1, (1, 1, *variant["input_size"])).astype(np.float32)
    session = ort.InferenceSession(str(run.output_dir / status["folder"] / variant["file"]),
                                   providers=["CPUExecutionProvider"])
    exported = session.run([manifest["io"]["output"]], {manifest["io"]["input"]: x})[0]
    network, finetuned, _ = export_web.finetuned_network(teacher.yaml)
    geometry = finetuned.base_model_config.geometry
    student = distill.make_student(network, geometry, 2, "unbounded", parameters)
    served_model = model_config_from_entry(export_web.serving_entry(teacher.yaml)[1]).config.model
    with torch.no_grad():
        start = student.network(torch.from_numpy(x)).numpy()
        student.network.load_state_dict(torch.load(run.output_dir / distill.CHECKPOINT_NAME)["state_dict"])
        kept = student.network(torch.from_numpy(x)).numpy()
        served = served_model(torch.from_numpy(_middle(x, geometry.input_shape))).numpy()
    return exported, kept, start, _middle(served, exported.shape[2:])


@pytest.mark.finetune
def test_a_funlib_unet_distills_from_its_pruned_copy_into_one_for_the_browser(teacher, launched):
    pytest.importorskip("cellmap_models.model_export.student")
    pytest.importorskip("onnxruntime")

    run = distill.start_distill(teacher.yaml, teacher.corrections, teacher.run)
    assert export_web.export_status(run)["status"] == "running"
    assert _run(teacher, run, "--epochs", "4", "--steps-per-epoch", "15", "--learning-rate", "1e-3") == 0
    status = export_web.export_status(run)
    assert status["status"] == "done", status["error"]

    # A tenth of the teacher or less, in a folder of the catalog's beside the
    # finetuned model's own, and closer to the teacher on the held-out
    # blocks than the pruned network it started from.
    assert status["folder"] == export_web.folder_name("big_finetuned_20261026_120000") + distill.SUFFIX
    distillation = status["manifest"]["distillation"]
    assert distillation["init"] == "pruned"
    assert status["manifest"]["params"] <= distillation["teacher_params"] / 10
    assert distillation["distilled"]["mse"] < 0.7 * distillation["start"]["mse"]

    # Its fp32 file computes what the student it kept computes, and nearer
    # what the server serves than the pruned network does.
    exported, kept, start, served = _outputs(teacher, run, status)
    assert np.abs(exported - kept).max() <= 1e-5 * np.abs(kept).max()
    assert np.mean((exported - served) ** 2) < np.mean((start - served) ** 2)

    # Started again, it carries on from the student it kept.
    run = distill.start_distill(teacher.yaml, teacher.corrections, teacher.run)
    assert _run(teacher, run, "--epochs", "1", "--steps-per-epoch", "2") == 0
    again = export_web.export_status(run)["manifest"]["distillation"]
    assert again["steps"] >= distillation["steps"] > 2
    assert again["start"] == distillation["start"]


@pytest.mark.finetune
@pytest.mark.parametrize("teacher", ["plain2d"], indirect=True)
def test_any_other_network_distills_from_a_random_standard_unet_for_the_browser(teacher, launched):
    pytest.importorskip("cellmap_models.model_export.student")
    pytest.importorskip("onnxruntime")

    from cellmap_flow.models.configs.onnx import parse_manifest

    run = distill.start_distill(teacher.yaml, teacher.corrections, teacher.run, parameters=20_000)
    assert _run(teacher, run, "--parameters", "20000", "--epochs", "3", "--steps-per-epoch", "10",
                "--learning-rate", "3e-3") == 0
    status = export_web.export_status(run)
    assert status["status"] == "done", status["error"]
    distillation = status["manifest"]["distillation"]
    assert distillation["init"] == "random"
    assert distillation["distilled"]["mse"] < 0.7 * distillation["start"]["mse"]

    # A 2D StandardUnet whose output, the middle of the teacher's, is the
    # teacher's 1x24x24, read from more raw around it than the teacher reads;
    # exported at larger slices too.
    exported, kept, start, served = _outputs(teacher, run, status, parameters=20_000)
    assert exported.shape == served.shape == (1, 2, 1, 24, 24)
    assert np.abs(exported - kept).max() <= 1e-5 * np.abs(kept).max()
    assert np.mean((exported - served) ** 2) < np.mean((start - served) ** 2)
    sizes = [v["input_size"] for v in parse_manifest(status["manifest"])["variants"]]
    assert len(sizes) > 1 and all(size[0] == 1 for size in sizes)


@pytest.mark.parametrize("teacher", ["coarse"], indirect=True)
def test_a_model_a_small_copy_cannot_be_made_of_is_refused_before_training(teacher, launched):
    pytest.importorskip("cellmap_models.model_export.student")

    run = distill.start_distill(teacher.yaml, teacher.corrections, teacher.run)
    assert _run(teacher, run) == 1
    status = export_web.export_status(run)
    assert status["status"] == "failed" and status["error"]
    # The student is kept before the first step: there is none.
    assert not (run.output_dir / distill.CHECKPOINT_NAME).exists()
