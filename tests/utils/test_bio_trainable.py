"""The bioimage type's finetuning, against a stand-in ``bioimageio.core``
whose pipeline runs a small torch network the way the real one does: the
input's preprocessing in numpy, the network on the input's axes, each
output's postprocessing. ``trainable_model()`` must give what process_chunk
serves, for a 3D and a 2D model, with normalization, a halo and a sigmoid;
``serve_trained`` must make the inferencer's own forward serve the same; a
model it cannot train is refused with the reason; gradients and a LoRA
adapter reach the network."""

import copy
import io
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from funlib.geometry import Roi

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from cellmap_flow.models.models_config import BioModelConfig  # noqa: E402
from tests.utils.test_bio_model import (  # noqa: E402,F401  (fake_bioimageio is a fixture)
    ArrayIDI,
    FakeSample,
    FakeTensor,
    axis,
    fake_bioimageio,
    parameterized,
    reference,
)


# --- descriptions with processing and weights ------------------------------------

def op(id_, **kwargs):
    return SimpleNamespace(id=id_, kwargs=SimpleNamespace(**kwargs))


def zmuv(axes=None):
    return op("zero_mean_unit_variance", axes=axes, eps=1e-6)


def tensor(id_, axes, dtype="float32", preprocessing=(), postprocessing=()):
    return SimpleNamespace(id=id_, axes=axes, data=SimpleNamespace(type=dtype), optional=False,
                           preprocessing=list(preprocessing), postprocessing=list(postprocessing))


def weights(net, formats=("pytorch_state_dict",)):
    """The description's weights: each format a stand-in for ``net``'s file."""
    buffer = io.BytesIO()
    if "torchscript" in formats:
        torch.jit.save(torch.jit.script(net), buffer)
    specs = {f: SimpleNamespace(net=net, get_reader=lambda: io.BytesIO(buffer.getvalue())) for f in formats}
    return SimpleNamespace(**specs)


def unet_3d(net, channels=2, halo=2, preprocessing=(zmuv(["channel", "z", "y", "x"]),),
            postprocessing=(op("sigmoid"),), formats=("pytorch_state_dict",), dtype="float32"):
    """Like "conscientious-dromedary": fixed sizes, batch fixed at 1, zero mean unit variance and a sigmoid."""
    space = [axis("space", a, size) for a, size in zip("zyx", (4, 16, 16))]
    out_space = [axis("space", a, size, halo=halo if a != "z" else None) for a, size in zip("zyx", (4, 16, 16))]
    return SimpleNamespace(
        inputs=[tensor("input0", [axis("batch", size=1), axis("channel", channel_names=["raw"]), *space],
                       preprocessing=preprocessing)],
        outputs=[tensor("output0", [axis("batch", size=1),
                                    axis("channel", channel_names=[f"c{i}" for i in range(channels)]), *out_space],
                        dtype=dtype, postprocessing=postprocessing)],
        weights=weights(net, formats),
    )


def two_outputs_2d(net, batch=None, axes=("channel", "y", "x"), halo=4):
    """A 2D model of two outputs: a foreground with a sigmoid, and a distance without a channel axis, scaled."""
    batch_axes = [axis("batch", size=batch)]
    space = [axis("space", a, parameterized(32, 8)) for a in "yx"]
    out = [axis("space", a, reference("raw", a), halo=halo) for a in "yx"]
    return SimpleNamespace(
        inputs=[tensor("raw", [*batch_axes, axis("channel", channel_names=["raw"]), *space],
                       preprocessing=[op("ensure_dtype", dtype="float32"), zmuv(None if axes is None else list(axes))])],
        outputs=[tensor("fg", [*batch_axes, axis("channel", channel_names=["fg"]), *out],
                        postprocessing=[op("sigmoid")]),
                 tensor("dist", [*batch_axes, *out], postprocessing=[op("scale_linear", gain=2.0, offset=-1.0)])],
        weights=weights(net),
    )


class Net3d(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.conv = nn.Conv3d(1, 8, 3, padding=1)
        self.head = nn.Conv3d(8, channels, 1)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)))


class Net2dTwoOutputs(nn.Module):
    """(fg (N, 1, Y, X), dist (N, Y, X)), as the description's two outputs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        out = self.head(torch.relu(self.conv(x)))
        return out[:, :1], out[:, 1]


# --- a stand-in pipeline that runs the network -------------------------------------

def _apply(ops, array, dims):
    """The processing bioimageio.core applies, in numpy: statistics of this call's tensor."""
    for step in ops:
        kwargs = step.kwargs
        if step.id == "zero_mean_unit_variance":
            over = tuple(dims.index(a) for a in kwargs.axes) if kwargs.axes else None
            mean, std = array.mean(axis=over, keepdims=True), array.std(axis=over, keepdims=True)
            array = (array - mean) / (std + kwargs.eps)
        elif step.id == "sigmoid":
            array = 1 / (1 + np.exp(-array))
        elif step.id == "scale_linear":
            array = array * kwargs.gain + kwargs.offset
        elif step.id == "ensure_dtype":
            array = array.astype(kwargs.dtype)
        else:
            raise NotImplementedError(step.id)
    return array


class NetworkPipeline:
    built = []

    def __init__(self, description, weights_format=None):
        self.description = description
        spec = getattr(description.weights, "pytorch_state_dict", None) or description.weights.torchscript
        self.net = copy.deepcopy(spec.net).eval()
        self.calls = []
        NetworkPipeline.built.append(self)

    def predict_sample_without_blocking(self, sample, **flags):
        ((input_id, image),) = sample.members.items()
        self.calls.append(image.data.shape)
        (described,) = self.description.inputs
        data = _apply(described.preprocessing, image.data.astype(np.float32), list(image.dims))
        with torch.no_grad():
            out = self.net(torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)))
        out = out if isinstance(out, tuple) else (out,)
        members = {}
        for tensor_descr, array in zip(self.description.outputs, out):
            dims = [str(a.id) for a in tensor_descr.axes]
            members[str(tensor_descr.id)] = FakeTensor(_apply(tensor_descr.postprocessing, array.numpy(), dims), dims)
        return FakeSample(members, {}, sample.id)


@pytest.fixture
def fake_zoo(fake_bioimageio, monkeypatch):  # noqa: F811  (the fixture imported above)
    """fake_bioimageio, with a pipeline that runs the network and bioimageio's torch weight loader."""
    core = sys.modules["bioimageio.core"]
    monkeypatch.setattr(core, "create_prediction_pipeline", NetworkPipeline, raising=False)
    backends = types.ModuleType("bioimageio.core.backends")
    pytorch_backend = types.ModuleType("bioimageio.core.backends.pytorch_backend")
    # A network of its own, as load_torch_model builds one from the file.
    pytorch_backend.load_torch_model = lambda spec, load_state=True, devices=None: copy.deepcopy(spec.net).to(devices[0])
    backends.pytorch_backend = pytorch_backend
    monkeypatch.setitem(sys.modules, "bioimageio.core.backends", backends)
    monkeypatch.setitem(sys.modules, "bioimageio.core.backends.pytorch_backend", pytorch_backend)
    NetworkPipeline.built = []
    return fake_bioimageio


def _served_and_trained(model, idi, roi):
    """(process_chunk's chunk, the trainable model's output for the same read), as numpy."""
    served = model.config.process_chunk(idi, roi)
    module = model.trainable_model()
    read = np.asarray(idi.to_ndarray_ts(roi.grow(model.config.context, model.config.context)), dtype=np.float32)
    device = next(module.parameters()).device
    with torch.no_grad():
        trained = module(torch.from_numpy(read)[None, None].to(device))[0].cpu().numpy()
    return served, trained, module


# --- the module matches what is served ---------------------------------------------

def test_a_3d_models_trainable_module_gives_what_process_chunk_serves(fake_zoo):
    torch.manual_seed(0)
    fake_zoo["unet"] = unet_3d(Net3d())
    model = BioModelConfig(model="unet", voxel_size=8)
    idi = ArrayIDI(np.random.default_rng(0).normal(100, 20, (8, 32, 32)).astype(np.float32), (8, 8, 8))

    served, trained, module = _served_and_trained(model, idi, Roi((0, 16, 16), (32, 96, 96)))

    # The halo of 2 is cut off in y and x: 16 - 4 voxels written.
    assert served.shape == trained.shape == (2, 4, 12, 12)
    np.testing.assert_allclose(trained, served, atol=1e-5)
    assert 0 < trained.min() and trained.max() < 1  # the sigmoid is in the module


@pytest.mark.parametrize("batch, axes", [
    pytest.param(None, ("channel", "y", "x"), id="slices-in-one-call-each-normalized"),
    pytest.param(None, ("batch", "channel", "y", "x"), id="slices-in-one-call-normalized-together"),
    pytest.param(1, None, id="a-call-a-slice"),
])
def test_a_2d_models_trainable_module_gives_what_process_chunk_serves(fake_zoo, batch, axes):
    torch.manual_seed(0)
    fake_zoo["2d"] = two_outputs_2d(Net2dTwoOutputs(), batch=batch, axes=axes)
    model = BioModelConfig(model="2d", voxel_size=8, input_size=40, slices_per_chunk=3)
    # Slices of different brightness, so per-slice and joint statistics differ.
    rng = np.random.default_rng(1)
    array = (rng.normal(0, 1, (3, 40, 40)) * [[[5]], [[20]], [[50]]] + [[[0]], [[100]], [[300]]]).astype(np.float32)
    idi = ArrayIDI(array, (8, 8, 8))

    served, trained, _ = _served_and_trained(model, idi, Roi((0, 32, 32), (24, 32 * 8, 32 * 8)))

    # fg then dist, the halo of 4 cut off each side.
    assert model.config.channels == ["fg_fg", "dist"]
    assert served.shape == trained.shape == (2, 3, 32, 32)
    np.testing.assert_allclose(trained, served, atol=1e-5)
    assert len(NetworkPipeline.built[0].calls) == (1 if batch is None else 3)


def test_serve_trained_makes_the_inferencers_forward_serve_the_trained_module(fake_zoo):
    from cellmap_flow.inference.runner import ModelRunner

    torch.manual_seed(0)
    fake_zoo["2d"] = two_outputs_2d(Net2dTwoOutputs())
    model = BioModelConfig(model="2d", voxel_size=8, input_size=40, slices_per_chunk=3)
    idi = ArrayIDI(np.random.default_rng(2).normal(50, 10, (3, 40, 40)).astype(np.float32), (8, 8, 8))
    roi = Roi((0, 32, 32), (24, 32 * 8, 32 * 8))
    served = model.config.process_chunk(idi, roi)

    model.serve_trained(model.config, model.trainable_model())
    runner = ModelRunner(model)  # puts it on the device, warms it up and checks its output's shape
    predicted = runner.predict(idi, roi)

    assert model.config.process_chunk is None
    assert predicted.dtype == np.float32 and predicted.shape == served.shape
    np.testing.assert_allclose(predicted, served, atol=1e-5)


def test_torchscript_weights_are_trained_when_there_is_no_state_dict(fake_zoo):
    torch.manual_seed(0)
    fake_zoo["unet"] = unet_3d(Net3d(), formats=("torchscript", "onnx"))
    model = BioModelConfig(model="unet", voxel_size=8)
    idi = ArrayIDI(np.random.default_rng(0).normal(0, 1, (4, 16, 16)).astype(np.float32), (8, 8, 8))

    served, trained, module = _served_and_trained(model, idi, Roi((0, 16, 16), (32, 96, 96)))

    np.testing.assert_allclose(trained, served, atol=1e-5)
    assert any(isinstance(m, torch.jit.ScriptModule) for m in module.modules())


# --- what it refuses ----------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    pytest.param(dict(formats=("onnx", "tensorflow_saved_model_bundle")),
                 "onnx, tensorflow_saved_model_bundle weights only.*torch module", id="no-torch-weights"),
    pytest.param(dict(postprocessing=(op("sigmoid"), op("binarize", threshold=0.5))),
                 "binarize thresholds it: no gradient", id="binarize"),
    pytest.param(dict(dtype="uint16", postprocessing=()), "outputs are uint16 .labels", id="labels"),
    pytest.param(dict(postprocessing=(op("stardist_postprocessing"),)), "makes instances outside the network",
                 id="stardist"),
    pytest.param(dict(postprocessing=(zmuv(),)), "normalizes it by its own statistics", id="output-statistics"),
])
def test_what_cannot_be_finetuned_is_refused_with_the_reason(fake_zoo, kwargs, message):
    fake_zoo["unet"] = unet_3d(Net3d(), **kwargs)
    with pytest.raises(ValueError, match=f"unet cannot be finetuned: .*{message}"):
        BioModelConfig(model="unet", voxel_size=8).trainable_model()


# --- training -----------------------------------------------------------------------

def test_gradients_reach_every_parameter_of_the_network(fake_zoo):
    fake_zoo["2d"] = two_outputs_2d(Net2dTwoOutputs())
    module = BioModelConfig(model="2d", voxel_size=8, input_size=40, slices_per_chunk=3).trainable_model()
    device = next(module.parameters()).device

    module.train()
    module(torch.randn(2, 1, 3, 40, 40, device=device)).square().mean().backward()

    named = dict(module.named_parameters())
    assert named and all(p.grad is not None and p.grad.abs().sum() > 0 for p in named.values())


@pytest.mark.finetune
def test_a_lora_adapter_on_the_trainable_module_trains(fake_zoo):
    from cellmap_flow.finetune.adaptation import LoraStrategy
    from cellmap_flow.finetune.trainable import finetune_modes

    torch.manual_seed(0)
    fake_zoo["unet"] = unet_3d(Net3d())
    module = BioModelConfig(model="unet", voxel_size=8).trainable_model()
    assert finetune_modes(module) == ("lora", "full")
    device = next(module.parameters()).device
    model = LoraStrategy(r=4, alpha=8, dropout=0).prepare(module)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.5)
    x = torch.randn(2, 1, 4, 16, 16, device=device)
    target = torch.zeros(2, 2, 4, 12, 12, device=device)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert trainable and len(trainable) < len(list(model.parameters()))  # the adapter only
    assert losses[-1] < losses[0]
