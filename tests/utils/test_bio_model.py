"""The bioimage model type, against a stand-in ``bioimageio.core``: what it
serves for a chunk of a 3D model and of a 2D one (its description's sizes and
halo, batched as it allows), its model entry, and what it can be finetuned
with. The real bioimageio.core is in pixi's bioimageio environment only, not
in the test environments."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from cellmap_flow.models import registry
from cellmap_flow.models.models_config import BioModelConfig


# --- descriptions, as bioimageio.spec's 0.5 ModelDescr reads them -------------

def axis(type_, id_=None, size=None, scale=1.0, unit=None, halo=None, channel_names=None):
    return SimpleNamespace(type=type_, id=id_ or type_, size=size, scale=scale, unit=unit, halo=halo,
                           channel_names=channel_names)


def parameterized(min_, step):
    return SimpleNamespace(min=min_, step=step)


def reference(tensor_id, axis_id, offset=0):
    return SimpleNamespace(tensor_id=tensor_id, axis_id=axis_id, offset=offset)


def tensor(id_, axes, dtype="float32", optional=False):
    return SimpleNamespace(id=id_, axes=axes, data=SimpleNamespace(type=dtype), optional=optional)


def unet_3d(channels=2):
    """Like the zoo's 3D EM U-Nets ("conscientious-dromedary"): fixed sizes, batch fixed at 1, no halo."""
    space = [axis("space", a, size) for a, size in zip("zyx", (4, 16, 16))]
    return SimpleNamespace(
        inputs=[tensor("input0", [axis("batch", size=1), axis("channel", channel_names=["raw"]), *space])],
        outputs=[tensor("output0", [axis("batch", size=1),
                                    axis("channel", channel_names=[f"c{i}" for i in range(channels)]), *space])],
    )


def parameterized_2d(batch=None, halo=None, dtype="float32"):
    """Like Empanada's ("stupendous-sheep"): y and x of min + n * step, the output as large as the input."""
    batch_axes = [] if batch == "none" else [axis("batch", size=batch)]
    space = [axis("space", a, parameterized(200, 16)) for a in "yx"]
    out = [axis("space", a, reference("raw", a), halo=halo) for a in "yx"]
    return SimpleNamespace(inputs=[tensor("raw", [*batch_axes, *space])],
                           outputs=[tensor("labels", [*batch_axes, *out], dtype=dtype)])


# --- a stand-in bioimageio.core ------------------------------------------------

class FakeTensor:
    def __init__(self, data, dims):
        self.data, self.dims = data, tuple(dims)

    @classmethod
    def from_numpy(cls, array, dims):
        assert array.ndim == len(dims)
        return cls(array, dims)


class FakeSample:
    def __init__(self, members, stat, id):
        self.members, self.stat, self.id = members, stat, id


class FakePipeline:
    """Each output is its input plus 10 times its channel index, in the output's axes."""

    built = []

    def __init__(self, description, weights_format=None):
        self.description, self.weights_format = description, weights_format
        self.calls = []
        FakePipeline.built.append(self)

    def predict_sample_without_blocking(self, sample, **flags):
        ((_, image),) = sample.members.items()
        self.calls.append((image.data.shape, image.dims, flags))
        members = {}
        for out in self.description.outputs:
            # The input, in the output's axes (the stand-in descriptions'
            # outputs have all of their input's), the channels added.
            dims = [str(a.id) for a in out.axes]
            missing = [d for d in dims if d not in image.dims]
            values = image.data.astype(np.float32).reshape(image.data.shape + (1,) * len(missing))
            values = values.transpose([[*image.dims, *missing].index(d) for d in dims])
            for i, a in enumerate(out.axes):
                if a.type == "channel":
                    n = len(a.channel_names)
                    values = values + 10 * np.arange(n).reshape([n if j == i else 1 for j in range(len(dims))])
            members[str(out.id)] = FakeTensor(values, dims)
        return FakeSample(members, {}, sample.id)


@pytest.fixture
def fake_bioimageio(monkeypatch):
    """bioimageio.core with load_model_description serving ``descriptions`` by name."""
    descriptions = {}
    package = types.ModuleType("bioimageio")
    core = types.ModuleType("bioimageio.core")

    def load_model_description(source, format_version=None, perform_io_checks=None):
        assert format_version == "latest"  # read as 0.5, converting a 0.4 description
        return descriptions[source]

    core.load_model_description = load_model_description
    core.create_prediction_pipeline = FakePipeline
    core.Tensor, core.Sample = FakeTensor, FakeSample
    package.core = core
    monkeypatch.setitem(sys.modules, "bioimageio", package)
    monkeypatch.setitem(sys.modules, "bioimageio.core", core)
    FakePipeline.built = []
    return descriptions


class ArrayIDI:
    """A RawInput over a numpy array at ``voxel_size``, origin 0."""

    def __init__(self, array, voxel_size):
        self.array, self.voxel_size = array, Coordinate(voxel_size)
        self.reads = []

    def to_ndarray_ts(self, roi):
        self.reads.append(roi)
        begin, shape = roi.begin / self.voxel_size, roi.shape / self.voxel_size
        return self.array[tuple(slice(b, b + s) for b, s in zip(begin, shape))]


# --- chunks --------------------------------------------------------------------

def test_the_pipeline_is_built_once_and_each_chunk_is_one_call(fake_bioimageio):
    fake_bioimageio["unet"] = unet_3d()
    model = BioModelConfig(model="unet", voxel_size=8, weight_format="torchscript")
    idi = ArrayIDI(np.arange(8 * 32 * 32, dtype=np.uint8).reshape(8, 32, 32), (8, 8, 8))
    for z in (0, 4):
        out = model.config.process_chunk(idi, Roi((z * 8, 0, 128), (32, 128, 128)))
        assert out.shape == (2, 4, 16, 16) and out.dtype == np.float32
        np.testing.assert_array_equal(out[1], idi.array[z:z + 4, :16, 16:].astype(np.float32) + 10)

    (pipeline,) = FakePipeline.built
    assert pipeline.weights_format == "torchscript"
    # In the model's axes, read with its context already: not padded again.
    assert [call[:2] for call in pipeline.calls] == [((1, 1, 4, 16, 16), ("batch", "channel", "z", "y", "x"))] * 2
    assert pipeline.calls[0][2] == {"skip_input_padding": True, "skip_output_cropping": True}


@pytest.mark.parametrize("batch, calls", [
    pytest.param(None, 1, id="batch-of-any-size"),
    pytest.param(1, 3, id="batch-fixed-at-one"),
    pytest.param("none", 3, id="no-batch-axis"),
])
def test_a_2d_models_slices_are_batched_when_its_rdf_allows(fake_bioimageio, batch, calls):
    fake_bioimageio["2d"] = parameterized_2d(batch=batch, halo=4, dtype="uint16")
    model = BioModelConfig(model="2d", voxel_size=8, input_size=200, slices_per_chunk=3)
    idi = ArrayIDI(np.arange(3 * 200 * 200, dtype=np.uint16).reshape(3, 200, 200) % 1000, (8, 8, 8))

    out = model.config.process_chunk(idi, Roi((0, 32, 32), (24, 192 * 8, 192 * 8)))

    assert idi.reads == [Roi((0, 0, 0), (24, 1600, 1600))]
    assert len(FakePipeline.built[0].calls) == calls
    assert out.shape == (1, 3, 192, 192) and out.dtype == np.uint16
    np.testing.assert_array_equal(out[0], idi.array[:, 4:196, 4:196])


# --- the model entry -----------------------------------------------------------

@pytest.mark.parametrize("key", ["model", "model_name", "model_path"])
def test_an_entry_round_trips_and_keeps_the_old_keys(key):
    entry = {"type": "bioimage", key: "conscientious-dromedary", "voxel_size": [16, 8, 8], "input_size": [20, 256, 256],
             "context": 16, "slices_per_chunk": 4, "weight_format": "onnx", "name": "mito", "scale": "s1"}
    model = registry.build_model(entry, "mito")
    expected = {"type": "bioimage", "model": "conscientious-dromedary", **{k: v for k, v in entry.items()
                                                                           if k not in ("type", key)}}
    assert model.to_dict() == expected
    assert registry.build_model(model.launch_entry, "mito").to_dict() == expected

    # A bare entry writes only what it gave: the rest follows the model.
    assert registry.build_model({"type": "bioimage", key: "m"}, "m").to_dict() == {
        "type": "bioimage", "model": "m", "name": "m"}


# --- what it can be finetuned with ------------------------------------------------

@pytest.mark.parametrize("formats, weight_format, modes", [
    (["onnx", "pytorch"], None, ("lora", "full")),
    (["onnx", "torchscript"], None, ("full",)),  # compiled: no layers to attach adapters to
    (["onnx", "tensorflow"], None, ()),
    (["pytorch", "torchscript"], "torchscript", ("full",)),  # the TorchScript it was asked to run
    (["onnx", "pytorch"], "onnx", ("lora", "full")),  # served from ONNX, trained from the state dict
])
def test_finetune_modes_follow_the_catalogs_weight_formats(monkeypatch, formats, weight_format, modes):
    from cellmap_flow.models import bioimage_catalog

    monkeypatch.setattr(bioimage_catalog, "find_bioimage_model",
                        lambda key: {"key": key, "weight_formats": formats})
    assert BioModelConfig(model="m", weight_format=weight_format).finetune_modes() == modes