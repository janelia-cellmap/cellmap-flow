"""The bioimage model type, against a stand-in ``bioimageio.core``: the
geometry it reads from a model's description (fixed and parameterized sizes,
halo, 2D and 3D), the dtype it serves, its voxel size, that it builds the
prediction pipeline once, its model entry, and that nothing but building the
model imports bioimageio. The real bioimageio.core is in pixi's bioimageio
environment only, not in the test environments."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from funlib.geometry import Coordinate, Roi

from cellmap_flow.models import registry
from cellmap_flow.models.configs.bio import served_dtype
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


def unet_3d(channels=2, unit=None, scale=1.0):
    """Like the zoo's 3D EM U-Nets ("conscientious-dromedary"): fixed sizes, batch fixed at 1, no halo."""
    space = [axis("space", a, size, scale=scale, unit=unit) for a, size in zip("zyx", (4, 16, 16))]
    return SimpleNamespace(
        inputs=[tensor("input0", [axis("batch", size=1), axis("channel", channel_names=["raw"]), *space])],
        outputs=[tensor("output0", [axis("batch", size=1),
                                    axis("channel", channel_names=[f"c{i}" for i in range(channels)]),
                                    *[axis("space", a, size, scale=scale, unit=unit) for a, size in zip("zyx", (4, 16, 16))]])],
    )


def parameterized_2d(batch=None, halo=None, dtype="float32", unit=None, scale=1.0):
    """Like Empanada's ("stupendous-sheep"): y and x of min + n * step, the output as large as the input."""
    batch_axes = [] if batch == "none" else [axis("batch", size=batch)]
    space = [axis("space", a, parameterized(200, 16), scale=scale, unit=unit) for a in "yx"]
    out = [axis("space", a, reference("raw", a), scale=scale, unit=unit, halo=halo) for a in "yx"]
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
    """An ImageDataInterface over a numpy array at ``voxel_size``, origin 0."""

    def __init__(self, array, voxel_size):
        self.array, self.voxel_size = array, Coordinate(voxel_size)
        self.reads = []

    def to_ndarray_ts(self, roi):
        self.reads.append(roi)
        begin, shape = roi.begin / self.voxel_size, roi.shape / self.voxel_size
        return self.array[tuple(slice(b, b + s) for b, s in zip(begin, shape))]


# --- geometry ------------------------------------------------------------------

def test_a_3d_model_of_fixed_size_is_one_tile_a_chunk(fake_bioimageio):
    fake_bioimageio["unet"] = unet_3d()
    model = BioModelConfig(model="unet", voxel_size="16,8,8", input_size=128)
    config = model.config
    # Its fixed sizes win over input_size, and without a halo nothing is cut off.
    assert config.input_voxel_size == config.output_voxel_size == (16, 8, 8)
    assert config.read_shape == config.write_shape == (4 * 16, 16 * 8, 16 * 8)
    assert model.geometry.context == Coordinate(0, 0, 0)
    assert config.block_shape.tolist() == [4, 16, 16, 2] and config.channels == ["c0", "c1"]
    assert model.output_dtype == np.float32


@pytest.mark.parametrize("input_size, tile", [
    (None, 264),  # the smallest of 200 + n * 16 of at least 256, the 2D default
    (300, 312),
    ("100", 200),  # the smallest it takes
])
def test_a_parameterized_2d_model_takes_the_size_asked_for_rounded_up(fake_bioimageio, input_size, tile):
    fake_bioimageio["sheep"] = parameterized_2d(halo=8)
    model = BioModelConfig(model="sheep", voxel_size=8, input_size=input_size, slices_per_chunk=4)
    config = model.config
    # Its RDF's halo of 8 is cut off each side in y and x; each slice is on its own, so none in z.
    assert config.read_shape == (4 * 8, tile * 8, tile * 8)
    assert config.write_shape == (4 * 8, (tile - 16) * 8, (tile - 16) * 8)
    assert model.geometry.context == Coordinate(0, 64, 64)
    assert config.block_shape.tolist() == [4, tile - 16, tile - 16, 1]


def test_context_replaces_the_halo(fake_bioimageio):
    fake_bioimageio["sheep"] = parameterized_2d(halo=8)
    config = BioModelConfig(model="sheep", voxel_size=8, input_size=200, context="2,20").config
    assert config.write_shape == (8 * 8, 196 * 8, 160 * 8) and config.context == (0, 16, 160)

    fake_bioimageio["unet"] = unet_3d()
    with pytest.raises(ValueError, match="once its context is cut off"):
        BioModelConfig(model="unet", voxel_size=8, context=8).config


def test_the_voxel_size_comes_from_the_rdf_unless_given(fake_bioimageio):
    fake_bioimageio["um"] = unet_3d(unit="micrometer", scale=0.008)
    assert BioModelConfig(model="um").config.input_voxel_size == (8, 8, 8)
    assert BioModelConfig(model="um", voxel_size="5.24,4,4").config.input_voxel_size == (5.24, 4, 4)
    # A 2D model's slices are taken to be as far apart as its y voxels.
    fake_bioimageio["2d"] = parameterized_2d(unit="nanometer", scale=6)
    assert BioModelConfig(model="2d").config.output_voxel_size == (6, 6, 6)

    fake_bioimageio["no unit"] = unet_3d()
    with pytest.raises(ValueError, match="no physical voxel size .* give voxel_size"):
        BioModelConfig(model="no unit").config


@pytest.mark.parametrize("dtypes, served", [
    (["float32"], np.float32),
    (["float64"], np.float32),
    (["uint8"], np.uint8),
    (["uint16"], np.uint16),
    (["int64"], np.uint64),  # neuroglancer has no int64
    (["bool"], np.uint8),
    (["uint8", "float32"], np.float32),
])
def test_the_output_is_float32_unless_the_rdf_says_integers(dtypes, served):
    outputs = [SimpleNamespace(data=SimpleNamespace(type=t)) for t in dtypes]
    assert served_dtype(outputs) == np.dtype(served)


@pytest.mark.parametrize("description, message", [
    pytest.param(SimpleNamespace(inputs=[tensor("a", [axis("space", "y", 8), axis("space", "x", 8)])] * 2,
                                 outputs=[]), "needs 2 inputs", id="two-inputs"),
    pytest.param(SimpleNamespace(inputs=[tensor("image", [axis("channel", channel_names=["R", "G", "B"]),
                                                          axis("space", "y", 8), axis("space", "x", 8)])],
                                 outputs=[]), "takes 3 input channels", id="rgb"),
    # micro-SAM's masks: one per object, so many as there are objects.
    pytest.param(SimpleNamespace(inputs=[tensor("image", [axis("space", "y", 8), axis("space", "x", 8)])],
                                 outputs=[tensor("masks", [axis("index", "object", SimpleNamespace(min=1, max=None)),
                                                           axis("space", "y", 8), axis("space", "x", 8)])]),
                 "cannot map onto an image: object", id="object-axis"),
])
def test_what_it_cannot_run_is_refused_when_built(fake_bioimageio, description, message):
    fake_bioimageio["m"] = description
    with pytest.raises(ValueError, match=message):
        BioModelConfig(model="m", voxel_size=8).config


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
    assert model.to_dict() == expected and list(model.to_dict()) == list(expected)
    assert registry.build_model(model.launch_entry, "mito").to_dict() == expected

    # A bare entry writes only what it gave: the rest follows the model.
    assert registry.build_model({"type": "bioimage", key: "m"}, "m").to_dict() == {
        "type": "bioimage", "model": "m", "name": "m"}


def test_a_weight_format_it_does_not_know_is_refused():
    with pytest.raises(ValueError, match="weight_format must be one of"):
        BioModelConfig(model="m", weight_format="pytorch")


def test_nothing_but_building_the_model_imports_bioimageio(monkeypatch):
    # None in sys.modules makes `import bioimageio` raise ImportError.
    monkeypatch.setitem(sys.modules, "bioimageio", None)
    from cellmap_flow.cli import main
    from cellmap_flow.models.configs.base import ModelEnvError

    model = registry.build_model({"type": "bioimage", "model": "affable-shark", "voxel_size": 8}, "bio")
    model.to_dict(), model.launch_entry, model.command
    assert "BioModelConfig" in registry.describe_types()
    result = CliRunner().invoke(main.cli, ["infer", "bioimage", "--help"])
    assert result.exit_code == 0 and "--weight-format" in result.output
    with pytest.raises((ModelEnvError, ImportError)):
        model.config
