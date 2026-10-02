"""The cellpose model type, against a stand-in ``cellpose``: its geometry, how
it batches Cellpose's tiles, its two outputs, and that nothing but building
the model imports cellpose. The real Cellpose 4 is not in the test
environments (it has an environment of its own, cellpose4)."""

import os
import sys
import types

import numpy as np
import pytest
from click.testing import CliRunner
from funlib.geometry import Coordinate, Roi

from cellmap_flow.models import registry
from cellmap_flow.models.configs.cellpose import tiles_per_slice
from cellmap_flow.models.models_config import CellposeModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeCellposeModel:
    """cellpose.models.CellposeModel: the backbone Cellpose would read from
    the weights, and an eval that records its arguments."""

    built = []

    def __init__(self, gpu=False, pretrained_model="cpsam_v2"):
        self.pretrained_model = pretrained_model
        self.backbone = {"cpdino": "dino_vitl", "cpdino-vitb": "dino_vitb"}.get(
            pretrained_model, "dino_vitl" if str(pretrained_model).endswith("dino.pt") else "sam_vitl"
        )
        self.calls = []
        FakeCellposeModel.built.append(self)

    def eval(self, x, channel_axis=None, **kwargs):
        self.calls.append((x.shape, channel_axis, kwargs))
        z, y, xs, _ = x.shape
        # The logit is the voxel's x index, so a crop shows where it came from.
        logits = np.broadcast_to(np.arange(xs, dtype=np.float32) - xs / 2, (z, y, xs)).copy()
        # Two objects in every slice, numbered 1 and 2 from each slice.
        masks = np.zeros((z, y, xs), dtype=np.uint16)
        masks[:, : y // 2] = 1
        masks[:, y // 2 :] = 2
        return masks.squeeze(), [None, None, logits.squeeze()], None


@pytest.fixture
def fake_cellpose(monkeypatch):
    package = types.ModuleType("cellpose")
    models = types.ModuleType("cellpose.models")
    models.MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
    models.get_user_models = lambda: []
    models.CellposeModel = FakeCellposeModel
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    FakeCellposeModel.built = []
    return models


class ArrayIDI:
    """An ImageDataInterface over a numpy array at ``voxel_size``, origin 0."""

    def __init__(self, array, voxel_size):
        self.array, self.voxel_size = array, Coordinate(voxel_size)
        self.reads = []

    def to_ndarray_ts(self, roi):
        self.reads.append(roi)
        begin, shape = roi.begin / self.voxel_size, roi.shape / self.voxel_size
        return self.array[tuple(slice(b, b + s) for b, s in zip(begin, shape))]


@pytest.mark.parametrize("output, dtype", [("probability", np.float32), ("masks", np.uint64)])
def test_the_geometry_is_the_slices_with_context_in_y_and_x(fake_cellpose, output, dtype):
    model = CellposeModelConfig(voxel_size="16,8,8", output=output, slices_per_chunk=4, slice_size=100, context=10)
    config = model.config
    assert config.input_voxel_size == config.output_voxel_size == Coordinate(16, 8, 8)
    assert config.write_shape == Coordinate(4 * 16, 100 * 8, 100 * 8)
    assert config.read_shape == Coordinate(4 * 16, 120 * 8, 120 * 8)
    assert model.geometry.context == Coordinate(0, 80, 80)
    assert config.block_shape.tolist() == [4, 100, 100, 1] and config.output_channels == 1
    assert model.output_dtype is dtype
    assert config.eval_kwargs["compute_masks"] is (output == "masks")


def test_the_example_scripts_chunk_is_three_by_three_cellpose_sam_tiles():
    # 576 px (512 + 2 * 32) is padded to 592, then cut into ceil(1.2 * 592 / 256).
    assert tiles_per_slice(576, 256) == 9
    assert tiles_per_slice(576, 384) == 4
    assert tiles_per_slice(200, 256) == 1  # padded up to one tile
    assert tiles_per_slice(576, 256, rescale=0.5) == 4  # 288 px, padded to 304


@pytest.mark.parametrize(
    "kwargs, bsize, batch_size",
    [
        ({}, 256, 8 * 9),  # cpsam_v2: Cellpose-SAM's 256 px tiles
        ({"pretrained_model": "cpsam"}, 256, 8 * 9),
        ({"pretrained_model": "cpdino"}, 384, 8 * 4),  # DINO: 384 px
        ({"pretrained_model": "cpdino-vitb", "slices_per_chunk": 2}, 384, 2 * 4),
        ({"diameter": 60}, 256, 8 * 4),  # resized by 30 / 60 before tiling
        ({"batch_size": 16}, 256, 16),  # given, it is kept
    ],
    ids=["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb", "diameter", "given"],
)
def test_a_chunks_tiles_go_in_one_pass_at_the_models_tile_size(fake_cellpose, kwargs, bsize, batch_size):
    eval_kwargs = CellposeModelConfig(voxel_size=64, **kwargs).config.eval_kwargs
    assert (eval_kwargs["bsize"], eval_kwargs["batch_size"]) == (bsize, batch_size)


def test_finetuned_weights_are_a_path_and_tile_as_their_network(fake_cellpose, tmp_path):
    weights = tmp_path / "my_dino.pt"
    weights.write_bytes(b"")
    config = CellposeModelConfig(voxel_size=64, pretrained_model=str(weights)).config
    assert config.model.pretrained_model == str(weights) and config.eval_kwargs["bsize"] == 384

    # Cellpose would fall back to cpsam_v2 and serve that.
    with pytest.raises(ValueError, match="neither one of Cellpose's models .* nor an existing file"):
        CellposeModelConfig(voxel_size=64, pretrained_model=str(tmp_path / "missing")).config


def test_the_model_is_built_once(fake_cellpose):
    model = CellposeModelConfig(voxel_size=64)
    assert model.config is model.config and len(FakeCellposeModel.built) == 1


def test_the_probability_is_the_sigmoid_of_the_inner_logits(fake_cellpose):
    model = CellposeModelConfig(voxel_size=8, slices_per_chunk=3, slice_size=20, context=5)
    idi = ArrayIDI(np.zeros((10, 60, 60), dtype=np.uint8), (8, 8, 8))
    roi = Roi((8, 80, 160), (3 * 8, 20 * 8, 20 * 8))

    out = model.config.process_chunk(idi, roi)

    assert idi.reads == [roi.grow(Coordinate(0, 40, 40), Coordinate(0, 40, 40))]
    ((shape, channel_axis, kwargs),) = model.config.model.calls
    assert shape == (3, 30, 30, 1) and channel_axis == 3 and kwargs["batch_size"] == 3
    assert out.shape == (1, 3, 20, 20) and out.dtype == np.float32
    logits = np.arange(5, 25) - 15  # the fake's logit is x - 30 / 2
    np.testing.assert_allclose(out[0, 1, 7], 1 / (1 + np.exp(-logits)), rtol=1e-6)


def test_masks_are_numbered_through_the_chunk(fake_cellpose):
    model = CellposeModelConfig(voxel_size=8, output="masks", slices_per_chunk=3, slice_size=20, context=5)
    idi = ArrayIDI(np.zeros((3, 30, 30), dtype=np.uint8), (8, 8, 8))

    out = model.config.process_chunk(idi, Roi((0, 40, 40), (24, 160, 160)))

    assert out.shape == (1, 3, 20, 20) and out.dtype == np.uint64
    # Each slice's objects 1 and 2 get ids past the slices before.
    assert [sorted(np.unique(s).tolist()) for s in out[0]] == [[1, 2], [3, 4], [5, 6]]


def test_a_single_slice_chunk_is_unsqueezed(fake_cellpose):
    # Cellpose squeezes the output of a one-image batch to (y, x).
    model = CellposeModelConfig(voxel_size=8, output="masks", slices_per_chunk=1, slice_size=20, context=5)
    out = model.config.process_chunk(ArrayIDI(np.zeros((1, 30, 30)), (8, 8, 8)), Roi((0, 40, 40), (8, 160, 160)))
    assert out.shape == (1, 1, 20, 20)


def test_nothing_but_building_the_model_imports_cellpose(monkeypatch):
    # None in sys.modules makes `import cellpose` raise ImportError.
    monkeypatch.setitem(sys.modules, "cellpose", None)
    from cellmap_flow.cli import main

    model = registry.build_model({"type": "cellpose", "voxel_size": 64}, "cp")
    model.to_dict(), model.launch_entry, model.command
    assert "CellposeModelConfig" in registry.describe_types()
    result = CliRunner().invoke(main.cli, ["infer", "cellpose", "--help"])
    assert result.exit_code == 0 and "--pretrained-model" in result.output
    # It runs in cellpose4: a process without cellpose refuses to build it, naming the env.
    from cellmap_flow.models.configs.base import ModelEnvError

    with pytest.raises(ModelEnvError, match="cellpose4"):
        model.config


def test_it_runs_in_the_cellpose4_environment_by_default():
    assert CellposeModelConfig.default_env == "cellpose4"


def test_an_entry_round_trips_through_to_dict_and_the_launch_entry():
    entry = {"type": "cellpose", "voxel_size": [16, 8, 8], "pretrained_model": "/w/my model", "output": "masks",
             "slices_per_chunk": 4, "slice_size": 256, "context": 16, "batch_size": 12, "diameter": 45.0,
             "flow_threshold": 0.5, "cellprob_threshold": -1.0, "name": "cp", "scale": "s2"}
    model = registry.build_model(entry, "cp")
    assert model.to_dict() == entry and list(model.to_dict()) == list(entry)
    assert registry.build_model(model.launch_entry, "cp").to_dict() == entry

    # A bare entry: the defaults, written out, so an exported YAML says what ran.
    assert registry.build_model({"type": "cellpose", "voxel_size": 64}, "cp").to_dict() == {
        "type": "cellpose", "voxel_size": [64, 64, 64], "pretrained_model": "cpsam_v2", "output": "probability",
        "slices_per_chunk": 8, "slice_size": 512, "context": 32, "flow_threshold": 0.4, "cellprob_threshold": 0.0,
        "name": "cp",
    }


def test_the_form_builds_one_from_its_strings():
    model = registry.instantiate_model_config(
        "CellposeModelConfig", {"voxel_size": "8,8,8", "slices_per_chunk": "4", "batch_size": "", "diameter": "40"},
    )
    assert (model.voxel_size, model.slices_per_chunk, model.batch_size, model.diameter) == ((8, 8, 8), 4, None, 40.0)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"type": "cellpose"}, "missing required parameter 'voxel_size'"),
        ({"type": "cellpose", "voxel_size": 64, "output": "labels"}, "output must be one of probability, masks"),
        ({"type": "cellpose", "voxel_size": 64, "slice_size": 0}, "slice_size must be at least 1"),
    ],
    ids=["no-voxel-size", "bad-output", "empty-slices"],
)
def test_a_bad_entry_is_a_config_error(entry, message):
    from cellmap_flow.config.yaml import ConfigError

    with pytest.raises(ConfigError, match=message):
        registry.build_model(entry, "cp")


def test_the_example_yaml_is_a_cellpose_model():
    import yaml

    with open(os.path.join(ROOT, "example", "cellpose_sam.yaml")) as f:
        (model,) = registry.build_models(yaml.safe_load(f)["models"])
    assert isinstance(model, CellposeModelConfig)
    assert (model.pretrained_model, model.output, model.voxel_size) == ("cpsam", "probability", (64, 64, 64))


def test_a_non_integer_voxel_size_is_kept(fake_cellpose):
    """_as_int_tuple truncated 5.24 nm to 5, putting every chunk on the wrong grid."""
    from cellmap_flow.models.configs.cellpose import CellposeModelConfig

    config = CellposeModelConfig(voxel_size="5.24,4,4", slices_per_chunk=2, slice_size=64, context=8).config
    assert (config.input_voxel_size, config.write_shape, config.read_shape) == (
        (5.24, 4, 4), (10.48, 256, 256), (10.48, 320, 320))
