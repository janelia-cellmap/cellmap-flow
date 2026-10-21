"""The cellpose model type, against a stand-in ``cellpose``: what it serves
for a chunk of slices (flows, masks) and for a 3D chunk, that nothing but
building the model imports cellpose, and how its flows layer is shown. The real Cellpose 4 is not in the test
environments (it has an environment of its own, cellpose4)."""

import sys
import types

import numpy as np
import pytest
from click.testing import CliRunner
from funlib.geometry import Coordinate, Roi

from cellmap_flow.models import registry
from cellmap_flow.models.configs.cellpose import OUTPUT_CHANNELS
from cellmap_flow.models.models_config import CellposeModelConfig


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
        # The logit is the voxel's x index, so a crop shows where it came from;
        # the flows are its z (flowY) and y (flowX) index, likewise, and in 3D
        # its z, y and x index (flowZ, flowY, flowX).
        logits = np.broadcast_to(np.arange(xs, dtype=np.float32) - xs / 2, (z, y, xs)).copy()
        index = [np.broadcast_to(np.arange(n, dtype=np.float32).reshape(shape), (z, y, xs))
                 for n, shape in ((z, (-1, 1, 1)), (y, (1, -1, 1)), (xs, (1, 1, -1)))]
        dp = np.stack(index if kwargs.get("do_3D") else index[:2])
        # Two objects in every slice, numbered 1 and 2 from each slice; linked
        # (stitch_threshold), Cellpose gives them one id through the stack.
        masks = np.zeros((z, y, xs), dtype=np.uint16)
        masks[:, : y // 2] = 1
        masks[:, y // 2 :] = 2
        # Cellpose squeezes all three, so a one-slice batch is (y, x).
        return masks.squeeze(), [None, dp.squeeze(), logits.squeeze()], None


@pytest.fixture
def fake_cellpose(monkeypatch):
    package = types.ModuleType("cellpose")
    models = types.ModuleType("cellpose.models")
    models.MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
    models.get_user_models = lambda: []
    models.CellposeModel = FakeCellposeModel
    package.models = models
    # A Cellpose 4: when an earlier test imported the environment's real
    # Cellpose 3, its cellpose.version stayed loaded, and the version check
    # refused the stand-in.
    version = types.ModuleType("cellpose.version")
    version.version = "4.0.0"
    package.version = version
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    monkeypatch.setitem(sys.modules, "cellpose.version", version)
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


def test_masks_are_numbered_through_the_chunk(fake_cellpose):
    model = CellposeModelConfig(voxel_size=8, output="masks", slices_per_chunk=3, slice_size=20, context=5)
    idi = ArrayIDI(np.zeros((3, 30, 30), dtype=np.uint8), (8, 8, 8))

    out = model.config.process_chunk(idi, Roi((0, 40, 40), (24, 160, 160)))

    assert out.shape == (1, 3, 20, 20) and out.dtype == np.uint64
    # Each slice's objects 1 and 2 get ids past the slices before.
    assert [sorted(np.unique(s).tolist()) for s in out[0]] == [[1, 2], [3, 4], [5, 6]]


def test_flows_are_cellposes_dp_then_the_probability_inside_the_context(fake_cellpose):
    model = CellposeModelConfig(voxel_size=8, output="flows", slices_per_chunk=3, slice_size=20, context=5)
    idi = ArrayIDI(np.zeros((10, 60, 60), dtype=np.uint8), (8, 8, 8))

    out = model.config.process_chunk(idi, Roi((8, 80, 160), (3 * 8, 20 * 8, 20 * 8)))

    assert out.shape == (1 + 2, 3, 20, 20) and out.dtype == np.float32
    assert model.config.model.calls[0][2]["compute_masks"] is False
    # flowY is the fake's z index, flowX its y index, cut to the inner 20 voxels.
    np.testing.assert_array_equal(out[0, :, 0, 0], [0, 1, 2])
    np.testing.assert_array_equal(out[1, 1, :, 0], np.arange(5, 25))
    logits = np.arange(5, 25) - 15
    np.testing.assert_allclose(out[2, 1, 7], 1 / (1 + np.exp(-logits)), rtol=1e-6)


def test_in_3d_the_chunk_is_one_image_cut_to_its_inside_on_every_axis(fake_cellpose):
    model = CellposeModelConfig(voxel_size=8, output="flows", do_3d=True, slice_size=20, context=5)
    idi = ArrayIDI(np.zeros((60, 60, 60), dtype=np.uint8), (8, 8, 8))
    roi = Roi((80, 80, 160), (20 * 8, 20 * 8, 20 * 8))

    out = model.config.process_chunk(idi, roi)

    assert idi.reads == [roi.grow(Coordinate(40, 40, 40), Coordinate(40, 40, 40))]
    ((shape, channel_axis, kwargs),) = model.config.model.calls
    assert shape == (30, 30, 30, 1) and channel_axis == 3 and kwargs["do_3D"] and kwargs["z_axis"] == 0
    assert out.shape == (4, 20, 20, 20) and out.dtype == np.float32
    # flowZ, flowY and flowX are the fake's z, y and x index, cut to the inner 20.
    np.testing.assert_array_equal(out[0, :, 0, 0], np.arange(5, 25))
    np.testing.assert_array_equal(out[1, 0, :, 0], np.arange(5, 25))
    np.testing.assert_array_equal(out[2, 0, 0, :], np.arange(5, 25))


def test_nothing_but_building_the_model_imports_cellpose(monkeypatch):
    # None in sys.modules makes `import cellpose` raise ImportError.
    monkeypatch.setitem(sys.modules, "cellpose", None)
    from cellmap_flow.cli import main

    model = registry.build_model({"type": "cellpose", "voxel_size": 64}, "cp")
    model.to_dict(), model.launch_entry, model.command
    assert "cellpose" in registry.describe_types()
    result = CliRunner().invoke(main.cli, ["infer", "cellpose", "--help"])
    assert result.exit_code == 0 and "--pretrained-model" in result.output
    # Only building it imports cellpose. Where pixi provides cellpose4, the
    # error names that env (ModelEnvError); where it cannot (CI, conda-only),
    # the model falls back to this environment and the import itself fails.
    from cellmap_flow.models.configs.base import ModelEnvError

    with pytest.raises((ModelEnvError, ImportError)):
        model.config


def test_all_channels_are_served_by_default_and_the_layer_shows_them_together():
    """One at a time, on a slider, flow_y looked like noise and the three
    channels like three unrelated images. The layer shows the flows' direction
    as colour, dimmed by the cell probability."""
    from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor
    from cellmap_flow.viewer.layers import display_channel, flow_channels, prediction_layer

    model = CellposeModelConfig(voxel_size=8)
    assert model.output == "flows" and model.display_channel == 2
    assert CellposeModelConfig(voxel_size=8, output="probability").display_channel is None
    info = {"display_channel": 2, "output_channels": 3, "has_channel": True, "channels": OUTPUT_CHANNELS["flows"],
            "output_class": "unit"}
    assert display_channel(info, []) == 2 and flow_channels(info, []) == (0, 1)
    # Masks made of the flows are one channel: nothing to choose.
    masks = [CellposeMasksPostprocessor()]
    assert display_channel(info, masks) is None and flow_channels(info, masks) is None
    assert display_channel({**info, "display_channel": None}, []) is None
    assert flow_channels({**info, "channels": ["cell"], "output_channels": 1}, []) is None

    layer = prediction_layer("cellpose_sam_v2", "http://gpu1:8000", "blob", dataset_path=None, postprocess=[],
                             color="red", info=info).to_json()
    # The served c' slider renamed c^, a channel axis the shader reads all of.
    (source,) = layer["source"]
    assert source["transform"] == {"outputDimensions": {"c^": [1, ""]}, "inputDimensions": {"c'": [1, ""]}}
    shader = layer["shader"]
    assert "getDataValue(0)" in shader and "getDataValue(1)" in shader and "channel=2" in shader
    assert 'color(default="red")' in shader and "localPosition" not in layer
