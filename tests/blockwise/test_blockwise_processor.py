"""The blockwise processor: what the master creates and checks, what a worker
writes and marks, and how it all fails. LSF is conftest's ``fake_lsf``; the
worker bsub argv itself is pinned in test_bsub_argv_snapshot."""

import contextlib
import hashlib
import logging
import subprocess

import daisy
import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate
from starlette.testclient import TestClient

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import (
    CellMapFlowBlockwiseProcessor,
    check_block,
    precheck,
    spawn_worker,
)
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.models.models_config import ModelConfig, ScriptModelConfig
from cellmap_flow.jobs.settings import launcher_settings
from cellmap_flow.post.postprocessors import MortonSegmentationRelabeling, ThresholdPostprocessor
from cellmap_flow.jobs.site import current_site
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.process_chain import process_chain
from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
from tests.utils.serving_helpers import layer, read_chunk

JSON_DATA = {
    "input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}],
    "postprocess": [{"name": "ThresholdPostprocessor", "threshold": 0.5}],
}


def test_a_models_scale_picks_its_level_of_a_multiscale_group(ome_pyramid, pooling_model, task_yaml):
    """Blockwise ignored ``scale`` and read the level nearest the model's input
    voxel size: the same YAML read other data than under cellmap_flow yaml."""
    path = task_yaml(ome_pyramid(((8, 0), (16, 4))), pooling_model(), model_extra={"scale": "s1"})
    processor = CellMapFlowBlockwiseProcessor(path, create=True)
    assert processor.input_path.rstrip("/").endswith("pyramid.zarr/s1")
    assert processor.idi_raw.path.rstrip("/").endswith("s1")


SPACE = [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"]


def ome(axes, scale, translation):
    """The OME multiscales attribute of an output's group, for its one level s0."""
    return [{
        "axes": axes,
        "coordinateTransformations": [{"scale": [1.0] * len(axes), "type": "scale"}],
        "datasets": [{"coordinateTransformations": [{"scale": scale, "type": "scale"},
                                                    {"translation": translation, "type": "translation"}],
                      "path": "s0"}],
        "name": "",
        "version": "0.4",
    }]


# What an output is, written from raw whose corner is (8, 16, 24): the array a
# worker opens (ROI, voxel size, dtype), the zarr chunks of s0, the attributes
# blockwise gives funlib for s0 (FUNLIB_ATTRS; funlib versions add others of
# their own), the OME multiscales on the group (the translation is voxel
# 0's centre; a channel axis has none), and the sum and hash of what two
# blocks write: one at the output's corner, and the 32 nm inside the output of
# one overhanging its end in z. ONE_CHANNEL holds one model channel; STACKED
# holds both, on a leading channel axis.
FUNLIB_ATTRS = ("axis_names", "offset", "units", "voxel_size")
ONE_CHANNEL = (
    daisy.Roi((8, 16, 24), (128, 128, 128)), Coordinate(16, 16, 16), np.uint8, (4, 4, 4),
    {"axis_names": ["z", "y", "x"], "offset": [8, 16, 24], "units": ["nanometer"] * 3,
     "voxel_size": [16, 16, 16]},
    ome(SPACE, [16, 16, 16], [16.0, 24.0, 32.0]),
    [(7, "f8b67fe4a95c8a04"), (16, "de50b53da232c274")],
)
STACKED = (
    daisy.Roi((0, 8, 16, 24), (2, 128, 128, 128)), Coordinate(1, 16, 16, 16), np.uint8, (2, 4, 4, 4),
    {"axis_names": ["c", "z", "y", "x"], "offset": [0, 8, 16, 24], "units": ["", *["nanometer"] * 3],
     "voxel_size": [1, 16, 16, 16]},
    ome([{"name": "c", "type": "channel"}, *SPACE], [1, 16, 16, 16], [0.0, 16.0, 24.0, 32.0]),
    [(14, "ae5ec9ead2c8d143"), (32, "5e5e84fe00189cd0")],
)


@pytest.mark.parametrize("output_channels, outputs", [
    pytest.param(None, {"a": ONE_CHANNEL, "b": ONE_CHANNEL}, id="one-per-model-channel"),
    pytest.param({"both": [0, 1], "second": 1}, {"both": STACKED, "second": ONE_CHANNEL},
                 id="a-dict-of-channel-indices"),
])
def test_the_outputs_and_the_blocks_written_are_unchanged(raw_zarr, pooling_model, task_yaml,
                                                          output_channels, outputs):
    """Where blockwise writes: each output's array and attributes, and what a
    worker writes there for two blocks through a json_data chain. Without
    output_channels there is one output per model channel; a dict names each
    output's channel indices, and one listing several stacks them. The output
    starts at the raw data's corner (8, 16, 24); it was created at 0 and the
    prediction written shifted."""
    overrides = {"output_channels": output_channels} if output_channels else {}
    path = task_yaml(raw_zarr(offset=(8, 16, 24)), pooling_model(8, 16), json_data=JSON_DATA, **overrides)
    master = CellMapFlowBlockwiseProcessor(path, create=True)
    worker = CellMapFlowBlockwiseProcessor(path, create=False)
    for offset in ((8, 16, 24), (104, 16, 24)):  # at the output's corner; overhanging its end in z by 32 nm
        roi = daisy.Roi(offset, (64, 64, 64))
        worker.process_fn(daisy.Block(roi, roi, roi, task_id="t"))

    assert master.output_channels == list(outputs)
    for (channel, expected), array in zip(outputs.items(), worker.output_arrays):
        s0 = zarr.open(str(master.output_path / channel / "s0"), mode="r")
        attrs = zarr.open_group(str(master.output_path / channel), mode="r").attrs
        corner, far = s0[..., :4, :4, :4], s0[..., 6:, :4, :4]  # 16 nm voxels
        written = [(int(v.sum()), hashlib.sha256(v.tobytes()).hexdigest()[:16]) for v in (corner, far)]
        funlib = {key: s0.attrs[key] for key in FUNLIB_ATTRS}
        assert (array.roi, array.voxel_size, array.dtype, s0.chunks, funlib, attrs["multiscales"],
                written) == expected, channel


@pytest.mark.parametrize("names", [
    pytest.param('classes = ["a", "b"]', id="a-scripts-classes"),
    pytest.param('channels_names = ["a", "b"]', id="hugging-faces-channels_names"),
])
def test_the_outputs_are_named_however_the_model_names_its_channels(raw_zarr, pooling_model, task_yaml, names):
    """Blockwise read config.channels alone, so a model naming its channels
    another way failed with AttributeError."""
    path = task_yaml(raw_zarr(), pooling_model(names=names))
    master = CellMapFlowBlockwiseProcessor(path, create=True)
    worker = CellMapFlowBlockwiseProcessor(path, create=False)
    roi = daisy.Roi((0, 0, 0), (32, 32, 32))
    worker.process_fn(daisy.Block(roi, roi, roi, task_id="t"))
    assert master.output_channels == ["a", "b"]
    assert all(array.to_ndarray(roi).any() for array in worker.output_arrays), "each output was written"


def test_a_model_naming_no_channels_needs_them_given(raw_zarr, pooling_model, task_yaml):
    """Without names blockwise cannot name the outputs, or pick a listed channel."""
    path = task_yaml(raw_zarr(), pooling_model(names=""))
    with pytest.raises(ConfigError, match="names no channels"):
        CellMapFlowBlockwiseProcessor(path, create=True)


def test_the_whole_volume_task_covers_the_output_on_its_chunk_grid(raw_zarr, pooling_model, task_yaml, monkeypatch):
    """With no context, the task's blocks start on the output chunk grid; they
    straddled output chunks while the output sat at 0 and the task at the raw's offset."""
    processor = CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(offset=(8, 16, 24)), pooling_model(8, 16)), create=True)
    scheduled = []

    class Stop(Exception):
        pass

    def task(name, **kwargs):
        scheduled.append(kwargs["total_roi"])
        raise Stop

    monkeypatch.setattr(blockwise_processor.daisy, "Task", task)
    with pytest.raises(Stop):
        processor.run()
    assert scheduled == [processor.output_arrays[0].roi]


@pytest.mark.parametrize("corner, voxel_size, resample", [
    pytest.param(0, 8, False, id="corner-at-0"),
    pytest.param(-4, 8, False, id="a-janelia-corner"),
    # 16x4x4 nm data resampled to the model's 8 nm: the grid starts at its corner.
    pytest.param(-4, (16, 4, 4), True, id="resampled-from-anisotropic-data"),
])
def test_a_block_gets_the_same_label_ids_served_or_blockwise(ome_pyramid, pooling_model, task_yaml, corner,
                                                             voxel_size, resample):
    """Each chunk's ids are offset by the Morton code of its index on the
    output grid, which starts at the raw data's corner. The index was taken
    from 0, so on data whose corner is -4 nm the first chunk was
    (-1, -1, -1), which Morton reads as 1023 on every axis."""
    centre = (corner + np.broadcast_to(voxel_size, 3) / 2).tolist()
    raw = ome_pyramid(((voxel_size, centre),)) + "/s0"  # every voxel is at least 1
    model = pooling_model()  # 4^3 output voxels a chunk
    posts = [ThresholdPostprocessor(threshold=0.5), MortonSegmentationRelabeling()]
    path = task_yaml(raw, model, json_data={"input_norm": [], "postprocess": [p.to_dict() for p in posts]},
                     resample=resample)
    CellMapFlowBlockwiseProcessor(path, create=True)
    worker = CellMapFlowBlockwiseProcessor(path, create=False)
    assert worker.idi_raw.resampled is resample
    server = ChunkmirageServer(raw, ScriptModelConfig(script_path=model), resample=resample)
    client = TestClient(server.app)
    for index in range(2):  # along z
        roi = daisy.Roi((corner + 32 * index, corner, corner), (32, 32, 32))
        worker.process_fn(daisy.Block(roi, roi, roi, task_id="t"))
        written = worker.output_arrays[0].to_ndarray(roi)
        served = read_chunk(client, layer(posts=posts), (index, 0, 0))[0]
        assert np.all(written == served) and np.all(served == 1 + 64 * index), index


# --- bounding boxes ---------------------------------------------------------------
#
# The raw is ome_pyramid's: 16^3 voxels of 8 nm from 0, each holding its z index
# + 1, which the pooling model (8 nm in and out, no chain) writes as it is, so an
# output voxel at z nm holds z / 8 + 1 and a voxel left unwritten holds 0. Its
# blocks are 32 nm (4 voxels) a side, with no context.


def run_in_process(path, monkeypatch):
    """Create the outputs of the task YAML at ``path`` and run it, daisy's
    serial server handing every block to a worker's process_fn in this
    process instead of to LSF workers. The master, and each block's task id
    and write ROI."""
    master = CellMapFlowBlockwiseProcessor(path, create=True)
    worker = CellMapFlowBlockwiseProcessor(path, create=False)
    blocks = []

    def process(block):
        blocks.append((block.task_id, block.write_roi))
        worker.process_fn(block)

    monkeypatch.setattr(blockwise_processor, "spawn_worker", lambda *a, **k: process)
    monkeypatch.setattr(blockwise_processor.daisy, "Server", lambda stop_event: daisy.SerialServer())
    assert master.run()
    return master, blocks


def written(output, channel="a"):
    """Output ``channel`` of the zarr at ``output``: s0 as stored, its funlib
    offset and its OME translation."""
    s0 = zarr.open(str(output / channel / "s0"), mode="r")
    (multiscales,) = zarr.open_group(str(output / channel), mode="r").attrs["multiscales"]
    translation = multiscales["datasets"][0]["coordinateTransformations"][1]["translation"]
    return s0, s0.attrs["offset"], translation


def raw_at(roi):
    """What the model writes over ``roi`` (nm, z, y, x): each voxel's raw z index + 1."""
    shape = np.array(roi.shape) // 8
    z = np.arange(roi.offset[0] // 8, roi.offset[0] // 8 + shape[0]) + 1
    return np.broadcast_to(z[:, None, None], tuple(shape)).astype(np.float32)


def box(offset, shape):
    return {"offset": list(offset), "shape": list(shape)}


def test_a_box_runs_output_is_the_box(ome_pyramid, pooling_model, task_yaml, monkeypatch):
    """The output was an array the size of the whole volume, with the box's
    blocks written into it, the last ones up to a block past its end. Now it
    is the box, snapped out to every 8 nm output voxel it touches: (12, 20,
    4) + (50, 30, 70) nm is (8, 16, 0) + (56, 40, 80), 7 x 5 x 10 voxels,
    not whole 4-voxel chunks. Its blocks start at its corner, and the last
    along each axis are cut at its end."""
    path = task_yaml(ome_pyramid(((8, 4),)) + "/s0", pooling_model(), bounding_boxes=[box((12, 20, 4), (50, 30, 70))])
    master, blocks = run_in_process(path, monkeypatch)

    s0, offset, translation = written(master.output_path)
    the_box = daisy.Roi((8, 16, 0), (56, 40, 80))
    assert (s0.shape, s0.chunks, offset, translation) == ((7, 5, 10), (4, 4, 4), [8, 16, 0], [12.0, 20.0, 4.0])
    assert master.output_arrays[0].roi == the_box
    np.testing.assert_array_equal(s0[:], raw_at(the_box))
    # 2 x 2 x 3 blocks from the box's corner, 32 nm apart; those at its end overhang it.
    begins = sorted(roi.offset for _, roi in blocks)
    assert begins == sorted(Coordinate(z, y, x) for z in (8, 40) for y in (16, 48) for x in (0, 32, 64))
    assert {roi.shape for _, roi in blocks} == {Coordinate(32, 32, 32)}
    # Readers place it at the box: by the group's OME translation, or by s0's funlib offset.
    for read in (master.output_path / "a", master.output_path / "a" / "s0"):
        assert tuple(ImageDataInterface(str(read)).offset) == (8, 16, 0), read


def test_a_box_is_cut_to_the_data(ome_pyramid, pooling_model, task_yaml):
    """Past the raw data there is nothing to compute an output from."""
    raw, model = ome_pyramid(((8, 4),)) + "/s0", pooling_model()
    master = CellMapFlowBlockwiseProcessor(task_yaml(raw, model, bounding_boxes=[box((96, -20, 0), (64, 50, 16))]),
                                           create=True)
    assert master.output_arrays[0].roi == daisy.Roi((96, 0, 0), (32, 32, 16))

    outside = task_yaml(raw, model, bounding_boxes=[box((200, 0, 0), (8, 8, 8))])
    with pytest.raises(ConfigError, match="bounding box 1 .* is outside the data"):
        CellMapFlowBlockwiseProcessor(outside, create=True)


# Two boxes: the second one starts off the first one's block grid.
FIRST, SECOND = box((8, 16, 8), (24, 24, 24)), box((72, 64, 40), (40, 32, 48))
FIRST_ROI, SECOND_ROI = daisy.Roi((8, 16, 8), (24, 24, 24)), daisy.Roi((72, 64, 40), (40, 32, 48))


def test_several_boxes_share_one_output_over_their_bounding_box(ome_pyramid, pooling_model, task_yaml, monkeypatch):
    """One output from the lowest corner to the highest far end; each box's
    blocks run, on the output's block grid from its corner (the second box's
    from y = 48, not its own 64, so two blocks cover its 32 nm in y), and
    write only inside their box: between the boxes the output stays 0."""
    path = task_yaml(ome_pyramid(((8, 4),)) + "/s0", pooling_model(), bounding_boxes=[FIRST, SECOND])
    master, blocks = run_in_process(path, monkeypatch)

    s0, offset, translation = written(master.output_path)
    union = daisy.Roi((8, 16, 8), (104, 80, 80))
    assert (s0.shape, offset, translation) == ((13, 10, 10), [8, 16, 8], [12.0, 20.0, 12.0])
    expected = np.zeros(s0.shape, np.float32)
    for roi in (FIRST_ROI, SECOND_ROI):
        inside = tuple(slice(b, e) for b, e in zip((roi.begin - union.begin) // 8, (roi.end - union.begin) // 8))
        expected[inside] = raw_at(roi)
    np.testing.assert_array_equal(s0[:], expected)
    assert sorted(roi.offset for task, roi in blocks if task == "predict_m_t_roi2") == sorted(
        Coordinate(z, y, x) for z in (72, 104) for y in (48, 80) for x in (40, 72))
    assert [roi.offset for task, roi in blocks if task == "predict_m_t_roi1"] == [Coordinate(8, 16, 8)]


def test_separate_bounding_boxes_zarrs_gives_each_box_its_own_output(ome_pyramid, pooling_model, task_yaml,
                                                                      monkeypatch):
    """It took one box; the dashboard wrote a task per box to get each its own
    zarr. Each is <output_path>/box_<n>, the size of its box, a group of the
    output zarr, written by its box's task."""
    path = task_yaml(ome_pyramid(((8, 4),)) + "/s0", pooling_model(), bounding_boxes=[FIRST, SECOND],
                     separate_bounding_boxes_zarrs=True)
    master, blocks = run_in_process(path, monkeypatch)

    for n, roi in ((1, FIRST_ROI), (2, SECOND_ROI)):
        s0, offset, translation = written(master.output_path / f"box_{n}")
        assert (offset, translation) == (list(roi.offset), [o + 4.0 for o in roi.offset]), n
        np.testing.assert_array_equal(s0[:], raw_at(roi))
    assert sorted(zarr.open_group(str(master.output_path), mode="r").group_keys()) == ["box_1", "box_2"]
    assert sorted(roi.offset for task, roi in blocks if task == "predict_m_t_roi2") == sorted(
        Coordinate(z, 64, x) for z in (72, 104) for x in (40, 72))


def test_a_block_of_another_task_is_refused(ome_pyramid, pooling_model, task_yaml):
    """With several boxes a worker finds a block's box by its task id."""
    path = task_yaml(ome_pyramid(((8, 4),)) + "/s0", pooling_model(), bounding_boxes=[FIRST, SECOND],
                     separate_bounding_boxes_zarrs=True)
    CellMapFlowBlockwiseProcessor(path, create=True)
    worker = CellMapFlowBlockwiseProcessor(path, create=False)
    roi = daisy.Roi((8, 16, 8), (32, 32, 32))
    with pytest.raises(ValueError, match="none of this run's"):
        worker.process_fn(daisy.Block(roi, roi, roi, task_id="predict_m_other"))


MORTON = [ThresholdPostprocessor(threshold=0.5), MortonSegmentationRelabeling()]


def test_a_boxs_blocks_get_label_ids_of_their_own(ome_pyramid, pooling_model, task_yaml, monkeypatch):
    """A block's ids are keyed to its index on the volume's grid, floored for
    a box's blocks, which start at the box's corner, 8 nm off that grid: the
    three blocks along z, the last cut at the box's end, still get an id each
    (one object a block), 1 + 64 x Morton(index)."""
    json_data = {"input_norm": [], "postprocess": [p.to_dict() for p in MORTON]}
    path = task_yaml(ome_pyramid(((8, 4),)) + "/s0", pooling_model(), json_data=json_data,
                     bounding_boxes=[box((8, 8, 8), (72, 32, 32))])
    master, blocks = run_in_process(path, monkeypatch)

    s0, _, _ = written(master.output_path)
    ids = [np.unique(s0[z:z + 4]).tolist() for z in (0, 4, 8)]  # each block's voxels, along z
    assert ids == [[1], [1 + 64 * 1], [1 + 64 * 8]]  # Morton of (0, 0, 0), (1, 0, 0), (2, 0, 0)


def test_a_box_on_the_volumes_grid_gets_the_ids_the_server_does(ome_pyramid, pooling_model, task_yaml, monkeypatch):
    """Where the blocks are the server's chunks, as for a box on the volume's
    block grid, the ids are the same as served."""
    raw, model = ome_pyramid(((8, 4),)) + "/s0", pooling_model()
    json_data = {"input_norm": [], "postprocess": [p.to_dict() for p in MORTON]}
    path = task_yaml(raw, model, json_data=json_data, bounding_boxes=[box((32, 0, 32), (64, 32, 32))])
    master, _ = run_in_process(path, monkeypatch)

    s0, _, _ = written(master.output_path)
    client = TestClient(ChunkmirageServer(raw, ScriptModelConfig(script_path=model)).app)
    for n, index in enumerate(((1, 0, 1), (2, 0, 1))):
        served = read_chunk(client, layer(posts=MORTON), index)[0]
        assert np.all(s0[4 * n:4 * n + 4] == served), index


@pytest.mark.parametrize("first, then", [
    pytest.param({"bounding_boxes": [box((0, 0, 0), (32, 32, 32))]}, {"bounding_boxes": [box((32, 0, 0), (32, 32, 32))]},
                 id="another-box-of-the-same-shape"),
    pytest.param({}, {"bounding_boxes": [box((0, 0, 0), (32, 32, 32))]}, id="a-box-into-a-whole-volume-output"),
])
def test_resuming_into_an_output_over_another_region_is_refused(ome_pyramid, pooling_model, task_yaml, first, then):
    """Said so: funlib's prepare_ds takes an array of the same shape
    elsewhere, keeping its offset, and the OME check after it blamed the
    other translation on an older cellmap-flow's half-voxel placement; one
    of another shape it refused as "not compatible". An output of a box run
    by an older cellmap-flow is the whole volume's."""
    raw, model = ome_pyramid(((8, 4),)) + "/s0", pooling_model()
    CellMapFlowBlockwiseProcessor(task_yaml(raw, model, **first), create=True)
    CellMapFlowBlockwiseProcessor(task_yaml(raw, model, **first), create=True)  # the same region resumes
    with pytest.raises(ConfigError, match="already holds an output covering"):
        CellMapFlowBlockwiseProcessor(task_yaml(raw, model, **then), create=True)


@pytest.mark.parametrize("failing", [pytest.param(True, id="blocks-failed"), pytest.param(False, id="clean-run")])
def test_a_run_succeeds_only_if_every_block_did(raw_zarr, pooling_model, task_yaml, monkeypatch, caplog, failing):
    """daisy.run_blockwise counts failed blocks as done, so the master said
    True, and exited 0, however many had failed. The blocks run here in
    process, on daisy's serial server, instead of on LSF workers."""
    def process(block):
        if failing and block.write_roi.offset[0] == 0:
            block.status = daisy.BlockStatus.FAILED

    master = CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model()), create=True)
    monkeypatch.setattr(blockwise_processor, "spawn_worker", lambda *a, **k: process)
    monkeypatch.setattr(blockwise_processor.daisy, "Server", lambda stop_event: daisy.SerialServer())
    with caplog.at_level(logging.ERROR, logger=blockwise_processor.logger.name):
        assert master.run() is not failing
    # 16 of the 64 blocks, named by their task.
    assert ("predict_m_t: 16" in caplog.text) is failing


class FakeClient:
    """daisy.Client handing out a fixed list of blocks."""

    blocks = []

    def __init__(self):
        self._blocks = list(self.blocks)

    @contextlib.contextmanager
    def acquire_block(self):
        yield self._blocks.pop(0) if self._blocks else None


def _block(task_id, offset, size=32):
    roi = daisy.Roi(offset, (size,) * 3)
    return daisy.Block(daisy.Roi(offset, (size * 4,) * 3), roi, roi, task_id=task_id)


@pytest.fixture
def worker(raw_zarr, pooling_model, task_yaml, tmp_path, monkeypatch):
    """A processor that tracks progress, taking its blocks from FakeClient."""
    path = task_yaml(raw_zarr(), pooling_model(), track_progress=True, tmp_dir=str(tmp_path / "progress"))
    processor = CellMapFlowBlockwiseProcessor(path, create=True)
    monkeypatch.setattr(processor, "process_fn", lambda block: None)
    monkeypatch.setattr(blockwise_processor.daisy, "Client", FakeClient)
    return processor


def test_a_progress_marker_names_one_block_of_one_task(worker):
    """Markers were named by block_id[1] alone, one directory per YAML task:
    every bounding box is its own daisy task numbering its blocks from 0, so
    once ROI 1 was done ROI 2's blocks were skipped, and a changed block size
    reused the numbers for other regions."""
    done, other_task = _block("predict_m_roi1", (0, 0, 0)), _block("predict_m_roi2", (640, 640, 640))
    bigger = _block("predict_m_roi1", (0, 0, 0), size=64)
    assert done.block_id[1] == other_task.block_id[1] and done.block_id == bigger.block_id
    FakeClient.blocks = [done]
    worker.client()
    assert [check_block(worker.tmp_dir, b) for b in (done, other_task, bigger)] == [True, False, False]


def test_a_failed_block_is_logged_with_its_traceback_and_leaves_no_marker(worker, monkeypatch, caplog):
    def boom(block):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker, "process_fn", boom)
    block = _block("predict_m", (0, 0, 0))
    FakeClient.blocks = [block]
    with caplog.at_level(logging.ERROR, logger=blockwise_processor.logger.name):
        worker.client()
    assert block.status == daisy.BlockStatus.FAILED and not check_block(worker.tmp_dir, block)
    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.exc_info is not None and "CUDA out of memory" in caplog.text


@pytest.fixture
def no_model_loading(monkeypatch):
    def refuse(self):
        raise AssertionError("the precheck loaded a model")

    monkeypatch.setattr(ScriptModelConfig, "_get_config", refuse)


@pytest.mark.parametrize("overrides, message", [
    pytest.param({"output_path": "/somewhere/out"}, ".zarr", id="output-not-a-zarr"),
    pytest.param({"workers": 0}, "workers", id="no-workers"),
    pytest.param({"track_progress": True}, "tmp_dir", id="progress-without-a-tmp-dir"),
    pytest.param({"model_mode": "SOMETIMES"}, "SOMETIMES", id="unknown-model-mode"),
    pytest.param({"output_channels": ["a", "a"]}, "duplicated", id="duplicated-channels"),
    pytest.param({"json_data": {"input_norm": {}}}, "json_data", id="json-data-without-postprocess"),
    pytest.param({"data_path": "/nope.zarr/raw"}, "does not exist", id="missing-data"),
    pytest.param({"bounding_boxes": [{"offset": [0, 0, 0], "shape": [8, 8]}]}, "shape should be three numbers",
                 id="a-box-of-two-sides"),
    pytest.param({"bounding_boxes": [{"offset": [0, 0, 0], "shape": [8, 0, 8]}]}, "above 0", id="an-empty-box"),
    pytest.param({"bounding_boxes": [{"offset": [0, 0, 0]}]}, "offset, shape", id="a-box-without-a-shape"),
])
def test_the_precheck_reports_a_bad_setting(raw_zarr, pooling_model, task_yaml, no_model_loading, overrides, message):
    overrides = dict(overrides)
    data = overrides.pop("data_path", None) or raw_zarr()
    with pytest.raises(ConfigError, match=message):
        precheck(task_yaml(data, pooling_model(), **overrides))


def test_the_precheck_takes_several_boxes_each_in_its_own_output(raw_zarr, pooling_model, task_yaml, no_model_loading):
    """separate_bounding_boxes_zarrs was refused with more than one box."""
    boxes = [{"offset": [0, 0, 0], "shape": [32, 32, 32]}, {"offset": [64, 0, 0], "shape": [32, 32, 32]}]
    precheck(task_yaml(raw_zarr(), pooling_model(), bounding_boxes=boxes, separate_bounding_boxes_zarrs=True))


def test_the_processor_refuses_the_same_settings(raw_zarr, pooling_model, task_yaml):
    with pytest.raises(ConfigError, match="tmp_dir"):
        CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model(), track_progress=True), create=True)


@pytest.mark.parametrize("in_yaml, saved, expected", [
    pytest.param("36:00", "10:00", "36:00", id="the-yamls"),
    pytest.param(None, "10:00", "10:00", id="else-the-saved-setting"),
    pytest.param(None, None, current_site().default_walltime, id="else-the-sites"),
])
def test_the_workers_walltime(raw_zarr, pooling_model, task_yaml, in_yaml, saved, expected):
    launcher_settings().walltime = saved
    overrides = {"walltime": in_yaml} if in_yaml else {}
    assert CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model(), **overrides), create=True).walltime == expected


THRESHOLD_ONLY = PipelineSpec(postprocess=[{"name": "ThresholdPostprocessor", "threshold": 0.5}])


@pytest.mark.parametrize("json_data, chain", [
    pytest.param(JSON_DATA, PipelineSpec.from_json_data(JSON_DATA), id="the-tasks-json-data-becomes-the-processs"),
    pytest.param(None, THRESHOLD_ONLY, id="else-the-processs-own-is-kept-and-used"),
])
def test_the_chain_a_task_runs_and_the_dtype_it_writes(raw_zarr, pooling_model, task_yaml, json_data, chain):
    """A task's json_data is installed as the process's chain; without one the
    process's own is kept. Either way the output's dtype is the chain's: here
    ThresholdPostprocessor's uint8 over the model's float32."""
    process_chain().set(THRESHOLD_ONLY)
    overrides = {"json_data": json_data} if json_data else {}
    processor = CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model(), **overrides), create=True)
    assert process_chain().spec == chain
    assert [type(s).__name__ for s in process_chain().postprocess] == ["ThresholdPostprocessor"]
    assert processor.dtype == np.uint8


def test_a_worker_submitted_without_a_walltime_gets_the_default(fake_lsf, tmp_path):
    """The worker bsub had no -W, so the GPU queues killed every worker at two hours."""
    fake_lsf.answers["bsub"] = ["Job <77> is submitted to queue <gpu_h100>.\n"]
    spawn_worker("w", "/t.yaml", "grp", "gpu_h100", log_dir=tmp_path)()
    (argv,) = fake_lsf.commands("bsub")
    assert argv[argv.index("-W") + 1] == current_site().default_walltime


def test_a_refused_worker_submission_raises(fake_lsf, tmp_path):
    """Its result was never checked: the master waited forever for a worker that would never connect."""
    fake_lsf.answers["bsub"] = [(255, "", "Project grp is not valid")]
    with pytest.raises(subprocess.CalledProcessError):
        spawn_worker("w", "/t.yaml", "grp", "gpu_h100", log_dir=tmp_path)()


def test_only_the_workers_load_the_model(raw_zarr, pooling_model, task_yaml, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the master ran the model")

    monkeypatch.setattr(blockwise_processor, "Inferencer", refuse)
    monkeypatch.setattr(ModelConfig, "_validate_model_shapes", refuse)
    master = CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model()), create=True)
    assert master.inferencers == [] and master.output_arrays, "it still creates the outputs"
    with pytest.raises(RuntimeError, match="worker"):
        master.process_fn(None)


def test_a_worker_checks_its_model_on_the_warmup_forward_only(raw_zarr, pooling_model, task_yaml, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a forward besides the warmup")

    monkeypatch.setattr(ModelConfig, "_validate_model_shapes", refuse)
    path = task_yaml(raw_zarr(), pooling_model())
    CellMapFlowBlockwiseProcessor(path, create=True)  # creates the outputs a worker opens
    (inferencer,) = CellMapFlowBlockwiseProcessor(path, create=False).inferencers
    assert inferencer.output_class is not None, "the warmup forward ran"
