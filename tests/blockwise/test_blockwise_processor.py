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

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import (
    CellMapFlowBlockwiseProcessor,
    check_block,
    precheck,
    spawn_worker,
)
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ModelConfig, ScriptModelConfig
from cellmap_flow.jobs.site import current_site
from cellmap_flow.config.yaml import ConfigError

JSON_DATA = {
    "input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}],
    "postprocess": [{"name": "ThresholdPostprocessor", "threshold": 0.5}],
}


def test_a_models_scale_picks_its_level_of_a_multiscale_group(ome_pyramid, pooling_model, task_yaml):
    """Blockwise ignored ``scale`` and read the level nearest the model's input
    voxel size: the same YAML read other data than under cellmap_flow_yaml."""
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
# worker opens (ROI, voxel size, dtype), the zarr chunks of s0, funlib's
# attributes on s0, the OME multiscales on the group (the translation is voxel
# 0's centre; a channel axis has none), and the sum and hash of what one block
# writes. ONE_CHANNEL holds one model channel; STACKED holds both, on a
# leading channel axis.
ONE_CHANNEL = (
    daisy.Roi((8, 16, 24), (128, 128, 128)), Coordinate(16, 16, 16), np.uint8, (4, 4, 4),
    {"axis_names": ["z", "y", "x"], "offset": [8, 16, 24], "units": ["nanometer"] * 3,
     "voxel_size": [16, 16, 16]},
    ome(SPACE, [16, 16, 16], [16.0, 24.0, 32.0]),
    (7, "f8b67fe4a95c8a04"),
)
STACKED = (
    daisy.Roi((0, 8, 16, 24), (2, 128, 128, 128)), Coordinate(1, 16, 16, 16), np.uint8, (2, 4, 4, 4),
    {"axis_names": ["c", "z", "y", "x"], "offset": [0, 8, 16, 24], "units": ["", *["nanometer"] * 3],
     "voxel_size": [1, 16, 16, 16]},
    ome([{"name": "c", "type": "channel"}, *SPACE], [1, 16, 16, 16], [0.0, 16.0, 24.0, 32.0]),
    (14, "ae5ec9ead2c8d143"),
)


@pytest.mark.parametrize("output_channels, outputs", [
    pytest.param(None, {"a": ONE_CHANNEL, "b": ONE_CHANNEL}, id="one-per-model-channel"),
    pytest.param({"both": [0, 1], "second": 1}, {"both": STACKED, "second": ONE_CHANNEL},
                 id="a-dict-of-channel-indices"),
])
def test_the_outputs_and_a_block_written_are_unchanged(raw_zarr, pooling_model, task_yaml, output_channels,
                                                       outputs):
    """Where blockwise writes: each output's array and attributes, and what a
    worker writes there for one block through a json_data chain. Without
    output_channels there is one output per model channel; a dict names each
    output's channel indices, and one listing several stacks them. The output
    starts at the raw data's corner (8, 16, 24); it was created at 0 and the
    prediction written shifted."""
    overrides = {"output_channels": output_channels} if output_channels else {}
    path = task_yaml(raw_zarr(offset=(8, 16, 24)), pooling_model(8, 16), json_data=JSON_DATA, **overrides)
    master = CellMapFlowBlockwiseProcessor(path, create=True)
    worker = CellMapFlowBlockwiseProcessor(path, create=False)
    roi = daisy.Roi((8, 16, 24), (64, 64, 64))
    worker.process_fn(daisy.Block(roi, roi, roi, task_id="t"))

    assert master.output_channels == list(outputs)
    for (channel, expected), array in zip(outputs.items(), worker.output_arrays):
        s0 = zarr.open(str(master.output_path / channel / "s0"), mode="r")
        attrs = zarr.open_group(str(master.output_path / channel), mode="r").attrs
        block = s0[..., :4, :4, :4]  # the block's 64 nm at 16 nm, at the output's corner
        written = (int(block.sum()), hashlib.sha256(block.tobytes()).hexdigest()[:16])
        assert (array.roi, array.voxel_size, array.dtype, s0.chunks, dict(s0.attrs), attrs["multiscales"],
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
])
def test_the_precheck_reports_a_bad_setting(raw_zarr, pooling_model, task_yaml, no_model_loading, overrides, message):
    overrides = dict(overrides)
    data = overrides.pop("data_path", None) or raw_zarr()
    with pytest.raises(ConfigError, match=message):
        precheck(task_yaml(data, pooling_model(), **overrides))


def test_the_processor_refuses_the_same_settings(raw_zarr, pooling_model, task_yaml):
    with pytest.raises(ConfigError, match="tmp_dir"):
        CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model(), track_progress=True), create=True)


@pytest.mark.parametrize("in_yaml, saved, expected", [
    pytest.param("36:00", "10:00", "36:00", id="the-yamls"),
    pytest.param(None, "10:00", "10:00", id="else-the-saved-setting"),
])
def test_the_workers_walltime(raw_zarr, pooling_model, task_yaml, monkeypatch, in_yaml, saved, expected):
    monkeypatch.setattr(g, "walltime", saved)
    overrides = {"walltime": in_yaml} if in_yaml else {}
    assert CellMapFlowBlockwiseProcessor(task_yaml(raw_zarr(), pooling_model(), **overrides), create=True).walltime == expected


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
