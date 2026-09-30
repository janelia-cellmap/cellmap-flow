"""Fixtures for the finetune tests.

- ``ome_zarr``: an OME-NGFF group with the given levels.
- ``annotation_volume``: an annotation volume over a raw zarr, and the
  VirtualPatchDataset that reads it.
- ``make_trainer``: a LoRAFinetuner on the CPU, in fp32, without TensorBoard.
- ``tiny_script`` and ``run_cli``: finetune_cli.main() on a script model, with
  a fake data loader and inference server, and restarts delivered the way the
  job's server delivers them.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader, TensorDataset


def _triple(value):
    return [float(v) for v in value] if isinstance(value, (list, tuple)) else [float(value)] * 3


def _v3_array(path, data, chunks=None, attributes=None):
    """A zarr v3 array (written with tensorstore; zarr-python 2 cannot)."""
    import tensorstore as ts

    metadata = {"shape": list(data.shape), "data_type": str(data.dtype), "attributes": attributes or {},
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks or data.shape)}}}
    ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}, "create": True,
             "metadata": metadata}).result()[:] = data
    return str(path)


@pytest.fixture
def ome_zarr():
    """``ome_zarr(path, (name, data, scale, translation), ...)``: a group with OME
    multiscales, zarr v2 (0.4) or, with ``zarr_format=3``, v3 (0.5). ``ome_zarr.v3_array``
    writes a plain v3 array."""

    def write(path, *levels, chunks=None, zarr_format=2):
        multiscales = [{
            "version": "0.4" if zarr_format == 2 else "0.5",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [{"path": name, "coordinateTransformations": [
                {"type": "scale", "scale": _triple(scale)},
                {"type": "translation", "translation": _triple(translation)},
            ]} for name, _, scale, translation in levels],
        }]
        if zarr_format == 2:
            group = zarr.open_group(str(path), mode="w")
            for name, data, _, _ in levels:
                group.create_dataset(name, data=data, chunks=chunks or data.shape)
            group.attrs["multiscales"] = multiscales
            return str(path)
        path.mkdir(parents=True)
        (path / "zarr.json").write_text(json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"multiscales": multiscales}}}
        ))
        for name, data, _, _ in levels:
            _v3_array(path / name, data, chunks)
        return str(path)

    write.v3_array = _v3_array
    return write


@pytest.fixture
def annotation_volume(tmp_path, ome_zarr):
    """``annotation_volume(labels, raw=None, crops=())``: a 16 nm volume holding
    ``labels`` over a raw of the same grid (constant 128 unless given), and
    ``.dataset(**kw)``, the VirtualPatchDataset over them (8^3 in, 4^3 out)."""
    from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

    def make(labels, raw=None, crops=(), name="vol"):
        raw = np.full(labels.shape, 128, np.uint8) if raw is None else raw
        raw_path = ome_zarr(tmp_path / f"{name}_raw.zarr", ("s0", raw, 16.0, 0.0), chunks=(16, 16, 16))
        path = str(tmp_path / f"{name}.zarr")
        root = zarr.open_group(path, mode="w")
        root.create_group("annotation").create_dataset(
            "s0", shape=labels.shape, chunks=(16, 16, 16), dtype="uint8", fill_value=0
        )
        root["annotation"]["s0"][:] = labels
        root.attrs.update(dataset_offset_nm=[0.0] * 3, imported_crops=list(crops))
        root["annotation"].attrs["multiscales"] = zarr.open_group(raw_path).attrs["multiscales"]

        def dataset(**kw):
            args = dict(volume_zarr_path=path, raw_dataset_path=raw_path, input_size_voxels=(8, 8, 8),
                        output_size_voxels=(4, 4, 4), input_voxel_size_nm=(16, 16, 16),
                        output_voxel_size_nm=(16, 16, 16), seed=0)
            return VirtualPatchDataset(**{**args, **kw})

        return SimpleNamespace(path=path, raw=raw_path, dataset=dataset)

    return make


@pytest.fixture
def make_trainer(tmp_path):
    """``make_trainer(model, data, **kw)``: ``data`` is a DataLoader, or (raw, annotation[, anchor]) tensors."""
    from cellmap_flow.finetune.lora_trainer import LoRAFinetuner

    def make(model, data, batch_size=1, **kw):
        if not isinstance(data, DataLoader):
            data = DataLoader(TensorDataset(*data), batch_size=batch_size)
        args = dict(output_dir=str(tmp_path / "run"), num_epochs=1, device="cpu",
                    use_mixed_precision=False, loss_type="bce", tensorboard=False)
        return LoRAFinetuner(model, data, **{**args, **kw})

    return make


SCRIPT = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

input_voxel_size = output_voxel_size = Coordinate(8, 8, 8)
read_shape = write_shape = Coordinate(4, 4, 4) * input_voxel_size
output_channels = {channels}
torch.manual_seed(0)
model = nn.Sequential(nn.Conv3d(1, 4, 1), nn.ReLU(), nn.Conv3d(4, {channels}, 1))
"""


@pytest.fixture
def tiny_script(tmp_path):
    """``tiny_script(channels=1)``: the path of a script model, 4^3 in and out at 8 nm."""

    def write(channels=1):
        path = tmp_path / f"model_{channels}.py"
        path.write_text(SCRIPT.format(channels=channels))
        return path

    return write


def patches(n=2, fill=None):
    """``n`` patches of 4^3 raw and annotation (background below z=2, foreground above)."""
    ann = torch.full((n, 1, 4, 4, 4), 2.0)
    ann[:, :, :2] = 1
    raw = torch.rand(n, 1, 4, 4, 4) if fill is None else torch.full((n, 1, 4, 4, 4), fill)
    return DataLoader(TensorDataset(raw, ann), batch_size=2)


MARKERS = ("TRAINING_ITERATION_COMPLETE:", "FINETUNED_MODEL_YAML:", "RESTART_FAILED:",
           "INFERENCE_SERVER_FAILED:", "TRAINING_DIVERGED", "RESTARTING_TRAINING",
           "WAITING_FOR_RESTART", "RESTART_STATUS:")


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys, tiny_script):
    """``run_cli(*flags, ...)``: finetune_cli.main() on the tiny script, one epoch, rank 0.

    ``loaders``: what each create_dataloader call gives, in turn: a DataLoader,
    an exception (raised), or a callable of the run's record returning either.
    ``server``: replaces starting the inference server. ``restarts``: the
    requests the job receives in turn, through the restart controller the
    server would hand them to; None (and running out) is a malformed signal
    file, which ends the job. ``manifest``: the corrections' manifest, None for
    none. ``run_dir``: the output dir, <tmp>/session/runs/run by default.
    """
    from cellmap_flow.finetune import finetune_cli

    def run(*flags, channels=1, loaders=None, server=None, restarts=(), manifest=True,
            run_dir=None, tensorboard=False):
        session = tmp_path / "session"
        corrections = session / "corrections"
        corrections.mkdir(parents=True, exist_ok=True)
        if manifest is True:
            manifest = {"kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr"}
        if manifest is not None:
            (corrections / "_virtual_sources.json").write_text(json.dumps(manifest))
        record = SimpleNamespace(session=session, run=run_dir or session / "runs" / "run",
                                 loaded=[], served=[], waited=0)
        loads = iter(loaders or [])
        requests = iter(restarts)

        def dataloader(*args, **kwargs):
            record.loaded.append(kwargs)
            item = next(loads, None) or patches()
            item = item(record) if callable(item) else item
            if isinstance(item, Exception):
                raise item
            return item

        real_wait = finetune_cli._wait_for_restart_signal

        def wait(**kwargs):
            record.waited += 1
            request = next(requests, None)
            if request is None:
                kwargs["signal_file"].write_text("not json")
            else:
                kwargs["restart_controller"].request_restart(request)
            return real_wait(**kwargs)

        def serve(args, model_config, model, **kwargs):
            record.served.append(model)
            return server(args, model_config, model) if server else (None, 0)

        monkeypatch.setattr(finetune_cli, "create_dataloader", dataloader)
        monkeypatch.setattr(finetune_cli, "_start_inference_server_background", serve)
        monkeypatch.setattr(finetune_cli, "_wait_for_restart_signal", wait)
        monkeypatch.setattr("sys.argv", [
            "finetune_cli", "--model-type", "script", "--model-script", str(tiny_script(channels)),
            "--model-name", "tiny", "--corrections", str(corrections), "--output-dir", str(record.run),
            "--lora-r", "0", "--num-epochs", "1", "--loss-type", "bce", "--no-mixed-precision",
            "--num-workers", "0", *([] if tensorboard else ["--no-tensorboard"]), *flags,
        ])
        record.code = finetune_cli.main()
        record.out = capsys.readouterr().out
        record.markers = [line for line in record.out.splitlines() if line.startswith(MARKERS)]
        return record

    run.patches = patches
    return run
