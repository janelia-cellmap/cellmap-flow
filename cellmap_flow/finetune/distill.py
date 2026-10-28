"""Distill a finetuned model into a small copy the browser can run, and export the copy for the browser.

The large cellmap UNets (about 800M parameters, 3.2 GB, such as
mito-aff-unet-setup-16) are too big for a browser to load, and export_web
gives them no browser form. This trains a small network to compute what a
finetuned model's network computes, and exports it as export_web exports a
model's browser form. The network surgery is cellmap-models' (its
``distill_student``); this is the job around it:

- The teacher is the network the serving YAML serves, as export_web builds
  it (``export_web.finetuned_network``): the base as the trainer builds it,
  the finetune folded in, every layer a real one.
- The student is cellmap-models' ``distill_student``: for a funlib UNet
  (what ``prune_funlib_unet`` covers, such as the cellmap StandardUnets) the
  same network with fewer channels at every level, started from the
  teacher's strongest channels (``init`` "pruned"); for any other network
  (ResUNet, Unet3dIso...) a fly_organelles StandardUnet with the teacher's
  channels and output activation, started at random (``init`` "random"),
  at the largest block whose output fits inside the teacher's. A random
  start has more to learn, so it gets more epochs and hours by default.
- The distillation trains the student to reproduce the teacher's output as
  the network emits it, the mean squared difference over every voxel and
  channel (losses.distillation_loss, the finetune's teacher term), on
  blocks of the raw the model was finetuned on, read and normalized as the
  trainer reads its patches (data.loader.dataset_from_manifest): half
  centred on what was annotated, half at random points of the volume. It
  needs no annotations. Each block is read large enough for both networks'
  inputs; both read and write around the same centre, as the inference
  runner places a model's output in the middle of what it reads
  (ModelGeometry.context), so the teacher's input and the student's are
  the middle of the block, and the student's target is the middle of the
  teacher's output. Squared difference of the raw output because the
  output may be logits, affinities or distances, which the network cannot
  tell apart (probe_output_class calls all three unbounded): what is used
  downstream of affinities and distances is their value, every level of it
  (watershed and agglomeration thresholds, distance cut-offs), and a
  student that matches the values matches any threshold or postprocessing
  on them, logits' included. A sigmoid's cross-entropy would read distances
  as logits and stop caring where both are far from 0. Adam takes the
  loss's scale out of its steps, so logits of about 5 or about 1400
  (export_merged's finetunes) train alike.
- It runs on the GPU when there is one (utils.device.best_device), the
  training steps in mixed precision (trainer.recovery.Precision: bf16
  where the GPU has it), the held-out comparison in float32 with TF32 off,
  so that its numbers are the networks' own.
- How close: on held-out blocks drawn apart from the training's,
  cellmap-models' ``output_difference`` of the student from the teacher
  (the export's metrics: the mean squared, mean and largest absolute
  difference, and the share of voxels on the other side of the model's
  decision boundary, 0.5 for a model that ends in a sigmoid and else 0, as
  portable.label_it.probability reads it), for the start and for the
  student exported.

What it writes in ``<output>``:

- ``<folder>-small/manifest.json`` and its ONNX files, as export_web writes
  a model's browser form (``<folder>`` is export_web's folder for the
  finetuned model), with a ``distillation`` block: the teacher's folder (so
  that a publish can say "distilled from" its ref), the teacher's
  parameters, the student's ``init``, num_fmaps and fmap_inc_factor, the
  steps it trained, and the held-out closeness of the ``start`` and of the
  student (``distilled``);
- ``student_best.pt``: the best student so far, written each time the
  held-out difference improves, so a job killed before it exports keeps
  it; a run that makes the same student continues from it;
- ``result.json`` and ``export_log.txt``, as export_web's: ``{"folder",
  "manifest"}`` or ``{"error"}``, also printed as the output's last line.

The progress lines are the trainer's, "Starting epoch N of M" and "Epoch
N/M - Loss: ..." (markers' patterns), each epoch's with the held-out
closeness after it.

The dashboard starts it and follows it as an export, with export_web's
``export_status``::

    run = start_distill(yaml_path, corrections_dir, run_dir)
    export_web.export_status(run)   # the answer start_export's gets

``start_distill`` submits the job as start_export does (job_manager.submit's
module_command and launch, in the environment the model finetunes in, on
its YAML's queue), into ``<run_dir>/web_export/<YAML's name>-small/``, with
the training's time limit (the longer, a random start's, unless given) and
an hour for the rest as its walltime, and returns at once. From the command
line::

    python -m cellmap_flow.finetune.distill --yaml <serving yaml> \\
        --corrections <session>/corrections --output <dir> \\
        [--parameters 12000000] [--epochs 100] [--steps-per-epoch 100] [--hours 3]

The student's size: at most ``--parameters``, by default a tenth of the
teacher's and no more than BROWSER_PARAMETERS, about what the browser
catalog's largest models have (distill_student says how the widths are
chosen). For mito-aff-unet-setup-16 (792M parameters) that is num_fmaps 13
and fmap_inc_factor 3, 10.8M parameters, 43 MB in fp32.

A 2D model (a block one z-slice thick) gets a 2D student (distill_student's
1x3x3 kernels and 1x2x2 pooling), trained and exported on slices. The
browser copy is exported at the block the student trained at and, for a
StandardUnet student, at the larger blocks it computes too (BROWSER_SIDES,
cellmap-models' Student.blocks), each checked against the student on real
raw.

What it refuses, before it trains: a model that writes at another voxel size
than it reads (the student writes at the voxel size it reads); one that
takes more than one input channel (the browser feeds one); a teacher served
other than by its network's forward (Cellpose); and a teacher whose output
block holds no StandardUnet block.

Needs cellmap-models with ``model_export.student`` and ``model_export.onnx_web``
(the pin in pixi.toml's finetune feature) and its export extra, onnx and
onnxruntime, where it runs.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from cellmap_flow.finetune import export_web
from cellmap_flow.portable.publish import SMALL_SUFFIX

logger = logging.getLogger(__name__)

MODULE = "cellmap_flow.finetune.distill"
CHECKPOINT_NAME = "student_best.pt"
# The browser catalog's models (cellmap/onnx-web-models) have 5.6M
# (setup_53) and 10.7M (setup_54) parameters, 22 and 43 MB in fp32, and run
# in WebGPU at blocks of 116 to 172.
BROWSER_PARAMETERS = 12_000_000
# The default student has at most this share of the teacher's parameters.
TEACHER_SHARE = 0.1
# The share of training blocks centred at a random point of the volume; the
# rest are centred on what was annotated, where the model was finetuned.
RANDOM_SHARE = 0.5
# Blocks the student is measured on and never trains on, drawn with a seed
# of their own.
HELD_OUT_BLOCKS = 8
HELD_OUT_SEED = 1_000_000_007
# The default budget, by how the student starts: a pruned one is close to
# the teacher already, a random one learns it from nothing.
EPOCHS = {"pruned": 100, "random": 300}
HOURS = {"pruned": 3.0, "random": 8.0}
STEPS_PER_EPOCH = 100
LEARNING_RATE = 1e-4
WORKERS = 4
# The sides of the blocks the browser copy is exported at besides the one it
# trained at, where it computes them (cellmap-models' Student.blocks: a
# StandardUnet): as cellmap/onnx-web-models' models are, 3D blocks of up to
# about 240 voxels a side (setup_53's largest is 172, setup-16's served block
# 178), and 2D slices of a few hundred to a thousand (setup_53_2d's 300, 556
# and 1068). The browser runs the largest 3D block its GPU holds and the 2D
# side nearest 512 (web/src/models/variant.ts), and most of a 178 -> 56
# block is the context around its output.
BROWSER_SIDES = {"3d": (210, 242), "2d": (300, 556, 1068)}
# The training's time limit leaves this long, in the job's walltime, for
# building the teacher, reading the held-out blocks and the export.
SETUP_HOURS = 1.0


def _check_distillable(finetuned, geometry) -> None:
    """Raise ValueError for a model a small copy cannot be made of (see the
    module's docstring)."""
    input_voxel_size = tuple(finetuned.input_voxel_size or geometry.input_voxel_size)
    output_voxel_size = tuple(finetuned.output_voxel_size or geometry.output_voxel_size)
    if len(geometry.input_shape) != 3:
        raise ValueError(f"The model's block is {'x'.join(map(str, geometry.input_shape))} voxels: the small "
                         "copy is a network of (z, y, x) blocks, one z-slice thick for a 2D model")
    if input_voxel_size != output_voxel_size:
        raise ValueError(f"The model reads {input_voxel_size} nm voxels and writes {output_voxel_size} nm ones: "
                         "the small copy writes at the voxel size it reads")
    if geometry.input_channels != 1:
        raise ValueError(f"The model takes {geometry.input_channels} input channels; the browser feeds one")


def make_student(teacher, geometry, out_channels, output_class, parameters=None):
    """cellmap-models' student of ``teacher`` (its ``Student``), for the
    model of ``geometry``, whose output has ``out_channels`` channels and is
    of ``output_class`` (probe_output_class): at most ``parameters``
    parameters, by default a tenth of the teacher's and at most
    BROWSER_PARAMETERS.

    Raises:
        ValueError: no student is that small, or one that large would not
            fit an .onnx file, or no StandardUnet block fits in the teacher's.
    """
    import torch

    try:
        from cellmap_models.model_export.student import distill_student
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. The small copy needs cellmap-models' model_export.student (the commit pixi.toml's "
            "finetune feature pins)."
        ) from e

    from cellmap_flow.portable.label_it import SIGNED_UNIT, UNIT

    if parameters is None:
        teacher_parameters = sum(p.numel() for p in teacher.parameters())
        parameters = min(int(TEACHER_SHARE * teacher_parameters), BROWSER_PARAMETERS)
    if 4 * parameters >= export_web.ONNX_FILE_LIMIT:
        raise ValueError(f"A student of {parameters:,} parameters would not fit an .onnx file "
                         "(protobuf caps one at 2 GB)")
    # A random student ends as the teacher ends, so that its output reads as
    # the teacher's (the manifest's "output", the browser's thresholds).
    activation = {UNIT: torch.nn.Sigmoid, SIGNED_UNIT: torch.nn.Tanh}.get(output_class)
    return distill_student(
        teacher, parameters, geometry.input_shape, geometry.output_shape, out_channels,
        activation=activation() if activation else None,
    )


def _centre(block, shape):
    """The middle ``shape`` voxels of ``block``'s spatial axes: where a
    network that reads ``block`` places an output of ``shape``, as the
    inference runner grows a model's output by its context on each side
    (ModelGeometry.context), and what a network of input ``shape`` reads
    around the same centre."""
    starts = [(have - want) // 2 for have, want in zip(block.shape[-3:], shape)]
    return block[(..., *(slice(start, start + want) for start, want in zip(starts, shape)))]


def closeness(network, blocks, targets, input_size, threshold: float) -> dict:
    """How far ``network``'s output on ``blocks`` (the middle ``input_size``
    of each) is from ``targets``, the teacher's: cellmap-models'
    output_difference (``mse``, ``mean_abs``, ``max_abs``, ``flip_frac`` on
    ``threshold``) over all of them, computed in float32 with TF32 off."""
    import numpy as np
    import torch
    from cellmap_models.model_export.onnx_web import output_difference

    device = next(network.parameters()).device
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        each = [output_difference(network(_centre(block, input_size).to(device)).float().cpu().numpy(),
                                  target.numpy(), threshold)
                for block, target in zip(blocks, targets)]
    # The blocks are of one size, so the mean of their means is the mean.
    return {key: float((np.max if key == "max_abs" else np.mean)([e[key] for e in each])) for key in each[0]}


def _blocks(loader):
    """The loader's raw blocks, one batch after another, epoch after epoch."""
    while True:
        for batch in loader:
            yield batch[0]


def _save(path: Path, checkpoint: dict) -> None:
    """``checkpoint`` at ``path``, written whole before it replaces the one there."""
    import torch

    partial = path.with_name(f".{path.name}.partial")
    torch.save(checkpoint, partial)
    os.replace(partial, path)


def _format(numbers: dict) -> str:
    return (f"mse {numbers['mse']:.6g}, mean |diff| {numbers['mean_abs']:.4g}, "
            f"max |diff| {numbers['max_abs']:.4g}, flips {numbers['flip_frac']:.2%}")


def distill(yaml_path, corrections_dir, output_dir, *, parameters=None, epochs=None,
            steps_per_epoch=STEPS_PER_EPOCH, hours=None, learning_rate=LEARNING_RATE,
            workers=WORKERS, variants=None) -> dict:
    """Distill the model ``yaml_path`` serves into a small copy and export
    it into ``output_dir``; ``{"folder", "manifest"}`` (see the module's
    docstring).

    Trains for ``epochs`` of ``steps_per_epoch`` blocks, or until ``hours``
    have passed, whichever is first (by default EPOCHS and HOURS of how the
    student starts), and exports the student that came closest to the
    teacher on the held-out blocks.

    Raises:
        ValueError: the model is one this cannot distill or export (see
            the module's docstring), or its session has no training manifest.
    """
    import numpy as np
    import torch

    from cellmap_flow.finetune.data.loader import dataset_from_manifest, make_training_loader
    from cellmap_flow.finetune.losses import distillation_loss
    from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME, read_manifest
    from cellmap_flow.finetune.trainer.recovery import Precision
    from cellmap_flow.finetune.trainer.supervised import probe_output_class
    from cellmap_flow.models.geometry import ModelGeometry
    from cellmap_flow.portable.label_it import UNIT
    from cellmap_flow.utils.device import best_device

    output_dir = Path(output_dir)
    teacher, finetuned, input_norm = export_web.finetuned_network(yaml_path)
    geometry = finetuned.base_model_config.geometry
    _check_distillable(finetuned, geometry)
    manifest = read_manifest(str(corrections_dir))
    if manifest is None:
        raise ValueError(f"{corrections_dir} has no {VIRTUAL_MANIFEST_FILENAME}, so the raw the model "
                         "was trained on is not known")

    # The teacher, on the GPU when there is one, and what its output is.
    device = best_device()
    teacher = teacher.to(device).eval().requires_grad_(False)
    probe = torch.from_numpy(export_web.training_block(
        corrections_dir, geometry.input_shape, geometry.output_shape, input_norm)).to(device)
    precision = Precision(device, enabled=True)
    with torch.no_grad(), precision.autocast():
        output = teacher(probe)
    if not torch.isfinite(output).all():
        precision.fall_back_to_fp32()
    out_channels = int(output.shape[1])
    output_class = probe_output_class(teacher, probe)
    # The decision boundary label_it.probability reads the output by.
    threshold = 0.5 if output_class == UNIT else 0.0

    student = make_student(teacher, geometry, out_channels, output_class, parameters)
    teacher_parameters = sum(p.numel() for p in teacher.parameters())
    epochs = EPOCHS[student.init] if epochs is None else epochs
    hours = HOURS[student.init] if hours is None else hours
    logger.info(f"Student: {student.init} start, num_fmaps {student.num_fmaps}, fmap_inc_factor "
                f"{student.fmap_inc_factor}, {student.parameters:,} parameters, "
                f"{student.parameters / teacher_parameters:.1%} of the teacher's {teacher_parameters:,}; "
                f"{'x'.join(map(str, student.input_size))} voxels in, "
                f"{'x'.join(map(str, student.output_size))} out")

    # Blocks large enough for both inputs, around the student's output.
    read = {**manifest, "input_norm": input_norm, "anchor_fraction": RANDOM_SHARE, "augment": False}
    block = (tuple(max(t, s) for t, s in zip(geometry.input_shape, student.input_size)), student.output_size)
    training = dataset_from_manifest({**read, "patches_per_epoch": steps_per_epoch}, patch_voxels=block)
    held_out = dataset_from_manifest({**read, "seed": HELD_OUT_SEED}, patch_voxels=block)
    blocks = [held_out[i][0][None] for i in range(HELD_OUT_BLOCKS)]

    def target(raw):
        """The teacher's output on ``raw``, the middle the student writes."""
        return _centre(teacher(_centre(raw, geometry.input_shape)), student.output_size)

    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        targets = [target(b.to(device)).float().cpu() for b in blocks]

    checkpoint_path = output_dir / CHECKPOINT_NAME
    made = {"init": student.init, "num_fmaps": student.num_fmaps, "fmap_inc_factor": student.fmap_inc_factor}
    earlier = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.exists() else None
    if earlier and {key: earlier.get(key) for key in made} != made:
        logger.info(f"The student in {checkpoint_path} is another; starting from a new one")
        earlier = None
    network = student.network.to(device)
    start = earlier["start"] if earlier else closeness(network, blocks, targets, student.input_size, threshold)
    steps = 0
    if earlier:
        network.load_state_dict(earlier["state_dict"])
        steps = earlier["steps"]
        logger.info(f"Continuing from the student in {checkpoint_path}, after {steps} step(s)")
    best = closeness(network, blocks, targets, student.input_size, threshold)
    logger.info(f"Held out, {'the student so far' if earlier else 'at the start'}: {_format(best)}")

    def save():
        _save(checkpoint_path, {"state_dict": network.state_dict(), **made, "steps": steps,
                                "held_out": best, "start": start})

    save()
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    # Persistent, spawned workers, as the trainer's (make_training_loader).
    batches = _blocks(make_training_loader(training, 1, workers))
    began = time.monotonic()
    deadline = began + 3600 * hours
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1} of {epochs}...")
        losses = []
        network.train()
        for _ in range(steps_per_epoch):
            raw = next(batches).to(device, non_blocking=True)
            with precision.autocast():
                with torch.no_grad():
                    goal = target(raw)
                loss = distillation_loss(network(_centre(raw, student.input_size)).float(), goal.float(), "all")
            optimizer.zero_grad(set_to_none=True)
            precision.scaler.scale(loss).backward()
            precision.scaler.step(optimizer)
            precision.scaler.update()
            losses.append(float(loss))
            steps += 1
            if time.monotonic() > deadline:
                break
        network.eval()
        held = closeness(network, blocks, targets, student.input_size, threshold)
        loss = float(np.mean(losses))
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f} - Held out: {_format(held)} - "
                    f"{(time.monotonic() - began) / steps:.2f} s a step")
        if not math.isfinite(loss):
            logger.warning("The loss is not finite; stopping, with the best student so far")
            break
        if held["mse"] < best["mse"]:
            best = held
            save()
        if time.monotonic() > deadline:
            logger.info(f"The time limit ({hours:g} h) is up; stopping, with the best student so far")
            break

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(best_checkpoint["state_dict"])
    teacher_folder = export_web.folder_name(finetuned.name)
    folder = f"{teacher_folder}{SMALL_SUFFIX}"
    student_geometry = ModelGeometry(
        input_voxel_size=geometry.input_voxel_size, output_voxel_size=geometry.output_voxel_size,
        read_shape=[n * v for n, v in zip(student.input_size, geometry.input_voxel_size)],
        write_shape=[n * v for n, v in zip(student.output_size, geometry.output_voxel_size)],
        output_channels=geometry.output_channels, channel_names=geometry.channel_names,
    )
    # The block it trained at, and the larger (and for a slice, smaller)
    # ones it computes too, as the catalog's models are exported.
    blocks = [student.input_size, *student.blocks(BROWSER_SIDES["2d" if student.input_size[0] == 1 else "3d"])]
    largest = tuple(max(b[i] for b in blocks) for i in range(3))
    logger.info(f"Exporting the student of step {best_checkpoint['steps']} ({_format(best)}) at "
                f"{', '.join('x'.join(map(str, b)) for b in blocks)} voxels")
    validation = export_web.training_block(corrections_dir, largest, student.output_size, input_norm)
    network.eval()
    with torch.no_grad():
        student_class = probe_output_class(network, torch.from_numpy(_centre(validation, student.input_size)).to(device))
        scale = float(network(torch.from_numpy(validation).to(device)).abs().max())
    description = {
        **export_web.describe(finetuned, student_geometry, folder, student_class, input_norm,
                              checkpoint=f"distilled with cellmap-flow from {teacher_folder}"),
        "distillation": {
            "teacher": teacher_folder,
            "teacher_params": teacher_parameters,
            "init": student.init,
            "num_fmaps": student.num_fmaps,
            "fmap_inc_factor": student.fmap_inc_factor,
            "steps": best_checkpoint["steps"],
            "held_out_blocks": HELD_OUT_BLOCKS,
            "teacher_mean_abs": float(np.mean([t.abs().mean() for t in targets])),
            "start": start,
            "distilled": best,
        },
    }
    manifest = export_web._export_browser(network, output_dir / folder, folder, student_geometry, validation,
                                          scale, description, variants, blocks)
    return {"folder": folder, "manifest": manifest}


def _walltime(hours: float) -> str:
    """The job's walltime, "HH:MM": the training's time limit and SETUP_HOURS."""
    minutes = math.ceil(60 * (hours + SETUP_HOURS))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def start_distill(yaml_path, corrections_dir, run_dir, *, parameters=None, hours=None) -> export_web.WebExport:
    """Start distilling the model ``yaml_path`` serves, trained on the
    session ``corrections_dir``, into ``<run_dir>/web_export/<YAML's
    name>-small/``; follow it with export_web.export_status.

    ``parameters``: the student's largest size (by default make_student's);
    ``hours``: the training's time limit (by default HOURS of how the
    student starts, which the job decides; the walltime allows the longer).
    The job runs as start_export's does, in the environment the model
    finetunes in on the YAML's queue and charge group, and a result an
    earlier run left there is removed first, so that export_status does not
    read it as this one's; its best student is kept, for this run to
    continue from.

    Raises:
        ValueError: the YAML serves no finetuned model, or its environment
            cannot run the job.
        RuntimeError: the job could not be submitted.
    """
    from cellmap_flow.finetune.job_manager import submit
    from cellmap_flow.finetune.model_loading import model_config_from_entry

    config, entry = export_web.serving_entry(yaml_path)
    # The student's folder, and its export's directory, are the finetuned model's with SMALL_SUFFIX after them.
    output_dir = Path(run_dir) / "web_export" / f"{Path(yaml_path).stem}{SMALL_SUFFIX}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / export_web.RESULT_NAME).unlink(missing_ok=True)
    args = ["--yaml", str(yaml_path), "--corrections", str(corrections_dir), "--output", str(output_dir),
            *(["--hours", str(hours)] if hours else []), *(["--parameters", str(parameters)] if parameters else [])]
    command = submit.module_command(model_config_from_entry(entry), MODULE, args, output_dir / export_web.LOG_NAME)
    job = submit.launch(
        f"distill_{export_web.folder_name(entry.get('name') or 'model')}", command,
        queue=config.get("queue"), charge_group=config.get("charge_group"),
        walltime=_walltime(hours or max(HOURS.values())),
    )
    return export_web.WebExport(output_dir, job)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yaml", required=True, help="the finetuned model's serving YAML")
    parser.add_argument("--corrections", required=True,
                        help="the session's corrections directory, which the model was trained on")
    parser.add_argument("--output", required=True, help="the directory to distill and export into")
    parser.add_argument("--parameters", type=int, default=None,
                        help="the student's largest number of parameters (default: a tenth of the teacher's, "
                             f"at most {BROWSER_PARAMETERS:,})")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"epochs (default: {EPOCHS['pruned']} from a pruned start, {EPOCHS['random']} from "
                             "a random one)")
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH, help="blocks per epoch, one a step")
    parser.add_argument("--hours", type=float, default=None,
                        help=f"the training's time limit (default: {HOURS['pruned']:g} from a pruned start, "
                             f"{HOURS['random']:g} from a random one)")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="AdamW's")
    parser.add_argument("--workers", type=int, default=WORKERS, help="data loader workers")
    parser.add_argument("--variants", default=None,
                        help="comma-separated onnx_web precisions (fp32,fp16,int8w,int8w_fp16; all by default)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    try:
        result = distill(
            args.yaml, args.corrections, output_dir, parameters=args.parameters, epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch, hours=args.hours, learning_rate=args.learning_rate,
            workers=args.workers, variants=variants,
        )
    except Exception as e:
        logger.exception("The distillation failed")
        result = {"error": export_web._said(e)}
    export_web._write_result(output_dir, result)
    print(json.dumps(result), flush=True)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
