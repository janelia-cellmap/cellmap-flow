"""Distill a finetuned model too big for the browser into a small copy, and export the copy for the browser.

The large cellmap UNets (about 800M parameters, 3.2 GB, such as
mito-aff-unet-setup-16) are too big for a browser to load, and export_web
refuses them. This trains a narrower copy of a finetuned model's network to
compute what the network computes, and exports the copy as export_web
exports a model:

- The teacher is the network the serving YAML serves, built as export_web
  builds it: the base as the trainer builds it
  (model_loading.load_trainable_model), the finetune folded in
  (export_merged.apply_finetune), every layer a real one.
- The student is that network with fewer channels at every level. A funlib
  UNet (``funlib.learn.torch``'s, alone or with one final convolution after
  it, as fly_organelles' StandardUnet has) has ``num_fmaps *
  fmap_inc_factor**level`` channels at each level; the student has the
  same levels, convolutions and kernels with a smaller num_fmaps and
  fmap_inc_factor (``student_widths``), and starts from the teacher's own
  weights, each level's strongest channels (cellmap-models'
  ``prune_funlib_unet``, structured L1 selection). It is the teacher's own
  module tree narrowed, not a new instance of its class: a cellmap or
  Hugging Face model trains as torch.export's unflattened program, and its
  class (fly_organelles') is not installed where models finetune. Its
  weights load into ``StandardUnet(channels, num_fmaps=..,
  fmap_inc_factor=..)`` all the same.
- The distillation trains the student to reproduce the teacher's output as
  the network emits it, the mean squared difference over every voxel and
  channel (losses.distillation_loss, the finetune's teacher term), on
  blocks of the raw the model was finetuned on, read and normalized as the
  trainer reads its patches (data.loader.dataset_from_manifest), at the
  block the model is served at: half centred on what was annotated, half at
  random points of the volume. It needs no annotations. Squared difference
  of the raw output because the output may be logits, affinities or
  distances, which the network cannot tell apart (probe_output_class calls
  all three unbounded): what is used downstream of affinities and
  distances is their value, every level of it (watershed and agglomeration
  thresholds, distance cut-offs), and a student that matches the values
  matches any threshold or postprocessing on them, logits' included. A
  sigmoid's cross-entropy would read distances as logits and stop caring
  where both are far from 0. Adam takes the loss's scale out of its steps,
  so logits of about 5 or about 1400 (export_merged's finetunes) train
  alike.
- How close: on held-out blocks drawn apart from the training's, the
  student's mean squared, mean and largest absolute difference from the
  teacher, and the share of voxels on the other side of the model's
  decision boundary (``flip_frac``, export_onnx_web's metric; the boundary
  is 0.5 for a model that ends in a sigmoid, else 0, as
  portable.label_it.probability reads it), for the pruned start and for
  the student exported.

What it writes in ``<output>``:

- ``<folder>-small/manifest.json`` and its ONNX files, as export_web writes
  a model's (``<folder>`` is export_web's folder for the finetuned model),
  with a ``distillation`` block: the teacher's folder (so that a publish
  can say "distilled from" its ref), the teacher's parameters, the
  student's num_fmaps and fmap_inc_factor, the steps it trained, and the
  held-out closeness of the pruned start and of the student;
- ``student_best.pt``: the best student so far, written each time the
  held-out difference improves, so a job killed before it exports keeps
  it; a run with the same widths continues from it;
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
the training's time limit and an hour for the rest as its walltime, and
returns at once. From the command line::

    python -m cellmap_flow.finetune.distill --yaml <serving yaml> \\
        --corrections <session>/corrections --output <dir> \\
        [--parameters 12000000] [--epochs 100] [--steps-per-epoch 100] [--hours 3]

The student's size: at most ``--parameters``, by default a tenth of the
teacher's and no more than BROWSER_PARAMETERS, about what the browser
catalog's largest models have. Of the widths that fit, those of at least
half the largest that fits are candidates, and the one with the most
channels at full resolution is taken: the deep levels hold most of the
parameters and add context, while the top level, where the output is
decided voxel by voxel, costs few. For mito-aff-unet-setup-16 (num_fmaps 16,
fmap_inc_factor 6, 792M parameters) that is num_fmaps 13 and
fmap_inc_factor 3, 10.8M parameters, 43 MB in fp32.

What it refuses, before it trains or moves anything to the GPU: a network
that is not a funlib UNet with one head (prune_funlib_unet's), alone or
with one final convolution; one whose levels' widths are not ``num_fmaps *
fmap_inc_factor**level``; and whatever export_web refuses to export (a
model that is not a cube in and out, more than one input channel).

Needs cellmap-models with ``model_export.prune`` and ``model_export.onnx_web``
(the onnx-web-export-and-pruning branch, which pixi.toml's finetune feature
installs) and its export extra, onnx and onnxruntime, where it runs.
"""

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cellmap_flow.finetune import export_web

logger = logging.getLogger(__name__)

MODULE = "cellmap_flow.finetune.distill"
# The student's folder, and its export's directory, are the finetuned
# model's with this after them.
SUFFIX = "-small"
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
EPOCHS = 100
STEPS_PER_EPOCH = 100
HOURS = 3.0
LEARNING_RATE = 1e-4
WORKERS = 4
# The training's time limit leaves this long, in the job's walltime, for
# building the teacher, reading the held-out blocks and the export.
SETUP_HOURS = 1.0


@dataclass(frozen=True)
class UNetLayout:
    """Where a funlib UNet's tensors are in a network's state dict, and its widths.

    ``prefix``: the UNet's key prefix ("unet_backbone." in StandardUnet,
    "" for a bare UNet). ``final_key``: the convolution after it, or None.
    ``encoder``, ``decoder``: the indices of each level's convolutions in
    its ``conv_pass``, by level (the decoder has one level fewer).
    ``outputs``: the decoder's top level's channels, which are the network's
    output when there is no final convolution.
    """

    prefix: str
    final_key: Optional[str]
    encoder: tuple
    decoder: tuple
    in_channels: int
    num_fmaps: int
    fmap_inc_factor: int
    outputs: int

    @property
    def levels(self) -> int:
        return len(self.encoder)


def _refuse(why: str):
    return ValueError(f"{why}. The small copy is made by prune_funlib_unet, which prunes a funlib UNet "
                      "with one head, alone or with one final convolution after it, and nothing else")


def _conv_indices(state, block: str) -> tuple:
    """The indices of the convolutions in ``<block>.conv_pass``, in order."""
    pattern = re.compile(re.escape(block) + r"\.conv_pass\.(\d+)\.weight")
    return tuple(sorted(int(m.group(1)) for key in state if (m := pattern.fullmatch(key))))


def unet_layout(state) -> UNetLayout:
    """The funlib UNet in the state dict ``state``, as a UNetLayout.

    Raises:
        ValueError: ``state`` is not a funlib UNet's that prune_funlib_unet
            prunes: no UNet in it, or more than one, or more than one head;
            levels whose widths are not ``num_fmaps * fmap_inc_factor**level``;
            any tensor besides the UNet's and one final convolution's, or one
            of a shape funlib's UNet would not give it.
    """
    prefixes = {m.group(1) for key in state if (m := re.fullmatch(r"(.*)l_conv\.\d+\.conv_pass\.\d+\.weight", key))}
    if len(prefixes) != 1:
        raise _refuse("The network has no funlib UNet in it (no l_conv.<level>.conv_pass convolutions)"
                      if not prefixes else f"The network has {len(prefixes)} funlib UNets in it")
    prefix = prefixes.pop()
    if any(key.startswith(f"{prefix}r_conv.1.") for key in state):
        raise _refuse("The network's UNet has more than one head")
    levels = 0
    while _conv_indices(state, f"{prefix}l_conv.{levels}"):
        levels += 1
    encoder = tuple(_conv_indices(state, f"{prefix}l_conv.{level}") for level in range(levels))
    decoder = tuple(_conv_indices(state, f"{prefix}r_conv.0.{level}") for level in range(levels - 1))
    if levels < 2 or not all(decoder):
        raise _refuse(f"The network's UNet has {levels} encoder level(s) and decoder levels "
                      f"{[level for level, convs in enumerate(decoder) if convs]}")
    weights = [state[f"{prefix}l_conv.{level}.conv_pass.{encoder[level][-1]}.weight"] for level in range(levels)]
    widths = [int(w.shape[0]) for w in weights]
    num_fmaps, factor = widths[0], widths[1] // widths[0]
    if any(width != num_fmaps * factor**level for level, width in enumerate(widths)):
        raise _refuse(f"The network's UNet has {', '.join(map(str, widths))} channels at its levels, "
                      "not num_fmaps * fmap_inc_factor**level")
    others = sorted({key.rpartition(".")[0] for key in state if not key.startswith(prefix)})
    if len(others) > 1 or (others and f"{others[0]}.weight" not in state):
        raise _refuse(f"The network has layers besides its UNet and one final convolution: {', '.join(others)}")
    top = state[f"{prefix}r_conv.0.0.conv_pass.{decoder[0][-1]}.weight"]
    layout = UNetLayout(
        prefix=prefix, final_key=others[0] if others else None, encoder=encoder, decoder=decoder,
        in_channels=int(state[f"{prefix}l_conv.0.conv_pass.{encoder[0][0]}.weight"].shape[1]),
        num_fmaps=num_fmaps, fmap_inc_factor=factor, outputs=int(top.shape[0]),
    )
    # The teacher is the student of its own widths: the shapes funlib's UNet
    # gives every tensor at them, compared with its own, check the rest of
    # it, from the decoder's inputs to the transposed convolutions.
    expected = student_shapes(state, layout, num_fmaps, factor)
    for key in sorted(set(expected) | set(state)):
        if key not in expected or key not in state or tuple(state[key].shape) != expected[key]:
            have = tuple(state[key].shape) if key in state else "nothing"
            raise _refuse(f"The network is not laid out as a funlib UNet: {key} is {have}, where funlib's "
                          f"UNet would have {expected.get(key, 'nothing')}")
    return layout


def student_shapes(state, layout: UNetLayout, num_fmaps: int, fmap_inc_factor: int) -> dict:
    """The shape of each of ``state``'s tensors in a copy of its UNet with
    ``num_fmaps * fmap_inc_factor**level`` channels at each level, as
    funlib's UNet lays them out: each level's convolutions from the level
    above's channels (the input's at the top) to its own, the decoder's from
    the skip's channels and the upsampled level's below, concatenated, a
    transposed convolution's (constant_upsample=False) from and to the level
    below's, and the final convolution's from the decoder's top level."""
    width = [num_fmaps * fmap_inc_factor**level for level in range(layout.levels)]
    # Without a final convolution, the decoder's top level is the output,
    # whose channels the copy keeps.
    top = width[0] if layout.final_key else layout.outputs
    shapes = {}

    def conv(key, outputs, inputs, transposed=False):
        kernel = tuple(state[f"{key}.weight"].shape[2:])
        shapes[f"{key}.weight"] = ((inputs, outputs) if transposed else (outputs, inputs)) + kernel
        if f"{key}.bias" in state:
            shapes[f"{key}.bias"] = (outputs,)

    def conv_pass(block, convs, outputs, inputs):
        for j, index in enumerate(convs):
            conv(f"{block}.conv_pass.{index}", outputs, inputs if j == 0 else outputs)

    p = layout.prefix
    for level, convs in enumerate(layout.encoder):
        conv_pass(f"{p}l_conv.{level}", convs, width[level], width[level - 1] if level else layout.in_channels)
    for level, convs in enumerate(layout.decoder):
        up = f"{p}r_up.0.{level}.up"
        if f"{up}.weight" in state:
            conv(up, width[level + 1], width[level + 1], transposed=True)
        conv_pass(f"{p}r_conv.0.{level}", convs, top if level == 0 else width[level],
                  width[level] + width[level + 1])
    if layout.final_key:
        final = state[f"{layout.final_key}.weight"]
        conv(layout.final_key, int(final.shape[0]), top)
    return shapes


def _parameters(shapes: dict) -> int:
    return sum(math.prod(shape) for shape in shapes.values())


def student_widths(state, layout: UNetLayout, parameters: int) -> tuple:
    """The student's ``(num_fmaps, fmap_inc_factor)``, at most ``parameters`` parameters.

    Each at most the teacher's, so that every level keeps a subset of the
    teacher's channels. Of the widths that fit, those of at least half the
    largest that fits are candidates, and the one with the most channels at
    full resolution is taken (the module's docstring says why), the larger
    of two with as many.

    Raises:
        ValueError: no student is that small, or one that large would not
            fit an .onnx file.
    """
    if 4 * parameters >= export_web.ONNX_FILE_LIMIT:
        raise ValueError(f"A student of {parameters:,} parameters would not fit an .onnx file "
                         "(protobuf caps one at 2 GB)")
    fitting = {}
    for num_fmaps in range(1, layout.num_fmaps + 1):
        for factor in range(1, layout.fmap_inc_factor + 1):
            count = _parameters(student_shapes(state, layout, num_fmaps, factor))
            if count <= parameters and (num_fmaps, factor) != (layout.num_fmaps, layout.fmap_inc_factor):
                fitting[num_fmaps, factor] = count
    if not fitting:
        raise ValueError(f"No copy of the network with fewer channels has at most {parameters:,} parameters")
    largest = max(fitting.values())
    return max((widths for widths, count in fitting.items() if count >= largest / 2),
               key=lambda widths: (widths[0], fitting[widths]))


def _narrowed(network, state):
    """A copy of ``network`` whose tensors are ``state``'s, each convolution's
    channel counts made its new weight's: onnx_web's rewrites build their
    layers from them."""
    import torch

    narrow = copy.deepcopy(network)
    for key, tensor in state.items():
        path, _, name = key.rpartition(".")
        setattr(narrow.get_submodule(path), name, torch.nn.Parameter(tensor.detach().clone()))
    convs = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    transposed = (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
    for module in narrow.modules():
        if isinstance(module, transposed):
            module.in_channels, module.out_channels = module.weight.shape[0], module.weight.shape[1] * module.groups
        elif isinstance(module, convs):
            module.in_channels, module.out_channels = module.weight.shape[1] * module.groups, module.weight.shape[0]
    return narrow


@dataclass(frozen=True)
class Student:
    """A small copy of a network (``small_copy``) and what it was made as."""

    network: object
    num_fmaps: int
    fmap_inc_factor: int
    parameters: int
    teacher_parameters: int


def small_copy(teacher, parameters: Optional[int] = None) -> Student:
    """The student of the funlib UNet ``teacher``: the same network with
    ``student_widths``' channels, at most ``parameters`` parameters (by
    default a tenth of the teacher's, at most BROWSER_PARAMETERS), its
    weights the teacher's strongest channels (prune_funlib_unet).

    Raises:
        ValueError: ``teacher`` is not a network prune_funlib_unet prunes
            (``unet_layout``), or no student is that small.
    """
    import torch

    try:
        from cellmap_models.model_export.prune import prune_funlib_unet
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. The small copy needs cellmap-models' prune (its onnx-web-export-and-pruning branch, "
            "which pixi.toml's finetune feature installs)."
        ) from e

    state = teacher.state_dict()
    layout = unet_layout(state)
    teacher_parameters = _parameters({key: tuple(t.shape) for key, t in state.items()})
    if parameters is None:
        parameters = min(int(TEACHER_SHARE * teacher_parameters), BROWSER_PARAMETERS)
    num_fmaps, factor = student_widths(state, layout, parameters)
    shapes = student_shapes(state, layout, num_fmaps, factor)
    # prune_funlib_unet reads only the student's shapes.
    pruned = prune_funlib_unet(
        state, {key: torch.empty(shape, device="meta") for key, shape in shapes.items()},
        prefix=layout.prefix, final_key=layout.final_key,
    )
    return Student(_narrowed(teacher, pruned), num_fmaps, factor, _parameters(shapes), teacher_parameters)


def teacher_network(yaml_path):
    """The finetuned network ``yaml_path`` serves, on the CPU, as export_web
    exports one: its base as the trainer builds it, the finetune folded in,
    every layer a real one; ``(network, config, finetuned)``, with the
    serving YAML's config and the finetuned model's config.

    Raises:
        ValueError: the YAML serves no finetuned model, or one that is not a
            cube in and out or takes more than one input channel.
    """
    from cellmap_flow.finetune.export_merged import apply_finetune
    from cellmap_flow.finetune.lora_wrapper import _replace_interpreter_modules
    from cellmap_flow.finetune.model_loading import load_trainable_model, model_config_from_entry

    config, entry = export_web.serving_entry(yaml_path)
    finetuned = model_config_from_entry(entry)
    geometry = finetuned.base_model_config.geometry
    export_web._side(geometry.input_shape, "input")
    export_web._side(geometry.output_shape, "output")
    if geometry.input_channels != 1:
        raise ValueError(f"The model takes {geometry.input_channels} input channels; the browser feeds one")
    network = apply_finetune(
        load_trainable_model(finetuned.base_model_config), finetuned.lora_adapter_path, finetuned.weights_path
    ).cpu()
    # Real layers where torch.export's unflatten left interpreters (a
    # TorchScript base's), sharing the weights: the student is a narrowed
    # copy of them, and onnx_web rewrites only real ones.
    _replace_interpreter_modules(network, transposed=True)
    return network, config, finetuned


def closeness(network, blocks, targets, threshold: float) -> dict:
    """How far ``network``'s output on ``blocks`` is from ``targets``, the
    teacher's: the mean squared difference (``mse``), the mean and largest
    absolute difference (``mean_abs``, ``max_abs``), and the share of
    voxels and channels on the other side of ``threshold`` (``flip_frac``),
    in float32."""
    import torch

    device = next(network.parameters()).device
    squared = absolute = largest = flipped = voxels = 0.0
    with torch.no_grad():
        for block, target in zip(blocks, targets):
            output = network(block.to(device)).float().cpu()
            difference = (output - target).abs()
            squared += float((difference**2).sum())
            absolute += float(difference.sum())
            largest = max(largest, float(difference.max()))
            flipped += float(((output > threshold) != (target > threshold)).sum())
            voxels += difference.numel()
    return {"mse": squared / voxels, "mean_abs": absolute / voxels, "max_abs": largest, "flip_frac": flipped / voxels}


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


def distill(yaml_path, corrections_dir, output_dir, *, parameters=None, epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH, hours=HOURS, learning_rate=LEARNING_RATE,
            workers=WORKERS, variants=None) -> dict:
    """Distill the model ``yaml_path`` serves into a small copy and export
    it into ``output_dir``; ``{"folder", "manifest"}`` (see the module's
    docstring).

    Trains for ``epochs`` of ``steps_per_epoch`` blocks, or until ``hours``
    have passed, whichever is first, and exports the student that came
    closest to the teacher on the held-out blocks.

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
    from cellmap_flow.portable.label_it import UNIT
    from cellmap_flow.utils.device import best_device

    output_dir = Path(output_dir)
    teacher, config, finetuned = teacher_network(yaml_path)
    student = small_copy(teacher, parameters)
    logger.info(f"Student: num_fmaps {student.num_fmaps}, fmap_inc_factor {student.fmap_inc_factor}, "
                f"{student.parameters:,} parameters, {student.parameters / student.teacher_parameters:.1%} "
                f"of the teacher's {student.teacher_parameters:,}")

    manifest = read_manifest(str(corrections_dir))
    if manifest is None:
        raise ValueError(f"{corrections_dir} has no {VIRTUAL_MANIFEST_FILENAME}, so the raw the model "
                         "was trained on is not known")
    geometry = finetuned.base_model_config.geometry
    input_norm = _input_norm(config)
    read = {**manifest, "input_norm": input_norm, "anchor_fraction": RANDOM_SHARE, "augment": False}
    block = (geometry.input_shape, geometry.output_shape)
    training = dataset_from_manifest({**read, "patches_per_epoch": steps_per_epoch}, patch_voxels=block)
    held_out = dataset_from_manifest({**read, "seed": HELD_OUT_SEED}, patch_voxels=block)
    blocks = [held_out[i][0][None] for i in range(HELD_OUT_BLOCKS)]

    checkpoint_path = output_dir / CHECKPOINT_NAME
    earlier = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.exists() else None
    if earlier and (earlier["num_fmaps"], earlier["fmap_inc_factor"]) != (student.num_fmaps, student.fmap_inc_factor):
        logger.info(f"The student in {checkpoint_path} has other widths; starting from the pruned network")
        earlier = None

    device = best_device()
    teacher = teacher.to(device).eval().requires_grad_(False)
    network = student.network.to(device)
    precision = Precision(device, enabled=True)
    first = blocks[0].to(device)
    with torch.no_grad(), precision.autocast():
        if not torch.isfinite(teacher(first)).all():
            precision.fall_back_to_fp32()
    output_class = probe_output_class(teacher, first)
    # The decision boundary label_it.probability reads the output by.
    threshold = 0.5 if output_class == UNIT else 0.0
    with torch.no_grad():
        targets = [teacher(b.to(device)).float().cpu() for b in blocks]
    pruned = earlier["pruned"] if earlier else closeness(network, blocks, targets, threshold)
    steps = 0
    if earlier:
        network.load_state_dict(earlier["state_dict"])
        steps = earlier["steps"]
        logger.info(f"Continuing from the student in {checkpoint_path}, after {steps} step(s)")
    best = closeness(network, blocks, targets, threshold)
    logger.info(f"Held out, {'the student so far' if earlier else 'pruned'}: {_format(best)}")

    def save():
        _save(checkpoint_path, {"state_dict": network.state_dict(), "num_fmaps": student.num_fmaps,
                                "fmap_inc_factor": student.fmap_inc_factor, "steps": steps,
                                "held_out": best, "pruned": pruned})

    save()
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    # Persistent, spawned workers, as the trainer's (make_training_loader).
    batches = _blocks(make_training_loader(training, 1, workers))
    deadline = time.monotonic() + 3600 * hours
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1} of {epochs}...")
        losses = []
        network.train()
        for _ in range(steps_per_epoch):
            raw = next(batches).to(device, non_blocking=True)
            with precision.autocast():
                with torch.no_grad():
                    target = teacher(raw)
                loss = distillation_loss(network(raw).float(), target.float(), "all")
            optimizer.zero_grad(set_to_none=True)
            precision.scaler.scale(loss).backward()
            precision.scaler.step(optimizer)
            precision.scaler.update()
            losses.append(float(loss))
            steps += 1
            if time.monotonic() > deadline:
                break
        network.eval()
        held = closeness(network, blocks, targets, threshold)
        loss = float(np.mean(losses))
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f} - Held out: {_format(held)}")
        if not math.isfinite(loss):
            logger.warning("The loss is not finite; stopping, with the best student so far")
            break
        if held["mse"] < best["mse"]:
            best = held
            save()
        if time.monotonic() > deadline:
            logger.info(f"The time limit ({hours:g} h) is up; stopping, with the best student so far")
            break

    best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    network.load_state_dict(best_checkpoint["state_dict"])
    logger.info(f"Exporting the student of step {best_checkpoint['steps']}: {_format(best)}")
    teacher_folder = export_web.folder_name(finetuned.name)
    return _export_student(
        network.cpu().eval(), config, finetuned, corrections_dir, output_dir, variants,
        folder=f"{teacher_folder}{SUFFIX}",
        checkpoint=f"distilled with cellmap-flow from {teacher_folder}",
        distillation={
            "teacher": teacher_folder,
            "teacher_params": student.teacher_parameters,
            "num_fmaps": student.num_fmaps,
            "fmap_inc_factor": student.fmap_inc_factor,
            "steps": best_checkpoint["steps"],
            "held_out_blocks": HELD_OUT_BLOCKS,
            "teacher_mean_abs": float(np.mean([t.abs().mean() for t in targets])),
            "pruned": pruned,
            "distilled": best,
        },
    )


def _input_norm(config) -> list:
    """The input normalization the serving YAML's ``config`` serves its model with, as steps."""
    from cellmap_flow.pipeline_spec import normalize_steps

    return list(normalize_steps((config.get("json_data") or {}).get("input_norm")))


def _export_student(network, config, finetuned, corrections_dir, output_dir, variants, *, folder, checkpoint,
                    distillation) -> dict:
    """``network``, the student of the model ``finetuned`` (served by the
    YAML whose config is ``config``), exported as export_web.export exports
    that model, into ``<output_dir>/<folder>``, its manifest saying where it
    came from (``checkpoint``) and how close it came (``distillation``);
    ``{"folder", "manifest"}``.

    export_web.export builds its network from the YAML itself, so its
    export, from the probe of the output to the manifest's check, is
    repeated here for a network built elsewhere.

    Raises:
        ValueError: its fp32 file does not agree with it.
    """
    import torch

    try:
        from cellmap_models.model_export.onnx_web import VARIANTS, export_onnx_web
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. The web export needs cellmap-models' onnx_web (its onnx-web-export-and-pruning branch, "
            "which pixi.toml's finetune feature installs) and its export extra, onnx and onnxruntime."
        ) from e

    from cellmap_flow.finetune.model_loading import root_base_model_dict
    from cellmap_flow.finetune.trainer.supervised import probe_output_class
    from cellmap_flow.models.configs.onnx import parse_manifest
    from cellmap_flow.models.entry import numbers

    input_norm = _input_norm(config)
    geometry = finetuned.base_model_config.geometry
    block = export_web.training_block(corrections_dir, geometry.input_shape, geometry.output_shape, input_norm)
    with torch.no_grad():
        output_class = probe_output_class(network, torch.from_numpy(block))
        scale = float(network(torch.from_numpy(block)).abs().max())
    root = root_base_model_dict(finetuned)
    channels = list(geometry.channel_names or (f"channel_{i}" for i in range(geometry.output_channels)))
    rows = export_onnx_web(
        network, os.path.join(output_dir, folder), folder, [export_web._side(geometry.input_shape, "input")],
        variants=variants or VARIANTS, validation_input=block,
        manifest_extra={
            "model": folder,
            "task": export_web._task(channels),
            "checkpoint": checkpoint,
            "channels": channels,
            "input_voxel_size_nm": numbers(finetuned.input_voxel_size or geometry.input_voxel_size),
            "output_voxel_size_nm": numbers(finetuned.output_voxel_size or geometry.output_voxel_size),
            "output": export_web.OUTPUT_DESCRIPTIONS[output_class],
            "input_norm": input_norm,
            **({"source_repo": root["repo"]} if root.get("repo") else {}),
            "torch_version": torch.__version__,
            "exporter": export_web.EXPORTER,
            "distillation": distillation,
        },
    )
    for row in rows:
        if row["variant"] == "fp32" and row["max_abs"] > export_web.FP32_REL_TOL * max(scale, 1e-6):
            raise ValueError(f"The fp32 export differs from the network by up to {row['max_abs']:.3g}, "
                             f"{row['max_abs'] / max(scale, 1e-6):.2e} of its output's scale ({scale:.3g}), "
                             f"over the {export_web.FP32_REL_TOL:.0e} tolerance")
    manifest_path = os.path.join(output_dir, folder, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    parse_manifest(manifest, manifest_path)
    return {"folder": folder, "manifest": manifest}


def _walltime(hours: float) -> str:
    """The job's walltime, "HH:MM": the training's time limit and SETUP_HOURS."""
    minutes = math.ceil(60 * (hours + SETUP_HOURS))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def start_distill(yaml_path, corrections_dir, run_dir, *, parameters=None, hours=HOURS) -> export_web.WebExport:
    """Start distilling the model ``yaml_path`` serves, trained on the
    session ``corrections_dir``, into ``<run_dir>/web_export/<YAML's
    name>-small/``; follow it with export_web.export_status.

    ``parameters``: the student's largest size (by default small_copy's);
    ``hours``: the training's time limit. The job runs as start_export's
    does, in the environment the model finetunes in on the YAML's queue and
    charge group, and a result an earlier run left there is removed first,
    so that export_status does not read it as this one's; its best student
    is kept, for this run to continue from.

    Raises:
        ValueError: the YAML serves no finetuned model, or its environment
            cannot run the job.
        RuntimeError: the job could not be submitted.
    """
    from cellmap_flow.finetune.job_manager import submit
    from cellmap_flow.finetune.model_loading import model_config_from_entry

    config, entry = export_web.serving_entry(yaml_path)
    output_dir = Path(run_dir) / "web_export" / f"{Path(yaml_path).stem}{SUFFIX}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / export_web.RESULT_NAME).unlink(missing_ok=True)
    args = ["--yaml", str(yaml_path), "--corrections", str(corrections_dir), "--output", str(output_dir),
            "--hours", str(hours), *(["--parameters", str(parameters)] if parameters else [])]
    command = submit.module_command(model_config_from_entry(entry), MODULE, args, output_dir / export_web.LOG_NAME)
    job = submit.launch(
        f"distill_{export_web.folder_name(entry.get('name') or 'model')}", command,
        queue=config.get("queue"), charge_group=config.get("charge_group"), walltime=_walltime(hours),
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
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH, help="blocks per epoch, one a step")
    parser.add_argument("--hours", type=float, default=HOURS, help="the training's time limit")
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
        result = {"error": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"}
    export_web._write_result(output_dir, result)
    print(json.dumps(result), flush=True)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
