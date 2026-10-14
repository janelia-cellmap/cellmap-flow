"""``CellposeModelConfig``: Cellpose 4 (Cellpose-SAM and its successors), in 2D or 3D.

By default each z slice of a chunk is segmented in 2D by
``cellpose.models.CellposeModel``, read with ``context`` voxels of margin in
y and x that are cut off again, so that objects at the chunk's edge are seen
whole. With ``do_3d`` the chunk is segmented in 3D, as Cellpose's
``do_3D``: its 2D network runs on every YX, ZY and ZX plane of the chunk,
their flows are summed into 3D flows (flow_z, flow_y, flow_x) and their
cell probabilities likewise, and masks follow the 3D flows, so an object
is one id through z. The chunk is then a cube with context on all six
sides (``chunk_voxels``), its planes about one tile each, and costs about
three times what the same voxels do slice by slice. What the layer shows is
``output``:

- ``"flows"`` (the default): all of what Cellpose's network predicts, three
  float32 channels named flow_y, flow_x and cell: the flows towards each
  object's centre (Cellpose's ``dP``, about -5 to 5) and the cell
  probability, 0 to 1, the sigmoid of its logit; in 3D a flow_z channel
  comes first, and the logit is the sum of the three planes' (Cellpose's
  own, which its 3D thresholds are on). The layer colours each
  voxel by its flow's direction, dimmed by the cell probability
  (``display_channel``; viewer.layers.flow_channels); the
  CellposeMasksPostprocessor makes masks of them on the server, with
  thresholds changeable from the dashboard; a seed reads the probability.
  Computed per voxel, so it joins up across chunks.
- ``"probability"``: the cell probability alone, one channel.
- ``"masks"``: Cellpose's instance masks (uint64), ids unique within a
  chunk. Masks are made per chunk: an object that crosses a chunk's edge is
  cut there, with another id on each side. Objects are joined from slice to
  slice only with ``stitch_threshold`` (Cellpose's own: a mask takes the id
  of the one in the slice before that it overlaps by at least that IoU),
  and only within a chunk. The ``MortonSegmentationRelabeling``
  postprocessor makes the ids unique across chunks and shows the layer as a
  segmentation; it is not applied here (see ``_masks``). In 3D the masks
  are Cellpose's 3D objects, one id through the chunk's z.

Cellpose 4 cannot share cellmap-flow's default environment, whose cellpose 3
pins an older numpy, so this type runs in the ``cellpose4`` pixi
environment unless its entry names another (``default_env``).

It can be finetuned, with LoRA or in full, on painted instances
(``trainable_model``): the network learns Cellpose's own outputs, the flows
towards each instance's centre and the cell probability, from flow targets
(``finetune.instance_flows``), and is served afterwards as before, through
Cellpose's eval, which runs the network training changed in place.

``cellpose`` is imported only when the model is built, as every type imports
its framework: the CLIs, ``--help`` and the dashboard's model form import
every type, and none of them has (or needs) Cellpose 4.

Licence: the Cellpose-SAM weights were trained on data that includes
datasets licensed CC-BY-NC, so they are for non-commercial use.
"""

import contextlib
import logging
import math
import os

import numpy as np

from cellmap_flow.models.configs.base import Config, ModelConfig, _voxel_size
from cellmap_flow.models.geometry import _numbers

logger = logging.getLogger(__name__)

# What Cellpose 4 ships (cellpose.models.MODEL_NAMES), downloaded from the
# Hugging Face Hub on first use; any other value is a path to weights. The
# DINO models also need the dinov3 package, which the cellpose4 environment
# does not install (see _load_model).
PRETRAINED_MODELS = ("cpsam_v2", "cpsam", "cpdino", "cpdino-vitb")
OUTPUTS = ("probability", "flows", "masks")
# The channels each output serves: Cellpose's dP (flowY, flowX) and its cell
# probability, as the sigmoid of the logit it predicts. In 3D, dP has a z
# flow first.
OUTPUT_CHANNELS = {"probability": ["cell"], "flows": ["flow_y", "flow_x", "cell"], "masks": ["cell"]}
OUTPUT_CHANNELS_3D = {**OUTPUT_CHANNELS, "flows": ["flow_z", "flow_y", "flow_x", "cell"]}

# A chunk's voxels a side and the context read around it, in y and x, when
# not given. In 2D, 512 + 2 * 32 = 576 px slices, 3 x 3 Cellpose-SAM tiles.
# In 3D every plane of the chunk, in all three orientations, goes through
# the network, so the chunk is kept to one tile a plane: 192 + 2 * 24 = 240,
# which Cellpose pads to 256. That is 720 tiles for 192^3 voxels, about 3x
# a 2D chunk's 72 for 8 x 512^2 per voxel (and ~5 s a chunk on an H100).
DEFAULT_SLICE_SIZE, DEFAULT_CONTEXT = 512, 32
DEFAULT_SLICE_SIZE_3D, DEFAULT_CONTEXT_3D = 192, 24

# Cellpose cuts each slice into square tiles, bsize pixels a side,
# overlapping by tile_overlap (its default, kept). Cellpose-SAM's ViT-L
# ("sam_vitl", cpsam and cpsam_v2) takes 256 px tiles and refuses any other;
# the DINO models default to 384. Which one a model is, Cellpose reads from
# its weights (CellposeModel.backbone), so finetuned weights are covered too.
SAM_TILE_SIZE = 256
DINO_TILE_SIZE = 384
TILE_OVERLAP = 0.1
# Cellpose's own "diameter" scale: objects about 30 px across.
CELLPOSE_DIAMETER = 30.0
# Tiles per GPU pass, by default. Timed on Cellpose-SAM's network in bfloat16
# with 256 px tiles (ms per tile at batch 1 / 8 / 16 / 72 / 128): an L4 takes
# 47 / 49 / 54 / ~50 / 50, an H100 16 / 7.0 / 6.6 / ~6.3 / 6.25. Peak memory
# was 1.5 GB at 8, 2.4 GB at 16, ~9 GB at 72 and 15 GB at 128. So a whole
# chunk's tiles in one pass (72 for the default chunk), the old default,
# bought nothing over 16 and cost about 6.5 GB more.
DEFAULT_BATCH_SIZE = 16


def tile_size(backbone: str) -> int:
    """The tile edge, in pixels, Cellpose runs ``backbone`` on (``CellposeModel._run_net``)."""
    return SAM_TILE_SIZE if backbone == "sam_vitl" else DINO_TILE_SIZE


def _rescale(diameter) -> float:
    """The factor Cellpose resizes each slice by before tiling it (``CellposeModel.eval``)."""
    return CELLPOSE_DIAMETER / diameter if diameter is not None and diameter > 0 else 1.0


def _tiles_per_side(length: int, tile: int, rescale: float = 1.0) -> int:
    """How many tiles Cellpose cuts one side of a plane ``length`` px long into.

    ``cellpose.core.run_net``'s arithmetic: the side is resized by
    ``rescale``, padded by ``transforms.get_pad_yx`` (to a multiple of 16
    plus 16, or up to one tile), then cut into
    ``ceil((1 + 2 * overlap) * padded / tile)`` tiles, or one when it fits
    in a tile.
    """
    resized = int(length * rescale) if rescale != 1.0 else int(length)
    div = 16
    if resized >= tile:
        padded = div * math.ceil(resized / div) + div
    else:
        padded = max(resized + div, tile)
    return 1 if padded <= tile else math.ceil((1.0 + 2 * TILE_OVERLAP) * padded / tile)


def tiles_per_slice(length: int, tile: int, rescale: float = 1.0) -> int:
    """How many tiles Cellpose cuts a ``length`` x ``length`` slice into."""
    return _tiles_per_side(length, tile, rescale) ** 2


def tiles_per_volume(shape, tile: int, rescale: float = 1.0, anisotropy: float = 1.0) -> int:
    """How many tiles Cellpose's ``do_3D`` runs on a (z, y, x) volume of ``shape``.

    ``CellposeModel._run_net`` resizes the volume by ``rescale`` (and z by
    ``anisotropy`` too), then ``core.run_3D`` runs the network on each of its
    YX, ZY and ZX planes.
    """
    z, y, x = int(shape[0] * anisotropy * rescale), int(shape[1] * rescale), int(shape[2] * rescale)

    def per_plane(height, width):
        return _tiles_per_side(height, tile) * _tiles_per_side(width, tile)

    return z * per_plane(y, x) + y * per_plane(z, x) + x * per_plane(z, y)


def _check_cellpose_4():
    """Raise when the installed Cellpose is older than 4.

    Cellpose 3 (the default environment's) does not know cpsam: it would
    warn and segment with another model, so the layer would look plausible
    and be wrong.
    """
    try:
        from cellpose.version import version
    except ImportError:  # a stand-in cellpose (the tests), or a build without it
        return
    major = str(version).split(".")[0]
    if major.isdigit() and int(major) < 4:
        raise RuntimeError(
            f"The cellpose model type needs Cellpose 4; this environment has cellpose {version}. "
            "Run it in the cellpose4 environment (its default; leave out env, or give env: cellpose4)."
        )


_BLOCKS = None


def _network_blocks():
    """The torch module classes below, defined on first use: this module is
    imported by every CLI and the dashboard, which do not import torch."""
    global _BLOCKS
    if _BLOCKS is not None:
        return _BLOCKS
    import types

    from torch import nn

    class CellposeNetwork(nn.Module):
        """A Cellpose 4 network on (N, 1, tile, tile) images: (N, 3, tile, tile),
        flowY, flowX and the cell probability logit.

        One channel, as Cellpose's eval gives a grayscale image (its network
        reads only the first ``x.shape[1]`` channels of its patch embedding),
        and without the style vector Cellpose returns beside the output.
        ``fixed`` refuses any other size, as Cellpose-SAM's position
        embeddings fit one tile only, with a message rather than a shape
        error from deep inside the ViT.
        """

        def __init__(self, net, tile: int, fixed: bool = True):
            super().__init__()
            self.net = net
            self.tile = int(tile)
            self.fixed = bool(fixed)

        def forward(self, x):
            if self.fixed and tuple(x.shape[-2:]) != (self.tile, self.tile):
                raise RuntimeError(
                    f"Cellpose-SAM's network takes {self.tile} x {self.tile} tiles, not "
                    f"{tuple(x.shape[-2:])}: Cellpose serves larger slices by tiling them "
                    "in its eval, which serving uses; the trainer reads tiles "
                    "(CellposeModelConfig.training_patch_voxels)"
                )
            return self.net(x)[0]

    _BLOCKS = types.SimpleNamespace(CellposeNetwork=CellposeNetwork)
    return _BLOCKS


@contextlib.contextmanager
def _serving(model):
    """While ``model`` (a CellposeModel) segments: in bfloat16, and with its
    network's train/eval mode put back afterwards.

    - Cellpose serves in bfloat16. When ``trainable_model`` has put the
      network in float32 for training, the batch of a whole chunk's tiles
      took twice the memory, and in the trainer's live server, beside a full
      finetune's weights, gradients and optimizer state, an L4 ran out.
      Under autocast the matrix products are bfloat16 again, LoRA's
      included, while the weights stay float32.
      Its attention scores stay float32 under autocast, though, so the batch
      of tiles is halved as well: a chunk's 72 tiles at once (the default
      then, and still a batch_size one can give) peaked at 19 GB after
      training, against 9 GB before.
    - Cellpose's eval switches the network to eval mode and leaves it there,
      so in the trainer's live server training went on without its
      stochastic depth until the next epoch's ``train()``.

    Yields a function of the eval kwargs giving those to change.
    """
    import torch

    net = getattr(model, "net", None)
    training = getattr(net, "training", False)
    autocast = torch.cuda.is_available() and getattr(net, "dtype", None) == torch.float32

    def override(eval_kwargs):
        if not autocast:
            return {}
        return {"batch_size": max(1, int(eval_kwargs.get("batch_size", 8)) // 2)}

    try:
        with torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext():
            yield override
    finally:
        if net is not None and training:
            net.train(True)


class CellposeModelConfig(ModelConfig):
    """Cellpose 4 run on each z slice of a chunk, or on the chunk in 3D.

    Args:
        voxel_size: nm per voxel, input and output (one number or one per
            axis). Cellpose-SAM sees objects best about 30 voxels across, so
            pick the scale at which yours are roughly that, or give
            ``diameter``.
        pretrained_model: "cpsam_v2" (default), "cpsam", "cpdino",
            "cpdino-vitb", or the path of finetuned Cellpose weights.
        output: "flows" (the default; float32: flow_y, flow_x and the cell
            probability, with flow_z first in 3D), "probability" (float32,
            0 to 1, one channel) or "masks" (uint64, ids unique within a
            chunk).
        do_3d: segment each chunk in 3D (Cellpose's ``do_3D``) rather than
            slice by slice.
        slices_per_chunk: z slices in a chunk (2D only; a 3D chunk's z
            follows from ``slice_size``, ``chunk_voxels``).
        slice_size: voxels a side, in y and x, of each chunk's slices; None
            is 512 in 2D, 192 in 3D (``DEFAULT_SLICE_SIZE``).
        context: voxels read on each side in y and x beyond those, and cut
            off, and in z too in 3D; None is 32 in 2D, 24 in 3D.
        batch_size: tiles per GPU pass (``DEFAULT_BATCH_SIZE``; None, as a
            blank form field sends it, is that too).
        diameter: object diameter in input voxels; None keeps the model's
            own scale (about 30). Cellpose resizes each slice by
            30 / diameter.
        flow_threshold, cellprob_threshold: Cellpose's mask thresholds (masks
            only).
        stitch_threshold: Cellpose's own slice linking (masks only, 2D
            only): a mask takes the id of the mask in the slice before that
            it overlaps by at least this IoU, within a chunk. 0, the
            default, leaves each slice's masks apart.
        flow3d_smooth: in 3D, the sigma, in voxels, of a Gaussian Cellpose
            smooths the 3D flows with before following them (its
            ``flow3D_smooth``), against objects broken up along z; 0, the
            default, is none.
    """

    cli_name = "cellpose"
    # Cellpose 4 cannot share cellmap-flow's default environment (see the
    # module docstring); an entry's explicit env still wins.
    default_env = "cellpose4"
    finetunable = True
    # Cellpose's eval normalizes each slice to its 1st-99th percentile, and
    # its trainable module does the same (trainable_model's ScaleRange).
    normalizes_input = True
    # What a finetune trains it on: Cellpose predicts flows and a cell
    # probability, so the painted instances become flow targets (the
    # dashboard reads this to pick the target; finetune.cli's --output-type).
    finetune_output_type = "flows"

    @property
    def channels(self):
        """The names of the channels ``output`` serves, in 2D or 3D."""
        return list((OUTPUT_CHANNELS_3D if self.do_3d else OUTPUT_CHANNELS)[self.output])

    @property
    def display_channel(self):
        """The cell probability's channel when the output has several (flows):
        the layer's flow colours are dimmed by it, and its range is the
        layer's."""
        channels = self.channels
        return channels.index("cell") if len(channels) > 1 else None

    def __init__(
        self,
        voxel_size,
        pretrained_model: str = "cpsam_v2",
        output: str = "flows",
        do_3d: bool = False,
        slices_per_chunk: int = 8,
        slice_size: int = None,
        context: int = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        diameter: float = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        stitch_threshold: float = 0.0,
        flow3d_smooth: float = 0.0,
        name=None,
        scale=None,
    ):
        super().__init__()
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")
        # The server CLI and the model form pass the voxel size as a string
        # ("64" or "16,8,8"); a YAML gives a number or a list.
        # Not _as_int_tuple: that truncated a 5.24 nm voxel to 5.
        self.voxel_size = _voxel_size(voxel_size)
        self.pretrained_model = str(pretrained_model)
        self.output = output
        # The model form sends a checkbox as a string.
        self.do_3d = do_3d if isinstance(do_3d, bool) else str(do_3d).strip().lower() in ("1", "true", "yes", "on")
        self.slices_per_chunk = int(slices_per_chunk)
        if slice_size is None:
            slice_size = DEFAULT_SLICE_SIZE_3D if self.do_3d else DEFAULT_SLICE_SIZE
        if context is None:
            context = DEFAULT_CONTEXT_3D if self.do_3d else DEFAULT_CONTEXT
        self.slice_size = int(slice_size)
        self.context = int(context)
        if self.slices_per_chunk < 1 or self.slice_size < 1 or self.context < 0:
            raise ValueError(
                "slices_per_chunk and slice_size must be at least 1 and context at least 0; got "
                f"{self.slices_per_chunk}, {self.slice_size} and {self.context}"
            )
        # None is the default, not "the whole chunk" as it once was: the
        # model form sends a blank field as None, and a whole chunk's tiles
        # cost memory for no speed (DEFAULT_BATCH_SIZE).
        self.batch_size = DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
        self.diameter = None if diameter is None else float(diameter)
        self.flow_threshold = float(flow_threshold)
        self.cellprob_threshold = float(cellprob_threshold)
        self.stitch_threshold = float(stitch_threshold)
        if not 0.0 <= self.stitch_threshold <= 1.0:
            raise ValueError(f"stitch_threshold is an IoU, from 0 to 1; got {self.stitch_threshold}")
        if self.stitch_threshold and output != "masks":
            # It links masks; with no masks it would only slow the chunk down
            # (Cellpose computes them to link them) and change nothing shown.
            raise ValueError(f"stitch_threshold links masks: it needs output masks, not {output!r}")
        if self.stitch_threshold and self.do_3d:
            raise ValueError("stitch_threshold links 2D slices' masks; in 3D (do_3d) the masks are 3D objects already")
        self.flow3d_smooth = float(flow3d_smooth)
        if self.flow3d_smooth < 0:
            raise ValueError(f"flow3d_smooth is a sigma, 0 or more; got {self.flow3d_smooth}")
        if self.flow3d_smooth and not self.do_3d:
            raise ValueError("flow3d_smooth smooths 3D flows: it needs do_3d")
        self.name = name
        self.scale = scale

    @property
    def read_size(self) -> int:
        """Voxels a side, in y and x, of each slice Cellpose is given."""
        return self.slice_size + 2 * self.context

    @property
    def anisotropy(self) -> float:
        """z's voxel size over y's: how much Cellpose stretches z in 3D (its
        ``anisotropy``) so that its planes are square in nm."""
        return float(self.voxel_size[0]) / float(self.voxel_size[1])

    def chunk_voxels(self):
        """``(chunk, context)``: a chunk's (z, y, x) voxels, and those read
        beyond it on each side.

        In 2D, ``slices_per_chunk`` slices of ``slice_size`` squared, with
        context in y and x only: each slice is segmented on its own. In 3D a
        cube in nm: ``slice_size`` and ``context`` in y and x, and as many
        nm in z, so that once Cellpose stretches z by ``anisotropy`` every
        plane is as wide as a YX one, about one tile.
        """
        if not self.do_3d:
            return (self.slices_per_chunk, self.slice_size, self.slice_size), (0, self.context, self.context)
        z = max(1, round(self.slice_size / self.anisotropy))
        z_context = round(self.context / self.anisotropy)
        return (z, self.slice_size, self.slice_size), (z_context, self.context, self.context)

    def _load_model(self):
        from cellpose import models

        _check_cellpose_4()
        known = list(getattr(models, "MODEL_NAMES", PRETRAINED_MODELS))
        get_user_models = getattr(models, "get_user_models", None)
        if get_user_models is not None:
            known += list(get_user_models())
        if self.pretrained_model not in known and not os.path.exists(self.pretrained_model):
            # Cellpose would warn and fall back to cpsam_v2, which serves a
            # plausible layer from weights nobody asked for.
            raise ValueError(
                f"pretrained_model {self.pretrained_model!r} is neither one of Cellpose's models "
                f"({', '.join(known)}) nor an existing file"
            )
        try:
            # gpu=True falls back to the CPU when there is none.
            return models.CellposeModel(gpu=True, pretrained_model=self.pretrained_model)
        except NameError as e:
            # Cellpose imports its DINO backbones from facebookresearch's
            # dinov3, which is not on PyPI, so neither the cellpose4
            # environment nor `pip install cellpose` has it; Cellpose only
            # logs a warning and fails here with a bare NameError.
            if "dinov3" not in str(e):
                raise
            raise RuntimeError(
                f"pretrained_model {self.pretrained_model!r} is a Cellpose DINO model, which needs the "
                "dinov3 package: `pip install git+https://github.com/facebookresearch/dinov3` "
                "in the model's environment"
            ) from e

    def _get_config(self):
        model = self._load_model()
        voxel_size = np.asarray(self.voxel_size, dtype=float)
        chunk, context = (np.asarray(v) for v in self.chunk_voxels())
        read = chunk + 2 * context

        config = Config()
        config.model = model
        # Plain numbers rather than Coordinates, which are integers: a
        # 5.24 nm z would be 5, and every chunk placed on the wrong grid.
        config.input_voxel_size = self.voxel_size
        config.output_voxel_size = self.voxel_size
        # In 2D no context in z: each slice is segmented on its own.
        config.read_shape = _numbers(read * voxel_size)
        config.write_shape = _numbers(chunk * voxel_size)
        config.context = _numbers((np.asarray(config.read_shape) - np.asarray(config.write_shape)) / 2)
        config.channels = self.channels
        config.output_channels = len(config.channels)
        config.block_shape = np.array((*chunk, config.output_channels))
        config.output_dtype = np.uint64 if self.output == "masks" else np.float32

        # Cellpose cuts each plane into tiles; given the chunk as one batch it
        # runs batch_size of them, from any of its planes, per GPU pass.
        # bsize is passed too, so Cellpose tiles as counted here.
        bsize = tile_size(getattr(model, "backbone", "sam_vitl"))
        rescale = _rescale(self.diameter)
        if self.do_3d:
            tiles = tiles_per_volume(read, bsize, rescale, self.anisotropy)
        else:
            tiles = int(read[0]) * tiles_per_slice(int(read[1]), bsize, rescale)
        config.eval_kwargs = {
            "batch_size": self.batch_size,
            "bsize": bsize,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "compute_masks": self.output == "masks",
        }
        if self.stitch_threshold:
            # Cellpose stitches only a stack it takes for 3D, which needs
            # its z axis named (and refuses one otherwise). It still segments
            # each slice in 2D, but normalizes the chunk's slices together.
            config.eval_kwargs.update(stitch_threshold=self.stitch_threshold, z_axis=0)
        if self.do_3d:
            # The chunk as one 3D image, normalized as a whole; Cellpose
            # stretches z by the anisotropy before its planes are cut, and
            # resizes the flows back.
            config.eval_kwargs.update(do_3D=True, z_axis=0)
            if self.anisotropy != 1.0:
                config.eval_kwargs["anisotropy"] = self.anisotropy
            if self.flow3d_smooth:
                config.eval_kwargs["flow3D_smooth"] = self.flow3d_smooth
        logger.info(
            f"Cellpose {self.pretrained_model} ({getattr(model, 'backbone', '?')}"
            f"{', 3D' if self.do_3d else ''}): {tiles} tiles of {bsize} px a chunk "
            f"({' x '.join(str(int(n)) for n in read)} voxels read), batch_size {self.batch_size}"
        )
        config.process_chunk = self.process_chunk
        return config

    def process_chunk(self, idi, output_roi):
        """``output_roi`` segmented: ``(channels, z, y, x)``, the ``output``'s channels."""
        return self._segment(self.config, idi, output_roi)

    def _segment(self, config, idi, output_roi):
        """``output_roi`` segmented by ``config``'s Cellpose model and geometry.

        ``config`` is this model's own, or a finetuned model's with the same
        voxel counts (``serve_trained``), whose context is measured from its
        own shapes, at the voxel size the finetune was trained at.
        """
        context = _numbers((np.asarray(config.read_shape) - np.asarray(config.write_shape)) / 2)
        data = idi.to_ndarray_ts(output_roi.grow(context, context))
        # (z, y, x, channel). In 2D a batch of 2D images: Cellpose still
        # normalizes and segments each slice on its own, but batches their
        # tiles together. A list of slices would be run image by image, a
        # pass or more per slice; z_axis is refused without do_3D or
        # stitching, and a 3D array is taken for one 2D image with channels.
        # In 3D (do_3D and z_axis in eval_kwargs) one 3D image.
        with _serving(config.model) as eval_kwargs_override:
            masks, flows, _ = config.model.eval(data[..., np.newaxis], channel_axis=3,
                                                **{**config.eval_kwargs, **eval_kwargs_override(config.eval_kwargs)})
        chunk, margin = self.chunk_voxels()
        inner = tuple(slice(m, m + w) for m, w in zip(margin, chunk))
        if self.output in ("probability", "flows"):
            # flows[2] is the cell probability as a logit, (z, y, x); Cellpose
            # squeezes a single slice to (y, x).
            logits = np.reshape(flows[2], data.shape)[inner].astype(np.float32)
            probability = (1.0 / (1.0 + np.exp(-logits)))[np.newaxis]
            if self.output == "probability":
                return probability
            # flows[1] is dP, (flowY, flowX) or in 3D (flowZ, flowY, flowX),
            # then (z, y, x), squeezed alike.
            dp = np.reshape(flows[1], (-1, *data.shape))[(slice(None), *inner)].astype(np.float32)
            return np.concatenate([dp, probability])
        masks = np.reshape(masks, data.shape)[inner]
        if config.eval_kwargs.get("stitch_threshold") or config.eval_kwargs.get("do_3D"):
            # Stitched or 3D, an id is one object through the chunk's slices
            # already; numbering each slice apart would split them again.
            return masks.astype(np.uint64)[np.newaxis]
        return self._masks(masks)[np.newaxis]

    # ---- finetuning -------------------------------------------------------

    def finetune_modes(self):
        """("lora", "full"), without building the network: every Cellpose 4
        network is a ViT whose Linear and Conv layers take adapters."""
        return ("lora", "full")

    def training_patch_voxels(self):
        """The patch the trainer reads, (input, output) in voxels: one slice of one tile.

        Not the serving geometry. Cellpose-SAM's ViT adds a fixed 32 x 32
        grid of position embeddings to its 8-pixel tokens, so it takes
        256 x 256 images and nothing else; Cellpose serves a slice by
        cutting it into such tiles, and trains on random tiles of that size
        too. The trainer does the same: it reads tiles, and supervises the
        whole of each, as Cellpose does (an instance cut by its edge is left
        out of the flow loss, ``instance_flows``). One slice: the network is
        2D and sees each slice alone, so more slices would cost a network
        pass each and add nothing, and a one-slice patch centred on a
        painted voxel always lands on paint. A 3D model's network is the
        same 2D one, run on the chunk's planes in all three orientations;
        it is trained on z slices, the YX planes.
        """
        tile = tile_size(getattr(self.config.model, "backbone", "sam_vitl"))
        patch = (1, tile, tile)
        return patch, patch

    def trainable_model(self):
        """Cellpose's network as the trainer trains it: (B, 1, Z, Y, X) -> (B, 3, Z, Y, X).

        Each slice normalized as Cellpose's eval does (1st to 99th
        percentile), then through the network, which gives flowY, flowX and
        the cell probability logit (``instance_flows``' channels). The
        network is the one ``config.model`` segments with, so what
        is trained is what is served; it is put in float32 (Cellpose loads
        it in bfloat16), which serving then uses too. Its patch embedding is
        kept out of LoRA (``lora_exclude_patterns``): Cellpose's forward
        reads that layer's weight tensor directly, so an adapter on it would
        never be used.
        """
        import torch

        from cellmap_flow.finetune.trainable import ScaleRange, SliceWise

        cellpose = self.config.model
        net = cellpose.net
        if getattr(net, "dtype", torch.float32) != torch.float32:
            net.dtype = torch.float32  # Cellpose's setter: converts it, and what eval feeds it
        backbone = getattr(cellpose, "backbone", "sam_vitl")
        module = torch.nn.Sequential(
            ScaleRange(1, 99, dims=(1, 3, 4)),  # each slice of (B, 1, Z, Y, X)
            SliceWise(_network_blocks().CellposeNetwork(net, tile_size(backbone), backbone == "sam_vitl")),
        )
        module.lora_exclude_patterns = ["patch_embed"]
        return module

    def serve_trained(self, config, module):
        """Serve ``module`` (the trained ``trainable_model()``) through Cellpose's eval.

        Training changed the network in place, LoRA adapters and all, and
        Cellpose's eval runs that network, so the chunks are segmented as
        before. ``config`` is this model's own config (the trainer's live
        server) or a finetuned model's new one with this geometry, which
        gets what segmenting needs.

        ``config.model`` stays Cellpose's model object, as it is for the
        base model, so the inferencer neither moves nor forwards it. Were it
        the module, the warmup and the shape check would forward it, and a
        3-channel tile network fails the 1-channel serving geometry: as a
        hard error whenever the read slice happens to be one tile. The
        module is kept as ``config.trained_module``.
        """
        own = self.config
        cellpose = own.model
        if not any(part is cellpose.net for part in module.modules()):
            raise ValueError(
                "The trained module does not hold this model's Cellpose network: it was not "
                "built by this config's trainable_model(), so serving through Cellpose's eval "
                "would not serve it."
            )
        config.model = cellpose
        config.trained_module = module
        if config is own:
            return
        config.eval_kwargs = dict(own.eval_kwargs)
        config.output_dtype = own.output_dtype

        def process_chunk(idi, output_roi):
            return self._segment(config, idi, output_roi)

        config.process_chunk = process_chunk

    @staticmethod
    def _masks(masks):
        """Each slice's masks, renumbered so an id means one object in the whole chunk.

        Unique within the chunk only. Morton relabeling is left to the
        ``MortonSegmentationRelabeling`` postprocessor rather than done here:
        the viewer shows a layer as a segmentation only when its postprocess
        chain ends in labels, so the postprocessor is needed anyway, and it
        numbers chunks on the grid the server and blockwise share (from the
        data's corner), which a chunk's ROI alone does not give.
        """
        output = np.zeros(masks.shape, dtype=np.uint64)
        next_id = 0
        for z, mask in enumerate(masks):
            mask = mask.astype(np.uint64)
            # Each slice numbers its objects from 1; shift them past the
            # slices before.
            output[z] = np.where(mask > 0, mask + np.uint64(next_id), 0)
            next_id = max(next_id, int(output[z].max()))
        return output

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        result = {
            "type": "cellpose",
            "voxel_size": list(self.voxel_size),
            "pretrained_model": self.pretrained_model,
            "output": self.output,
        }
        if self.do_3d:
            result["do_3d"] = True
        result.update({
            "slices_per_chunk": self.slices_per_chunk,
            "slice_size": self.slice_size,
            "context": self.context,
        })
        result["batch_size"] = self.batch_size
        if self.diameter is not None:
            result["diameter"] = self.diameter
        result["flow_threshold"] = self.flow_threshold
        result["cellprob_threshold"] = self.cellprob_threshold
        if self.stitch_threshold:
            result["stitch_threshold"] = self.stitch_threshold
        if self.flow3d_smooth:
            result["flow3d_smooth"] = self.flow3d_smooth
        return self._with_name_scale(result)
