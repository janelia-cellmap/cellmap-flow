"""``CellposeModelConfig``: Cellpose 4 (Cellpose-SAM and its successors), slice by slice.

Each z slice of a chunk is segmented in 2D by ``cellpose.models.CellposeModel``,
read with ``context`` voxels of margin in y and x that are cut off again, so
that objects at the chunk's edge are seen whole. What the layer shows is
``output``:

- ``"flows"`` (the default): all of what Cellpose's network predicts, three
  float32 channels named flow_y, flow_x and cell: the flows towards each
  object's centre (Cellpose's ``dP``, about -5 to 5) and the cell
  probability, 0 to 1, the sigmoid of its logit. The layer opens on the
  cell probability (``display_channel``), the flows a channel away; the
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
  segmentation; it is not applied here (see ``_masks``).

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
# probability, as the sigmoid of the logit it predicts.
OUTPUT_CHANNELS = {"probability": ["cell"], "flows": ["flow_y", "flow_x", "cell"], "masks": ["cell"]}

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


def tiles_per_slice(length: int, tile: int, rescale: float = 1.0) -> int:
    """How many tiles Cellpose cuts a ``length`` x ``length`` slice into.

    ``cellpose.core.run_net``'s arithmetic: the slice is resized by
    ``rescale``, padded by ``transforms.get_pad_yx`` (to a multiple of 16
    plus 16, or up to one tile), then cut into
    ``ceil((1 + 2 * overlap) * padded / tile)`` tiles a side, or one when it
    fits in a tile.
    """
    resized = int(length * rescale) if rescale != 1.0 else int(length)
    div = 16
    if resized >= tile:
        padded = div * math.ceil(resized / div) + div
    else:
        padded = max(resized + div, tile)
    per_side = 1 if padded <= tile else math.ceil((1.0 + 2 * TILE_OVERLAP) * padded / tile)
    return per_side * per_side


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
    """Cellpose 4 run on each z slice of a chunk.

    Args:
        voxel_size: nm per voxel, input and output (one number or one per
            axis). Cellpose-SAM sees objects best about 30 voxels across, so
            pick the scale at which yours are roughly that, or give
            ``diameter``.
        pretrained_model: "cpsam_v2" (default), "cpsam", "cpdino",
            "cpdino-vitb", or the path of finetuned Cellpose weights.
        output: "flows" (the default; float32: flow_y, flow_x and the cell
            probability), "probability" (float32, 0 to 1, one channel) or
            "masks" (uint64, ids unique within a chunk).
        slices_per_chunk: z slices in a chunk.
        slice_size: voxels a side, in y and x, of each chunk's slices.
        context: voxels read on each side in y and x beyond those, and cut off.
        batch_size: tiles per GPU pass (``DEFAULT_BATCH_SIZE``; None, as a
            blank form field sends it, is that too).
        diameter: object diameter in input voxels; None keeps the model's
            own scale (about 30). Cellpose resizes each slice by
            30 / diameter.
        flow_threshold, cellprob_threshold: Cellpose's mask thresholds (masks
            only).
        stitch_threshold: Cellpose's own slice linking (masks only): a mask
            takes the id of the mask in the slice before that it overlaps by
            at least this IoU, within a chunk. 0, the default, leaves each
            slice's masks apart.
    """

    cli_name = "cellpose"
    # Cellpose 4 cannot share cellmap-flow's default environment (see the
    # module docstring); an entry's explicit env still wins.
    default_env = "cellpose4"
    finetunable = True
    # What a finetune trains it on: Cellpose predicts flows and a cell
    # probability, so the painted instances become flow targets (the
    # dashboard reads this to pick the target; finetune.cli's --output-type).
    finetune_output_type = "flows"

    @property
    def display_channel(self):
        """The cell probability's channel when the output has several (flows):
        the layer opens on it rather than on flow_y."""
        channels = OUTPUT_CHANNELS[self.output]
        return channels.index("cell") if len(channels) > 1 else None

    def __init__(
        self,
        voxel_size,
        pretrained_model: str = "cpsam_v2",
        output: str = "flows",
        slices_per_chunk: int = 8,
        slice_size: int = 512,
        context: int = 32,
        batch_size: int = DEFAULT_BATCH_SIZE,
        diameter: float = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        stitch_threshold: float = 0.0,
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
        self.slices_per_chunk = int(slices_per_chunk)
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
        self.name = name
        self.scale = scale

    @property
    def read_size(self) -> int:
        """Voxels a side, in y and x, of each slice Cellpose is given."""
        return self.slice_size + 2 * self.context

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
        slices, size, read = self.slices_per_chunk, self.slice_size, self.read_size

        config = Config()
        config.model = model
        # Plain numbers rather than Coordinates, which are integers: a
        # 5.24 nm z would be 5, and every chunk placed on the wrong grid.
        config.input_voxel_size = self.voxel_size
        config.output_voxel_size = self.voxel_size
        # No context in z: each slice is segmented on its own.
        config.read_shape = _numbers(np.array((slices, read, read)) * voxel_size)
        config.write_shape = _numbers(np.array((slices, size, size)) * voxel_size)
        config.context = _numbers((np.asarray(config.read_shape) - np.asarray(config.write_shape)) / 2)
        config.channels = list(OUTPUT_CHANNELS[self.output])
        config.output_channels = len(config.channels)
        config.block_shape = np.array((slices, size, size, config.output_channels))
        config.output_dtype = np.uint64 if self.output == "masks" else np.float32

        # Cellpose cuts each slice into tiles; given the chunk as one batch it
        # runs batch_size of them, from any of its slices, per GPU pass.
        # bsize is passed too, so Cellpose tiles as counted here.
        bsize = tile_size(getattr(model, "backbone", "sam_vitl"))
        tiles = tiles_per_slice(read, bsize, _rescale(self.diameter))
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
        logger.info(
            f"Cellpose {self.pretrained_model} ({getattr(model, 'backbone', '?')}): {tiles} tiles "
            f"of {bsize} px a slice, {slices * tiles} a chunk, batch_size {self.batch_size}"
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
        # A batch of 2D images, (z, y, x, channel): Cellpose still normalizes
        # and segments each slice on its own, but batches their tiles
        # together. A list of slices would be run image by image, a pass or
        # more per slice; z_axis is refused without do_3D, and a 3D array is
        # taken for one 2D image with channels.
        with _serving(config.model) as eval_kwargs_override:
            masks, flows, _ = config.model.eval(data[..., np.newaxis], channel_axis=3,
                                                **{**config.eval_kwargs, **eval_kwargs_override(config.eval_kwargs)})
        inner = (
            slice(None),
            slice(self.context, self.context + self.slice_size),
            slice(self.context, self.context + self.slice_size),
        )
        if self.output in ("probability", "flows"):
            # flows[2] is the cell probability as a logit, (z, y, x); Cellpose
            # squeezes a single slice to (y, x).
            logits = np.reshape(flows[2], data.shape)[inner].astype(np.float32)
            probability = (1.0 / (1.0 + np.exp(-logits)))[np.newaxis]
            if self.output == "probability":
                return probability
            # flows[1] is dP, flowY and flowX, (2, z, y, x), squeezed alike.
            dp = np.reshape(flows[1], (2, *data.shape))[(slice(None), *inner)].astype(np.float32)
            return np.concatenate([dp, probability])
        masks = np.reshape(masks, data.shape)[inner]
        if config.eval_kwargs.get("stitch_threshold"):
            # Stitched, an id is one object through the chunk's slices
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
        painted voxel always lands on paint.
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
            "slices_per_chunk": self.slices_per_chunk,
            "slice_size": self.slice_size,
            "context": self.context,
        }
        result["batch_size"] = self.batch_size
        if self.diameter is not None:
            result["diameter"] = self.diameter
        result["flow_threshold"] = self.flow_threshold
        result["cellprob_threshold"] = self.cellprob_threshold
        if self.stitch_threshold:
            result["stitch_threshold"] = self.stitch_threshold
        return self._with_name_scale(result)
