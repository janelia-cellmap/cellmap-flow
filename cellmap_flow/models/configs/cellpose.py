"""``CellposeModelConfig``: Cellpose 4 (Cellpose-SAM and its successors), slice by slice.

Each z slice of a chunk is segmented in 2D by ``cellpose.models.CellposeModel``,
read with ``context`` voxels of margin in y and x that are cut off again, so
that objects at the chunk's edge are seen whole. What the layer shows is
``output``:

- ``"probability"``: Cellpose's cell probability, from 0 to 1 (float32), the
  sigmoid of its logit. It is computed per voxel, so it joins up across
  chunks, and the mask dynamics are skipped, which makes it the faster of
  the two.
- ``"masks"``: Cellpose's instance masks (uint64), ids unique within a
  chunk. Masks are made per chunk: an object that crosses a chunk's edge is
  cut there, with another id on each side, and objects are not joined from
  slice to slice. The ``MortonSegmentationRelabeling`` postprocessor makes
  the ids unique across chunks and shows the layer as a segmentation; it is
  not applied here (see ``_masks``).

Cellpose 4 cannot share cellmap-flow's default environment, whose cellpose 3
pins an older numpy, so this type runs in the ``cellpose4`` pixi
environment unless its entry names another (``default_env``).

``cellpose`` is imported only when the model is built, as every type imports
its framework: the CLIs, ``--help`` and the dashboard's model form import
every type, and none of them has (or needs) Cellpose 4.

Licence: the Cellpose-SAM weights were trained on data that includes
datasets licensed CC-BY-NC, so they are for non-commercial use.
"""

import logging
import math
import os

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import Config, ModelConfig, _as_int_tuple

logger = logging.getLogger(__name__)

# What Cellpose 4 ships (cellpose.models.MODEL_NAMES), downloaded from the
# Hugging Face Hub on first use; any other value is a path to weights. The
# DINO models also need the dinov3 package, which the cellpose4 environment
# does not install (see _load_model).
PRETRAINED_MODELS = ("cpsam_v2", "cpsam", "cpdino", "cpdino-vitb")
OUTPUTS = ("probability", "masks")

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


class CellposeModelConfig(ModelConfig):
    """Cellpose 4 run on each z slice of a chunk.

    Args:
        voxel_size: nm per voxel, input and output (one number or one per
            axis). Cellpose-SAM sees objects best about 30 voxels across, so
            pick the scale at which yours are roughly that, or give
            ``diameter``.
        pretrained_model: "cpsam_v2" (default), "cpsam", "cpdino",
            "cpdino-vitb", or the path of finetuned Cellpose weights.
        output: "probability" (float32, 0 to 1) or "masks" (uint64, ids
            unique within a chunk).
        slices_per_chunk: z slices in a chunk.
        slice_size: voxels a side, in y and x, of each chunk's slices.
        context: voxels read on each side in y and x beyond those, and cut off.
        batch_size: tiles per GPU pass; None puts the whole chunk's in one.
        diameter: object diameter in input voxels; None keeps the model's
            own scale (about 30). Cellpose resizes each slice by
            30 / diameter.
        flow_threshold, cellprob_threshold: Cellpose's mask thresholds (masks
            only).
    """

    cli_name = "cellpose"
    # Cellpose 4 cannot share cellmap-flow's default environment (see the
    # module docstring); an entry's explicit env still wins.
    default_env = "cellpose4"

    def __init__(
        self,
        voxel_size,
        pretrained_model: str = "cpsam_v2",
        output: str = "probability",
        slices_per_chunk: int = 8,
        slice_size: int = 512,
        context: int = 32,
        batch_size: int = None,
        diameter: float = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        name=None,
        scale=None,
    ):
        super().__init__()
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")
        # The server CLI and the model form pass the voxel size as a string
        # ("64" or "16,8,8"); a YAML gives a number or a list.
        self.voxel_size = _as_int_tuple(voxel_size)
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
        self.batch_size = None if batch_size is None else int(batch_size)
        self.diameter = None if diameter is None else float(diameter)
        self.flow_threshold = float(flow_threshold)
        self.cellprob_threshold = float(cellprob_threshold)
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
        voxel_size = Coordinate(self.voxel_size)
        slices, size, read = self.slices_per_chunk, self.slice_size, self.read_size

        config = Config()
        config.model = model
        config.input_voxel_size = voxel_size
        config.output_voxel_size = voxel_size
        # No context in z: each slice is segmented on its own.
        config.read_shape = Coordinate((slices, read, read)) * voxel_size
        config.write_shape = Coordinate((slices, size, size)) * voxel_size
        config.context = (config.read_shape - config.write_shape) / 2
        config.output_channels = 1
        config.channels = ["cell"]
        config.block_shape = np.array((slices, size, size, config.output_channels))
        config.output_dtype = np.uint64 if self.output == "masks" else np.float32

        # Cellpose cuts each slice into tiles; given the chunk as one batch it
        # runs the tiles of as many slices as fit in batch_size per GPU pass,
        # so this many is the whole chunk in one. Lower it if a GPU runs out
        # of memory. bsize is passed too, so Cellpose tiles as counted here.
        bsize = tile_size(getattr(model, "backbone", "sam_vitl"))
        tiles = tiles_per_slice(read, bsize, _rescale(self.diameter))
        config.eval_kwargs = {
            "batch_size": self.batch_size or slices * tiles,
            "bsize": bsize,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "compute_masks": self.output == "masks",
        }
        logger.info(
            f"Cellpose {self.pretrained_model} ({getattr(model, 'backbone', '?')}): {tiles} tiles "
            f"of {bsize} px a slice, batch_size {config.eval_kwargs['batch_size']}"
        )
        config.process_chunk = self.process_chunk
        return config

    def process_chunk(self, idi, output_roi):
        """``output_roi`` segmented: ``(1, z, y, x)``, probability or masks."""
        config = self.config
        data = idi.to_ndarray_ts(output_roi.grow(config.context, config.context))
        # A batch of 2D images, (z, y, x, channel): Cellpose still normalizes
        # and segments each slice on its own, but batches their tiles
        # together. A list of slices would be run image by image, a pass or
        # more per slice; z_axis is refused without do_3D, and a 3D array is
        # taken for one 2D image with channels.
        masks, flows, _ = config.model.eval(data[..., np.newaxis], channel_axis=3, **config.eval_kwargs)
        inner = (
            slice(None),
            slice(self.context, self.context + self.slice_size),
            slice(self.context, self.context + self.slice_size),
        )
        if self.output == "probability":
            # flows[2] is the cell probability as a logit, (z, y, x); Cellpose
            # squeezes a single slice to (y, x).
            logits = np.reshape(flows[2], data.shape)[inner].astype(np.float32)
            return (1.0 / (1.0 + np.exp(-logits)))[np.newaxis]
        return self._masks(np.reshape(masks, data.shape)[inner])[np.newaxis]

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
        if self.batch_size is not None:
            result["batch_size"] = self.batch_size
        if self.diameter is not None:
            result["diameter"] = self.diameter
        result["flow_threshold"] = self.flow_threshold
        result["cellprob_threshold"] = self.cellprob_threshold
        return self._with_name_scale(result)
