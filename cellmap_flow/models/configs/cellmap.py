"""``CellMapModelConfig``: a model that cellmap_models exported, from its folder.

The folder's metadata gives the shapes, voxel sizes and channel names; the
model is its TorchScript export.
"""

import json
import logging
import os
from types import SimpleNamespace

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import Config, ModelConfig, _get_device
from cellmap_flow.models.geometry import channel_names_of
from cellmap_flow.portable.label_it import BYTE_UNIT, SIGNED_UNIT, UNBOUNDED, UNIT
from cellmap_flow.serving.probe import metadata_input_norm

logger = logging.getLogger(__name__)


def runs_at(model, input_shape, output_shape) -> bool:
    """Whether the TorchScript ``model`` runs on an input of ``input_shape``
    voxels and answers ``output_shape``.

    cellmap_models traces the export (``torch.jit.trace``) at whichever shape
    the exporter passed, and a traced UNet's crops are baked in, so it runs
    only at that shape: fed another, it either fails (jrc_mus-livers_16nm_to_8nm_mito,
    traced at its inference 288³, fails at its training 216³ with "padded
    input size (1 x 1 x 1)") or answers the baked-in output
    (mito_aff_unet_setup_16, traced at its training 178³ → 56³, answers 56³
    to a 378³ input). The metadata records both shapes and not which one
    the export was traced at, so this asks the model.
    """
    import torch

    try:
        first = next(model.parameters(), None)
        device = first.device if first is not None else "cpu"
        with torch.no_grad():
            out = model(torch.zeros((1, 1, *[int(s) for s in input_shape]), device=device))
    except (RuntimeError, TypeError, ValueError):
        return False
    return tuple(int(s) for s in out.shape[-len(output_shape):]) == tuple(int(s) for s in output_shape)


def export_finetune_modes(file_names) -> tuple:
    """How a cellmap_models export whose top-level files are ``file_names``
    can be finetuned, from their names alone: LoRA and full with a
    ``model.pt2``, which cellmap_models' ``train()`` unflattens into a module
    adapters attach to; full only without one, as ``train()`` then hands the
    trainer the TorchScript, a compiled network that takes no adapter.

    The one rule for an export's folder (CellMapModelConfig.finetune_modes),
    its Hugging Face repository (HuggingFaceModelConfig's) and the model
    lists' facts (hf_catalog.weight_facts). fly_organelles' copies under
    /nrs/cellmap/models/saalfeldlab (model.pt and model.ts; their Hugging
    Face copies have a model.pt2) and cellmap/jrc_mus-livers_16nm_to_8nm_mito
    (model.pt, model.ts and model.onnx) have none: offered LoRA, their jobs
    failed at their start.
    """
    return ("lora", "full") if "model.pt2" in file_names else ("full",)


def declared_offsets(value):
    """``affinity_offsets`` as an export's metadata declares them: one
    ``[dz, dy, dx]`` of whole voxels per affinity channel, in channel order,
    as lists of ints; None when ``value`` is not that (absent, empty, or
    not three whole numbers each)."""
    try:
        offsets = [[int(v) for v in offset] for offset in value]
        whole = all(int(v) == v for offset in value for v in offset)
    except (TypeError, ValueError):
        return None
    return offsets if offsets and whole and all(len(offset) == 3 for offset in offsets) else None


def _integer_dtype(dtype):
    """``dtype`` (an export's ``output_dtype``) as a numpy dtype when it is
    an integer one; None for a float, nothing, or no dtype."""
    try:
        dtype = np.dtype(dtype) if dtype else None
    except TypeError:
        return None
    return dtype if dtype is not None and dtype.kind in "iub" else None


def integer_output_refusal(name, output_dtype):
    """Why a model whose export declares ``output_dtype`` cannot be
    finetuned, or None when it can: an integer output (a probability x 255,
    rounded) passes no gradient back to the network. The finetune_refusal
    of an export's folder and of its Hugging Face repository."""
    dtype = _integer_dtype(output_dtype)
    if dtype is None:
        return None
    return (f"{name} cannot be finetuned: its output is {dtype} (rounded to whole numbers, a probability x 255 "
            "for a uint8 export), through which no gradient reaches the network")


# The activation an export's inference_output_activation names, by the class
# of the output it gives once applied: a sigmoid's and a softmax's are
# probabilities, a tanh's [-1, 1].
_ACTIVATION_CLASSES = {"sigmoid": UNIT, "softmax": UNIT, "tanh": SIGNED_UNIT}


def metadata_output_class(output_dtype, activation):
    """What an export's output is (portable/label_it.py's classes), as its
    metadata.json declares it, or None when it does not: BYTE_UNIT for a
    uint8 ``output_dtype``, a probability x 255 (cellmap_models writes one
    so); else by its ``inference_output_activation`` (``activation``, its
    dict, or a string such as "sigmoid"), as cellmap_models takes it: a
    sigmoid, softmax or tanh with ``apply_after_model`` false is in the
    export (UNIT, UNIT, SIGNED_UNIT), and with it true is to come after it,
    so the export gives logits (UNBOUNDED; setup_53's, 54's, 56's and 61's
    say so). "identity", which ModelMetadata writes when nothing is said,
    and any other name declare nothing."""
    if _integer_dtype(output_dtype) == np.uint8:
        return BYTE_UNIT
    if isinstance(activation, str):
        # cellmap_models' reading of a string: applied after the model
        # unless it says it is in it.
        text = activation.strip().lower()
        name = next((n for n in _ACTIVATION_CLASSES if n in text), None)
        after = not any(phrase in text for phrase in ("included", "in exported model", "in model"))
    elif isinstance(activation, dict):
        name = str(activation.get("name") or "").strip().lower()
        after = bool(activation.get("apply_after_model"))
    else:
        return None
    if name not in _ACTIVATION_CLASSES:
        return None
    return UNBOUNDED if after else _ACTIVATION_CLASSES[name]


def _other_shapes(metadata):
    """The metadata's inference shapes (input, output), when it gives them and
    they are not its training ones; else None. The export runs at one of the
    two, and which, only the model can say (runs_at)."""
    inference_in = getattr(metadata, "inference_input_shape", None)
    inference_out = getattr(metadata, "inference_output_shape", None)
    if inference_in and inference_out and list(inference_in) != list(metadata.input_shape):
        return list(inference_in), list(inference_out)
    return None


def metadata_geometry(metadata):
    """``ModelConfig.declared_geometry`` from an export's metadata (``metadata``
    has its fields as attributes): its training output, when the metadata
    gives no other shape (``_other_shapes``), so that it is the one the
    export runs at; else None, as when a field is missing. setup_16's
    metadata gives inference shapes too, and its export runs at the training
    ones; jrc_mus-livers_16nm_to_8nm_mito's runs at its inference ones."""
    try:
        if _other_shapes(metadata) is not None:
            return None
        voxel_size = Coordinate(metadata.output_voxel_size)
        names = channel_names_of(metadata)
        return {
            "write_shape": list(Coordinate(metadata.output_shape) * voxel_size),
            "input_voxel_size": list(Coordinate(metadata.input_voxel_size)),
            "output_voxel_size": list(voxel_size),
            "output_channels": int(metadata.out_channels),
            "channels": list(names) if names else None,
        }
    except (AttributeError, TypeError, ValueError):
        return None


class CellMapModelConfig(ModelConfig):
    """Configuration class for a CellmapModel."""

    cli_name = "cellmap"

    finetunable = True

    def __init__(self, folder_path, name=None, scale=None):
        super().__init__()
        from cellmap_models.model_export.cellmap_model import CellmapModel

        self.cellmap_model = CellmapModel(folder_path=folder_path)
        if name is None:
            # folder name 
            name = folder_path.rstrip("/").split("/")[-1]
        self.name = name
        self.scale = scale
        self._metadata_file = None

    @classmethod
    def from_metadata(cls, metadata, name, scale=None) -> "CellMapModelConfig":
        """The config of an export not made yet, from the metadata it will
        have (``metadata``: cellmap_models' ModelMetadata, as it will be
        written into its metadata.json). It answers what the export declares
        (declared_geometry, declared_input_norm, declared_affinity_offsets,
        finetune_refusal, finetune_modes, runs_at_native_voxel_size) from that
        metadata, by the code an export's folder is answered by, and has no
        model to serve. A mia model's before its server converts it
        (configs/mia.py)."""
        self = cls.__new__(cls)
        ModelConfig.__init__(self)
        # The metadata and no folder: finetune_modes says both then, as for a
        # folder that cannot be listed, unless finetune_refusal refuses.
        self.cellmap_model = SimpleNamespace(metadata=metadata, folder_path=None)
        self.name, self.scale = name, scale
        # Its metadata.json as cellmap_models' write_metadata will write it (_written_metadata).
        self._metadata_file = metadata.model_dump()
        return self

    def _get_config(self) -> Config:
        config = Config()
        metadata = self.cellmap_model.metadata

        # Populate config from metadata
        config.model_name = metadata.model_name
        config.model_type = metadata.model_type
        config.framework = metadata.framework
        config.spatial_dims = metadata.spatial_dims
        config.in_channels = metadata.in_channels
        config.output_channels = metadata.out_channels
        config.iteration = metadata.iteration
        config.input_voxel_size = Coordinate(metadata.input_voxel_size)
        config.output_voxel_size = Coordinate(metadata.output_voxel_size)
        config.channels_names = metadata.channels_names
        config.channels = metadata.channels_names  # alias for compatibility

        config.model = self.cellmap_model.ts_model
        config.model.to(_get_device())
        config.model.eval()

        # The shape the export runs at: its training shapes (input_shape →
        # output_shape) if a forward at them answers the training output,
        # else its inference shapes (what the Pipeline builder shows). The
        # metadata carries both; which the TorchScript was traced at, only
        # the model can say (runs_at). The probe at the training shape is the
        # cheap one: the smaller input, and a mismatch fails fast.
        training = (list(metadata.input_shape), list(metadata.output_shape))
        inference = _other_shapes(metadata)
        if inference is not None and not runs_at(config.model, *training):
            input_shape, output_shape = inference
            logger.info(f"{self.name} runs at its inference shape {input_shape} → {output_shape}, not its training shape {training[0]}")
        else:
            input_shape, output_shape = training
        config.read_shape = Coordinate(input_shape) * config.input_voxel_size
        config.write_shape = Coordinate(output_shape) * config.output_voxel_size
        config.inference_input_shape = config.read_shape
        config.inference_output_shape = config.write_shape

        config.block_shape = [*output_shape, metadata.out_channels]

        # An overlap, for a model whose output depends on its whole input,
        # with how its blocks lie and are blended (ModelGeometry.overlap,
        # block_phase, blend), and the dtype its output is in when it is not
        # float32 (a uint8 export keeps an overlap's cached blocks a quarter
        # of the size): fields a cellmap_models before its mia_model had not
        # parsed (_metadata_field).
        for key in ("overlap", "block_phase", "blend"):
            value = self._metadata_field(key)
            if value:
                setattr(config, key, value)
        output_dtype = self._metadata_field("output_dtype")
        if output_dtype:
            config.output_dtype = np.dtype(output_dtype)
            # A uint8 export's output is a probability x 255 (cellmap_models
            # writes one so), which its server reports as that, not as the
            # unbounded range its probe would see (inference/runner.py).
            if config.output_dtype == np.uint8:
                config.output_class = BYTE_UNIT
        return config

    def _metadata_field(self, key):
        """``key`` of the export's metadata: cellmap_models' parsed metadata's
        when it has the field and a value for it, else the metadata.json's as
        written (``_written_metadata``); None when neither has it. Never
        raises."""
        try:
            parsed = getattr(self.cellmap_model.metadata, key, None)
        except (OSError, ValueError, AttributeError):
            parsed = None
        return parsed if parsed is not None else self._written_metadata().get(key)

    def _written_metadata(self) -> dict:
        """The export's metadata.json as written, read once; {} when it cannot
        be read. cellmap_models' parsed metadata leaves out the fields it does
        not know: input_normalization, and before its mia_model overlap,
        block_phase, blend, output_dtype, affinity_offsets and
        native_voxel_size."""
        if self._metadata_file is None:
            try:
                with open(os.path.join(self.cellmap_model.folder_path, "metadata.json")) as f:
                    written = json.load(f)
                self._metadata_file = written if isinstance(written, dict) else {}
            except (OSError, ValueError, AttributeError):
                self._metadata_file = {}
        return self._metadata_file

    def finetune_modes(self):
        """``export_finetune_modes`` of its folder's files, read from their
        names without loading them: the model list asks on every poll. Both
        while the folder cannot be listed; none when its output cannot be
        trained through (finetune_refusal)."""
        if self.finetune_refusal():
            return ()
        folder = getattr(self.cellmap_model, "folder_path", None)
        try:
            return export_finetune_modes(os.listdir(folder)) if folder else ("lora", "full")
        except OSError:  # not a folder, or not one this process may list
            return ("lora", "full")

    def declared_geometry(self):
        """Its export's metadata.json's (``metadata_geometry``), read without the model."""
        try:
            metadata = self.cellmap_model.metadata
        except (OSError, ValueError):
            return None
        return metadata_geometry(metadata)

    def declared_input_norm(self, data_range=None):
        """Its export's metadata.json's input_normalization
        (``metadata_input_norm``). cellmap_models' parsed metadata leaves that
        field out, so the file is read for it, once (_written_metadata)."""
        return metadata_input_norm(self._written_metadata().get("input_normalization") or {}, data_range)

    def finetune_refusal(self):
        """``integer_output_refusal`` of its export's ``output_dtype``."""
        return integer_output_refusal(self.name, self._metadata_field("output_dtype"))

    def declared_affinity_offsets(self):
        """The offsets its export's metadata declares (``affinity_offsets``,
        ``declared_offsets``), read without the model; None without them."""
        return declared_offsets(self._metadata_field("affinity_offsets"))

    def declared_output_class(self):
        """``metadata_output_class`` of its export's metadata.json, read
        without the model: as written, as cellmap_models' parsed metadata
        gives inference_output_activation its default when none is written."""
        written = self._written_metadata()
        return metadata_output_class(written.get("output_dtype"), written.get("inference_output_activation"))

    def runs_at_native_voxel_size(self) -> bool:
        """Whether its export's metadata says it runs at the data's own voxel
        size (``native_voxel_size``), its input_voxel_size being a default only."""
        return bool(self._metadata_field("native_voxel_size"))

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        return self._with_name_scale(
            {"type": "cellmap", "folder_path": self.cellmap_model.folder_path}
        )
