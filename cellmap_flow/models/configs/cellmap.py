"""``CellMapModelConfig``: a model that cellmap_models exported, from its folder.

The folder's metadata gives the shapes, voxel sizes and channel names; the
model is its TorchScript export.
"""

import logging

from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import Config, ModelConfig, _get_device

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
        inference = (getattr(metadata, "inference_input_shape", None), getattr(metadata, "inference_output_shape", None))
        if inference[0] and inference[1] and list(inference[0]) != training[0] and not runs_at(config.model, *training):
            input_shape, output_shape = list(inference[0]), list(inference[1])
            logger.info(f"{self.name} runs at its inference shape {input_shape} → {output_shape}, not its training shape {training[0]}")
        else:
            input_shape, output_shape = training
        config.read_shape = Coordinate(input_shape) * config.input_voxel_size
        config.write_shape = Coordinate(output_shape) * config.output_voxel_size
        config.inference_input_shape = config.read_shape
        config.inference_output_shape = config.write_shape

        config.block_shape = [*output_shape, metadata.out_channels]
        return config

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        return self._with_name_scale(
            {"type": "cellmap", "folder_path": self.cellmap_model.folder_path}
        )
