"""``CellMapModelConfig``: a model that cellmap_models exported, from its folder.

The folder's metadata gives the shapes, voxel sizes and channel names; the
model is its TorchScript export.
"""

from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import Config, ModelConfig, _get_device


class CellMapModelConfig(ModelConfig):
    """Configuration class for a CellmapModel."""

    cli_name = "cellmap"

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

        config.read_shape = Coordinate(metadata.input_shape) * config.input_voxel_size
        config.write_shape = Coordinate(metadata.output_shape) * config.output_voxel_size
        config.inference_input_shape = Coordinate(metadata.inference_input_shape)* config.input_voxel_size
        config.inference_output_shape = Coordinate(metadata.inference_output_shape)* config.output_voxel_size
        
        config.block_shape = [*metadata.output_shape, metadata.out_channels]

        config.model = self.cellmap_model.ts_model
        config.model.to(_get_device())
        config.model.eval()
        return config

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        return self._with_name_scale(
            {"type": "cellmap", "folder_path": self.cellmap_model.folder_path}
        )
