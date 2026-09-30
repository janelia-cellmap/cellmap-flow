"""``ScriptModelConfig``: a model that a Python script defines.

``load_safe_config`` checks the script, runs it and returns its globals as
a Config: the model (or a ``predict``) and its geometry. Sizes in voxels
(``input_size``, ``output_size``) and shapes in nm (``read_shape``,
``write_shape``) are each derived from the other when only one is given.
"""

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import ModelConfig


class ScriptModelConfig(ModelConfig):

    cli_name = "script"

    def __init__(self, script_path, name=None, scale=None):
        super().__init__()
        self.script_path = script_path
        self.name = name
        self.scale = scale

    def _get_config(self):
        from cellmap_flow.utils.load_py import load_safe_config

        config = load_safe_config(self.script_path)

        # Derive read_shape/write_shape from input_size/output_size or vice versa
        has_input_size = hasattr(config, "input_size")
        has_output_size = hasattr(config, "output_size")
        has_read_shape = hasattr(config, "read_shape")
        has_write_shape = hasattr(config, "write_shape")

        if not has_read_shape and has_input_size:
            config.read_shape = Coordinate(config.input_size) * Coordinate(
                config.input_voxel_size
            )
        if not has_write_shape and has_output_size:
            config.write_shape = Coordinate(config.output_size) * Coordinate(
                config.output_voxel_size
            )
        # Reverse: derive input_size/output_size from read_shape/write_shape
        if not has_input_size and has_read_shape:
            config.input_size = tuple(
                int(s)
                for s in Coordinate(config.read_shape)
                / Coordinate(config.input_voxel_size)
            )
        if not has_output_size and has_write_shape:
            config.output_size = tuple(
                int(s)
                for s in Coordinate(config.write_shape)
                / Coordinate(config.output_voxel_size)
            )

        if not hasattr(config, "block_shape"):
            config.block_shape = np.array(
                tuple(config.output_size) + (config.output_channels,)
            )
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        return self._with_name_scale({"type": "script", "script_path": self.script_path})
