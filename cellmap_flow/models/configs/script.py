"""``ScriptModelConfig``: a model that a Python script defines.

``load_safe_config`` checks the script, runs it and returns its globals as
a Config: the model (or a ``predict``) and its geometry. Sizes in voxels
(``input_size``, ``output_size``) and shapes in nm (``read_shape``,
``write_shape``) are each derived from the other when only one is given.
"""

import ast
import logging
import os

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.plugins import analyze_script

logger = logging.getLogger(__name__)


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def _force_safe_from_env() -> bool:
    """FORCE_SAFE_CONFIG as a boolean, read now rather than at import.

    Unset, empty, 0/false/no/off mean false; 1/true/yes/on mean true. Anything
    else is treated as true, since this switch exists to refuse scripts.
    """
    value = os.environ.get("FORCE_SAFE_CONFIG", "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning(
        f"FORCE_SAFE_CONFIG={os.environ['FORCE_SAFE_CONFIG']!r} is not a boolean; "
        "treating it as true"
    )
    return True


def load_safe_config(config_path, force_safe=None):
    """
    Loads the configuration script at `config_path` after verifying its safety.
    If `force_safe` is True, raises an error if the script is deemed unsafe.
    When it is None, the FORCE_SAFE_CONFIG environment variable decides.
    """
    if force_safe is None:
        force_safe = _force_safe_from_env()
    # print(f"Analyzing script for obvious security liabilities:\n\t{config_path}")
    # print(
    #     "Keep in mind that this is not a foolproof security measure. Use caution using code from untrusted sources."
    # )
    is_safe, issues = analyze_script(config_path)
    if not is_safe:
        print("Script contains unsafe elements:")
        for issue in issues:
            print(f" - {issue}")
        if force_safe:
            raise ValueError(
                "Unsafe script detected; loading aborted. You can set the environment variable FORCE_SAFE_CONFIG=False or pass force_safe=False to override."
            )

    # Here, not at module level: upath brings fsspec (~1.5s), and the CLIs
    # import every model type to build their commands.
    from upath import UPath

    # Load the config module if script is safe
    config_path = UPath(config_path)
    # Create a dedicated namespace for the config
    config_namespace = {}
    try:
        with open(config_path, "r") as config_file:
            code = config_file.read()
            # Parse the code into an AST
            tree = ast.parse(code)

            # Define a node transformer to replace __file__ with the config path
            class ReplaceFileNode(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "__file__":
                        return ast.Constant(value=str(config_path), kind=None)
                    return node

            # Transform the AST
            transformer = ReplaceFileNode()
            tree = transformer.visit(tree)

            # Convert the modified AST back to source code
            code = ast.unparse(tree)

            exec(code, config_namespace)
        # Extract the config object from the namespace
        config = Config(**config_namespace)
    except Exception as e:
        error_msg = (
            f"Failed to execute configuration file: {config_path}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error details: {str(e)}"
        )
        raise RuntimeError(error_msg) from e

    return config


class ScriptModelConfig(ModelConfig):

    cli_name = "script"

    def __init__(self, script_path, name=None, scale=None):
        super().__init__()
        self.script_path = script_path
        self.name = name
        self.scale = scale

    def _get_config(self):
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
