"""``FlyModelConfig``: a fly_organelles checkpoint.

The checkpoint holds no geometry, so the constructor takes the voxel sizes
and the input and output sizes (178 and 56 voxels a side when not given). A
TorchScript file (``.ts``) or a ``StandardUnet`` state dict gets a sigmoid
on top; a whole pickled model (``model.pt``) is used as it is, and only
when ``CELLMAP_FLOW_ALLOW_PICKLE`` allows it.
"""

import logging

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import ModelConfig, _as_int_tuple, _get_device
from cellmap_flow.utils.serialize_config import Config

logger = logging.getLogger(__name__)


class FlyModelConfig(ModelConfig):

    cli_name = "fly"

    def __init__(
        self,
        checkpoint_path: str,
        channels: list[str],
        input_voxel_size: tuple,
        output_voxel_size: tuple,
        name: str = None,
        input_size=None,
        output_size=None,
        scale=None,
    ):
        super().__init__()
        self.name = name
        self.checkpoint_path = checkpoint_path
        if isinstance(channels, str):
            channels = [c.strip() for c in channels.split(",") if c.strip()]
        self.channels = channels
        if isinstance(input_voxel_size, str):
            input_voxel_size = _as_int_tuple(input_voxel_size)
        if isinstance(output_voxel_size, str):
            output_voxel_size = _as_int_tuple(output_voxel_size)
        self.input_voxel_size = input_voxel_size
        self.output_voxel_size = output_voxel_size
        self.scale = scale
        self._model = None
        if input_size is None or output_size is None:
            input_size = (178, 178, 178)
            output_size = (56, 56, 56)
            logger.warning(
                "Input and output size not provided, defaulting to (178, 178, 178) and (56, 56, 56)"
            )
        # The server CLI passes these as "178,178,178" strings.
        self.input_size = _as_int_tuple(input_size)
        self.output_size = _as_int_tuple(output_size)

    def load_eval_model(self, num_channels, checkpoint_path):
        """Load evaluation model from checkpoint (TorchScript or PyTorch)."""
        # Imported here rather than at module scope: importing torch costs
        # ~7s, and the CLI builds its command list from this module, so
        # `cellmap_flow --help` paid that before printing anything.
        import torch

        device = _get_device()

        if checkpoint_path.endswith(".ts"):
            model_backbone = torch.jit.load(checkpoint_path, map_location=device)
        elif checkpoint_path.endswith("model.pt"):
            # Load full model directly (for trusted fly_organelles models).
            # torch.load(weights_only=False) unpickles arbitrary objects, which
            # is a code-exec sink if the checkpoint comes from an untrusted
            # location. Require the operator to opt-in explicitly.
            import os
            trusted = os.environ.get("CELLMAP_FLOW_ALLOW_PICKLE", "").lower() in ("1", "true", "yes")
            if not trusted:
                raise ValueError(
                    f"Refusing to torch.load (weights_only=False) checkpoint {checkpoint_path}. "
                    "This unpickles arbitrary objects. If the checkpoint is from a trusted "
                    "source, set CELLMAP_FLOW_ALLOW_PICKLE=1 in the environment."
                )
            model = torch.load(checkpoint_path, weights_only=False, map_location=device)
            model.to(device)
            model.eval()
            return model
        else:
            from fly_organelles.model import StandardUnet

            model_backbone = StandardUnet(num_channels)
            checkpoint = torch.load(
                checkpoint_path, weights_only=True, map_location="cpu"
            )
            model_backbone.load_state_dict(checkpoint["model_state_dict"])

        model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
        model.to(device)
        model.eval()
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_eval_model(len(self.channels), self.checkpoint_path)
        return self._model

    def _get_config(self):
        config = Config()
        config.model = self.model
        config.input_voxel_size = Coordinate(self.input_voxel_size)
        config.output_voxel_size = Coordinate(self.output_voxel_size)
        config.read_shape = Coordinate(self.input_size) * config.input_voxel_size
        # Output voxels are output_voxel_size wide; using the input voxel size
        # here made the context and the shape check wrong whenever they differ.
        config.write_shape = Coordinate(self.output_size) * config.output_voxel_size
        config.channels = self.channels
        config.output_channels = len(self.channels)
        config.block_shape = np.array(
            tuple(self.output_size) + (config.output_channels,)
        )
        # Add axes_names for server compatibility
        config.axes_names = ["x", "y", "z", "c^"]
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {
            "type": "fly",
            "checkpoint_path": self.checkpoint_path,
            "channels": self.channels,
            "input_voxel_size": list(self.input_voxel_size),
            "output_voxel_size": list(self.output_voxel_size),
        }
        if self.name is not None:
            result["name"] = self.name
        if self.input_size is not None:
            result["input_size"] = list(self.input_size)
        if self.output_size is not None:
            result["output_size"] = list(self.output_size)
        if self.scale is not None:
            result["scale"] = self.scale
        return result
