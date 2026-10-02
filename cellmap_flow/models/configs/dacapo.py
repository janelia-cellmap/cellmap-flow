"""``DaCapoModelConfig``: a DaCapo run's model, with one iteration's weights.

The shapes come from the model, the voxel size from the run's first
training dataset, and the channel names from its task.
"""

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.models.configs.base import Config, ModelConfig, _get_device


class DaCapoModelConfig(ModelConfig):

    cli_name = "dacapo"

    def __init__(self, run_name: str, iteration: int, name=None, scale=None):
        super().__init__()
        self.run_name = run_name
        self.iteration = iteration
        self.name = name
        self.scale = scale

    def _get_config(self):
        config = Config()
        run = self._load_dacapo_run()

        run.model.to(_get_device())
        run.model.eval()
        config.model = run.model

        in_shape = run.model.eval_input_shape
        # (number of output channels, spatial output shape)
        out_channels, out_shape = run.model.compute_output_shape(in_shape)
        voxel_size = run.datasplit.train[0].raw.voxel_size

        config.input_voxel_size = Coordinate(voxel_size)
        config.output_voxel_size = Coordinate(run.model.scale(voxel_size))
        config.read_shape = Coordinate(in_shape) * config.input_voxel_size
        # Output voxels are output_voxel_size wide (run.model.scale may make
        # them differ from the input's).
        config.write_shape = Coordinate(out_shape) * config.output_voxel_size
        config.channels = self._get_channels(run.task, out_channels)
        config.output_channels = len(config.channels)
        config.block_shape = np.array(tuple(out_shape) + (config.output_channels,))
        return config

    def _load_dacapo_run(self):
        """Load DaCapo run with optional weights."""
        from dacapo.experiments import Run
        from dacapo.store.create_store import create_config_store, create_weights_store

        config_store = create_config_store()
        run_config = config_store.retrieve_run_config(self.run_name)
        run = Run(run_config)

        if self.iteration > 0:
            weights_store = create_weights_store()
            weights = weights_store.retrieve_weights(run, self.iteration)
            run.model.load_state_dict(weights.model)
        return run

    @staticmethod
    def _get_channels(task, num_channels=None):
        """Channel names for the task, one per channel the model outputs.

        Tasks without names get the old guesses (x/y/z for affinities,
        "membrane" otherwise) only when those have the right length; an
        affinity task with 9 offsets outputs 9 channels, not 3.
        """
        if hasattr(task, "channels"):
            names = list(task.channels)
        elif type(task).__name__ == "AffinitiesTask":
            names = ["x", "y", "z"]
        else:
            names = ["membrane"]
        if num_channels is not None and len(names) != int(num_channels):
            neighborhood = getattr(getattr(task, "predictor", None), "neighborhood", None)
            if neighborhood is not None and len(neighborhood) == int(num_channels):
                names = [
                    "aff_" + "_".join(str(int(v)) for v in offset)
                    for offset in neighborhood
                ]
            else:
                names = [f"channel_{i}" for i in range(int(num_channels))]
        return names

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        return self._with_name_scale(
            {"type": "dacapo", "run_name": self.run_name, "iteration": self.iteration}
        )
