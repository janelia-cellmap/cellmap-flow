IP_PATTERN = "CELLMAP_FLOW_SERVER_IP(ip_address)CELLMAP_FLOW_SERVER_IP"

import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(self):
        self._config = None

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
        check_config(self._config)
        return self._config



class BioModelConfig(ModelConfig):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name


class ScriptModelConfig(ModelConfig):

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path

    def _get_config(self):
        from cellmap_flow.utils.load_py import load_safe_config
        config = load_safe_config(self.script_path)
        return config


class DaCapoModelConfig(ModelConfig):

    def __init__(self, run_name: str, iteration: int):
        super().__init__()
        self.run_name = run_name
        self.iteration = iteration

    def _get_config(self):
        from daisy.coordinate import Coordinate
        import numpy as np
        import torch

        config = Config()

        run = get_dacapo_run_model(self.run_name, self.iteration)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device:", device)

        run.model.to(device)
        run.model.eval()
        config.model = run.model

        in_shape = run.model.eval_input_shape
        out_shape = run.model.compute_output_shape(in_shape)[1]

        voxel_size = run.datasplit.train[0].raw.voxel_size
        config.input_voxel_size = voxel_size
        config.read_shape = Coordinate(in_shape) * Coordinate(voxel_size)
        config.write_shape = Coordinate(out_shape) * Coordinate(voxel_size)
        config.output_voxel_size = Coordinate(run.model.scale(voxel_size))
        channels = get_dacapo_channels(run.task)
        config.output_channels = len(
            channels
        )  # 0:all_mem,1:organelle,2:mito,3:er,4:nucleus,5:pm,6:vs,7:ld
        config.block_shape = np.array(tuple(out_shape) + (len(channels),))

        return config


def check_config(config):
    assert hasattr(config, "model"), f"Model not found in config"
    assert hasattr(
        config, "read_shape"
    ), f"read_shape not found in config"
    assert hasattr(
        config, "write_shape"
    ), f"write_shape not found in config"
    assert hasattr(
        config, "output_voxel_size"
    ), f"output_voxel_size not found in config"
    assert hasattr(
        config, "output_channels"
    ), f"output_channels not found in config"
    assert hasattr(
        config, "block_shape"
    ), f"block_shape not found in config"

class Config:
    pass


def get_dacapo_channels(task):
    if hasattr(task, "channels"):
        return task.channels
    elif type(task).__name__ == "AffinitiesTask":
        return ["x", "y", "z"]
    else:
        return ["membrane"]


def get_dacapo_run_model(run_name, iteration):
    from dacapo.experiments import Run
    from dacapo.store.create_store import create_config_store, create_weights_store

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)
    if iteration > 0:

        weights_store = create_weights_store()
        weights = weights_store.retrieve_weights(run, iteration)
        run.model.load_state_dict(weights.model)

    return run
