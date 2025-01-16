from pydantic import BaseModel

from enum import Enum

IP_PATTERN = "CELLMAP_FLOW_SERVER_IP(ip_address)CELLMAP_FLOW_SERVER_IP"


class ModelConfig:
    def __init__(self):
        self._config = None

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
        self.check_config(self._config)
        return self._config

    def check_config(self):
        raise NotImplementedError()


class BioModelConfig(ModelConfig):
    def __init__(self, model_name: str):
        self.model_name = model_name


class ScriptModelConfig(ModelConfig):

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path

    def check_config(self, config):
        assert hasattr(config, "model"), f"Model not found in config {self.script_path}"
        assert hasattr(
            config, "read_shape"
        ), f"read_shape not found in config {self.script_path}"
        assert hasattr(
            config, "write_shape"
        ), f"write_shape not found in config {self.script_path}"
        assert hasattr(
            config, "output_voxel_size"
        ), f"output_voxel_size not found in config {self.script_path}"
        assert hasattr(
            config, "output_channels"
        ), f"output_channels not found in config {self.script_path}"
        assert hasattr(
            config, "block_shape"
        ), f"block_shape not found in config {self.script_path}"

    def _get_config(self):
        from cellmap_flow.utils.load_py import load_safe_config

        config = load_safe_config(self.script_path)
        self.check_config(config)
        return config


class DaCapoModelConfig(ModelConfig):

    def __init__(self, run_name: str, iteration="best"):
        self.run_name = run_name
        self.iteration = iteration
