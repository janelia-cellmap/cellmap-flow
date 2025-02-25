IP_PATTERN = "CELLMAP_FLOW_SERVER_IP(ip_address)CELLMAP_FLOW_SERVER_IP"

import logging
from typing import List
import yaml

logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(self):
        self._config = None

    def __str__(self) -> str:
        attributes = vars(self)
        elms = ", ".join(f"{key}: {value}" for key, value in attributes.items())
        return f"{type(self)} : {elms}"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
        check_config(self._config)
        return self._config


class BioModelConfig(ModelConfig):
    def __init__(self, model_name: str, name=None):
        super().__init__()
        self.model_name = model_name
        self.name = name

    @property
    def command(self):
        return f"bioimage -m {self.model_name}"

    def _get_config(self):
        from bioimageio.core import load_description

        config = Config()
        config.model = load_description(self.model_name)
        return config


class ScriptModelConfig(ModelConfig):

    def __init__(self, script_path, name=None):
        super().__init__()
        self.script_path = script_path
        self.name = name

    @property
    def command(self):
        return f"script -s {self.script_path}"

    def _get_config(self):
        from cellmap_flow.utils.load_py import load_safe_config

        config = load_safe_config(self.script_path)
        return config


class DaCapoModelConfig(ModelConfig):

    def __init__(self, run_name: str, iteration: int, name=None):
        super().__init__()
        self.run_name = run_name
        self.iteration = iteration
        self.name = name

    @property
    def command(self):
        return f"dacapo -r {self.run_name} -i {self.iteration}"

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
    assert hasattr(config, "read_shape"), f"read_shape not found in config"
    assert hasattr(config, "write_shape"), f"write_shape not found in config"
    assert hasattr(config, "input_voxel_size"), f"input_voxel_size not found in config"
    assert hasattr(
        config, "output_voxel_size"
    ), f"output_voxel_size not found in config"
    assert hasattr(config, "output_channels"), f"output_channels not found in config"
    assert hasattr(config, "block_shape"), f"block_shape not found in config"


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


def parse_model_configs(yaml_file_path: str) -> List[ModelConfig]:
    """
    Reads a YAML file that defines a list of model configs.
    Validates them manually, then returns a list of constructed ModelConfig objects.
    """
    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level YAML structure must be a list.")

    configs: List[ModelConfig] = []

    for idx, model_def in enumerate(data):
        # Common checks:
        if "type" not in model_def:
            raise ValueError(f"Missing 'type' field in model definition #{idx+1}")

        model_type = model_def["type"]
        name = model_def.get("name")

        if model_type == "bio":
            # Expect "model_name"
            if "model_name" not in model_def:
                raise ValueError(f"Missing 'model_name' in bio model #{idx+1}")
            config = BioModelConfig(
                model_name=model_def["model_name"],
                name=name,
            )

        elif model_type == "script":
            # Expect "script_path"
            if "script_path" not in model_def:
                raise ValueError(f"Missing 'script_path' in script model #{idx+1}")
            config = ScriptModelConfig(
                script_path=model_def["script_path"],
                name=name,
            )

        elif model_type == "dacapo":
            # Expect "run_name" and "iteration"
            if "run_name" not in model_def or "iteration" not in model_def:
                raise ValueError(
                    f"Missing 'run_name' or 'iteration' in dacapo model #{idx+1}"
                )
            config = DaCapoModelConfig(
                run_name=model_def["run_name"],
                iteration=model_def["iteration"],
                name=name,
            )

        else:
            raise ValueError(
                f"Invalid 'type' field '{model_type}' in model definition #{idx+1}"
            )

        configs.append(config)

    return configs

from cellmap_flow.models.cellmap_models import CellmapModel
from typing import Optional

class CellMapModelConfig(ModelConfig):
    """
    Configuration class for a CellmapModel.
    Similar to DaCapoModelConfig, but uses a CellmapModel object
    to populate the necessary metadata and define a prediction function.
    """

    def __init__(self, cellmap_model: CellmapModel, name: Optional[str] = None):
        """
        :param cellmap_model: An instance of CellmapModel containing metadata 
                              and references to ONNX, TorchScript, or PyTorch models.
        :param name: Optional name for this configuration.
        """
        super().__init__()
        self.cellmap_model = cellmap_model
        self.name = name

    @property
    def command(self) -> str:
        """
        You can either return a placeholder command or remove this property if not needed.
        For consistency with your DaCapoModelConfig, we return something minimal here.
        """
        if self.name:
            return f"cellmap-run {self.name}"
        return "cellmap-run"

    def _get_config(self) -> Config:
        """
        Build and return a `Config` object populated using the CellmapModel's metadata and ONNX runtime.
        """
        config = Config()

        # Access metadata from the CellmapModel
        metadata = self.cellmap_model.metadata

        # If you want to store any of these metadata fields into your config object, do so here:
        config.model_name = metadata.model_name
        config.model_type = metadata.model_type
        config.framework = metadata.framework
        config.spatial_dims = metadata.spatial_dims
        config.in_channels = metadata.in_channels
        config.out_channels = metadata.out_channels
        config.iteration = metadata.iteration
        config.input_voxel_size = metadata.input_voxel_size
        config.output_voxel_size = metadata.output_voxel_size
        config.channels_names = metadata.channels_names
        config.input_shape = metadata.input_shape
        config.output_shape = metadata.output_shape
        # ... add or remove as needed ...

        # If you need a block shape or other derived parameters:
        # config.block_shape = some_computed_value

        # Define the predict function using the ONNX Runtime session
        def predict(input_data: np.ndarray) -> np.ndarray:
            """
            Run inference on a NumPy array using the loaded ONNX model.
            Assumes a single input and single output; adjust as needed.
            """
            onnx_session = self.cellmap_model.onnx_model
            if onnx_session is None:
                raise RuntimeError("No ONNX model loaded or onnxruntime not installed.")

            # Typically, you retrieve input/output names from the session
            input_name = onnx_session.get_inputs()[0].name
            output_names = [o.name for o in onnx_session.get_outputs()]

            # Run the inference
            result = onnx_session.run(output_names, {input_name: input_data})

            # If there's only one output, result might be [output_ndarray]. 
            # Return as you see fit
            return result[0] if len(result) == 1 else result

        # Attach predict method to the config
        config.predict = predict

        # If you also want to set `config.model` to be the PyTorch model or something else:
        # config.model = self.cellmap_model.pytorch_model
        # or
        # config.model = self.cellmap_model.ts_model

        return config
