"""
Templates for generating finetuned model scripts and YAML configurations.

This module provides functions to auto-generate the necessary files for serving
finetuned models, based on the patterns in my_yamls/jrc_c-elegans-bw-1_finetuned.py/yaml.
"""

import ast
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def _extract_tuple_from_ast(node):
    """Try to extract a tuple of ints from an AST node, handling Coordinate(...) calls."""
    # Direct literal: (196, 196, 196)
    try:
        val = ast.literal_eval(node)
        if isinstance(val, (tuple, list)):
            return tuple(int(v) for v in val)
    except (ValueError, TypeError):
        pass

    # Coordinate((196, 196, 196)) or Coordinate((...))
    if isinstance(node, ast.Call) and len(node.args) == 1:
        return _extract_tuple_from_ast(node.args[0])

    return None


def extract_shapes_from_script(script_path: str) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Safely extract input_size and output_size from a Python script using AST parsing.

    Also derives input_size/output_size from read_shape/write_shape and voxel sizes
    if the direct variables are not defined.

    This avoids executing the script (which may try to load models on GPU).

    Args:
        script_path: Path to the Python script

    Returns:
        Tuple of (input_size, output_size) or (None, None) if extraction fails
    """
    try:
        with open(script_path, 'r') as f:
            source = f.read()

        # Parse the source code into an AST
        tree = ast.parse(source)

        input_size = None
        output_size = None
        # Also track read_shape/write_shape and voxel sizes for derivation
        read_shape_voxels = None
        write_shape_voxels = None
        input_voxel_size = None
        output_voxel_size = None

        # Walk through all assignment nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'input_size':
                            input_size = _extract_tuple_from_ast(node.value)
                        elif target.id == 'output_size':
                            output_size = _extract_tuple_from_ast(node.value)
                        elif target.id == 'read_shape':
                            # Only capture the first assignment (before multiplication)
                            if read_shape_voxels is None:
                                read_shape_voxels = _extract_tuple_from_ast(node.value)
                        elif target.id == 'write_shape':
                            if write_shape_voxels is None:
                                write_shape_voxels = _extract_tuple_from_ast(node.value)
                        elif target.id == 'input_voxel_size':
                            input_voxel_size = _extract_tuple_from_ast(node.value)
                        elif target.id == 'output_voxel_size':
                            output_voxel_size = _extract_tuple_from_ast(node.value)

        # Derive input_size from read_shape if not found directly
        if input_size is None and read_shape_voxels is not None:
            if input_voxel_size is not None and read_shape_voxels == input_voxel_size:
                # read_shape was already in nm (unlikely but possible)
                pass
            else:
                # Assume read_shape was defined in voxels (common pattern)
                input_size = read_shape_voxels
                logger.info(f"Derived input_size={input_size} from read_shape")

        if output_size is None and write_shape_voxels is not None:
            output_size = write_shape_voxels
            logger.info(f"Derived output_size={output_size} from write_shape")

        logger.info(f"Extracted shapes from {script_path}: input_size={input_size}, output_size={output_size}")
        return input_size, output_size

    except Exception as e:
        logger.warning(f"AST extraction failed for {script_path}: {e}")

        # Fallback to regex parsing
        try:
            with open(script_path, 'r') as f:
                content = f.read()

            # Match patterns like: input_size = (56, 56, 56)
            input_match = re.search(r'input_size\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)', content)
            output_match = re.search(r'output_size\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)', content)

            if input_match:
                input_size = tuple(map(int, input_match.groups()))
            if output_match:
                output_size = tuple(map(int, output_match.groups()))

            # Also try read_shape/write_shape patterns
            if not input_size:
                read_match = re.search(r'read_shape\s*=\s*(?:Coordinate\s*\()?\s*\((\d+),\s*(\d+),\s*(\d+)\)', content)
                if read_match:
                    input_size = tuple(map(int, read_match.groups()))
            if not output_size:
                write_match = re.search(r'write_shape\s*=\s*(?:Coordinate\s*\()?\s*\((\d+),\s*(\d+),\s*(\d+)\)', content)
                if write_match:
                    output_size = tuple(map(int, write_match.groups()))

            if input_size or output_size:
                logger.info(f"Regex extracted shapes from {script_path}: input_size={input_size}, output_size={output_size}")
                return input_size, output_size

        except Exception as e2:
            logger.warning(f"Regex extraction also failed for {script_path}: {e2}")

    return None, None


def generate_finetuned_model_script(
    base_checkpoint: str,
    lora_adapter_path: str,
    model_name: str,
    channels: List[str],
    input_voxel_size: Tuple[int, int, int],
    output_voxel_size: Tuple[int, int, int],
    lora_r: int,
    lora_alpha: int,
    num_epochs: int,
    learning_rate: float,
    output_path: Path,
    base_script_path: str = None
) -> Path:
    """
    Generate .py script for loading and serving a finetuned model.

    Based on template: my_yamls/jrc_c-elegans-bw-1_finetuned.py

    Args:
        base_checkpoint: Path to base model checkpoint (for checkpoint-based models)
        lora_adapter_path: Path to LoRA adapter directory
        model_name: Name of the finetuned model
        channels: List of output channels (e.g., ["mito"])
        input_voxel_size: Input voxel size (z, y, x) in nm
        output_voxel_size: Output voxel size (z, y, x) in nm
        lora_r: LoRA rank used in training
        lora_alpha: LoRA alpha used in training
        num_epochs: Number of training epochs
        learning_rate: Learning rate used
        output_path: Where to write the .py file
        base_script_path: Path to base model script (for script-based models)

    Returns:
        Path to the generated script file
    """
    # Calculate lora_dropout (typically 0.0 or 0.1)
    lora_dropout = 0.0  # Default used in training

    # Format voxel sizes as tuples
    input_voxel_str = f"({input_voxel_size[0]}, {input_voxel_size[1]}, {input_voxel_size[2]})"
    output_voxel_str = f"({output_voxel_size[0]}, {output_voxel_size[1]}, {output_voxel_size[2]})"

    # Format channels list
    channels_str = ", ".join([f'"{c}"' for c in channels])

    # Determine if this is checkpoint-based or script-based
    is_script_based = bool(base_script_path and not base_checkpoint)

    # Handle model source info
    if is_script_based:
        base_model_info = f"Script: {base_script_path}"
        base_checkpoint_var = ""
        base_script_var = base_script_path
    else:
        base_model_info = base_checkpoint if base_checkpoint else "N/A (trained from scratch)"
        base_checkpoint_var = base_checkpoint if base_checkpoint else ""
        base_script_var = ""

    # Get shapes from base script using safe AST parsing (doesn't execute the script)
    if is_script_based and base_script_path:
        extracted_input_size, extracted_output_size = extract_shapes_from_script(base_script_path)
        base_input_size = extracted_input_size if extracted_input_size else (178, 178, 178)
        base_output_size = extracted_output_size if extracted_output_size else (56, 56, 56)

        if not extracted_input_size or not extracted_output_size:
            logger.warning(
                f"Could not extract shapes from {base_script_path}. "
                f"Using defaults: input_size={base_input_size}, output_size={base_output_size}"
            )
    else:
        base_input_size = (178, 178, 178)
        base_output_size = (56, 56, 56)

    # Format shapes as strings
    input_size_str = f"{base_input_size}"
    output_size_str = f"{base_output_size}"

    # Generate different templates based on model type
    if is_script_based:
        # Template for script-based models
        script_content = f'''"""
LoRA finetuned model: {model_name}

This model is based on:
{base_model_info}

Finetuned with LoRA on user corrections with parameters:
- LoRA rank (r): {lora_r}
- LoRA alpha: {lora_alpha}
- LoRA dropout: {lora_dropout}
- Training epochs: {num_epochs}
- Learning rate: {learning_rate}

Auto-generated by CellMap-Flow finetuning workflow.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import time

import gunpowder as gp
import numpy as np
from funlib.geometry.coordinate import Coordinate
from cellmap_flow.utils.load_py import load_safe_config

logger = logging.getLogger(__name__)

# Model configuration
classes = [{channels_str}]
output_channels = len(classes)

# Paths
BASE_SCRIPT = "{base_script_var}"
LORA_ADAPTER_PATH = "{lora_adapter_path}"

# Voxel sizes and shapes
input_voxel_size = Coordinate{input_voxel_str}
output_voxel_size = Coordinate{output_voxel_str}

# Model input/output shapes (from base model)
input_size = {input_size_str}
output_size = {output_size_str}

# Gunpowder shapes
read_shape = gp.Coordinate(*input_size) * Coordinate(input_voxel_size)
write_shape = gp.Coordinate(*output_size) * Coordinate(output_voxel_size)

# Block shape for processing
block_shape = np.array((*output_size, output_channels))

# Load base model ONCE at module level
logger.info(f"Loading base model from: {{BASE_SCRIPT}}")
_load_t0 = time.perf_counter()
_base_config = load_safe_config(BASE_SCRIPT, force_safe=False)
_base_model = _base_config.model
_base_elapsed = time.perf_counter() - _load_t0
logger.info(f"Base model/script load time: {{_base_elapsed:.2f}}s")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {{device}}")

# Apply LoRA adapter to base model
from cellmap_flow.finetune.lora_wrapper import load_lora_adapter
logger.info(f"Loading LoRA adapter from: {{LORA_ADAPTER_PATH}}")
_lora_t0 = time.perf_counter()
model = load_lora_adapter(
    _base_model,
    LORA_ADAPTER_PATH,
    is_trainable=False  # Inference mode
)
_lora_elapsed = time.perf_counter() - _lora_t0
model = model.to(device)
model.eval()
_total_elapsed = time.perf_counter() - _load_t0

logger.info("LoRA finetuned model loaded successfully")
logger.info(
    f"Model load timings (s): base={{_base_elapsed:.2f}}, "
    f"lora={{_lora_elapsed:.2f}}, total={{_total_elapsed:.2f}}"
)
logger.info(f"Model classes: {{classes}}")
logger.info(f"Input shape: {{input_size}}, Output shape: {{output_size}}")
logger.info(f"Voxel sizes - Input: {{input_voxel_size}}, Output: {{output_voxel_size}}")
'''
    else:
        # Template for checkpoint-based models (original template)
        script_content = f'''"""
LoRA finetuned model: {model_name}

This model is based on:
{base_model_info}

Finetuned with LoRA on user corrections with parameters:
- LoRA rank (r): {lora_r}
- LoRA alpha: {lora_alpha}
- LoRA dropout: {lora_dropout}
- Training epochs: {num_epochs}
- Learning rate: {learning_rate}

Auto-generated by CellMap-Flow finetuning workflow.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import time

import gunpowder as gp
import numpy as np
from funlib.geometry.coordinate import Coordinate

logger = logging.getLogger(__name__)

# Model configuration
classes = [{channels_str}]
output_channels = len(classes)

# Paths
BASE_CHECKPOINT = "{base_checkpoint_var}"
LORA_ADAPTER_PATH = "{lora_adapter_path}"

# Voxel sizes and shapes
input_voxel_size = Coordinate{input_voxel_str}
output_voxel_size = Coordinate{output_voxel_str}

# Model input/output shapes (fly model defaults)
# Note: These may need adjustment based on your specific model architecture
input_size = (178, 178, 178)
output_size = (56, 56, 56)

# Gunpowder shapes
read_shape = gp.Coordinate(*input_size) * Coordinate(input_voxel_size)
write_shape = gp.Coordinate(*output_size) * Coordinate(output_voxel_size)

# Block shape for processing
block_shape = np.array((*output_size, output_channels))


def load_base_model(checkpoint_path: str, num_channels: int, device) -> nn.Module:
    """Load the base fly model from checkpoint."""
    from fly_organelles.model import StandardUnet

    logger.info(f"Loading base model from: {{checkpoint_path}}")
    t0 = time.perf_counter()

    # Load the base model
    model_backbone = StandardUnet(num_channels)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model_backbone.load_state_dict(checkpoint["model_state_dict"])

    # Wrap with sigmoid
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    elapsed = time.perf_counter() - t0
    logger.info(f"Base checkpoint load time: {{elapsed:.2f}}s")

    return model


def load_finetuned_model(device) -> nn.Module:
    """Load the base model and apply LoRA adapter."""
    from cellmap_flow.finetune.lora_wrapper import load_lora_adapter
    t0 = time.perf_counter()

    # Load base model
    if BASE_CHECKPOINT:
        base_t0 = time.perf_counter()
        base_model = load_base_model(BASE_CHECKPOINT, len(classes), device)
        base_elapsed = time.perf_counter() - base_t0
    else:
        # Model was trained from scratch - create fresh model
        logger.warning("No base checkpoint specified - model was trained from scratch")
        base_t0 = time.perf_counter()
        from fly_organelles.model import StandardUnet
        model_backbone = StandardUnet(len(classes))
        base_model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
        base_model.to(device)
        base_elapsed = time.perf_counter() - base_t0

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from: {{LORA_ADAPTER_PATH}}")
    lora_t0 = time.perf_counter()
    model = load_lora_adapter(
        base_model,
        LORA_ADAPTER_PATH,
        is_trainable=False  # Inference mode
    )
    lora_elapsed = time.perf_counter() - lora_t0
    total_elapsed = time.perf_counter() - t0
    logger.info(
        f"Model load timings (s): base={{base_elapsed:.2f}}, "
        f"lora={{lora_elapsed:.2f}}, total={{total_elapsed:.2f}}"
    )

    return model


# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {{device}}")

model = load_finetuned_model(device)
model = model.to(device)
model.eval()

logger.info("LoRA finetuned model loaded successfully")
logger.info(f"Model classes: {{classes}}")
logger.info(f"Input shape: {{input_size}}, Output shape: {{output_size}}")
logger.info(f"Voxel sizes - Input: {{input_voxel_size}}, Output: {{output_voxel_size}}")
'''

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(script_content)

    logger.info(f"Generated finetuned model script: {output_path}")

    return output_path


def generate_finetuned_model_yaml(
    script_path: Path,
    model_name: str,
    resolution: int,
    output_path: Path,
    data_path: str,
    queue: str = "gpu_h100",
    charge_group: str = "cellmap",
    json_data: dict = None,
    scale: str = "s0"
) -> Path:
    """
    Generate .yaml configuration for serving a finetuned model.

    Based on template: my_yamls/jrc_c-elegans-bw-1_finetuned.yaml

    Args:
        script_path: Path to the generated .py script
        model_name: Name of the finetuned model
        resolution: Voxel resolution in nm
        output_path: Where to write the .yaml file
        data_path: Path to actual dataset used for training (REQUIRED - no placeholders)
        queue: LSF queue name
        charge_group: LSF charge group
        json_data: Optional dict with input_norm and postprocess from base model
        scale: Scale level (e.g., "s0", "s1") from base model

    Returns:
        Path to the generated YAML file
    """
    # Validate inputs - no placeholders allowed!
    if not data_path or data_path == "/path/to/your/data.zarr":
        raise ValueError(
            "data_path is required and cannot be a placeholder. "
            "Must provide actual dataset path from training corrections."
        )

    # Data path comment (always from corrections)
    data_path_comment = "# Data path from training corrections\n#\n"

    # Format json_data - use provided or warn if missing
    import yaml as yaml_lib
    if json_data:
        json_data_comment = "# Normalization and postprocessing from base model\n"
        json_data_str = yaml_lib.dump({'json_data': json_data}, default_flow_style=False, sort_keys=False).strip()
    else:
        # Missing json_data is a warning case - provide generic defaults
        # but log a warning (already done in job_manager)
        json_data_comment = "# WARNING: No normalization found in base model!\n# Using generic defaults - model may not work correctly.\n# Update these values based on your data.\n"
        json_data_str = '''json_data:
  input_norm:
    MinMaxNormalizer:
      min_value: 0
      max_value: 65535
      invert: false
    LambdaNormalizer:
      expression: x*2-1
  postprocess:
    DefaultPostprocessor:
      clip_min: 0
      clip_max: 1.0'''

    # Convert script_path to absolute path
    script_path_abs = Path(script_path).resolve()

    yaml_content = f'''# Finetuned model configuration: {model_name}
# Auto-generated by CellMap-Flow finetuning workflow
#
{data_path_comment}
data_path: "{data_path}"

charge_group: "{charge_group}"
queue: "{queue}"

{json_data_comment}{json_data_str}

# Model configuration
models:
  - type: "script"
    scale: "{scale}"
    resolution: {resolution}
    script_path: "{script_path_abs}"
    name: "{model_name}"
'''

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"Generated finetuned model YAML: {output_path}")

    return output_path


def register_finetuned_model(yaml_path: Path):
    """
    Load YAML config and register the finetuned model in g.models_config.

    This allows the model to appear in the dashboard immediately.

    Args:
        yaml_path: Path to the generated YAML config

    Returns:
        The newly created ScriptModelConfig object
    """
    from cellmap_flow.utils.config_utils import build_model_from_entry
    from cellmap_flow import globals as g
    import yaml

    logger.info(f"Registering finetuned model from: {yaml_path}")

    # Load YAML
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract model entry
    if "models" not in config or len(config["models"]) == 0:
        raise ValueError(f"No models found in YAML config: {yaml_path}")

    model_entry = config["models"][0]

    # Build ModelConfig object
    try:
        model_config = build_model_from_entry(model_entry)

        # Add to global models config
        if not hasattr(g, "models_config"):
            g.models_config = []

        g.models_config.append(model_config)

        logger.info(f"Successfully registered finetuned model: {model_config.name}")

        return model_config

    except Exception as e:
        logger.error(f"Failed to register finetuned model: {e}")
        raise RuntimeError(f"Model registration failed: {e}")
