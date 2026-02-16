"""Registry and introspection tools for ModelConfig subclasses."""

import inspect
from typing import Dict, Any, List
from cellmap_flow.models.models_config import (
    ScriptModelConfig,
    DaCapoModelConfig,
    FlyModelConfig,
    BioModelConfig,
    CellMapModelConfig,
)

# Registry of available model config classes
MODEL_CONFIG_CLASSES = {
    'ScriptModelConfig': ScriptModelConfig,
    'DaCapoModelConfig': DaCapoModelConfig,
    'FlyModelConfig': FlyModelConfig,
    'BioModelConfig': BioModelConfig,
    'CellMapModelConfig': CellMapModelConfig,
}


def get_parameter_info(cls) -> Dict[str, Any]:
    """
    Extract parameter information from a ModelConfig subclass __init__ method.
    
    Returns a dict with parameter names as keys and info dicts with:
    - 'type': the type hint (str, int, list, tuple, etc.)
    - 'required': whether the parameter is required (no default value)
    - 'default': the default value if provided
    - 'description': parameter name for UI labels
    """
    sig = inspect.signature(cls.__init__)
    params = {}
    
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue
            
        param_info = {
            'name': param_name,
            'required': param.default == inspect.Parameter.empty,
            'description': param_name.replace('_', ' ').title(),
        }
        
        # Extract type hint
        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
            # Handle generic types like list[str], tuple, etc.
            if hasattr(annotation, '__origin__'):
                param_info['type'] = str(annotation.__origin__.__name__)
            else:
                param_info['type'] = annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)
        else:
            param_info['type'] = 'string'
        
        # Set default value
        if param.default != inspect.Parameter.empty:
            param_info['default'] = param.default
        
        params[param_name] = param_info
    
    return params


def get_all_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all available ModelConfig subclasses.
    
    Returns a dict like:
    {
        'ScriptModelConfig': {
            'display_name': 'Script Model',
            'description': 'Load model from Python script',
            'parameters': {
                'script_path': {
                    'name': 'script_path',
                    'type': 'string',
                    'required': True,
                    'description': 'Script Path',
                    'input_type': 'file'
                },
                'name': {
                    'name': 'name',
                    'type': 'string',
                    'required': False,
                    'description': 'Name',
                    'default': None
                },
                ...
            }
        },
        ...
    }
    """
    registry = {}
    
    for class_name, cls in MODEL_CONFIG_CLASSES.items():
        display_name = class_name.replace('Config', '').replace('ModelConfig', '')
        # Insert spaces before capitals: ScriptModel -> Script Model
        display_name = ''.join([f' {c}' if c.isupper() and i > 0 else c 
                               for i, c in enumerate(display_name)])
        
        params = get_parameter_info(cls)
        
        # Add input type hints for specific parameters
        for param_name, param_info in params.items():
            if 'path' in param_name.lower() or 'checkpoint' in param_name.lower():
                param_info['input_type'] = 'file'
            elif 'channels' in param_name.lower() or 'voxel_size' in param_name.lower():
                param_info['input_type'] = 'textarea'  # for multi-line JSON
            elif param_name in ('input_size', 'output_size', 'edge_length_to_process', 'iteration'):
                param_info['input_type'] = 'number'
            else:
                param_info['input_type'] = 'text'
        
        registry[class_name] = {
            'display_name': display_name,
            'description': f'Create a {display_name} model configuration',
            'parameters': params,
            'class_name': class_name,
        }
    
    return registry


def instantiate_model_config(class_name: str, params: Dict[str, Any]) -> Any:
    """
    Instantiate a ModelConfig subclass with the provided parameters.
    
    Args:
        class_name: Name of the ModelConfig subclass (e.g., 'ScriptModelConfig')
        params: Dictionary of parameters to pass to __init__
        
    Returns:
        An instance of the ModelConfig subclass
        
    Raises:
        ValueError: If class_name is not recognized or params are invalid
    """
    if class_name not in MODEL_CONFIG_CLASSES:
        raise ValueError(f"Unknown model config class: {class_name}")
    
    cls = MODEL_CONFIG_CLASSES[class_name]
    
    # Parse parameter values based on their types
    parsed_params = {}
    sig = inspect.signature(cls.__init__)
    
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls') or param_name not in params:
            continue
        
        value = params[param_name]
        if value is None or value == '':
            # Skip None/empty values for optional parameters
            if param.default != inspect.Parameter.empty:
                continue
            else:
                raise ValueError(f"Required parameter '{param_name}' is missing")
        
        # Parse based on type annotation
        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
            
            # Handle list types
            if hasattr(annotation, '__origin__') and annotation.__origin__ in (list, tuple):
                if isinstance(value, str):
                    # Try to parse as JSON
                    import json
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # Try simple comma-separated parsing
                        value = [v.strip() for v in value.split(',')]
                if annotation.__origin__ == tuple:
                    value = tuple(value)
            
            # Handle numeric types
            elif annotation in (int, float):
                value = annotation(value)
            
            # Handle tuple from string (e.g., "16, 16, 16" -> (16, 16, 16))
            elif 'tuple' in str(annotation).lower():
                if isinstance(value, str):
                    import json
                    try:
                        value = json.loads(value)
                    except:
                        value = tuple(float(v.strip()) for v in value.split(','))
                if not isinstance(value, tuple):
                    value = tuple(value)
        
        parsed_params[param_name] = value
    
    try:
        return cls(**parsed_params)
    except Exception as e:
        raise ValueError(f"Failed to instantiate {class_name}: {str(e)}")
