"""
Utility functions for CLI generation and management.
"""

import re
import inspect
import click
from typing import Type, get_type_hints, Any, Dict, Tuple


def get_all_subclasses(base_class: Type) -> Dict[str, Type]:
    """
    Get all subclasses of a base class and convert their names to CLI-friendly format.
    
    Args:
        base_class: The base class to find subclasses for
        
    Returns:
        Dictionary mapping CLI-friendly names to class objects
        
    Example:
        >>> from cellmap_flow.models.models_config import ModelConfig
        >>> configs = get_all_subclasses(ModelConfig)
        >>> # Returns: {'dacapo': DaCapoModelConfig, 'script': ScriptModelConfig, ...}
    """
    subclasses = {}
    for subclass in base_class.__subclasses__():
        # Convert class name to CLI-friendly name
        # e.g., DaCapoModelConfig -> dacapo, CellMapModelConfig -> cellmap-model
        name = subclass.__name__
        cli_name = name.replace(base_class.__name__, '').lower()
        # Handle camelCase to kebab-case
        cli_name = re.sub('([a-z0-9])([A-Z])', r'\1-\2', cli_name).lower()
        subclasses[cli_name] = subclass
    
    return subclasses


def get_all_model_configs():
    """
    Discover all ModelConfig subclasses using __subclasses__().
    
    Returns:
        Dictionary mapping CLI-friendly names to ModelConfig classes
    """
    from cellmap_flow.models.models_config import ModelConfig
    return get_all_subclasses(ModelConfig)


def print_available_models(cli_command_name: str = "cellmap_flow"):
    """
    Print a formatted list of all available model configurations.
    
    Args:
        cli_command_name: Name of the CLI command for help text
    """
    model_configs = get_all_model_configs()
    
    click.echo("Available model configurations:\n")
    for cli_name, config_class in sorted(model_configs.items()):
        click.echo(f"  {cli_name:20s} - {config_class.__name__}")
        
        # Show parameters
        sig = inspect.signature(config_class.__init__)
        params = [p for p in sig.parameters.keys() if p != 'self']
        if params:
            click.echo(f"                       Parameters: {', '.join(params)}")
    
    click.echo(f"\nUse '{cli_command_name} <model-name> --help' for detailed parameter information.")


def parse_type_annotation(annotation) -> Tuple[type, bool]:
    """
    Parse type annotations to determine the base type and if it's optional.
    
    Args:
        annotation: The type annotation to parse
        
    Returns:
        Tuple of (base_type, is_optional)
        
    Example:
        >>> parse_type_annotation(str)
        (str, False)
        >>> parse_type_annotation(Optional[int])
        (int, True)
        >>> parse_type_annotation(list[str])
        (str, False)
    """
    # Handle string annotations
    if isinstance(annotation, str):
        if annotation == "str":
            return str, False
        elif annotation == "int":
            return int, False
        elif annotation == "float":
            return float, False
        elif annotation == "bool":
            return bool, False
        return str, False  # Default to string
    
    # Handle typing annotations
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    
    # Check for Optional/Union with None
    if origin is type(None) or (hasattr(annotation, '__name__') and annotation.__name__ == 'NoneType'):
        return str, True
    
    # Union types (including Optional)
    if origin is type(None.__class__.__bases__[0]):  # Union
        if type(None) in args:
            # It's Optional
            non_none_types = [t for t in args if t is not type(None)]
            if non_none_types:
                base_type, _ = parse_type_annotation(non_none_types[0])
                return base_type, True
        return str, False
    
    # List types
    if origin is list:
        if args:
            base_type, _ = parse_type_annotation(args[0])
            return base_type, False  # Return the element type, we'll handle as comma-separated
        return str, False
    
    # Tuple types
    if origin is tuple:
        return str, False  # Will parse as comma-separated values
    
    # Basic types
    if annotation in (str, int, float, bool):
        return annotation, False
    
    # Default to string
    return str, False


def parse_comma_separated_values(value: str, target_type: type) -> Any:
    """
    Parse comma-separated string into list of target type.
    
    Args:
        value: Comma-separated string (e.g., "1,2,3" or "a,b,c")
        target_type: Type to convert elements to (str, int, float)
        
    Returns:
        List of parsed values
        
    Example:
        >>> parse_comma_separated_values("1,2,3", int)
        [1, 2, 3]
        >>> parse_comma_separated_values("a,b,c", str)
        ['a', 'b', 'c']
    """
    if value is None:
        return None
    if ',' in value:
        values = [v.strip() for v in value.split(',')]
        if target_type == int:
            return [int(v) for v in values]
        elif target_type == float:
            return [float(v) for v in values]
        return values
    # Single value - still return as list if expected
    if target_type == int:
        return [int(value)]
    elif target_type == float:
        return [float(value)]
    return [value]


def create_click_option_from_param(param_name: str, param_info: inspect.Parameter) -> Dict[str, Any]:
    """
    Create a click.option configuration from a parameter.
    
    Args:
        param_name: Name of the parameter
        param_info: Parameter information from inspect.signature
        
    Returns:
        Dictionary with option configuration for click.option, or None if should skip
        
    Example:
        >>> sig = inspect.signature(MyClass.__init__)
        >>> param = sig.parameters['my_param']
        >>> config = create_click_option_from_param('my_param', param)
        >>> # Returns: {'param_decls': ['-m', '--my-param'], 'required': True, ...}
    """
    annotation = param_info.annotation
    default = param_info.default
    
    # Skip 'self'
    if param_name == 'self':
        return None
    
    # Parse the type
    base_type, is_optional = parse_type_annotation(annotation)
    
    # Determine if required
    is_required = default is inspect.Parameter.empty and param_name not in ['name', 'scale']
    
    # Create option name (convert underscore to hyphen)
    option_name = param_name.replace('_', '-')
    short_name = '-' + param_name[0] if len(param_name) > 0 else None
    long_name = '--' + option_name
    
    # Build the option config
    option_config = {
        'param_decls': [short_name, long_name] if short_name else [long_name],
        'required': is_required,
        'type': base_type if base_type in (str, int, float, bool) else str,
        'help': f"Parameter: {param_name}",
    }
    
    # Handle default values
    if default is not inspect.Parameter.empty and default is not None:
        option_config['default'] = default
        option_config['required'] = False
        option_config['help'] += f" (default: {default})"
    elif not is_required:
        option_config['default'] = None
        option_config['help'] += " (optional)"
    
    # Handle list/tuple types - convert to comma-separated strings
    if annotation != inspect.Parameter.empty:
        origin = getattr(annotation, '__origin__', None)
        if origin in (list, tuple):
            option_config['help'] += " [comma-separated values]"
    
    return option_config


def process_constructor_args(config_class: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process kwargs to match the constructor signature.
    Handle list/tuple type conversions from comma-separated strings.
    
    Args:
        config_class: The class whose constructor to match
        kwargs: Dictionary of keyword arguments
        
    Returns:
        Processed kwargs with proper type conversions
        
    Example:
        >>> class MyConfig:
        ...     def __init__(self, items: list[int], name: str):
        ...         pass
        >>> kwargs = {'items': '1,2,3', 'name': 'test'}
        >>> result = process_constructor_args(MyConfig, kwargs)
        >>> # Returns: {'items': [1, 2, 3], 'name': 'test'}
    """
    sig = inspect.signature(config_class.__init__)
    processed_kwargs = {}
    
    try:
        type_hints = get_type_hints(config_class.__init__)
    except:
        type_hints = {}
    
    for param_name, param_info in sig.parameters.items():
        if param_name == 'self':
            continue
        
        value = kwargs.get(param_name)
        if value is None:
            continue
        
        # Check if this should be a list/tuple
        annotation = type_hints.get(param_name, param_info.annotation)
        origin = getattr(annotation, '__origin__', None)
        
        if origin in (list, tuple) and isinstance(value, str):
            # Get the element type
            args = getattr(annotation, '__args__', ())
            element_type = args[0] if args else str
            base_element_type, _ = parse_type_annotation(element_type)
            parsed_values = parse_comma_separated_values(value, base_element_type)
            # Convert to tuple if that's what's expected
            processed_kwargs[param_name] = tuple(parsed_values) if origin == tuple else parsed_values
        elif annotation == tuple and isinstance(value, str):
            # Handle plain tuple annotation without type args (e.g., tuple instead of tuple[int, ...])
            # Try to parse as int first, fall back to str
            try:
                parsed_values = parse_comma_separated_values(value, int)
            except (ValueError, TypeError):
                parsed_values = parse_comma_separated_values(value, str)
            processed_kwargs[param_name] = tuple(parsed_values)
        else:
            processed_kwargs[param_name] = value
    
    return processed_kwargs
