"""
Utility functions for CLI generation and management.
"""

import re
import inspect
import click
from typing import Type, Any, Dict

from cellmap_flow.models import registry

# Moved to the model registry; importable from here as before.
from cellmap_flow.models.registry import (  # noqa: F401
    _parse_type_annotation as parse_type_annotation,
    _parse_comma_separated_values as parse_comma_separated_values,
)


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
        # Allow classes to define their own CLI name via a class attribute
        if hasattr(subclass, 'cli_name') and subclass.cli_name:
            cli_name = subclass.cli_name
        else:
            # Convert class name to CLI-friendly name
            # e.g., DaCapoModelConfig -> dacapo, ScriptModelConfig -> script
            name = subclass.__name__
            cli_name = name.replace(base_class.__name__, '').lower()
            # Handle camelCase to kebab-case
            cli_name = re.sub('([a-z0-9])([A-Z])', r'\1-\2', cli_name).lower()
        subclasses[cli_name] = subclass
    
    return subclasses


def get_all_model_configs():
    """Every model type by CLI name, plugins included: ``registry.model_types()``."""
    return registry.model_types()


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


def create_click_option_from_param(param_name: str, param_info: inspect.Parameter, used_short_names: set = None) -> Dict[str, Any]:
    """The ``click.option`` arguments for one constructor argument, or None for self.

    See ``registry.click_options``, which builds a whole command's options.
    """
    return registry._click_option(param_name, param_info, used_short_names)


def process_constructor_args(config_class: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Command-line ``kwargs`` as constructor arguments: ``registry.coerce_cli_args``."""
    return registry.coerce_cli_args(config_class, kwargs)
