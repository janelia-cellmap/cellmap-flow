# CellMap Flow CLI v2 - Dynamic Command Generation

The new CLI v2 automatically discovers all `ModelConfig` subclasses and generates CLI commands based on their constructor parameters. This eliminates the need to manually write CLI commands for each model type.

## Features

- **Auto-discovery**: Automatically finds all `ModelConfig` subclasses
- **Dynamic generation**: Creates CLI commands from class constructors
- **Type-aware**: Handles different parameter types (str, int, list, tuple, optional)
- **Self-documenting**: Auto-generates help text from parameters
- **Extensible**: Add new model configs without touching CLI code

## Installation

After updating the code, reinstall the package:

```bash
pip install -e .
```

## Basic Usage

### List Available Models

```bash
cellmap_flow_v2 list-models
```

This shows all available model configurations and their parameters.

### Get Help for a Specific Model

```bash
cellmap_flow_v2 dacapo --help
cellmap_flow_v2 script --help
cellmap_flow_v2 cellmap-model --help
```

### Run Inference

Each model type has its own command with auto-generated options:

#### DaCapo Model
```bash
cellmap_flow_v2 dacapo \
  -r my_run_name \
  -i 100 \
  -n "My Model" \
  -d /path/to/data \
  -q gpu_h100 \
  -P my_project
```

#### Script Model
```bash
cellmap_flow_v2 script \
  -s /path/to/script.py \
  -n "My Script Model" \
  -d /path/to/data
```

#### CellMap Model
```bash
cellmap_flow_v2 cellmap-model \
  -f /path/to/model/folder \
  -n mymodel \
  -d /path/to/data
```

#### Fly Model
```bash
cellmap_flow_v2 fly \
  -c /path/to/checkpoint.ts \
  --channels mito,er,nucleus \
  --input-voxel-size 4,4,4 \
  --output-voxel-size 4,4,4 \
  -d /path/to/data
```

#### BioImage Model
```bash
cellmap_flow_v2 bio \
  -m "model_name_or_path" \
  --voxel-size 4,4,4 \
  --edge-length-to-process 512 \
  -d /path/to/data
```

### Generic Run Command

For maximum flexibility, use the generic `run` command:

```bash
cellmap_flow_v2 run \
  -m dacapo \
  -c run_name=my_run \
  -c iteration=100 \
  -c name="My Model" \
  -d /path/to/data
```

### Server Check Mode

Test the server without running full inference:

```bash
cellmap_flow_v2 dacapo \
  -r my_run \
  -i 100 \
  -d /path/to/data \
  --server-check
```

## How It Works

### 1. Auto-Discovery

The CLI scans `cellmap_flow.utils.data` for all classes that inherit from `ModelConfig`:

```python
from cellmap_flow.utils.data import ModelConfig

class MyNewModelConfig(ModelConfig):
    def __init__(self, model_path: str, threshold: float = 0.5, name: str = None):
        super().__init__()
        self.model_path = model_path
        self.threshold = threshold
        self.name = name
    
    @property
    def command(self):
        return f"mynewmodel -m {self.model_path} -t {self.threshold}"
```

### 2. Command Generation

The CLI automatically creates a command based on the constructor:

- Class name `MyNewModelConfig` → CLI command `mynewmodel`
- Parameter `model_path: str` → Required option `-m/--model-path`
- Parameter `threshold: float = 0.5` → Optional option `-t/--threshold` with default
- Parameter `name: str = None` → Optional option `-n/--name`

### 3. Type Handling

The CLI intelligently handles different types:

- **Basic types**: `str`, `int`, `float`, `bool`
- **Lists/Tuples**: Comma-separated values (e.g., `--channels mito,er,nucleus`)
- **Optional types**: Parameters with default values or `Optional[]` annotation
- **Custom types**: Converted to strings

### 4. Common Options

All commands automatically get these common options:

- `-d/--data-path`: Path to the dataset (required)
- `-q/--queue`: Queue for job submission (default: gpu_h100)
- `-P/--project`: Project/chargeback group for billing
- `--server-check`: Run server check instead of full inference

## Adding New Model Types

To add a new model type, simply create a new `ModelConfig` subclass in `cellmap_flow/utils/data.py`:

```python
class NewModelConfig(ModelConfig):
    def __init__(self, 
                 required_param: str,
                 optional_param: int = 10,
                 name: str = None,
                 scale: str = None):
        super().__init__()
        self.required_param = required_param
        self.optional_param = optional_param
        self.name = name
        self.scale = scale
    
    @property
    def command(self):
        return f"newmodel -r {self.required_param} -o {self.optional_param}"
    
    def _get_config(self):
        # Implement your config logic
        pass
```

The CLI will automatically:
1. Discover the new class
2. Create a `newmodel` command
3. Generate options: `-r/--required-param`, `-o/--optional-param`, `-n/--name`, `-s/--scale`
4. Handle type conversions and validation

No CLI code changes needed!

## Comparison with Original CLI

### Original CLI (cli.py)
- Manual command definitions for each model type
- Hardcoded parameters
- Need to update CLI when adding new models
- Duplicate code for similar patterns

### New CLI v2 (cli_v2.py)
- Automatic command generation
- Parameters inferred from model constructors
- Add new models by just creating the class
- DRY principle - single source of truth

## Advanced Features

### List/Tuple Parameters

Parameters typed as `list[str]` or `tuple[int, int, int]` are handled as comma-separated values:

```bash
cellmap_flow_v2 fly --channels mito,er,nucleus
# Converts to: channels=['mito', 'er', 'nucleus']

cellmap_flow_v2 fly --input-voxel-size 4,4,4
# Converts to: input_voxel_size=[4, 4, 4]
```

### Type Annotations

The CLI uses Python type hints to determine parameter types:

```python
def __init__(self, 
             path: str,              # → str type
             count: int,             # → int type
             threshold: float,       # → float type
             enabled: bool,          # → bool type
             items: list[str],       # → comma-separated strings
             coords: tuple[int, int, int]):  # → comma-separated ints
```

### Optional Parameters

Parameters with defaults or `Optional[]` types are automatically optional:

```python
from typing import Optional

def __init__(self,
             required: str,                    # Required
             optional1: str = "default",       # Optional with default
             optional2: Optional[str] = None): # Optional, nullable
```

## Logging

Set the log level with `--log-level`:

```bash
cellmap_flow_v2 --log-level DEBUG dacapo -r my_run -i 100 -d /data
```

Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Migration Guide

To migrate from the old CLI to CLI v2:

### Old:
```bash
cellmap_flow dacapo -r my_run -i 100 -d /path/to/data -q gpu_h100 -P my_project
```

### New:
```bash
cellmap_flow_v2 dacapo -r my_run -i 100 -d /path/to/data -q gpu_h100 -P my_project
```

The syntax is the same! The difference is that v2 is auto-generated.

## Troubleshooting

### Command not found
```bash
pip install -e .
```

### "Unknown model type" error
Run `cellmap_flow_v2 list-models` to see available models.

### Type conversion errors
Check that comma-separated values match the expected type (e.g., `4,4,4` for int list).

### Missing required parameter
Run `cellmap_flow_v2 <command> --help` to see all required parameters.

## Future Enhancements

Possible future improvements:

- [ ] Add validation decorators to model configs
- [ ] Support for nested configurations
- [ ] Auto-complete support
- [ ] Configuration file support (YAML/JSON)
- [ ] Better error messages with suggestions
- [ ] Support for Union types beyond Optional
- [ ] Integration tests for all generated commands

## Contributing

When adding new model types:

1. Create the `ModelConfig` subclass in `cellmap_flow/utils/data.py`
2. Use type hints for all parameters
3. Implement the `command` property
4. Implement the `_get_config()` method
5. Test with `cellmap_flow_v2 list-models`
6. Run your command with `--help` to verify options

That's it! The CLI handles the rest automatically.
