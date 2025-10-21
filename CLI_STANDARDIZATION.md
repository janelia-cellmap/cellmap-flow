# CLI Standardization Summary

## Overview
This document summarizes the CLI argument standardization changes made to CellMap Flow.

## Key Changes

### 1. Argument Naming Convention
All multi-word arguments now use **hyphens** instead of underscores:
- ✓ `--data-path` (not `--data_path`)
- ✓ `--script-path` (not `--script_path`)
- ✓ `--model-path` (not `--model_path`)
- ✓ `--config-folder` (not `--config_folder`)

### 2. Terminology Standardization
- `--charge_group` → `--project` (with `-P` short form retained)
  - More intuitive naming
  - Consistent across all CLIs

### 3. Help Text Improvements
All help text has been standardized with:
- Proper punctuation (periods at end)
- Consistent formatting
- Clear, descriptive language

## Files Modified

### CLI Files
1. **`cellmap_flow/cli/cli.py`**
   - Updated all argument names to use hyphens
   - Changed `--charge_group` to `--project`
   - Improved help text for all commands
   - Fixed function parameter names

2. **`cellmap_flow/cli/server_cli.py`**
   - Updated all argument names to use hyphens
   - Improved help text consistency
   - Standardized across all subcommands

3. **`cellmap_flow/cli/fly_model.py`**
   - Added comprehensive module docstring
   - Documented YAML configuration format
   - Added usage examples

### Documentation Files
1. **`docs/source/cli_command.rst`**
   - Updated all examples to use new argument names
   - Added section for `cellmap_flow_server`
   - Added section for `cellmap_flow_fly`
   - Expanded usage examples
   - Added migration notes
   - Added usage tips

2. **`docs/source/cli_reference.rst`** (NEW)
   - Comprehensive argument reference
   - Organized by argument category
   - Includes all short forms, types, defaults
   - Usage examples for each argument
   - Migration guide with old vs new names
   - Use case examples

3. **`docs/source/index.rst`**
   - Added `cli_reference` to table of contents

## Argument Quick Reference

### Common Arguments
| Short | Long | Old Name | Description |
|-------|------|----------|-------------|
| `-d` | `--data-path` | `--data_path` | Path to dataset |
| `-P` | `--project` | `--charge_group` | Project/billing group |
| `-q` | `--queue` | (same) | Compute queue |
| `-n` | `--name` | (same) | Model name |

### Model-Specific
| Short | Long | Old Name | Used For |
|-------|------|----------|----------|
| `-s` | `--script-path` | `--script_path` | Script models |
| `-m` | `--model-path` | `--model_path` | Bioimage models |
| `-f` | `--config-folder` | `--config_folder` | CellMap models |
| `-r` | `--run-name` | (same) | DaCapo models |
| `-i` | `--iteration` | (same) | DaCapo models |
| `-c` | `--checkpoint` | (same) | Fly models |

### Advanced
| Short | Long | Old Name | Description |
|-------|------|----------|-------------|
| `-e` | `--edge-length-to-process` | `--edge_length_to_process` | Chunk size for 2D |
| `-ivs` | `--input-voxel-size` | `--input_voxel_size` | Fly input size |
| `-ovs` | `--output-voxel-size` | `--output_voxel_size` | Fly output size |

## Migration for Users

### Shell Scripts
Replace old arguments with new ones:
```bash
# Before
cellmap_flow script -s script.py -d /data/volume.zarr -P cellmap

# After (no change needed - hyphens work better but underscores may still work in Python's argparse)
cellmap_flow script -s script.py -d /data/volume.zarr -P cellmap

# But prefer the new consistent naming
cellmap_flow script --script-path script.py --data-path /data/volume.zarr --project cellmap
```

### YAML Files
YAML keys remain flexible but prefer consistency:
```yaml
# Recommended
data_path: "/data/volume.zarr"  # or data-path
project: "cellmap"               # or charge_group
queue: "gpu_h100"
```

## Benefits

1. **Consistency**: All CLIs use the same naming conventions
2. **Clarity**: Arguments are self-documenting
3. **Standards**: Follows common CLI best practices
4. **Maintainability**: Easier to understand and maintain codebase
5. **Documentation**: Comprehensive reference available

## Testing Recommendations

Test the following scenarios:
1. Single model runs with each model type
2. Multiple model runs via `cellmap_flow_multiple`
3. YAML-based configurations
4. Server commands with new arguments
5. Migration from old to new argument names

## Next Steps

1. Update any example scripts in the repository
2. Update CI/CD pipelines if they use CLI commands
3. Notify users of changes via release notes
4. Consider adding deprecation warnings for old argument names (future work)
