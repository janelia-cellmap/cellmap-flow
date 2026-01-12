# Comprehensive Unit Tests for cellmap-flow

This document outlines comprehensive unit tests for the cellmap-flow package using pytest. The tests cover all major components and functionality.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                     # Pytest configuration and fixtures
├── test_globals.py                 # Test global state management
├── test_inferencer.py              # Test inference engine
├── test_image_data_interface.py    # Test data loading interface
├── test_server.py                  # Test Flask server
├── test_dashboard_app.py           # Test dashboard application
├── test_norm/
│   ├── __init__.py
│   └── test_input_normalize.py     # Test normalization functions
├── test_post/
│   ├── __init__.py
│   └── test_postprocessors.py      # Test postprocessing functions
├── test_models/
│   ├── __init__.py
│   ├── test_model_configs.py       # Test model configurations
│   └── test_model_yaml.py          # Test YAML model loading
├── test_utils/
│   ├── __init__.py
│   ├── test_data.py                # Test data utilities
│   ├── test_ds.py                  # Test dataset utilities
│   ├── test_config_utils.py        # Test configuration utilities
│   ├── test_bsub_utils.py          # Test job submission utilities
│   └── test_web_utils.py           # Test web utilities
├── test_cli/
│   ├── __init__.py
│   ├── test_cli.py                 # Test main CLI
│   ├── test_multiple_cli.py        # Test multiple model CLI
│   └── test_server_cli.py          # Test server CLI
└── test_blockwise/
    ├── __init__.py
    └── test_blockwise_processor.py # Test blockwise processing
```

## Key Testing Areas

1. **Model Management**: DaCapo, BioImage.io, Script, and CellMap models
2. **Data Processing**: Normalization, postprocessing, zarr/n5 handling
3. **Inference Pipeline**: Real-time prediction, GPU optimization
4. **Web Interface**: Flask server, dashboard, neuroglancer integration
5. **CLI Tools**: Command-line interfaces for various workflows
6. **Utilities**: Configuration, serialization, job submission

## Coverage Goals

- **Unit Tests**: Individual functions and classes
- **Integration Tests**: Component interactions
- **End-to-End Tests**: Complete workflows
- **Mock Tests**: External dependencies (GPUs, file systems)
- **Error Handling**: Edge cases and failures
