Python Scripts
================================
CellMap Flow can be run from the command line or from a Python script. A script does what ``cellmap_flow yaml`` does with a YAML file (see :doc:`yaml_config`): it builds the model configs, submits one inference server per model, and then serves the dashboard.

First define your model configs: ``ScriptModelConfig``, ``DaCapoModelConfig``, ``BioModelConfig`` or ``FlyModelConfig``. Then pass them to ``run_multiple``.

Prerequisites
-------------

You need a valid model checkpoint and access to GPUs through LSF (e.g. the ``gpu_h100`` queue).

Script
------

.. code-block:: python

    from cellmap_flow.cli.yaml_cli import run_multiple
    from cellmap_flow.jobs.launch import install_cleanup_handlers
    from cellmap_flow.logging_setup import configure_logging
    from cellmap_flow.models.models_config import FlyModelConfig
    from cellmap_flow.pipeline_spec import PipelineSpec
    from cellmap_flow.process_chain import process_chain

    # The log lines cellmap_flow's commands print: each server as it starts,
    # and the dashboard's address.
    configure_logging()

    DATA_PATH = "/path/to/dataset.zarr/recon-1/em/fibsem-uint8/s1"

    model_config = FlyModelConfig(
        checkpoint_path="/path/to/fly_organelles/run07/model_checkpoint_432000",
        channels=["mito", "er", "nucleus"],
        input_voxel_size=(8, 8, 8),
        output_voxel_size=(8, 8, 8),
        name="fly_organelles",
    )

    # The normalization every layer is served with, as json_data.input_norm
    # would set it in a YAML file.
    process_chain().set(PipelineSpec(input_norm=[
        {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
        {"name": "LambdaNormalizer", "expression": "x*2-1"},
    ]))

    # Ctrl+C then kills the LSF jobs this script started.
    install_cleanup_handlers()

    run_multiple(
        [model_config],
        DATA_PATH,
        charge_group="CHARGE_GROUP",
        queue="gpu_h100",
    )

Explanation
-----------

- **configure_logging()**: the timestamped INFO lines. The ``cellmap_flow`` commands set up logging themselves; a script that skips this prints only warnings and errors.
- **FlyModelConfig**: a fly_organelles checkpoint; its channels, voxel sizes and tile are read from the
  checkpoint's folder when not given (see :ref:`fly`).
- **process_chain().set(PipelineSpec(...))**: the input normalization (``input_norm``) and postprocessing (``postprocess``) chains. Each step is ``{"name": <class name>, **its arguments}``, as in a YAML's ``json_data``; ``set()`` builds them, skipping a name it does not know with a warning. Left unset, the model sees the raw values.
- **queue**: the LSF queue the inference servers are submitted to.
- **charge_group**: the accounting group the GPU time is billed to.
- **run_multiple**: starts one inference server per model and waits for each to report its address. Then it serves the dashboard with a prediction layer per model. It does not return. Stop it with Ctrl+C: ``install_cleanup_handlers()`` makes that kill the jobs it started, which ``cellmap_flow yaml`` does for you.

A model type, normalizer or postprocessor from a plugin (:doc:`plugins`) exists only once the plugins are loaded. The ``cellmap_flow`` commands load them when they start; a script that uses one calls ``cellmap_flow.plugins.load_plugins()`` first.

If no model server starts, ``run_multiple`` raises ``JobStartError`` naming each failure.
