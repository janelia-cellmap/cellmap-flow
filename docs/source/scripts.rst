Python Scripts
================================
CellMap Flow can be run from the command line or from a Python script. A script does what ``cellmap_flow_yaml`` does with a YAML file (see :doc:`yaml_config`): it builds the model configs, submits one inference server per model, and then serves the dashboard.

First define your model configs: ``ScriptModelConfig``, ``DaCapoModelConfig``, ``BioModelConfig`` or ``FlyModelConfig``. Then pass them to ``run_multiple``.

Prerequisites
-------------

You need a valid model checkpoint and access to GPUs through LSF (e.g. the ``gpu_h100`` queue).

Script
------

.. code-block:: python

    from cellmap_flow.cli.yaml_cli import run_multiple
    from cellmap_flow.globals import g
    from cellmap_flow.models.models_config import FlyModelConfig
    from cellmap_flow.norm.input_normalize import LambdaNormalizer, MinMaxNormalizer
    from cellmap_flow.jobs.launch import install_cleanup_handlers

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
    g.input_norms = [
        MinMaxNormalizer(min_value=0, max_value=255),
        LambdaNormalizer(expression="x*2-1"),
    ]

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

- **FlyModelConfig**: the model, its checkpoint, and its input and output voxel sizes.
- **g.input_norms**: the input normalization chain. Left empty, the model sees the raw values.
- **queue**: the LSF queue the inference servers are submitted to.
- **charge_group**: the accounting group the GPU time is billed to.
- **run_multiple**: starts one inference server per model and waits for each to report its address. Then it serves the dashboard with a prediction layer per model. It does not return. Stop it with Ctrl+C: ``install_cleanup_handlers()`` makes that kill the jobs it started, which ``cellmap_flow_yaml`` does for you.

If no model server starts, ``run_multiple`` raises ``JobStartError`` naming each failure.
