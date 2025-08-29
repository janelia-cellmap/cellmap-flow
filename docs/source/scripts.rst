Python Scripts
================================
CellMap Flow can be executed from cli commands or programmatically via Python scripts.

First you need to define your model config, which can be ScriptModelConfig, DacapoModelConfig, BioModelConfig or FlyModelConfig. 
Then you can call Flow.run with the appropriate parameters.

Prerequisites
-------------

You should also have access to a valid model checkpoint and appropriate computational resources (e.g., H100 GPUs).

Script
------

.. code-block:: python

    from cellmap_flow.globals import Flow
    from cellmap_flow.utils.data import FlyModelConfig
    from cellmap_flow.norm.input_normalize import MinMaxNormalizer

    queue = "gpu_h100"
    charge_group = "CHARGE_GROUP"
    model_scale = (8, 8, 8)
    checkpoint_path = "../fly_organelles/run07/model_checkpoint_432000"

    model_config = FlyModelConfig(
        checkpoint_path=checkpoint_path,
        channels=["classes"] * 8,
        input_voxel_size=model_scale,
        output_voxel_size=model_scale,
        name="FLY",
    )

    url = Flow.run(
        zarr_path=DATA_PATH,
        model_configs=[model_config],
        queue=queue,
        charge_group=charge_group,
        input_normalizers=[
            MinMaxNormalizer(min_value=0, max_value=255, invert=False)
        ],
        post_processors=[],
    )

    print(url)

Explanation
-----------

- **queue**: Specifies the job queue (e.g., `gpu_h100`) used for resource allocation.
- **charge_group**: Accounting group for compute billing.
- **FlyModelConfig**: Configures the model parameters, including voxel size and checkpoint path.
- **MinMaxNormalizer**: Scales input data from [0, 255] to [0, 1].
- **Flow.run**: Launches the inference pipeline and returns a tracking URL.

Output
------

The script prints a tracking URL to monitor the job or retrieve results.

.. note::

   Ensure `DATA_PATH` is set to the path of your input Zarr volume before running the script.

