YAML Configuration
===================

``cellmap_flow_yaml`` lets you define and run multiple models from a single YAML file.
It is the recommended way to launch inference jobs, and the same YAML format is used by the blockwise processor (``cellmap_flow_blockwise``).

Usage
-----

.. code-block:: bash

    # Run inference
    cellmap_flow_yaml config.yaml

    # Validate without running
    cellmap_flow_yaml config.yaml --validate-only

    # List available model types
    cellmap_flow_yaml --list-types

    # Set log level
    cellmap_flow_yaml config.yaml --log-level DEBUG

YAML Structure
--------------

A configuration file has the following top-level fields:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Field
     - Required
     - Description
   * - ``data_path``
     - Yes
     - Path to the input dataset (zarr/n5).
   * - ``charge_group``
     - Yes
     - Project billing group.
   * - ``queue``
     - No
     - Job queue (default: ``gpu_h100``).
   * - ``models``
     - Yes
     - Dict or list of model entries (see below).
   * - ``json_data``
     - No
     - Input normalizers and postprocessors.
   * - ``wrap_raw``
     - No
     - Wrap raw data in neuroglancer (default: ``true``).
   * - ``output_path``
     - No
     - Output zarr path (used by blockwise processing).
   * - ``task_name``
     - No
     - Task name (used by blockwise processing).
   * - ``workers``
     - No
     - Number of GPU workers (blockwise).
   * - ``cpu_workers``
     - No
     - Number of CPU workers (blockwise).
   * - ``tmp_dir``
     - No
     - Temporary directory for intermediate files.
   * - ``bounding_boxes``
     - No
     - List of bounding boxes to process (blockwise).
   * - ``separate_bounding_boxes_zarrs``
     - No
     - Write each bounding box to a separate zarr (blockwise).

Model Entries
-------------

Each model entry requires a ``type`` field and the parameters for that model type.
Use ``cellmap_flow_yaml --list-types`` to see all available types and their required parameters.

Models can be specified as a **dict** (keys become model names) or a **list** (each entry must include a ``name`` field).

**Dict format** (recommended):

.. code-block:: yaml

    models:
      my_mito_model:
        type: fly
        checkpoint: /path/to/checkpoint
        resolution: 16
        classes:
          - mito
      my_dacapo_model:
        type: dacapo
        run_name: my_run
        iteration: 100

**List format**:

.. code-block:: yaml

    models:
      - name: my_mito_model
        type: fly
        checkpoint: /path/to/checkpoint
        resolution: 16
        classes:
          - mito

Available Model Types
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Type
     - Class
     - Key Parameters
   * - ``script``
     - ScriptModelConfig
     - ``script_path`` (required)
   * - ``dacapo``
     - DaCapoModelConfig
     - ``run_name`` (required), ``iteration`` (required)
   * - ``fly``
     - FlyModelConfig
     - ``checkpoint`` (required), ``classes`` (required), ``resolution`` (required)
   * - ``bio``
     - BioModelConfig
     - ``model_path`` (required)
   * - ``cellmap``
     - CellMapModelConfig
     - ``config_folder`` (required)
   * - ``huggingface``
     - HuggingFaceModelConfig
     - ``repo`` (required), ``revision`` (optional). See :doc:`huggingface`.

Common optional parameters: ``name``, ``scale``.

Normalizers and Postprocessors
------------------------------

Define input normalization and output postprocessing under ``json_data``:

.. code-block:: yaml

    json_data:
      input_norm:
        MinMaxNormalizer:
          min_value: 0
          max_value: 250
          invert: false
        LambdaNormalizer:
          expression: "x*2-1"
      postprocess:
        DefaultPostprocessor:
          clip_min: 0
          clip_max: 1.0
          bias: 0.0
          multiplier: 127.5
        ThresholdPostprocessor:
          threshold: 0.5

Normalizers are applied in order before inference. Postprocessors are applied in order after inference.

Bounding Boxes
--------------

For blockwise processing, you can specify regions of interest:

.. code-block:: yaml

    bounding_boxes:
      - offset: [59611, 52237, 5627]
        shape: [4674, 11566, 10067]
      - offset: [64285, 26408, 15695]
        shape: [11626, 12405, 26847]

Set ``separate_bounding_boxes_zarrs: true`` to write each bounding box to its own zarr subdirectory (``box_1``, ``box_2``, etc).

Examples
--------

Minimal configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    data_path: /nrs/cellmap/data/my_dataset/my_dataset.zarr/recon-1/em/fibsem-uint8
    charge_group: cellmap
    queue: gpu_h100

    models:
      my_model:
        type: dacapo
        run_name: my_run
        iteration: 50000

Full configuration with normalizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    data_path: /nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8
    queue: gpu_h100
    charge_group: cellmap

    json_data:
      input_norm:
        MinMaxNormalizer:
          min_value: 0
          max_value: 250
          invert: false
        LambdaNormalizer:
          expression: "x*2-1"
      postprocess:
        DefaultPostprocessor:
          clip_min: 0
          clip_max: 1.0
          bias: 0.0
          multiplier: 127.5
        ThresholdPostprocessor:
          threshold: 127.5

    models:
      model_tmp1:
        type: fly
        checkpoint: /path/to/model_checkpoint_362000
        resolution: 16
        classes:
          - mito

Blockwise processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    data_path: /nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8
    output_path: /path/to/output.zarr
    task_name: cellmap_flow_mito_task
    charge_group: cellmap
    queue: gpu_h100
    workers: 14
    cpu_workers: 12
    tmp_dir: /path/to/tmp

    models:
      - name: model_tmp1
        type: fly
        channels:
          - mito
        checkpoint_path: /path/to/model_checkpoint_362000
        input_size: [178, 178, 178]
        input_voxel_size: [16, 16, 16]
        output_size: [56, 56, 56]
        output_voxel_size: [16, 16, 16]

    bounding_boxes:
      - offset: [59611, 52237, 5627]
        shape: [4674, 11566, 10067]
      - offset: [64285, 26408, 15695]
        shape: [11626, 12405, 26847]

    json_data:
      input_norm:
        MinMaxNormalizer:
          invert: false
          max_value: 250
          min_value: 0
        LambdaNormalizer:
          expression: "x*2-1"
      postprocess:
        ThresholdPostprocessor:
          threshold: 0.5

Run blockwise processing with:

.. code-block:: bash

    cellmap_flow_blockwise config.yaml
    cellmap_flow_blockwise config.yaml --log-level DEBUG
