CLI Commands
=============================

CellMap Flow provides several CLI interfaces to run different model configurations and modes of execution. This section outlines the supported entrypoints, command syntax, and examples.

Available Entry points
----------------------

The following CLI entrypoints are available:

- ``cellmap_flow`` — the main interface for executing a single model using a script, Dacapo run, or Bioimage model.
- ``cellmap_flow_multiple`` — supports executing **multiple models** in one command, all on the same dataset.
- ``cellmap_flow_yaml`` — supports loading model configuration from a structured **YAML** file.

.. note::
   All CLI tools support `--queue` and `--charge_group` options for specifying compute resources.

Basic Syntax
------------

Main CLI Entrypoint (`cellmap_flow`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cellmap_flow <subcommand> [OPTIONS]

Supported subcommands:

- ``dacapo`` — run using a Dacapo training run.
- ``script`` — run using a Python script model definition.
- ``bioimage`` — run using a Bioimage.IO model.
- ``cellmap-model`` — run using a structured CellMap model config folder.

Examples:

.. code-block:: bash

    # Using a Dacapo run
    cellmap_flow dacapo -r run_name -i 60 -d /path/to/data -q gpu_h100 -P cellmap

    # Using a Python script
    cellmap_flow script -s /path/to/script.py -d /path/to/data -q gpu_h100 -P cellmap

    # Using a Bioimage.IO model
    cellmap_flow bioimage -m /path/to/model.bioimage.io -d /path/to/data -q gpu_h100 -P cellmap

    # Using a CellMap model folder
    cellmap_flow cellmap-model -f /path/to/model_dir -n model_name -d /path/to/data -q gpu_h100 -P cellmap

Chaining Multiple Models (`cellmap_flow_multiple`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This CLI allows running **multiple models sequentially** on the same dataset. Each subcommand must be prefixed with `--`.

Example:

.. code-block:: bash

    cellmap_flow_multiple \
        --data-path /nrs/path/zarr/volume.zarr \
        --project cellmap \
        --queue gpu_h100 \
        --dacapo -r mito_run -i 40000 \
        --script -s /path/to/script.py \
        --cellmap-model -f /path/to/model_dir -n mito_model

YAML-Based Configuration (`cellmap_flow_yaml`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run multiple models defined in a **YAML file** with structured metadata. This allows batch processing and version-controlled configuration.

YAML Example:

.. code-block:: yaml

   data_path: "/nrs/cellmap/data/jrc_mus-salivary-1/.../fibsem-uint8"
   charge_group: "cellmap"
   queue: "gpu_h100"

   models:
   - type: cellmap-model
      config_folder: "/groups/cellmap/cellmap/zouinkhim/salivary/train/binary/mito_16"
      name: "b_mito_16"
      scale: "s1"

   - type: cellmap-model
      config_folder: "/groups/cellmap/cellmap/zouinkhim/salivary/train/binary/lyso_16"
      name: "b_lyso_16"
      scale: "s1"

   - type: dacapo
      run_name: "nuclei_run"
      iteration: 36000
      name: "nuclei_model"

   - type: script
      script_path: "/groups/cellmap/scripts/mito_script.py"
      name: "mito_script_model"
      scale: "s2"

   - type: bioimage
      model_path: "/groups/cellmap/models/mito.bioimage.io"
      name: "bio_mito"


To execute:

.. code-block:: bash

    cellmap_flow_yaml /path/to/config.yaml

Advanced Options
----------------

All CLI variants support the following flags (unless overridden):

- ``--data-path/-d``: Path to the input dataset (required).
- ``--queue/-q``: Compute queue to submit the job (default: ``gpu_h100``).
- ``--project/-P``: Charge group for billing (required).
- ``--name/-n``: Custom model name (optional).
- ``--scale/-r``: Input scale (optional for structured models).
- ``--edge_length_to_process/-e``: For 2D models, desired chunk edge size (bioimage only).

Server Check Utility
--------------------

Validate a script-based model before submitting:

.. code-block:: bash

    cellmap_flow script-server-check -s /path/to/script.py -d /path/to/data

This will initialize the inference server and simulate a 2x2x2 chunk to confirm configuration validity.

