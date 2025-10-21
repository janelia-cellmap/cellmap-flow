CLI Commands
=============================

CellMap Flow provides several CLI interfaces to run different model configurations and modes of execution. This section outlines the supported entrypoints, command syntax, and examples.

Available Entry points
----------------------

The following CLI entrypoints are available:

- ``cellmap_flow`` — the main interface for executing a single model using a script, Dacapo run, or Bioimage model.
- ``cellmap_flow_server`` — runs inference servers for interactive prediction (used internally by cellmap_flow).
- ``cellmap_flow_multiple`` — supports executing **multiple models** in one command, all on the same dataset.
- ``cellmap_flow_yaml`` — supports loading model configuration from a structured **YAML** file.
- ``cellmap_flow_fly`` — specialized interface for running Fly models via YAML configuration.

.. note::
   All CLI tools use standardized argument naming: ``--data-path`` (not ``--data_path``), ``--project`` (for billing/chargeback groups), and ``--queue`` for compute resources.

Basic Syntax
------------

Main CLI Entrypoint (``cellmap_flow``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cellmap_flow <subcommand> [OPTIONS]

Supported subcommands:

- ``dacapo`` — run using a Dacapo training run.
- ``script`` — run using a Python script model definition.
- ``bioimage`` — run using a Bioimage.IO model.
- ``cellmap-model`` — run using a structured CellMap model config folder.
- ``script-server-check`` — validate a script-based model configuration.

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

    # Validate a script configuration
    cellmap_flow script-server-check -s /path/to/script.py -d /path/to/data

Chaining Multiple Models (``cellmap_flow_multiple``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This CLI allows running **multiple models sequentially** on the same dataset. Each subcommand must be prefixed with ``--``.

Example:

.. code-block:: bash

    cellmap_flow_multiple \
        --data-path /nrs/path/zarr/volume.zarr \
        --project cellmap \
        --queue gpu_h100 \
        --dacapo -r mito_run -i 40000 \
        --script -s /path/to/script.py \
        --cellmap-model -f /path/to/model_dir -n mito_model

YAML-Based Configuration (``cellmap_flow_yaml``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run multiple models defined in a **YAML file** with structured metadata. This allows batch processing and version-controlled configuration.

YAML Example:

.. code-block:: yaml

   data_path: "/nrs/cellmap/data/jrc_mus-salivary-1/.../fibsem-uint8"
   charge_group: "cellmap"  # or use 'project'
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

Fly Model YAML Configuration (``cellmap_flow_fly``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialized interface for Fly models with custom input/output sizes:

YAML Example:

.. code-block:: yaml

   input: "/path/to/input.zarr"
   queue: "gpu_h100"
   charge_group: "cellmap"
   input_size: [178, 178, 178]  # Optional
   output_size: [56, 56, 56]    # Optional
   json_data: "/path/to/norm.json"  # Optional preprocessing config
   
   runs:
     model_name:
       checkpoint: "/path/to/checkpoint.pth"
       classes: ["channel1", "channel2"]
       res: 8  # Resolution in nm
       scale: "s1"  # Optional

To execute:

.. code-block:: bash

    cellmap_flow_fly /path/to/fly_config.yaml

Inference Server CLI (``cellmap_flow_server``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``cellmap_flow_server`` CLI runs inference servers directly. This is typically used internally by the main CLI but can be invoked manually for debugging or custom setups.

Supported subcommands:

- ``dacapo`` — Run DaCapo model server
- ``script`` — Run custom script model server
- ``bioimage`` — Run Bioimage.IO model server
- ``cellmap-model`` — Run CellMap model server
- ``fly`` — Run Fly model server
- ``run-ui-server`` — Run the dashboard UI server

Examples:

.. code-block:: bash

    # Run a script model server
    cellmap_flow_server script -s /path/to/script.py -d /path/to/data -p 8080

    # Run a Fly model server
    cellmap_flow_server fly \
        -c /path/to/checkpoint.pth \
        -ch channel1,channel2 \
        -ivs 8,8,8 \
        -ovs 8,8,8 \
        -d /path/to/data

    # Run the UI server
    cellmap_flow_server run-ui-server \
        -n http://neuroglancer-url \
        -i http://inference-host

CLI Arguments Reference
-----------------------

Common Arguments
^^^^^^^^^^^^^^^^

The following arguments are standardized across all CLI interfaces:

**Data Path**
  - Flags: ``-d``, ``--data-path``
  - Type: ``str``
  - Required: Yes (for most commands)
  - Description: Path to the input dataset (typically a Zarr file/group)

**Project/Charge Group**
  - Flags: ``-P``, ``--project``
  - Type: ``str``
  - Required: Yes (for job submission commands)
  - Description: The project or chargeback group for billing purposes
  - Note: Previously called ``--charge_group``, now standardized to ``--project``

**Queue**
  - Flags: ``-q``, ``--queue``
  - Type: ``str``
  - Default: ``gpu_h100``
  - Description: The compute queue to submit jobs to

**Model Name**
  - Flags: ``-n``, ``--name``
  - Type: ``str``
  - Description: Custom name for the model (used for job naming and output identification)

**Scale**
  - Flags: ``-r``, ``--scale``
  - Type: ``str``
  - Description: Input scale level (e.g., ``s0``, ``s1``, ``s2``) for multi-resolution datasets

Model-Specific Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^

**DaCapo Models**
  - ``-r``, ``--run-name``: The name of the DaCapo run (required)
  - ``-i``, ``--iteration``: Training iteration to use (default: 0)

**Script Models**
  - ``-s``, ``--script-path``: Path to the Python script containing model specification (required)

**Bioimage.IO Models**
  - ``-m``, ``--model-path``: Path to the bioimage.io model (required)
  - ``-e``, ``--edge-length-to-process``: For 2D models, desired chunk edge size

**CellMap Models**
  - ``-f``, ``--config-folder``: Path to the model configuration folder (required)

**Fly Models**
  - ``-c``, ``--checkpoint``: Path to the model checkpoint (required)
  - ``-ch``, ``--channels``: Comma-separated list of channel names (required)
  - ``-ivs``, ``--input-voxel-size``: Comma-separated input voxel size (required)
  - ``-ovs``, ``--output-voxel-size``: Comma-separated output voxel size (required)

Server-Specific Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

These arguments are available for ``cellmap_flow_server`` commands:

- ``--debug``: Run server in debug mode (flag)
- ``-p``, ``--port``: Port number to listen on (default: 0 for auto-assign)
- ``--certfile``: Path to SSL certificate file for HTTPS
- ``--keyfile``: Path to SSL private key file for HTTPS

Migration Notes
^^^^^^^^^^^^^^^

If you have existing scripts using the old argument names, here are the changes:

- ``--data_path`` → ``--data-path``
- ``--charge_group`` → ``--project`` (``-P`` remains the same)
- ``--script_path`` → ``--script-path``
- ``--model_path`` → ``--model-path``
- ``--config_folder`` → ``--config-folder``
- ``--edge_length_to_process`` → ``--edge-length-to-process``
- ``--input_voxel_size`` → ``--input-voxel-size``
- ``--output_voxel_size`` → ``--output-voxel-size``
- ``--neuroglancer_url`` → ``--neuroglancer-url``
- ``--inference_host`` → ``--inference-host``

Usage Tips
----------

**Validate Configuration Before Submission**

Use the ``script-server-check`` command to validate your model configuration:

.. code-block:: bash

    cellmap_flow script-server-check -s /path/to/script.py -d /path/to/data

This initializes the inference server and simulates a 2×2×2 chunk to confirm validity.

**Running Multiple Models Efficiently**

For running multiple models on the same dataset, prefer ``cellmap_flow_yaml`` over chaining commands:

- ✓ Better: Use YAML configuration with ``cellmap_flow_yaml``
- ✗ Less optimal: Chain multiple ``cellmap_flow`` commands

**Choosing the Right CLI**

- Use ``cellmap_flow`` for single model runs
- Use ``cellmap_flow_multiple`` for multiple models via command line
- Use ``cellmap_flow_yaml`` for multiple models with version-controlled config
- Use ``cellmap_flow_fly`` for Fly-specific models with custom sizes
- Use ``cellmap_flow_server`` for manual server control or debugging

