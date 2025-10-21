CLI Argument Reference
======================

This document provides a comprehensive reference of all CLI arguments used across CellMap Flow's command-line interfaces.

.. contents:: Table of Contents
   :local:
   :depth: 2

Standardization Principles
---------------------------

All CellMap Flow CLIs follow these naming conventions:

1. **Hyphenated multi-word arguments**: Use hyphens (``--data-path``), not underscores (``--data_path``)
2. **Consistent terminology**: ``--project`` for billing groups (not ``--charge_group``)
3. **Clear, descriptive names**: Arguments should be self-documenting
4. **Short and long forms**: Commonly used arguments have single-letter shortcuts

Common Arguments
----------------

These arguments are used consistently across multiple CLI interfaces.

--data-path, -d
^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for most commands)
:Default: None
:Description: Path to the input dataset, typically a Zarr file or group

**Examples:**

.. code-block:: bash

    --data-path /nrs/cellmap/data/dataset.zarr
    -d /nrs/cellmap/data/dataset.zarr/s1

**Used in:**
  - ``cellmap_flow`` (all subcommands)
  - ``cellmap_flow_server`` (all subcommands)
  - ``cellmap_flow_multiple``
  - ``cellmap_flow_yaml`` (in YAML as ``data_path``)

--project, -P
^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for job submission)
:Default: None
:Description: Project or chargeback group for billing purposes

**Examples:**

.. code-block:: bash

    --project cellmap
    -P my_lab_group

**Used in:**
  - ``cellmap_flow`` (all subcommands)
  - ``cellmap_flow_multiple``
  - ``cellmap_flow_yaml`` (in YAML as ``charge_group`` or ``project``)

**Migration Note:** Previously ``--charge_group``. Updated to ``--project`` for consistency.

--queue, -q
^^^^^^^^^^^

:Type: ``string``
:Required: No
:Default: ``gpu_h100``
:Description: Compute queue to submit jobs to

**Examples:**

.. code-block:: bash

    --queue gpu_rtx
    -q local

**Used in:**
  - ``cellmap_flow`` (all subcommands)
  - ``cellmap_flow_multiple``
  - ``cellmap_flow_yaml`` (in YAML as ``queue``)

--name, -n
^^^^^^^^^^

:Type: ``string``
:Required: Depends on subcommand
:Default: Auto-generated from model/script path
:Description: Custom name for the model, used for job naming and output identification

**Examples:**

.. code-block:: bash

    --name mitochondria_model
    -n my_experiment_v2

**Used in:**
  - ``cellmap_flow cellmap-model``
  - ``cellmap_flow_server cellmap-model``
  - ``cellmap_flow_multiple`` (per-model in YAML)

Model Path Arguments
--------------------

--script-path, -s
^^^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for script-based models)
:Default: None
:Description: Path to Python script containing model specification

**Examples:**

.. code-block:: bash

    --script-path /groups/cellmap/scripts/my_model.py
    -s ./model_definition.py

**Used in:**
  - ``cellmap_flow script``
  - ``cellmap_flow_server script``
  - ``cellmap_flow_multiple --script``

--model-path, -m
^^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for bioimage.io models)
:Default: None
:Description: Path to bioimage.io model file or directory

**Examples:**

.. code-block:: bash

    --model-path /models/mito_segmentation.bioimage.io
    -m ./my_model.zip

**Used in:**
  - ``cellmap_flow bioimage``
  - ``cellmap_flow_server bioimage``
  - ``cellmap_flow_multiple --bioimage``

--config-folder, -f
^^^^^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for CellMap models)
:Default: None
:Description: Path to model configuration folder containing model definition files

**Examples:**

.. code-block:: bash

    --config-folder /groups/cellmap/models/mito_16
    -f ./model_configs/nuclei

**Used in:**
  - ``cellmap_flow cellmap-model``
  - ``cellmap_flow_server cellmap-model``
  - ``cellmap_flow_multiple --cellmap-model``

DaCapo-Specific Arguments
-------------------------

--run-name, -r
^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for DaCapo models)
:Default: None
:Description: Name of the DaCapo training run to use

**Examples:**

.. code-block:: bash

    --run-name mito_run_v3
    -r nuclei_training_20241015

**Used in:**
  - ``cellmap_flow dacapo``
  - ``cellmap_flow_server dacapo``
  - ``cellmap_flow_multiple --dacapo``

--iteration, -i
^^^^^^^^^^^^^^^

:Type: ``integer``
:Required: No
:Default: ``0``
:Description: Training iteration/checkpoint to use from the DaCapo run

**Examples:**

.. code-block:: bash

    --iteration 60000
    -i 40000

**Used in:**
  - ``cellmap_flow dacapo``
  - ``cellmap_flow_server dacapo``
  - ``cellmap_flow_multiple --dacapo``

Fly Model-Specific Arguments
-----------------------------

--checkpoint, -c
^^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for Fly models)
:Default: None
:Description: Path to the Fly model checkpoint file

**Examples:**

.. code-block:: bash

    --checkpoint /models/fly_checkpoint.pth
    -c ./checkpoints/epoch_100.pth

**Used in:**
  - ``cellmap_flow_server fly``
  - ``cellmap_flow_fly`` (in YAML as ``checkpoint``)

--channels, -ch
^^^^^^^^^^^^^^^

:Type: ``string`` (comma-separated)
:Required: Yes (for Fly models)
:Default: None
:Description: Comma-separated list of output channel names

**Examples:**

.. code-block:: bash

    --channels mito,nucleus,golgi
    -ch cell_membrane,er

**Used in:**
  - ``cellmap_flow_server fly``
  - ``cellmap_flow_fly`` (in YAML as ``classes``)

--input-voxel-size, -ivs
^^^^^^^^^^^^^^^^^^^^^^^^^

:Type: ``string`` (comma-separated)
:Required: Yes (for Fly models)
:Default: None
:Description: Comma-separated input voxel size in nanometers (z,y,x)

**Examples:**

.. code-block:: bash

    --input-voxel-size 8,8,8
    -ivs 4,4,4

**Used in:**
  - ``cellmap_flow_server fly``
  - ``cellmap_flow_fly`` (in YAML as ``input_voxel_size``)

--output-voxel-size, -ovs
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Type: ``string`` (comma-separated)
:Required: Yes (for Fly models)
:Default: None
:Description: Comma-separated output voxel size in nanometers (z,y,x)

**Examples:**

.. code-block:: bash

    --output-voxel-size 8,8,8
    -ovs 4,4,4

**Used in:**
  - ``cellmap_flow_server fly``
  - ``cellmap_flow_fly`` (in YAML as ``output_voxel_size``)

Bioimage-Specific Arguments
----------------------------

--edge-length-to-process, -e
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Type: ``integer``
:Required: No
:Default: Auto-calculated
:Description: For 2D models, desired edge length of chunks to process; batch size (z) will be adjusted accordingly

**Examples:**

.. code-block:: bash

    --edge-length-to-process 512
    -e 1024

**Used in:**
  - ``cellmap_flow bioimage``
  - ``cellmap_flow_server bioimage``
  - ``cellmap_flow_multiple --bioimage``

Server-Specific Arguments
-------------------------

--port, -p
^^^^^^^^^^

:Type: ``integer``
:Required: No
:Default: ``0`` (auto-assign)
:Description: Port number for the inference server to listen on

**Examples:**

.. code-block:: bash

    --port 8080
    -p 5000

**Used in:**
  - ``cellmap_flow_server`` (all model subcommands)

--debug
^^^^^^^

:Type: ``flag`` (boolean)
:Required: No
:Default: ``False``
:Description: Run server in debug mode with verbose logging

**Examples:**

.. code-block:: bash

    --debug

**Used in:**
  - ``cellmap_flow_server`` (all model subcommands)

--certfile
^^^^^^^^^^

:Type: ``string``
:Required: No (for HTTPS)
:Default: None
:Description: Path to SSL certificate file for HTTPS connections

**Examples:**

.. code-block:: bash

    --certfile /path/to/cert.pem

**Used in:**
  - ``cellmap_flow_server`` (all model subcommands)

--keyfile
^^^^^^^^^

:Type: ``string``
:Required: No (for HTTPS)
:Default: None
:Description: Path to SSL private key file for HTTPS connections

**Examples:**

.. code-block:: bash

    --keyfile /path/to/key.pem

**Used in:**
  - ``cellmap_flow_server`` (all model subcommands)

UI Server Arguments
-------------------

--neuroglancer-url
^^^^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for UI server)
:Default: None
:Description: Neuroglancer viewer URL to connect to

**Examples:**

.. code-block:: bash

    --neuroglancer-url http://localhost:8080/neuroglancer

**Used in:**
  - ``cellmap_flow_server run-ui-server``

--inference-host, -i
^^^^^^^^^^^^^^^^^^^^^

:Type: ``string``
:Required: Yes (for UI server)
:Default: None
:Description: Inference host address(es)

**Examples:**

.. code-block:: bash

    --inference-host http://gpu-node-01:5000
    -i http://localhost:8000,http://localhost:8001

**Used in:**
  - ``cellmap_flow_server run-ui-server``

Optional/Advanced Arguments
---------------------------

--scale, -r
^^^^^^^^^^^

:Type: ``string``
:Required: No
:Default: Auto-detected
:Description: Input scale level for multi-resolution datasets (e.g., ``s0``, ``s1``, ``s2``)

**Examples:**

.. code-block:: bash

    --scale s1
    -r s2

**Used in:**
  - ``cellmap_flow_multiple`` (per-model)
  - ``cellmap_flow_yaml`` (in YAML as ``scale``)

--log-level
^^^^^^^^^^^

:Type: ``choice`` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
:Required: No
:Default: ``INFO``
:Description: Logging verbosity level

**Examples:**

.. code-block:: bash

    --log-level DEBUG
    --log-level WARNING

**Used in:**
  - ``cellmap_flow`` (main CLI group)
  - ``cellmap_flow_server`` (main CLI group)

Migration Guide
---------------

Old vs New Argument Names
^^^^^^^^^^^^^^^^^^^^^^^^^

If you have scripts using old argument names, use this table to update them:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Old Argument
     - New Argument
     - Notes
   * - ``--data_path``
     - ``--data-path``
     - Standardized hyphenation
   * - ``--charge_group``
     - ``--project``
     - Clearer terminology
   * - ``--script_path``
     - ``--script-path``
     - Standardized hyphenation
   * - ``--model_path``
     - ``--model-path``
     - Standardized hyphenation
   * - ``--config_folder``
     - ``--config-folder``
     - Standardized hyphenation
   * - ``--edge_length_to_process``
     - ``--edge-length-to-process``
     - Standardized hyphenation
   * - ``--input_voxel_size``
     - ``--input-voxel-size``
     - Standardized hyphenation
   * - ``--output_voxel_size``
     - ``--output-voxel-size``
     - Standardized hyphenation
   * - ``--neuroglancer_url``
     - ``--neuroglancer-url``
     - Standardized hyphenation
   * - ``--inference_host``
     - ``--inference-host``
     - Standardized hyphenation
   * - ``--folder_path``
     - ``--folder-path``
     - Standardized hyphenation

Automated Migration Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large codebases, you can use this sed script to update your YAML and shell scripts:

.. code-block:: bash

    # Update argument names in shell scripts
    sed -i 's/--data_path/--data-path/g' *.sh
    sed -i 's/--charge_group/--project/g' *.sh
    sed -i 's/--script_path/--script-path/g' *.sh
    sed -i 's/--model_path/--model-path/g' *.sh
    sed -i 's/--config_folder/--config-folder/g' *.sh

Examples by Use Case
--------------------

Single Model Inference
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Script-based model
    cellmap_flow script \
        --script-path /path/to/model.py \
        --data-path /data/volume.zarr/s1 \
        --project cellmap \
        --queue gpu_h100

    # CellMap model
    cellmap_flow cellmap-model \
        --config-folder /models/mito_config \
        --name mito_v2 \
        --data-path /data/volume.zarr/s1 \
        --project cellmap

Multiple Models
^^^^^^^^^^^^^^^

.. code-block:: bash

    cellmap_flow_multiple \
        --data-path /data/volume.zarr/s1 \
        --project cellmap \
        --queue gpu_h100 \
        --script -s /models/mito.py -n mito \
        --script -s /models/nucleus.py -n nucleus \
        --cellmap-model -f /models/golgi -n golgi

YAML Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    data_path: "/data/volume.zarr/s1"
    project: "cellmap"  # or use 'charge_group'
    queue: "gpu_h100"
    
    models:
      - type: script
        script_path: "/models/mito.py"
        name: "mito"
      - type: cellmap-model
        config_folder: "/models/nucleus_config"
        name: "nucleus"
        scale: "s2"

Manual Server Control
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Start inference server
    cellmap_flow_server script \
        --script-path /models/model.py \
        --data-path /data/volume.zarr \
        --port 8080 \
        --debug

    # Start UI server
    cellmap_flow_server run-ui-server \
        --neuroglancer-url http://localhost:8080/ng \
        --inference-host http://localhost:5000
