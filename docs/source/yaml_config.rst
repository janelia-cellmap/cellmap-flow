YAML Configuration
===================

``cellmap_flow yaml`` lets you define and run multiple models from a single YAML file.
It is the recommended way to launch inference jobs, and the same YAML format is used by the blockwise processor (``cellmap_flow blockwise``).

Usage
-----

.. code-block:: bash

    # Run inference
    cellmap_flow yaml config.yaml

    # Validate without running
    cellmap_flow yaml config.yaml --validate-only

    # List available model types
    cellmap_flow yaml --list-types

    # Set log level
    cellmap_flow yaml config.yaml --log-level DEBUG

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
     - Path to the input dataset: zarr, N5 or precomputed, on disk or at an
       ``http(s)://``, ``s3://`` or ``gs://`` URL (see :doc:`data_paths`).
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
   * - ``resample``
     - No
     - ``true`` resamples the data to each model's input voxel size when it has no level at that size (default: ``false``; see :ref:`resampling`).
   * - ``extra_layers``
     - No
     - More volumes to show in the viewer beside the raw data (see below).
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
   * - ``output_channels``
     - No
     - Which model channels blockwise writes, and to which outputs (see :ref:`output-channels`).

Model Entries
-------------

Each model entry requires a ``type`` field and the parameters for that model type.
Use ``cellmap_flow yaml --list-types`` to see all available types and their required parameters.

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

Common optional parameters: ``name``, ``scale``, ``env`` (see :ref:`model-env`).

.. _model-env:

Running a model in its own environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model whose packages conflict with cellmap-flow's environment (Cellpose 4,
transformer models) can run in an environment of its own. Give its entry an
``env``:

.. code-block:: yaml

    models:
      cellpose_sam:
        type: script
        script_path: example/cellpose_sam_model.py
        env: cellpose4

``env`` is either

- the name of an environment in cellmap-flow's ``pixi.toml``. The server runs
  as ``pixi run --frozen --manifest-path <pixi.toml> -e cellpose4 cellmap_flow serve ...``,
  and pixi installs the environment from the lockfile the first time. The
  manifest is the one in the checkout cellmap-flow is installed from; set
  ``CELLMAP_FLOW_PIXI_MANIFEST`` to use another. cellmap-flow's ``pixi.toml``
  has ``cellpose4`` (Cellpose 4, for Cellpose-SAM), ``dacapo`` and
  ``bioimageio``; or
- the absolute path of a conda environment or virtualenv with cellmap-flow
  installed. The server runs as ``<path>/bin/python -P -m cellmap_flow.cli.main serve ...``.

A value with a ``/`` in it, or starting with ``~``, is a path. An unknown name,
or a path without ``bin/python``, is an error when the YAML is read.

Only the model's inference server and its finetuning job run there. The
dashboard reads the model's geometry from its running server rather than
building the model itself; blockwise runs models in its own process, so run
``pixi run -e cellpose4 cellmap_flow blockwise ...`` for such a model. A
finetuning job needs ``peft`` in the environment: a pixi environment that
does not install it is refused when the job is submitted.

``cellmap_flow infer <type> --env <env>`` and the dashboard's model form take
it too. ``example/cellpose_sam.yaml`` runs Cellpose-SAM this way.

Each model type's default
^^^^^^^^^^^^^^^^^^^^^^^^^

Some model types run in an environment of their own unless the entry says
otherwise, so their entries need no ``env``:

=========================================  ======================
Type                                       Default ``env``
=========================================  ======================
``cellpose``                               ``cellpose4``
``bioimage``                               ``bioimageio``
``dacapo``                                 ``dacapo``
``fly``, with a raw checkpoint             ``fly``
``finetune``                               its base model type's
``script``, ``cellmap``, ``huggingface``   none: this environment
=========================================  ======================

An entry's own ``env`` wins. ``env: current`` runs the model in this
environment whatever its type's default (``default`` is not that: it is
pixi's ``default`` environment). Exported YAMLs write only an ``env`` the
entry gave, never the type's default. A plugin type sets its own as
``default_env`` on its ``ModelConfig`` subclass.

A default this machine cannot provide does not stop the model: with no
``pixi.toml``, no environment of that name in it, or no pixi at all (a
conda-only account), the model runs in this environment, as before its type
had a default, with a warning that says how to give it one. A default that
is in ``pixi.toml`` but not installed yet is used: its first job installs it
(several minutes), and a warning says so. An ``env`` the entry names
itself is an error when it cannot be used.

Aliases
^^^^^^^

``~/.cellmap_flow/envs.yaml`` (or the file ``CELLMAP_FLOW_ENVS_FILE`` names)
maps environment names to conda environments or virtualenvs with cellmap-flow
installed. An alias wins over a pixi environment of the same name, both for
an entry's ``env`` and for a type's default, so this is how a machine without
pixi runs the types' defaults:

.. code-block:: yaml

    cellpose4: /groups/lab/home/me/miniconda3/envs/cellpose4
    dacapo: /groups/lab/home/me/miniconda3/envs/dacapo

A model whose ``env`` is an alias keeps the name in exported YAMLs, so the
YAML works on a machine that has the pixi environment instead. An alias to a
relative path, or to a directory without ``bin/python``, is an error.

``cellmap_flow envs``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cellmap_flow envs                  # each environment, as `envs list`
    cellmap_flow envs install cellpose4
    cellmap_flow envs check cellpose4

``envs list`` shows each environment of ``pixi.toml`` and each alias: where
it is, whether it is installed, whether it can run a finetuning job, and
which model types default to it (a type that decides per model, as ``fly``
does, is listed apart). ``envs install`` installs a pixi environment from the
lockfile now (``pixi install --frozen``), rather than on its first job; an
alias has nothing to install. ``envs check`` imports cellmap-flow with the
environment's python, to show that it works there.

.. _channel-names:

Channel Names
~~~~~~~~~~~~~

A model names its output channels, in order, with one of:

- ``channels``: a list in a model script, or a ``fly`` entry's parameter. In a ``fly`` entry, ``classes`` is the same parameter under another name, so give one of them; a ``fly`` entry may also give its names as one comma-separated string, ``classes: mito,er``.
- ``channels_names``: read from the ``metadata.json`` of a ``huggingface`` or ``cellmap`` model.
- ``classes``: a list in a model script.

A DaCapo model takes its names from its run's task. A ``bio`` model names none.

When a model gives more than one of these, the first that is not empty wins, in the order ``channels``, ``channels_names``, ``classes``. In a model script a single string is one name: ``channels = "mito"`` is one channel called ``mito``.

The inference server reports the names with the model's geometry, and blockwise names its outputs by them (see :ref:`output-channels`). A model that names no channels still serves; blockwise then needs ``output_channels`` as a mapping of channel indices.

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

.. _resampling:

Data at Another Voxel Size
--------------------------

A model reads the level of ``data_path`` at its input voxel size. When there is no such level, what happens depends on ``resample``:

- **Left out, or** ``resample: false``: the level closest to the model's voxel size that is not coarser on any axis (the finest level, when every one is) is read *as if* it were at the model's voxel size, voxel for voxel, with a warning. The model then sees data at the wrong scale, and its predictions are drawn where that level really is, at a proportionally different voxel size.
- ``resample: true``: a level is resampled to the model's voxel size, axis by axis, and the model sees the data at the size it was trained at. Its predictions are at its declared output voxel size.

.. code-block:: yaml

    data_path: /nrs/cellmap/data/my_dataset/my_dataset.zarr/recon-1/em/fibsem-uint8
    charge_group: cellmap
    resample: true   # levels 8x8x40, 16x16x80 ... nm; the model wants 16x16x16

    models:
      my_model:
        type: dacapo
        run_name: my_run
        iteration: 50000

How the data is resampled:

- **Which level.** The coarsest level that is no coarser than the model's voxel size on any axis, so every axis is downsampled from as close as the pyramid allows; when every level is coarser on some axis, the finest level. A level at exactly the model's voxel size is read as it is.
- **Each axis by its own factor.** By a whole number of voxels (8 nm to 16 nm), each voxel is the mean of the voxels it covers. Otherwise (40 nm to 16 nm, or 12 nm to 16 nm), it is a linear interpolation between the two nearest voxels. Label data (bool, or integers of 32 bits or more) takes the nearest voxel instead, so no label is invented. Intensities stay in their dtype (uint8 stays uint8, rounded), and go through ``json_data``'s normalizers after resampling, as a stored level would.
- **Where.** The resampled grid starts at the level's own corner, so the predictions lie over the data. Chunks are resampled on their own, and each gives exactly the voxels a read of the whole volume would.

``cellmap_flow yaml`` starts each server with ``--resample``, and ``cellmap_flow blockwise`` reads the data resampled the same way. ``cellmap_flow infer <type> --resample`` is the same for one model (:doc:`cli`).

In the dashboard it is the Models tab's *Resample if no scale matches the model* box, which ``resample: true`` (or ``cellmap_flow view --resample``) starts ticked. It applies to the models submitted from then on, to blockwise runs, and to annotation volumes made while it is on: those are at the model's own voxel sizes, and their finetunes train on the data resampled and are served resampled. When a running model reads a level as if it were at its voxel size, the banner above the tabs says so.

Extra Layers
------------

``cellmap_flow yaml`` can show more volumes in the viewer beside the raw data and the predictions, for instance an earlier prediction or an instance segmentation. Each is read as stored, without the input normalizers:

.. code-block:: yaml

    extra_layers:
      - name: base_mito
        path: /nrs/cellmap/predictions/mito.zarr/mito
        shader: |                      # optional, image layers only
          void main() { emitRGB(vec3(0, toNormalized(getDataValue()), 0)); }
        blend: additive                # optional, image layers only
      - name: instances
        path: /nrs/cellmap/predictions/instances.zarr/s0
        layer_type: segmentation       # default: image
        disable_meshes: true           # optional; no meshes computed when a segment is picked

A volume that cannot be opened is logged and left out. ``--validate-only`` checks that every entry has a unique ``name`` and a ``path``, and a known ``layer_type``.

Behind a Reverse Proxy
----------------------

When the dashboard is reached through a reverse proxy that sets ``X-Forwarded-Host``, the page loads the neuroglancer viewer from the same path on the proxy's host, so the proxy must route ``/v/`` on to the viewer. Inference servers are addressed as they report themselves (``http://<node>:<port>``) unless ``CELLMAP_FLOW_SERVER_URL_TEMPLATE`` is set in the dashboard's environment, for example to ``https://proxy.example.org/inf-{port}``; ``{url}``, ``{host}`` and ``{port}`` are the reported address and its parts.

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

.. _output-channels:

Output Channels
---------------

Blockwise writes each output to its own group, ``<output_path>/<output name>/s0``. ``output_channels`` says which of the model's channels go to which output:

- **Left out**: one output per model channel, named by the model's channel names (see :ref:`channel-names`).
- **A list of channel names** (or one name): one output for each, holding that model channel.

  .. code-block:: yaml

      output_channels: [mito, er]

- **A mapping of output names to channel indices**, counted from 0. An index, or a list of one, gives an output holding that channel. A list of several gives one output with those channels stacked, in the list's order, on a leading channel axis ``c``: its axes are ``c, z, y, x``.

  .. code-block:: yaml

      output_channels:
        affinities: [0, 1, 2]   # c, z, y, x: channels 0, 1 and 2
        mito: 3                 # z, y, x: channel 3

  A mapping names the outputs itself and picks channels by index, so it also works for a model that names no channels.

Outputs are named uniquely: a list naming a channel twice is refused.

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

    cellmap_flow blockwise config.yaml
    cellmap_flow blockwise config.yaml --log-level DEBUG
