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
     - ``checkpoint`` (required); ``classes``, ``resolution``, ``input_size`` and
       ``output_size`` when the run's folder does not say them; ``sigmoid``.
       See :ref:`fly`.
   * - ``bioimage``
     - BioModelConfig
     - ``model_name`` or ``model_path`` (required), ``voxel_size`` (required)
   * - ``cellmap``
     - CellMapModelConfig
     - ``config_folder`` (required)
   * - ``huggingface``
     - HuggingFaceModelConfig
     - ``repo`` (required), ``revision`` (optional). See :doc:`huggingface`.
   * - ``cellpose``
     - CellposeModelConfig
     - ``voxel_size`` (required), ``pretrained_model``, ``output``. See :ref:`cellpose`.

Common optional parameters: ``name``, ``scale``, ``env`` (see :ref:`model-env`).

.. _resolving-models:

An entry from what you have
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cellmap_flow add REF`` writes the entry for you. ``REF`` is whatever you
have in hand, and the entry is printed as YAML to paste under ``models:``,
with comments saying what ``REF`` was taken to be, where it runs, and what
it still needs:

.. code-block:: console

    $ cellmap_flow add cpsam
    # cellpose: a Cellpose 4 pretrained model
    # runs in the cellpose4 environment (its type's default)
    # still needs: voxel_size (add to the entry)
    models:
      cpsam:
        type: cellpose
        pretrained_model: cpsam

``-n`` names the model and ``-v 16,8,8`` gives the voxel size of a model
that does not say its own (Cellpose, bioimage.io). ``--run -d DATA`` (with
``-q``, ``-P`` and ``--resample`` as ``infer`` takes them) serves the model
right away, as ``cellmap_flow yaml`` would. The dashboard does the same for
a reference pasted into it (``POST /api/models/resolve``).

``REF`` is checked against these in order, and the first that matches wins:

1. A local file or folder: a ``.py`` is a ``script``; a folder with
   ``metadata.json`` (and ``model.ts``) a ``cellmap`` export; a folder with
   ``rdf.yaml`` or ``bioimageio.yaml``, such a file, or a ``.zip`` with one
   inside a ``bioimage`` model; ``model_checkpoint_<n>``, a ``.ts`` or a
   ``model.pt`` a ``fly`` model (its channels and voxel sizes read from its
   run's folder, see :ref:`fly`); a folder with ``adapter_config.json`` (a
   LoRA adapter) or a full finetune's ``model_state_dict.pt`` a
   ``finetune``, which still needs its ``base_model``; Cellpose-SAM weights a
   ``cellpose`` model.
2. A prefix that says outright what it is: ``hf:org/repo[@revision]``,
   ``bioimageio:<id, nickname, URL or path>``, ``cellpose:<name or path>``,
   ``fly:<path>``, ``dacapo:<run>@<iteration>``, ``script:<path>``.
3. A Cellpose model name: ``cpsam_v2``, ``cpsam``, ``cpdino``, ``cpdino-vitb``.
4. A URL: a bioimage.io model page, a Zenodo record, a DOI link or the URL of
   an ``rdf.yaml`` or zip is a ``bioimage`` model; ``huggingface.co/org/repo``
   is as 5.
5. ``org/repo``: a Hugging Face repo, which must be a cellmap-models export
   (``metadata.json`` and ``model.ts``). Any other model on the Hub is
   refused: wrap it in a script (:doc:`custom_script`).
6. A BioImage Model Zoo nickname or id (``affable-shark``,
   ``10.5281/zenodo.5764892``), looked up in the zoo's index.

5 and 6 are checked online; with ``--offline`` (or when the Hub or zoo
cannot be reached) a repo or zoo-shaped name is taken on trust, with a note.
Anything else is an error saying what was tried; a prefix settles it.

.. _model-env:

Running a model in its own environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model whose packages conflict with cellmap-flow's environment (Cellpose 4,
transformer models) can run in an environment of its own. Give its entry an
``env``:

.. code-block:: yaml

    models:
      my_model:
        type: script
        script_path: /path/to/my_model.py
        env: cellpose4

``env`` is either

- the name of an environment in cellmap-flow's ``pixi.toml``. The server runs
  as ``pixi run --frozen --manifest-path <pixi.toml> -e cellpose4 cellmap_flow serve ...``,
  and pixi installs the environment from the lockfile the first time. The
  manifest is the one in the checkout cellmap-flow is installed from; set
  ``CELLMAP_FLOW_PIXI_MANIFEST`` to use another. The environments of
  cellmap-flow's ``pixi.toml`` are listed below; or
- the absolute path of a conda environment or virtualenv with cellmap-flow
  installed. The server runs as ``<path>/bin/python -P -m cellmap_flow.cli.main serve ...``.

A value with a ``/`` in it, or starting with ``~``, is a path. An unknown name,
or a path without ``bin/python``, is an error when the YAML is read.

The environments of cellmap-flow's ``pixi.toml``. A model type with a default
runs there when its entry gives no ``env``; an explicit ``env`` wins, and
``env: current`` keeps it in the environment cellmap-flow runs in.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Environment
     - What it is for
     - Default for
   * - ``default``
     - The dashboard, the catalog, Cellpose 3 and finetuning; what Fileglancer
       deploys. ``test`` and ``dev`` add pytest and the linters to it.
     - types without a default of their own, which run where cellmap-flow runs
   * - ``fly``
     - fly_organelles (``mzouink/fly-organelles`` at ``ab89c10``), whose
       ``StandardUnet`` loads a raw training checkpoint and whose classes an
       eager ``model.pt`` unpickles. Solved on its own.
     - ``fly``, unless the checkpoint is TorchScript (``.ts``)
   * - ``bioimageio``
     - ``bioimageio.core`` 0.11 with its ONNX and PyTorch backends, for BioImage
       Model Zoo models. ``example/bioimage_em.yaml`` runs one.
     - ``bioimage``
   * - ``cellpose4``
     - Cellpose 4 (Cellpose-SAM), which needs a newer torch and numpy than
       Cellpose 3. Solved on its own.
     - none: give ``env: cellpose4``
   * - ``dacapo``
     - ``dacapo-ml``. Its lock does not import DaCapo yet (``dacapo-ml`` 0.3.0
       with fibsem-tools 7, which dropped ``fibsem_tools.metadata``), so a
       DaCapo model is served from a conda environment that has it.
     - none
   * - ``docs``
     - Building these docs.
     - none

Only the model's inference server and its finetuning job run there. The
dashboard reads the model's geometry from its running server rather than
building the model itself; blockwise runs models in its own process, so run
``pixi run -e cellpose4 cellmap_flow blockwise ...`` for such a model. A
finetuning job needs ``peft`` in the environment: a pixi environment that
does not install it is refused when the job is submitted.

``cellmap_flow infer <type> --env <env>`` and the dashboard's model form take
it too. The ``cellpose`` type runs in ``cellpose4`` without one (see
:ref:`cellpose`).

.. _fly:

fly_organelles checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~

``type: fly`` serves a network fly_organelles trained, from one checkpoint
file:

- a training checkpoint (``model_checkpoint_<iteration>``), loaded into
  fly_organelles' ``StandardUnet``; its feature maps, levels and kernel sizes
  are read from the weights;
- a TorchScript file (``.ts``);
- a whole pickled model (``model.pt``), unpickled only when
  ``CELLMAP_FLOW_ALLOW_PICKLE=1`` is set.

A training checkpoint or a ``model.pt`` runs in the ``fly`` pixi environment
unless the entry gives an ``env``; a ``.ts`` runs in any. A folder that
cellmap_models exported (``metadata.json`` and ``model.ts``) is served as
exported by ``type: cellmap``, ``folder_path: <the folder>``, and refused here.
The ``model.pt`` in such a folder can be served here, at a larger
``input_size`` than the one it was exported at.

What the entry does not give is read from the checkpoint's folder. A
fly_organelles training run is enough as it is:

.. code-block:: yaml

    models:
      mito_distance_16:
        type: fly
        checkpoint: /groups/cellmap/cellmap/zouinkhim/salevary/train/v2/distance/mito_16_all/model_checkpoint_20000

reads the channel names from the run's ``train.py`` and the voxel sizes and
tile from its training snapshots. Each value comes from the first of these
files that has it; a key the entry gives always wins:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - File
     - What is read
   * - ``metadata.json``
     - A cellmap_models export's: ``channels_names``, the voxel sizes,
       ``input_shape`` and ``output_shape``.
   * - ``config.yaml``
     - fly_organelles' run configuration: ``run.labels``, ``run.voxel_size``,
       ``checkpoint.input_shape`` and ``output_shape``.
   * - ``snapshots/``
     - The newest training snapshot's ``raw`` and ``output`` arrays: input and
       output size, their voxel sizes, and how many channels the network
       outputs.
   * - ``train.py``
     - Its top-level ``labels = [...]`` and ``voxel_size = ...``, when they are
       written out as literals. The script is parsed, never run.

Labels are taken as channel names only when the snapshots show one channel
per label; an affinity or LSD run's network outputs several per label, and
its entry has to name them. The voxel sizes are read only when the entry
gives neither, and the input and output size only when it gives neither: an
output size belongs to the input size it came with.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Meaning
   * - ``checkpoint`` (``checkpoint_path``)
     - (required)
     - The checkpoint file.
   * - ``classes`` (``channels``)
     - read from the folder
     - The name of each output channel, one per channel. An error when
       neither the entry nor the folder gives them.
   * - ``resolution`` (``input_voxel_size``), ``output_resolution``
       (``output_voxel_size``)
     - read from the folder
     - nm per voxel, one number or one per axis; fractions are kept. One
       stands for the other, as fly_organelles trains at one voxel size. An
       error when neither the entry nor the folder gives one.
   * - ``input_size``, ``output_size``
     - read from the folder, else 178 and computed
     - Voxels a side of a tile in and out. Without either, 178 goes in
       (fly_organelles' training tile) and what comes out is computed from the
       network. ``input_size`` alone is enough; ``output_size`` alone is an
       error. Larger tiles waste less context: a StandardUnet takes
       178 + 16k (194, 338, ...).
   * - ``sigmoid``
     - ``true``
     - Pass the output through a sigmoid, which is added unless the network
       ends in one already (a cellmap_models export does). fly_organelles
       trains on logits, so a training checkpoint gets one. ``false`` serves
       the network's output as it is.

.. _cellpose:

Cellpose
~~~~~~~~

``type: cellpose`` runs Cellpose 4 (Cellpose-SAM) on each z slice of a
chunk, in 2D, and serves its cell probability or its instance masks. It runs
in the ``cellpose4`` pixi environment unless the entry gives an ``env``:
Cellpose 4 cannot share the default environment, whose cellpose 3 pins an
older numpy. ``example/cellpose_sam.yaml`` serves it on jrc_mus-salivary-1.

.. code-block:: yaml

    models:
      cellpose_sam:
        type: cellpose
        voxel_size: 64
        output: probability

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Meaning
   * - ``voxel_size``
     - (required)
     - nm per voxel, read and written; one number or one per axis. Cellpose-SAM
       sees objects best about 30 voxels across, so pick the voxel size at
       which yours are roughly that (or give ``diameter``).
   * - ``pretrained_model``
     - ``cpsam_v2``
     - ``cpsam_v2``, ``cpsam``, ``cpdino``, ``cpdino-vitb``, or the path of
       finetuned Cellpose weights. The named ones are downloaded from the
       Hugging Face Hub (about 1 GB) to ``~/.cellpose/models``, or
       ``$CELLPOSE_LOCAL_MODELS_PATH``, when the server starts. The DINO
       models also need facebookresearch's ``dinov3`` package, which the
       ``cellpose4`` environment does not install.
   * - ``output``
     - ``probability``
     - ``probability``: the cell probability, 0 to 1 (float32). ``masks``:
       instance ids (uint64), unique within a chunk.
   * - ``slices_per_chunk``
     - 8
     - z slices in a chunk.
   * - ``slice_size``
     - 512
     - Voxels a side, in y and x, of each chunk's slices.
   * - ``context``
     - 32
     - Voxels read beyond them on each side in y and x, so that objects at
       a chunk's edge are seen whole, and cut off again. None in z.
   * - ``batch_size``
     - the whole chunk
     - Tiles per GPU pass. Cellpose cuts each slice into tiles (256 px for
       ``cpsam*``, 384 for the DINO models, overlapping by 10%); by default
       all of a chunk's tiles go in one pass. Lower it if the GPU runs out
       of memory.
   * - ``diameter``
     - none
     - Object diameter in voxels; Cellpose resizes each slice by
       30 / ``diameter``. None keeps the model's own scale.
   * - ``flow_threshold``, ``cellprob_threshold``
     - 0.4, 0.0
     - Cellpose's mask thresholds; ``masks`` only.

The probability is computed voxel by voxel, so it joins up across chunks,
and it skips the mask dynamics, which makes it the faster output. Masks are
made per chunk: an object that crosses a chunk's edge is cut there, with an
id on each side, and objects are not joined from slice to slice. Add the
``MortonSegmentationRelabeling`` postprocessor to make the ids unique across
chunks and show the layer as a segmentation:

.. code-block:: yaml

    json_data:
      postprocess:
        - name: MortonSegmentationRelabeling

For masks of a whole volume, write the probability with ``pixi run -e
cellpose4 cellmap_flow blockwise ...`` (blockwise runs the model in its own
process) and segment that, or run Cellpose's own distributed
segmentation (``cellpose.contrib.distributed_segmentation``), which stitches
objects across blocks.

Finetuned weights, from Cellpose's GUI or ``cellpose.train``, are a path:
``pretrained_model: /path/to/models/my_model``. Cellpose reads from the
weights which network they are, and the tiling follows.

Licence: the Cellpose-SAM weights were trained on data that includes
datasets licensed CC-BY-NC, so they are for non-commercial use.

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
