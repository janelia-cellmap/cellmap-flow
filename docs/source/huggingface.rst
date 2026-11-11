HuggingFace Models
==================

CellMap-Flow supports loading pre-trained models directly from `Hugging Face Hub <https://huggingface.co/cellmap>`_.
This is the easiest way to get started with inference — no need to train a model or manage checkpoints manually.

Available Models
----------------

The following models are available under the `cellmap <https://huggingface.co/cellmap>`_ organization on Hugging Face:

.. list-table::
   :header-rows: 1
   :widths: 30 30 10 10 10 10

   * - Model
     - Description
     - Channels
     - Input Voxel Size
     - Output Voxel Size
     - Iteration
   * - ``cellmap/fly_organelles_run07_432000``
     - Fly organelles segmentation (run07, 432k)
     - all_mem, organelle, mito, er, nucleus, pm, vs, ld
     - 8nm
     - 8nm
     - 432,000
   * - ``cellmap/fly_organelles_run07_700000``
     - Fly organelles segmentation (run07, 700k)
     - all_mem, organelle, mito, er, nucleus, pm, vs, ld
     - 8nm
     - 8nm
     - 700,000
   * - ``cellmap/fly_organelles_run08_438000``
     - Fly organelles segmentation (run08, 438k)
     - all_mem, organelle, mito, er, nucleus, pm, vs, ld
     - 8nm
     - 8nm
     - 438,000
   * - ``cellmap/jrc_mus-livers_16nm_to_8nm_mito``
     - Finetuned 3D UNet for mitochondria (mouse liver)
     - mito
     - 16nm
     - 8nm
     - 345,000
   * - ``cellmap/jrc_mus-livers_16nm_to_8nm_peroxisome``
     - Finetuned model for peroxisome (mouse liver)
     - peroxisome
     - 16nm
     - 8nm
     - 45,000
   * - ``cellmap/mito-aff-unet-setup-16``
     - Affinities for mitochondria segmentation (UNet)
     - mito_aff_1, mito_aff_2, mito_aff_3
     - 16nm
     - 16nm
     - 400,000
   * - ``cellmap/mito-aff-unet-setup-19-worms``
     - C. elegans affinities for mitochondria (UNet)
     - mito_aff_1, mito_aff_2, mito_aff_3
     - 16nm
     - 16nm
     - 370,000
   * - ``cellmap/ld-aff-unet-setup-48``
     - Affinities for lipid droplet segmentation (UNet)
     - ld_aff_1, ld_aff_2, ld_aff_3
     - 16nm
     - 16nm
     - 380,000

Loading a HuggingFace Model
----------------------------

There are three ways to use a HuggingFace model in CellMap-Flow:

Using a YAML configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended approach. Create a YAML file specifying the model by its HuggingFace repository ID:

.. code-block:: yaml

    data_path: /path/to/your/data.zarr/recon-1/em/fibsem-uint8
    charge_group: cellmap
    queue: gpu_h100

    models:
      mito_model:
        type: huggingface
        repo: cellmap/fly_organelles_run08_438000

Then run:

.. code-block:: bash

    cellmap_flow yaml config.yaml

You can also specify an optional ``revision`` to pin a specific version:

.. code-block:: yaml

    models:
      mito_model:
        type: huggingface
        repo: cellmap/fly_organelles_run08_438000
        revision: main

Using the CLI
~~~~~~~~~~~~~

Run a HuggingFace model directly from the command line:

.. code-block:: bash

    cellmap_flow infer huggingface --repo cellmap/fly_organelles_run08_438000 -d /path/to/your/data.zarr/recon-1/em/fibsem-uint8

Using the Dashboard
~~~~~~~~~~~~~~~~~~~

1. Launch the dashboard with ``cellmap_flow view -d /path/to/your/data.zarr``.
2. Go to the **Models** tab.
3. In the **HuggingFace Models** section, browse or search for a model.
4. Select the model(s) you want to run and click **Submit**.
5. The models running are listed under **Loaded models** at the top of the
   tab, where the ✕ beside one stops that model alone (see *Usage* on the
   :doc:`front page <index>`).

Parameters
----------

The ``huggingface`` model type accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Parameter
     - Required
     - Description
   * - ``repo``
     - Yes
     - HuggingFace repository ID (e.g., ``cellmap/fly_organelles_run08_438000``).
   * - ``revision``
     - No
     - Git revision, branch, or tag to use. Defaults to the latest version.
   * - ``name``
     - No
     - Custom name for the model. Defaults to the repository name.
   * - ``scale``
     - No
     - Optional scale factor for the output.

Multiple Models
---------------

You can run multiple HuggingFace models in a single configuration:

.. code-block:: yaml

    data_path: /path/to/your/data.zarr/recon-1/em/fibsem-uint8
    charge_group: cellmap
    queue: gpu_h100

    models:
      fly_organelles:
        type: huggingface
        repo: cellmap/fly_organelles_run08_438000
      mito_model:
        type: huggingface
        repo: cellmap/jrc_mus-livers_16nm_to_8nm_mito

You can also mix HuggingFace models with other model types:

.. code-block:: yaml

    models:
      hf_model:
        type: huggingface
        repo: cellmap/fly_organelles_run08_438000
      local_model:
        type: dacapo
        run_name: my_run
        iteration: 50000

Models trained with mia-train
-----------------------------

Models trained with `mia-train <https://github.com/AI-HHMI/mia-train>`_, the
training framework of HHMI's MIA team, are published as a ``config.json`` and
a ``model.safetensors`` rather than as a cellmap-models export. Give the repo
as for any other model: type it into **Add a model** on the dashboard's
**Models** tab (``ai-hhmi/lmd-vitl-p8-cellmap-lvd``, say), tick it in the
Hugging Face list, which shows the team's models that cellmap-flow can serve
beside cellmap's, or name it in a YAML:

.. code-block:: yaml

    models:
      organelles:
        type: mia
        repo: ai-hhmi/lmd-vitl-p8-cellmap-lvd

On the CLI, ``cellmap_flow infer mia --repo ai-hhmi/lmd-vitl-p8-cellmap-lvd -d ...``;
``path:`` instead of ``repo:`` names a local folder with the two files, and
``mia:`` before a repo or folder says what it is outright.

The first start converts the model, once, into a cellmap-models export, on
the GPU of the job that serves it, in the ``mia`` pixi environment (which has
mia-train; install it first with ``cellmap_flow envs install mia``). The
export is kept under ``~/.cellmap_flow/mia_exports`` (or
``CELLMAP_FLOW_MIA_CACHE``), one per commit of the repo, so every later start,
and a finetune of the model, serves it at once. The dashboard knows what the
export will declare before then, from the repo's ``config.json`` and model
card alone, so Submit sets the input normalization the model was trained on
(the raw data scaled onto [0, 1] for the ``ai-hhmi`` models) from the first
start. A model trained on each
dataset at its own voxel size runs at the finest voxel size of the data it is
served on, unless its entry gives ``run_voxel_size``. A repo of the framework
that is not a segmentation model (a pretraining checkpoint) is refused,
saying so.

These models run on a server only: the browser app's Add a model says so for
such a repo, as its vision transformers over 256³ windows are far too big for
the browser.

Listing Available Models
------------------------

You can list available HuggingFace models programmatically:

.. code-block:: python

    from cellmap_flow.models.hf_catalog import list_huggingface_models

    models = list_huggingface_models()
    for model_id, metadata in models.items():
        print(f"{model_id}: {metadata.get('description', '')}")

To force a refresh of the cached model list:

.. code-block:: python

    from cellmap_flow.models.hf_catalog import refresh_huggingface_models

    models = refresh_huggingface_models()
