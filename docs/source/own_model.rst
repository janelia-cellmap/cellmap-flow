Use your own model
==================

This page describes how to run a model that you trained yourself in
cellmap-flow.

Supported formats
-----------------

These formats run without writing any code. Enter the value from the second
column in **Add a model** on the dashboard's **Models** tab, or pass it to
``cellmap_flow add`` on the command line (see `Adding a model`_). Some
formats also need a voxel size or channel names, which **Add a model** asks
for.

.. list-table::
   :header-rows: 1
   :widths: 34 40 13 13

   * - Model
     - What to enter
     - Can be finetuned
     - Runs in the browser app
   * - cellmap-models export (`An export`_)
     - its folder, ``/path/to/exports/my_model``, or its Hugging Face repo,
       ``<you>/my-model``
     - yes
     - no; see `In the browser app`_
   * - fly_organelles checkpoint
     - ``/path/to/run08/model_checkpoint_438000``
     - yes
     - no
   * - TorchScript file (``.ts``)
     - ``/path/to/my_network.ts``, with its channels and input voxel size
     - whole network only
     - no
   * - DaCapo run
     - ``dacapo:my_run@100000`` (needs an environment with DaCapo installed)
     - yes
     - no
   * - mia-train model (:doc:`huggingface`)
     - its Hugging Face repo, ``ai-hhmi/lmd-vitl-p8-cellmap-lvd``
     - affinity models only
     - no
   * - Cellpose weights
     - ``cellpose:/path/to/weights``, or a built-in name such as ``cpsam``,
       with a voxel size
     - yes
     - no
   * - BioImage Model Zoo package (:doc:`bioimage`)
     - the folder or zip holding its ``rdf.yaml``
     - if it has PyTorch weights
     - no
   * - ONNX file (:ref:`onnx`)
     - ``/path/to/my_model.onnx``, with its input voxel size
     - only in the browser app
     - with a ``manifest.json``, if small enough

Other models need a short script or a plugin:

.. list-table::
   :header-rows: 1
   :widths: 34 40 13 13

   * - Model
     - What to do
     - Can be finetuned
     - Runs in the browser app
   * - PyTorch network with a weights file (a weights file alone is not
       accepted, because it does not define the network)
     - Write a script (`A script`_). To share the model, export it (`An
       export`_).
     - yes
     - no
   * - TensorFlow or another framework
     - Convert it to ONNX, or write a script that defines ``process_chunk``
       (`Models that are not PyTorch`_).
     - no
     - no
   * - A family of models in a format not listed here
     - Add a model type as a plugin (:doc:`plugins`).
     - if the type sets ``finetunable = True`` and provides a PyTorch network
     - no

Adding a model
--------------

In the dashboard:

1. Open the **Models** tab and enter the path, name or repo in **Add a
   model**.
2. Click *Resolve*. The dashboard shows the detected model type and a field
   for each value that the model file does not contain, such as the voxel
   size.
3. Fill in those fields and click *Run*.

On the command line, ``cellmap_flow add`` prints a YAML entry for the model
and lists any values that are still required. For example, a training
checkpoint contains neither channel names nor a voxel size:

.. code-block:: text

    $ cellmap_flow add /path/to/run08/model_checkpoint_438000
    # fly: a fly_organelles training checkpoint
    # runs in the fly environment (its type's default)
    # still needs: channels, input_voxel_size (add to the entry)
    models:
      run08_438000:
        type: fly
        checkpoint_path: /path/to/run08/model_checkpoint_438000

To also start the model, add ``--run -d /path/to/data.zarr/raw``. A model
that still requires values, as in this example, is started from the
dashboard or from a YAML configuration that contains the completed entry
(``cellmap_flow yaml``). For the complete list of accepted paths, names and
prefixes, see :ref:`resolving-models`.

A script
--------

A script is a Python file that builds the network and sets the variables
below at the top level. The variable names are fixed; ``channels`` is
optional:

.. code-block:: python

    # my_model.py
    import torch
    from my_package import MyNetwork        # must be an installed package

    input_voxel_size = (8, 8, 8)            # nm, in array axis order (z, y, x)
    output_voxel_size = (8, 8, 8)
    input_size = (178, 178, 178)            # input block size, in voxels
    output_size = (56, 56, 56)              # output block size, in voxels
    output_channels = 2
    channels = ["mito", "er"]               # optional channel names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork()
    model.load_state_dict(torch.load("/path/to/weights.pt", map_location=device, weights_only=True))
    model.to(device).eval()

cellmap-flow calls ``model`` with one input block at a time, after input
normalization: a float32 tensor with a batch of one and one input channel.
In this example the input has shape ``(1, 1, 178, 178, 178)``, and the model
must return a tensor of shape ``(1, 2, 56, 56, 56)``. The input and output
sizes may be equal.

Test the script on a single block before submitting a job. This command runs
locally, in the current environment, and prints ``Server check passed`` on
success. It checks that the model loads and returns the declared shape, not
that the prediction is good:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw --server-check

To run the model, enter the script's path in **Add a model**, or use the
command line:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw -q gpu_h100 -P your_charge_group

On a machine without a cluster, such as a Mac, omit ``-q`` and ``-P``: the
model then runs on that machine. Alternatively, add the script to a YAML
configuration (:doc:`yaml_config`):

.. code-block:: yaml

    models:
      my_model:
        type: script
        script_path: /path/to/my_model.py

Notes:

- **Dependencies.** By default the script runs in the cellmap-flow
  environment. If the model requires other packages, set ``env`` on the
  model entry to the path of a conda environment or virtualenv that contains
  both those packages and cellmap-flow (:ref:`model-env`).
- **Imports.** The script's folder is not on the Python path, so code that
  the script imports must be installed as a package.
- **Input normalization.** A script does not specify how the input is
  normalized. Select the normalization that the network was trained with
  (for example, scaling to [0, 1] or [-1, 1]) on the **Processing** tab, or
  set ``input_norm`` under ``json_data`` in the YAML (:doc:`yaml_config`).
- **Output activation.** If the network outputs logits, apply a sigmoid,
  either in the script
  (``model = torch.nn.Sequential(network, torch.nn.Sigmoid())``) or as a
  step on the **Processing** tab.
- **Voxel sizes that are not whole numbers.** Give ``read_shape`` and
  ``write_shape`` in nm instead of ``input_size`` and ``output_size``.
- **Several input channels.** ``model`` receives one channel of the data. A
  network that takes several needs its own ``predict`` or ``process_chunk``
  (:doc:`custom_script`).
- **Affinities.** Channels whose names contain ``_aff`` are treated as
  affinities, which the affinity postprocessing steps turn into a
  segmentation (:doc:`post`).

For all script variables and a TensorFlow example, see :doc:`custom_script`.

An export
---------

An export is a folder that contains the network and a ``metadata.json``
file. It runs without your source code, so it can be shared. The metadata
contains the same information as the script variables above, plus the input
normalization and output activation, which the dashboard applies when the
model starts unless you have already set them. Create an export with
`cellmap-models <https://github.com/janelia-cellmap/cellmap-models>`_:

.. code-block:: python

    import cellmap_models.model_export.config as c
    from cellmap_models.model_export import ModelMetadata, export_metadata, export_torch_model

    c.EXPORT_FOLDER = "/path/to/exports"

    model = ...  # your torch.nn.Module, weights loaded
    model.eval()

    metadata = ModelMetadata(
        model_name="my_model",
        model_type="UNet",
        framework="torch",
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels_names=["mito", "er"],
        iteration=100000,
        input_shape=[178, 178, 178],
        output_shape=[56, 56, 56],
        inference_input_shape=[178, 178, 178],
        inference_output_shape=[56, 56, 56],
        input_voxel_size=[8, 8, 8],
        output_voxel_size=[8, 8, 8],
        input_normalization={"method": "min_max", "min": -1.0, "max": 1.0},
        inference_output_activation={"name": "sigmoid", "apply_after_model": True},
        author="Your Name",
        description="Mitochondria and ER, trained on ...",
    )

    # Writes /path/to/exports/my_model/metadata.json (EXPORT_FOLDER/model_name).
    export_metadata(metadata, overwrite=True)
    # Writes the network into the same folder.
    export_torch_model(model, (1, 1, 178, 178, 178), "/path/to/exports/my_model")

To run the export, enter the folder's path in **Add a model**, or add it to
a YAML configuration:

.. code-block:: yaml

    models:
      my_model:
        type: cellmap
        folder_path: /path/to/exports/my_model

To share the export, upload the folder to a Hugging Face model repository.
Log in first, with ``hf auth login`` (``huggingface-cli login`` in older
versions of ``huggingface_hub``):

.. code-block:: python

    from cellmap_models.model_export import push_to_huggingface

    push_to_huggingface(folder_path="/path/to/exports/my_model", repo_id="<you>/my-model")

Other users can then run the model by entering ``<you>/my-model`` in **Add
a model** (:doc:`huggingface`).

.. note::

   An export runs only at the input size that it was exported with
   (178 voxels per side in this example), so export at the size that the
   model should run at.

Models that are not PyTorch
---------------------------

A model from another framework runs in one of two ways:

- **Convert it to ONNX.** cellmap-flow runs an ``.onnx`` file with ONNX
  Runtime. Enter its path and input voxel size in **Add a model**
  (:ref:`onnx`).
- **Write a script with its own prediction function.** Instead of ``model``,
  the script defines ``process_chunk``, which reads a block of the data and
  returns the prediction using any library. ``example/tensorflow_lsd.py`` in
  the repository is a TensorFlow example (:doc:`custom_script`). Set ``env``
  on the model entry to an environment that contains the framework and
  cellmap-flow (:ref:`model-env`).

Limitations compared with a PyTorch model:

- The model cannot be finetuned on the cluster, which trains PyTorch
  networks only. An ONNX model can be finetuned in the browser app.
- A script that defines ``process_chunk`` runs only at its own voxel size.

In the browser app
------------------

The `browser app <https://janelia-cellmap.github.io/cellmap-flow/app/>`_ runs
models on the user's own GPU. It supports ONNX models that are small enough
to run in a browser. Add a model from **Add a model** on the app's
**Models** tab (:doc:`browser`), in one of two ways:

- **From Hugging Face.** Enter ``<owner>/<repo>/<folder>`` for a public
  repository that has the same layout as
  `cellmap/onnx-web-models <https://huggingface.co/cellmap/onnx-web-models>`_:
  one folder per model, containing a ``manifest.json`` and the ``.onnx``
  files that it lists. The app lists added repositories under *Added* and
  remembers them.
- **From a local file.** Click *Open an .onnx file…* and select the ``.onnx``
  file together with its ``manifest.json``. The file is not uploaded.

The supported way to create this layout is *Publish to Hugging Face* on the
Finetune tab, for a model finetuned in cellmap-flow. It uploads the server
export and, if the model is small enough, the browser export with its
manifest to your Hugging Face account
(`docs/finetuning.md <https://github.com/janelia-cellmap/cellmap-flow/blob/main/docs/finetuning.md>`__).

Troubleshooting
---------------

- **The model fails to start.** Run the ``--server-check`` command above. It
  runs one block locally and prints the error in the terminal. A "shape
  validation failed" error means that ``output_size`` or ``output_channels``
  does not match what the network returns.
- **"Script contains unsafe elements".** This is a warning for scripts that
  import ``os``, ``sys`` or ``subprocess``, or call ``eval`` or ``exec``. The
  script still runs, unless ``FORCE_SAFE_CONFIG=1`` is set.
- **The job fails on the cluster.** Open **Job Logs** in the dashboard to see
  the job's output. An import error means that a required package is missing
  from the environment that the model runs in (:ref:`model-env`).
- **The prediction is blank, constant or noisy.** Check that the input
  normalization on the **Processing** tab matches the normalization that the
  network was trained with, that a network that outputs logits has its
  sigmoid, and that the voxel sizes are in z, y, x order.
- **The data has a different voxel size from the model.** No action is
  needed: by default the data is resampled to the model's input voxel size
  (:ref:`resampling`, :ref:`run-voxel-size`).
