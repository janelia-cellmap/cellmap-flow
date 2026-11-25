Use your own model
==================

This page shows how to run a model you trained yourself in cellmap-flow.

What to do for each kind of model
---------------------------------

Models in the formats below run as they are. Type the path, name or repo
into **Add a model** on the dashboard's **Models** tab, or run
``cellmap_flow add <path, name or repo>`` on the command line.

.. list-table::
   :header-rows: 1
   :widths: 34 40 13 13

   * - Model
     - What to type
     - Finetune
     - Browser app
   * - cellmap-models export (`An export`_)
     - its folder, ``/path/to/exports/my_model``, or its Hugging Face repo,
       ``<you>/my-model``
     - yes
     - with its web export
   * - fly_organelles checkpoint
     - ``/path/to/run08/model_checkpoint_438000``
     - yes
     - no
   * - DaCapo run
     - ``dacapo:my_run@100000``
     - yes
     - no
   * - mia-train model (:doc:`huggingface`)
     - its Hugging Face repo, ``ai-hhmi/lmd-vitl-p8-cellmap-lvd``
     - yes
     - no
   * - Cellpose weights
     - their path, or a built-in name such as ``cpsam``
     - yes
     - no
   * - BioImage Model Zoo package (:doc:`bioimage`)
     - the folder or zip holding its ``rdf.yaml``
     - with PyTorch weights
     - no
   * - ONNX file (:ref:`onnx`)
     - ``/path/to/my_model.onnx``
     - in the browser only
     - yes

Any other model needs a few lines of your own:

.. list-table::
   :header-rows: 1
   :widths: 34 40 13 13

   * - Model
     - What to write
     - Finetune
     - Browser app
   * - PyTorch network with a weights file
     - a script (`A script`_), or an export to share it (`An export`_)
     - yes
     - no
   * - TensorFlow model, or one with its own prediction code
     - a script that defines ``process_chunk`` (:doc:`custom_script`)
     - no
     - no
   * - Many models of one kind that none of the above covers
     - a model type, as a plugin (:doc:`plugins`)
     - if PyTorch
     - no

Adding a model
--------------

In the dashboard, **Add a model** shows what the model was recognised as
(*Resolve*), asks for anything the file does not record, and starts it
(*Run*).

On the command line, ``cellmap_flow add`` prints the same as an entry for a
YAML file. A training checkpoint, for example, records neither its channels
nor its voxel size, so both are listed as still needed:

.. code-block:: text

    $ cellmap_flow add /path/to/run08/model_checkpoint_438000
    # fly: a fly_organelles training checkpoint
    # runs in the fly environment (its type's default)
    # still needs: channels, input_voxel_size (add to the entry)
    models:
      run08_438000:
        type: fly
        checkpoint_path: /path/to/run08/model_checkpoint_438000

With ``--run -d /path/to/data.zarr/raw`` it starts the model instead of
printing the entry. :ref:`resolving-models` lists every form that is
recognised.

A script
--------

A script loads the network and states its voxel sizes, block sizes and
channels as variables. cellmap-flow reads them by these names:

.. code-block:: python

    # my_model.py
    import torch
    from my_package import MyNetwork        # your own code

    input_voxel_size = (8, 8, 8)            # nm, in the order of the array's axes (z, y, x)
    output_voxel_size = (8, 8, 8)
    input_size = (178, 178, 178)            # the block the network takes in, in voxels
    output_size = (56, 56, 56)              # the block it gives back for that
    output_channels = 2
    channels = ["mito", "er"]               # optional: the channels' names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork()
    model.load_state_dict(torch.load("/path/to/weights.pt", map_location=device, weights_only=True))
    model.to(device).eval()

``model`` is called on one block at a time: a float tensor of shape
``(1, 1, 178, 178, 178)`` here, already normalized, returning
``(1, 2, 56, 56, 56)``.

To check it, predict one block in the terminal, without a cluster job:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw --server-check

To run it, type the script's path into **Add a model**, or:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw -q gpu_h100 -P <charge group>

or list it in a YAML file (:doc:`yaml_config`):

.. code-block:: yaml

    models:
      my_model:
        type: script
        script_path: /path/to/my_model.py

Common problems:

- **Packages.** The script runs in cellmap-flow's environment. When the model
  needs packages that are not there, or that clash with it, give the entry an
  ``env``: the path of a conda environment or virtualenv that has both your
  packages and cellmap-flow (:ref:`model-env`).
- **Normalization.** Set the scaling the network was trained with (to
  [0, 1] or [-1, 1], say) on the **Processing** tab, or as
  ``input_normalizers`` in a YAML. A script does not record it.
- **Activation.** For a network that outputs logits, add the sigmoid in the
  script, ``model = torch.nn.Sequential(network, torch.nn.Sigmoid())``, or
  as a step on the Processing tab.

:doc:`custom_script` has the full list of variables and a TensorFlow example.

An export
---------

An export packs the network into one folder that runs without your code,
so it can be shared. Its ``metadata.json`` records what a script's variables
do, plus the input normalization and output activation, which the dashboard
then applies by itself. It is made with
`cellmap-models <https://github.com/janelia-cellmap/cellmap-models>`_:

.. code-block:: python

    import cellmap_models.model_export.config as c
    from cellmap_models.model_export import ModelMetadata, export_torch_model

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
        input_shape=[1, 1, 178, 178, 178],
        output_shape=[1, 2, 56, 56, 56],
        inference_input_shape=[178, 178, 178],
        inference_output_shape=[56, 56, 56],
        input_voxel_size=[8, 8, 8],
        output_voxel_size=[8, 8, 8],
        input_normalization={"method": "min_max", "min": -1.0, "max": 1.0},
        inference_output_activation={"name": "sigmoid", "apply_after_model": True},
        author="Your Name",
        description="Mitochondria and ER, trained on ...",
    )

    export_torch_model(model, (1, 1, 178, 178, 178), "/path/to/exports/my_model", metadata=metadata)

To run it, type the folder's path into **Add a model**, or list it in a YAML:

.. code-block:: yaml

    models:
      my_model:
        type: cellmap
        folder_path: /path/to/exports/my_model

To share it, push the folder to a Hugging Face model repo of your own
(``huggingface-cli login`` first):

.. code-block:: python

    from cellmap_models.model_export import push_to_huggingface

    push_to_huggingface(folder_path="/path/to/exports/my_model", repo_id="<you>/my-model")

Anyone can then run it as ``<you>/my-model`` (:doc:`huggingface`), and the
dashboard keeps added repos in its Hugging Face list.

The export runs only at the input size it was exported with (178 voxels a
side above).

In the browser app
------------------

The `browser app <https://janelia-cellmap.github.io/cellmap-flow/app/>`_ runs
models on the GPU of whoever opens the page, so a model must be an ONNX file
small enough for a browser. It is added under *Add a model* on the app's
Models tab (:doc:`browser`), as either:

- a public Hugging Face repo laid out as
  `cellmap/onnx-web-models <https://huggingface.co/cellmap/onnx-web-models>`_
  is (a folder per model, with its ``manifest.json`` and the ``.onnx`` files
  it names), added as ``<owner>/<repo>/<folder>``. cellmap-models'
  ``export_onnx_web`` writes such a folder;
- *Open an .onnx file…*, for a file on your machine, picked together with its
  ``manifest.json``. Nothing is uploaded.

A model finetuned in cellmap-flow is published from the Finetune tab's
*Publish to Hugging Face*, which writes both the server's export and, when
the model fits in a browser, the browser's to your Hugging Face account
(`docs/finetuning.md <https://github.com/janelia-cellmap/cellmap-flow/blob/main/docs/finetuning.md>`__).

When it does not work
---------------------

- Run ``--server-check`` (above) first: it predicts one block in your
  terminal, so you see the error directly.
- In the dashboard, **Job Logs** shows what the model's job printed. An
  import error there means a package your model needs is not installed in
  the environment it runs in (:ref:`model-env`).
- If the prediction is blank, all one value, or noise, check the input
  normalization first: the Processing tab should apply what the network was
  trained with.
- Your data does not have to be at the model's voxel size: it is resampled
  to what the model expects (:ref:`resampling`, :ref:`run-voxel-size`).
