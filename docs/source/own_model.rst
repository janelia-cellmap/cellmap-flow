Use your own model
==================

This page is for someone who has trained a model of their own and wants to
see it on their data in cellmap-flow. It says which route to take for what
you have, and links to the page with the details of each.

Whatever the route, cellmap-flow needs to know the same few things about a
model:

- how to load it;
- the voxel size it reads, and the voxel size it writes (in nm);
- how many voxels it reads at a time, and how many it writes for them (a
  network with valid convolutions writes fewer than it reads);
- how many channels it writes, and ideally their names.

The routes differ only in where those come from: a script you write, an
export's metadata, or the model file itself.

First, try pasting it
---------------------

cellmap-flow recognises many kinds of model from a path or a name alone.
Before writing anything, give it what you have:

- in the dashboard, type it into **Add a model** on the **Models** tab and
  click *Resolve*;
- or on the command line, ``cellmap_flow add REF`` (add ``--run -d
  /path/to/data.zarr/raw`` to serve it right away).

Both say what the reference was taken to be, the environment it runs in and
anything it still needs (a voxel size, say). :ref:`resolving-models` lists
everything that is recognised.

Which route
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 34 18 18

   * - What you have
     - What to do
     - Finetune it in cellmap-flow
     - Runs in the browser app
   * - A PyTorch network and its weights
     - A short Python script (`A script`_ below). The quickest way to see a
       model.
     - yes
     - no
   * - A PyTorch network you want to keep, share or list for your group
     - A cellmap-models export, in a folder or on Hugging Face (`An
       export`_ below)
     - yes
     - with its web export
   * - A TensorFlow model, or any model with its own way of predicting
     - A script that defines ``process_chunk`` (:doc:`custom_script`)
     - no
     - no
   * - An ``.onnx`` file
     - ``type: onnx`` with its ``input_voxel_size`` (:ref:`onnx`)
     - only in the browser
     - yes, with a ``manifest.json``
   * - A model in the BioImage Model Zoo format (``rdf.yaml`` or
       ``bioimageio.yaml``, a folder or a zip)
     - Paste its path, with its voxel size (:doc:`bioimage`)
     - yes, with PyTorch weights
     - no
   * - Cellpose weights, a fly_organelles checkpoint, a DaCapo run, a
       mia-train model
     - Paste its path, name or repo: each has a type of its own
       (:doc:`yaml_config`, :doc:`huggingface`)
     - yes
     - no
   * - A family of models none of these fit
     - A model type of your own, as a plugin (:doc:`plugins`)
     - when it gives a PyTorch network
     - no

A script
--------

A script is a Python file that builds the model and says what it reads and
writes, as variables at its top level:

.. code-block:: python

    # my_model.py
    import torch
    from my_package import MyNetwork        # your own code

    input_voxel_size = (8, 8, 8)            # nm, in the order of the array's axes (z, y, x)
    output_voxel_size = (8, 8, 8)
    input_size = (178, 178, 178)            # voxels the network reads at a time
    output_size = (56, 56, 56)              # voxels it writes for them
    output_channels = 2
    channels = ["mito", "er"]               # optional: the channels' names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork()
    model.load_state_dict(torch.load("/path/to/weights.pt", map_location=device, weights_only=True))
    model.to(device).eval()

``model`` is called with a float tensor of shape ``(1, 1, z, y, x)``, one
block of the data after the input normalization chosen in the dashboard, and
returns ``(1, channels, z, y, x)``.

Check it on one block, in the current process, without a job:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw --server-check

Then run it: paste the script's path into **Add a model**, or

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw -q gpu_h100 -P <charge group>

or name it in a YAML with other models (:doc:`yaml_config`):

.. code-block:: yaml

    models:
      my_model:
        type: script
        script_path: /path/to/my_model.py

Three things people run into:

- **Packages.** The script runs in cellmap-flow's environment. When the model
  needs packages that are not there, or that clash with it, give the entry an
  ``env``: the path of a conda environment or virtualenv that has both your
  packages and cellmap-flow (:ref:`model-env`).
- **Normalization.** What the network was trained on (the raw data scaled to
  [0, 1] or [-1, 1], say) is set on the **Processing** tab, or as
  ``input_normalizers`` in a YAML. A script does not declare it, so nothing
  sets it for you.
- **Activation.** A network that ends in logits needs its sigmoid: wrap it,
  ``torch.nn.Sequential(network, torch.nn.Sigmoid())``, or add a sigmoid step
  on the Processing tab.

:doc:`custom_script` has the full list of variables and a TensorFlow example.

An export
---------

An export is a folder holding the network and a ``metadata.json`` that says
everything cellmap-flow asks of a script, and also the input normalization
and the output activation, which the dashboard then sets up by itself. It is
made once, with `cellmap-models <https://github.com/janelia-cellmap/cellmap-models>`_:

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

The folder is served as it is: paste its path into **Add a model**, or

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

Anyone can then run it by its repo: ``<you>/my-model`` in **Add a model**,
``cellmap_flow add <you>/my-model``, or ``type: huggingface`` with ``repo:``
in a YAML (:doc:`huggingface`). The dashboard lists the repos you added in a
section of their own under Hugging Face.

The export is traced at the input shape you give it, and runs at that shape
only: export it at the shape you want it served at.

In the browser app
------------------

The `browser app <https://janelia-cellmap.github.io/cellmap-flow/app/>`_ runs
ONNX models on the visitor's own GPU, so a model has to be an ONNX export
small enough for a browser. There are two ways in, both from its Models
tab's *Add a model* (:doc:`browser`):

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

- ``--server-check`` (above) runs one block where you can read the error,
  before any job is submitted.
- In the dashboard, **Job Logs** shows the output of the model's job: an
  import error there means the model's packages are missing from the
  environment it runs in (:ref:`model-env`).
- A prediction that is all one value, or noise, is most often the input
  normalization: compare what the Processing tab applies with what the
  network was trained on.
- A model can run on data at another voxel size than its own; the data is
  resampled to what the model reads (:ref:`resampling`,
  :ref:`run-voxel-size`).
