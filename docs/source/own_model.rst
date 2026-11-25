Use your own model
==================

You trained a model and want to see its predictions on your data in
cellmap-flow. This page tells you what to do, depending on what form your
model is in, and links to the page with the details of each.

For any model, cellmap-flow has to know:

- how to load it;
- the voxel size of the data it expects, and of what it predicts (in nm);
- the size of the block it takes in, and of the block it gives back. A
  network with valid convolutions gives back a smaller block than it takes
  in: 178 voxels a side in and 56 out, say;
- how many channels it predicts, and ideally what each one is.

Some model files say all of this themselves. For the others, you say it in a
short script.

First, try pasting it
---------------------

Often there is nothing to write. Type the path of your model (or its name,
or its Hugging Face repo) into one of these, and cellmap-flow works out what
kind of model it is:

- in the dashboard: the **Add a model** box on the **Models** tab, then
  *Resolve*, then *Run*;
- on the command line: ``cellmap_flow add <what you have>``.

Things you can type, and what each is taken to be:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - You type
     - It is taken to be
   * - ``/path/to/my_model.py``
     - a script that defines the model (`A script`_ below)
   * - ``/path/to/exports/my_model`` (a folder holding ``metadata.json`` and
       ``model.ts``)
     - a cellmap-models export (`An export`_ below)
   * - ``cellmap/fly_organelles_run08_438000``
     - a Hugging Face repo holding such an export
   * - ``/path/to/run08/model_checkpoint_438000``
     - a fly_organelles training checkpoint
   * - ``/path/to/my_model.onnx``
     - an ONNX model
   * - ``/path/to/model_folder`` (a folder holding ``rdf.yaml``), or
       ``affable-shark``
     - a BioImage Model Zoo model, from disk or from the zoo by its nickname
   * - ``cpsam``, or the path of Cellpose weights
     - a Cellpose model
   * - ``dacapo:my_run@100000``
     - a DaCapo run at that iteration
   * - ``ai-hhmi/lmd-vitl-p8-cellmap-lvd``
     - a model trained with mia-train, on Hugging Face

The answer says what kind of model it is, which environment it will run in,
and anything you still have to give. On the command line it is printed as an
entry you can paste into a YAML file. For a script, nothing is missing:

.. code-block:: text

    $ cellmap_flow add /path/to/my_model.py
    # script: a Python script that defines the model
    # runs in this environment
    models:
      my_model:
        type: script
        script_path: /path/to/my_model.py

A training checkpoint, on the other hand, does not say what its channels are
or what voxel size it was trained at, so you are asked for them:

.. code-block:: text

    $ cellmap_flow add /path/to/run08/model_checkpoint_438000
    # fly: a fly_organelles training checkpoint
    # runs in the fly environment (its type's default)
    # still needs: channels, input_voxel_size (add to the entry)
    models:
      run08_438000:
        type: fly
        checkpoint_path: /path/to/run08/model_checkpoint_438000

In the dashboard, *Resolve* shows the same answer, with a box for each value
that is still needed, and *Run* starts the model.

To start the model straight from the command line instead of printing the
entry, add ``--run`` and the data:

.. code-block:: bash

    cellmap_flow add /path/to/my_model.py --run -d /path/to/data.zarr/raw

If what you typed is not recognised, the error says what was tried.
:ref:`resolving-models` has the full list of what is recognised.

If your model is just a PyTorch network and a file of weights, it is not
recognised by itself: nothing in a weights file says what network it belongs
to. Write a script (next section).

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

A script is a Python file that loads your network and states its sizes, as
plain variables. cellmap-flow runs the file and reads the variables by name,
so the names below matter:

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

cellmap-flow calls ``model`` on one block of your data at a time: a float
tensor of shape ``(1, 1, 178, 178, 178)`` here, already normalized as set in
the dashboard. It expects ``(1, 2, 56, 56, 56)`` back: the two channels, at
the output size.

Before running it for real, check that it loads and predicts one block. This
runs where you type it, without submitting a cluster job, and prints
``Server check passed``:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw --server-check

Then run it. Either type the script's path into **Add a model** in the
dashboard, or start it from the command line:

.. code-block:: bash

    cellmap_flow infer script -s my_model.py -d /path/to/data.zarr/raw -q gpu_h100 -P <charge group>

or list it in a YAML file, beside other models (:doc:`yaml_config`):

.. code-block:: yaml

    models:
      my_model:
        type: script
        script_path: /path/to/my_model.py

Three things commonly go wrong:

- **Packages.** The script runs in cellmap-flow's environment. When the model
  needs packages that are not there, or that clash with it, give the entry an
  ``env``: the path of a conda environment or virtualenv that has both your
  packages and cellmap-flow (:ref:`model-env`).
- **Normalization.** If your network was trained on data scaled to [0, 1] or
  [-1, 1], say, it has to get the same here. A script does not say what it
  was trained on, so you set it yourself: on the **Processing** tab, or as
  ``input_normalizers`` in a YAML.
- **Activation.** If your network outputs logits rather than probabilities,
  add the sigmoid: either in the script,
  ``model = torch.nn.Sequential(network, torch.nn.Sigmoid())``, or as a step
  on the Processing tab.

:doc:`custom_script` has the full list of variables and a TensorFlow example.

An export
---------

A script is quick, but it only works where your code and weights are. An
export packs the network into one folder that anyone can run without your
code. Its ``metadata.json`` holds the same facts as a script's variables,
and also the input normalization and the output activation, so the dashboard
sets those up for you. You make it once, with
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

Anyone can then run it by the repo's name: ``<you>/my-model`` typed into
**Add a model**, ``cellmap_flow add <you>/my-model``, or ``type:
huggingface`` with ``repo: <you>/my-model`` in a YAML (:doc:`huggingface`).
Once added, the dashboard keeps it in the Hugging Face list, in a section
for added repos.

One caution: the export only runs at the input size you exported it with
(178 voxels a side above), so pick the size you want it to run at.

In the browser app
------------------

The `browser app <https://janelia-cellmap.github.io/cellmap-flow/app/>`_ needs
nothing installed: the model runs on the GPU of whoever opens the page. For
that, the model has to be an ONNX file, and small enough for a browser.
There are two ways to add one, both under *Add a model* on the app's Models
tab (:doc:`browser`):

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
