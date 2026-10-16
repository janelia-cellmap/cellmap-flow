BioImage Model Zoo
==================

The Models tab lists the models of the `BioImage Model Zoo <https://bioimage.io>`_
beside cellmap's Hugging Face models, and runs the ones you tick as
``type: bioimage`` models. Their servers run in ``pixi.toml``'s ``bioimageio``
environment, the type's default (see :ref:`model-env`).

In the dashboard
----------------

1. Open the **Models** tab and expand **BioImage Model Zoo**. The list is
   read from bioimage.io's artifact server (``hypha.aicell.io``) the first
   time and cached in ``~/.cellmap_flow/bioimage/``; **Refresh** reads it
   again, and so does the page by itself once the cached list is an hour old,
   and every hour while it is open. Its listing holds each model's description, so 2D or 3D comes
   from the model's input axes. When the server cannot be reached the legacy
   ``collection.json`` index is read instead, which lists fewer models.
2. Narrow it with the search box (name, description and tags), **EM only**
   (on by default: models whose tags, name or description say electron
   microscopy) and **2D** / **3D**. The arrow beside a model opens its page
   on bioimage.io; hovering over its name shows the full description.
3. Tick a model and check its **Voxel (nm)**: ``z,y,x`` or one number for all
   three, the voxel size the model reads the data at. For the zoo's EM models
   it is filled in with the one the model was trained at, and the line below
   says on what data and where that number comes from (hover for the source).
   No zoo model says it in its description, so cellmap-flow keeps them in
   ``cellmap_flow/models/bioimage_voxel_sizes.yaml``, looked up by hand.
   Some were trained on downsampled data (the Platynereis nuclei model at
   100x80x80 nm, z,y,x), and a few across several resolutions, for which you enter
   one. Other models need one entered.
4. Click **Submit Models**. As for the other models, unticking one and
   submitting again stops it, and a running one is ticked when the page is
   reloaded.

A ticked model is entered as its nickname (``kind-seashell``), and its
layer and job are named after it (``kind_seashell``).

In a YAML file
--------------

The same model, without the dashboard:

.. code-block:: yaml

    models:
      mito:
        type: bioimage
        model: kind-seashell
        # voxel_size: [8, 8, 8]   # without it, the one it was trained at: 30x8x8 (MitoEM)

As in the dashboard, a zoo EM model reads at the voxel size it was trained at
unless ``voxel_size`` says otherwise; any other model needs it.
``example/bioimage_em.yaml`` is a complete one.

From Python
-----------

.. code-block:: python

    from cellmap_flow.models.bioimage_catalog import list_bioimage_models, refresh_bioimage_models

    for model in list_bioimage_models()["models"]:
        if model["em"]:
            print(model["key"], model["dims"], model["name"])

Each model has ``id`` (its alias on the server; in the legacy index a
Zenodo DOI for older models), ``nickname``, ``key`` (what it is loaded by),
``name``, ``description``, ``tags``, ``dims`` (``"2d"``, ``"3d"`` or None),
``em``, ``weight_formats``, ``license``, ``cover``, ``url`` (its bioimage.io
page), ``trained`` (its row of ``bioimage_voxel_sizes.yaml``, or None:
``voxel_size``, ``trained_on``, ``confidence``, ``source``, ``note``) and, from
the server, ``declared_voxel_size`` (nm, or None when its
axes carry no unit: none of the zoo's models declared one in October 2026,
so a blank voxel size is refused at Submit unless ``trained`` gives one). ``refresh_bioimage_models()``
fetches the list again; a failed fetch raises ``ZooIndexError`` and keeps
the cache.
