Viewer layer API
================

A running dashboard can add, rename and remove layers of its Neuroglancer
viewer over HTTP, so a script or a shell can put a dataset next to the
predictions without restarting anything. The dashboard's own pages don't use
these routes.

Every route takes a JSON ``POST`` body and answers with ``{"success": true,
...}``, or ``{"success": false, "error": "..."}`` and a 4xx or 5xx status.
Open the dashboard's Neuroglancer tab again (or reload it) to see a change
when the answer says ``"reload_page": true``.

In the examples, ``DASHBOARD`` is the address you open the dashboard at, the
``Running on http://...`` line it prints when it starts:

.. code-block:: bash

    DASHBOARD=http://10.36.1.20:43219

Add an image layer
------------------

``path`` is a zarr or n5 array or multiscale group, or a ``precomputed://``
URL, opened the same way as the raw dataset. ``shader`` and ``blend`` are
optional and passed to Neuroglancer as they are. A layer with the same name is
replaced.

.. code-block:: bash

    curl -X POST "$DASHBOARD/api/viewer/add-image-layer" \
         -H 'Content-Type: application/json' \
         -d '{"path": "/path/to/dataset.zarr/em/fibsem-uint8",
              "name": "em", "blend": "additive"}'

Add a segmentation layer
------------------------

The same, for label volumes. ``disable_meshes`` turns off Neuroglancer's
on-the-fly meshes, which can take a lot of the dashboard's memory for a large
label volume.

.. code-block:: bash

    curl -X POST "$DASHBOARD/api/viewer/add-segmentation-layer" \
         -H 'Content-Type: application/json' \
         -d '{"path": "/path/to/segmentation.zarr/mito",
              "name": "mito", "disable_meshes": true}'

Rename a layer
--------------

The layer keeps its place in the layer list. It answers 404 when
``old_name`` isn't a layer and 409 when ``new_name`` already is one.

.. code-block:: bash

    curl -X POST "$DASHBOARD/api/viewer/rename-layer" \
         -H 'Content-Type: application/json' \
         -d '{"old_name": "mito", "new_name": "mito (v2)"}'

Remove a layer
--------------

Removing a layer that isn't there is not an error; the answer says
``"removed": false``.

.. code-block:: bash

    curl -X POST "$DASHBOARD/api/viewer/remove-layer" \
         -H 'Content-Type: application/json' \
         -d '{"name": "mito (v2)"}'

A layer's shader settings go with it when it is renamed and are dropped when
it is removed, so a new layer with the old name starts from the defaults.

The routes open any path the dashboard's user can read, as the dashboard's
"set data" does, and anyone who can reach the dashboard can call them.
