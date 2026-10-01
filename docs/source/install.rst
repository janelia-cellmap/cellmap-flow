Installation
============

To install CellMapFlow, you can use pip:

.. code-block:: bash

   pip install cellmap-flow

Note that the basic installation does not include DaCapo and BioImage.io core dependencies. To install CellMapFlow with DaCapo support, use the following command:

.. code-block:: bash

   pip install cellmap-flow[dacapo]

To install CellMapFlow with BioImage.io support, use the following command:

.. code-block:: bash

   pip install cellmap-flow[bioimage]

To install CellMapFlow with both DaCapo and BioImage.io support, use the following command:

.. code-block:: bash

   pip install cellmap-flow[dacapo,bioimage]

Deployment with pixi
--------------------

The repository carries a ``pixi.toml`` and lockfile describing the environment
cellmap-flow is deployed from, including what PyPI cannot provide: the CUDA
toolkit, and the MinIO server and client that annotation painting needs.
From a checkout:

.. code-block:: bash

   pixi install          # the default environment: dashboard, catalog and cellpose models, finetuning
   pixi run cellmap_flow view -d /path/to/dataset.zarr

``runnables.yaml`` registers the same commands as Fileglancer apps. Inside a
pixi environment, inference servers submitted to the cluster start with
``pixi run cellmap_flow serve`` (``CELLMAP_FLOW_SERVER_COMMAND``), so they run
from the same lockfile. :doc:`cli` lists the commands.
