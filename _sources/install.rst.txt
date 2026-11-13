Installation
============

With pixi (recommended)
-----------------------

`pixi <https://pixi.sh>`_ installs cellmap-flow with everything it needs,
including what PyPI cannot provide (the MinIO server and client that
annotation painting needs), and the separate environments some model families
run in. It needs pixi 0.68 or later, which reads this lockfile's format: check
with ``pixi --version``, and update with ``pixi self-update``, or ``brew
upgrade pixi`` if Homebrew installed it. ``pixi install`` sets up the default
environment (the dashboard, the catalog's models and finetuning), and ``pixi
run`` runs a command in it:

.. code-block:: bash

   git clone https://github.com/janelia-cellmap/cellmap-flow.git
   cd cellmap-flow
   pixi install
   pixi run cellmap_flow view -d /path/to/dataset.zarr

It installs on Linux (x86-64) and on Apple-silicon Macs. On a Mac:

- **Compiler:** install Apple's command-line tools first (``xcode-select
  --install``). The neuroglancer fork used for painting annotations is compiled
  during ``pixi install``.
- **Model servers:** a Mac has no cluster to submit jobs to, so the dashboard
  runs each model's server on the Mac itself. Models serve and finetune on
  Apple's GPU, much more slowly than on the cluster: Cellpose-SAM takes about
  35 s for a chunk of 8 slices, so open one x–y panel rather than four, and for
  Cellpose consider ``slices_per_chunk: 1``.
- **CUDA:** it is Linux-only.

Model families whose dependencies clash with the default environment
(fly_organelles checkpoints, BioImage Model Zoo models, Cellpose 4) run in a
pixi environment of their own, chosen by the model's type; nobody picks one by
hand. Each is installed from the lockfile the first time a model needs it,
before that model's job is submitted (several minutes, once; the dashboard's
Job Logs shows pixi's output meanwhile), so a new user, Fileglancer's
included, needs nothing beforehand. ``pixi run cellmap_flow envs install
<name>`` installs one ahead of time, and ``cellmap_flow envs`` lists them and
what is installed. See :ref:`model-env`.

``runnables.yaml`` registers the same commands as Fileglancer apps. Inside a
pixi environment, inference servers submitted to the cluster start with
``pixi run cellmap_flow serve`` (``CELLMAP_FLOW_SERVER_COMMAND``), so they run
from the same lockfile. :doc:`cli` lists the commands.

With pip
--------

.. code-block:: bash

   pip install cellmap-flow

The basic installation does not include DaCapo's or BioImage.io's
dependencies; the extras add them, one or both:

.. code-block:: bash

   pip install "cellmap-flow[dacapo]"
   pip install "cellmap-flow[bioimageio]"
   pip install "cellmap-flow[dacapo,bioimageio]"

To get an unreleased branch without cloning:

.. code-block:: bash

   pip install "cellmap-flow @ git+https://github.com/janelia-cellmap/cellmap-flow.git"

Append ``@<branch>`` to the URL for a specific branch, and add extras in
brackets after the name, e.g. ``"cellmap-flow[dacapo] @
git+https://github.com/janelia-cellmap/cellmap-flow.git@my-branch"``. For
development, clone and install in editable mode instead (``pip install -e .``
in the checkout).

Finetuning
----------

Interactive finetuning needs two dependencies that cannot come from PyPI, a
MinIO server (conda-forge only) and a Neuroglancer fork with voxel-annotation
support, so ``pip install "cellmap-flow[finetune]"`` on its own is not enough;
pixi installs both. See ``docs/finetuning.md`` in the repository for the full
setup.

Any model whose network is plain PyTorch can be finetuned from the dashboard,
with LoRA or in full: cellmap and Hugging Face exports, fly checkpoints, DaCapo
runs, scripts, BioImage Model Zoo models with PyTorch weights, and Cellpose (on
its flows, from sparsely painted instances). A compiled (TorchScript) zoo model
is finetuned in full only, and ONNX or TensorFlow ones not at all; the Finetune
tab offers only what the selected model allows.
