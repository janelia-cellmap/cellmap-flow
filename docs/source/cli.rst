Command line
============

CellMap Flow has one command, ``cellmap_flow``, with a subcommand for each
job. ``cellmap_flow <command> --help`` lists a command's options, and
``--log-level`` before the command sets how much it logs (INFO by default):

.. code-block:: bash

    cellmap_flow view -d /path/to/data.zarr
    cellmap_flow --log-level DEBUG yaml config.yaml

Before 0.3.0 most of these were separate programs (``cellmap_flow_yaml`` and
the rest). Those names still work in 0.3.0; see `Deprecated names`_.

Commands
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - What it does
   * - ``infer <type>``
     - Start one model's inference server, then open the viewer on its
       predictions and serve the dashboard. There is a subcommand for each
       model type (``cellmap_flow models`` lists them) whose options are the
       type's constructor arguments: ``cellmap_flow infer script -s model.py
       -d data.zarr/raw``. Short flags go to the arguments in the order the
       constructor takes them, so an argument whose first letter an earlier
       one has has only its long flag (``script``'s ``--scale``). ``-d``,
       ``-q`` (queue), ``-P`` (billing project), ``--resample`` and
       ``--server-check`` are the command's own. ``--resample`` reads the
       data resampled to the model's input voxel size when it has no level
       at that size, instead of reading a level as if it were at that size
       (:ref:`resampling`).
   * - ``yaml CONFIG``
     - The same for every model a YAML file lists, with its normalization
       and postprocessing (:doc:`yaml_config`). ``--validate-only`` checks
       the file; ``--list-types`` lists the model types.
   * - ``view -d DATA``
     - Open a dataset in the viewer and serve the dashboard, where models are
       picked and submitted. ``-P`` bills them to a project; by default it is
       the launching job's own (``LSB_PROJECT_NAME``).
   * - ``dashboard``
     - Serve the dashboard on its own, for a viewer already running
       (``-n`` names the viewer the page embeds).
   * - ``blockwise YAML...``
     - Run a model over a whole volume, block by block, writing its
       predictions to disk (:doc:`yaml_config`). Several YAMLs run one after
       another; the command fails at the end if any left blocks unprocessed.
       ``--client`` runs one worker, which the run submits itself.
   * - ``serve --model ENTRY -d DATA``
     - Serve one model's predictions. The launchers (``infer``, ``yaml``, the
       dashboard) run it on a GPU node for you; see `Starting a server`_.
   * - ``finetune train``, ``export-merged``, ``build-corrections``
     - The finetune tools (``docs/finetuning.md`` in the repository). The
       dashboard's finetune tab runs ``train`` for you; each takes the flags of
       ``python -m cellmap_flow.finetune.<module>``, which ``--help`` lists.
   * - ``models``
     - List the model types and the arguments each takes.
   * - ``plugins register FILE``, ``unregister NAME``, ``list``
     - Manage plugins (:doc:`plugins`).
   * - ``doctor``
     - Check the environment: what is installed, what is missing, and the
       command that fixes each gap. ``--core-only`` skips the finetune checks.

Starting a server
-----------------

A launcher starts each model's server as an LSF job (or a local process
where there is no ``bsub``) with::

    $CELLMAP_FLOW_SERVER_COMMAND --model '<entry>' -d <data path>

The entry is the model as a YAML model entry, in JSON
(``ModelConfig.launch_entry``), and the server builds the model from it as
``cellmap_flow yaml`` builds a YAML's models:

.. code-block:: bash

    cellmap_flow serve -d /path/to/data.zarr/raw \
      --model '{"type": "script", "script_path": "/path/to/model.py", "name": "mito"}'

A launcher asked to resample (``infer --resample``, or ``resample: true``
in a YAML) adds ``--resample``; see :ref:`resampling`.

``CELLMAP_FLOW_SERVER_COMMAND`` is ``cellmap_flow serve`` by default. A
deployment whose environment is not on the compute nodes' ``PATH`` sets it:
the pixi environment sets ``pixi run cellmap_flow serve``
(:doc:`install`). The value before 0.3.0, ``pixi run cellmap_flow_server``,
still works, since ``cellmap_flow_server`` takes ``--model`` too.

Plugins
-------

Every command loads the registered plugins (``~/.cellmap_flow/plugins/``)
when it starts, so a plugin's model type has its ``infer`` and ``serve``
form and its normalizers and postprocessors are offered in the dashboard.
Importing ``cellmap_flow`` does not load them: a script that uses a plugin's
classes calls ``cellmap_flow.plugins.load_plugins()`` first.

Deprecated names
----------------

These still work in 0.3.0 and go in the release after it. Each prints, on
stderr, ```<old>` is deprecated and goes in the release after 0.3.0; use
`<new>`.`` and then does what ``<new>`` does.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Before 0.3.0
     - Since 0.3.0
   * - ``cellmap_flow_yaml``
     - ``cellmap_flow yaml``
   * - ``cellmap_flow_view``
     - ``cellmap_flow view``
   * - ``cellmap_flow_blockwise``
     - ``cellmap_flow blockwise``
   * - ``cellmap_flow_blockwise_multiple A.yaml B.yaml``
     - ``cellmap_flow blockwise A.yaml B.yaml``
   * - ``cellmap_flow_app``
     - ``cellmap_flow dashboard``
   * - ``cellmap_flow_server <type> ...``, ``cellmap_flow_server --model``
     - ``cellmap_flow serve --model`` (launchers before 0.3.0 still start
       servers with the per-type form)
   * - ``cellmap_flow_server list-models``, ``cellmap_flow list-models``
     - ``cellmap_flow models``
   * - ``cellmap_flow <type> ...``
     - ``cellmap_flow infer <type> ...``. The ``finetune`` type is the
       exception: ``cellmap_flow finetune`` is now the finetune tools, so a
       finetuned model is served with ``cellmap_flow infer finetune``.
   * - ``cellmap_flow run -m TYPE -c key=value``
     - ``cellmap_flow infer TYPE --key value``; ``run`` prints the exact
       command its arguments stand for.
   * - ``cellmap_flow register``, ``unregister``, ``list-plugins``
     - ``cellmap_flow plugins register``, ``unregister``, ``list``

Reference
---------

.. click:: cellmap_flow.cli.main:cli
   :prog: cellmap_flow
   :nested: full
