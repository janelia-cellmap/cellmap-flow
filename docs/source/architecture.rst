Architecture
============

This page is for whoever maintains cellmap-flow. It says what each module is
for, how the main requests travel through them, where state is kept, and the
conventions a change has to keep. It describes the package as the 0.3.0
cleanup (PR #103) leaves it. Each module's docstring says more about it, and
``CHANGELOG.md`` at the repository root says what changed and why.

Paths below are relative to ``cellmap_flow/``.

.. contents:: On this page
   :local:
   :depth: 1


Module map
----------

Reading data and running a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it is for
   * - ``io/``
     - Reading datasets. It imports nothing from the rest of the package (no
       ``globals``, Flask, neuroglancer or torch). Import the submodule you
       need:

       - ``paths``: where a container ends, and which format a path is;
       - ``metadata``: ``read_array_meta``, one ``ArrayMeta`` for zarr v2
         and v3, N5 and precomputed; ``list_levels`` for a pyramid;
       - ``multiscale``: the pyramid level for a voxel size, and
         ``closest_raw_scale``;
       - ``geometry``: ``Grid`` and ``Box``, world nanometres to voxels and
         back; ``list_populated_chunks``;
       - ``source``: ``open_array`` and ``read_padded``, tensorstore reads
         that pad what lies outside the array;
       - ``ome``: OME-NGFF translation to corner and back, and the
         ``multiscales`` attributes writers use.
   * - ``image_data_interface.py``
     - ``ImageDataInterface``: a dataset read in world coordinates through the
       input chain. It picks the level, keeps the grid and the array, and pads
       reads. Model scripts use it, so its path is public.
   * - ``norm/``
     - ``input_normalize``: the input normalizers (``InputNormalizer``
       subclasses) and ``SerializableInterface``. ``safe_expression``: the
       whitelist that Lambda normalizer and postprocessor expressions are
       checked against.
   * - ``post/``
     - ``postprocessors``: the postprocessors (``PostProcessor`` subclasses).
   * - ``pipeline_spec.py``
     - ``PipelineSpec``: the normalization and postprocessing chain as data,
       with a reader for every form it travels in (layer URL blob, YAML
       ``json_data``, finetune manifest, pipeline-builder nodes), its
       ``digest()`` and ``build()``. Also ``chain_output_dtype``,
       ``chain_num_channels`` and ``chain_is_segmentation``.
   * - ``process_chain.py``
     - ``process_chain()``: the process's chain, its live steps and their
       configured ``spec``, changed only by ``set()``. See `Where state lives`_.
   * - ``inference/runner.py``
     - ``ModelRunner``: a model on its device, the warmup forward (which
       checks the declared shapes, probes the output range and decides on
       fp16), and ``predict`` of an output region as ``(C, *spatial)``, with
       no postprocessing and no process-wide state. ``DeviceSlots``: how many
       chunks may use the device at once, in arrival order.
   * - ``inferencer.py``
     - ``Inferencer``: a ``ModelRunner`` that also applies a chain
       (``process_chunk``). ``predict`` and ``apply_postprocess`` stay
       importable here for model scripts.

Models
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it is for
   * - ``models/configs/``
     - One module per model type. ``base`` has ``ModelConfig`` (its
       ``config``, ``geometry``, ``to_dict()`` and ``command``) and
       ``Config``; then ``script``, ``dacapo``, ``fly``, ``bio``,
       ``cellmap``, ``finetune`` and ``huggingface``.
   * - ``models/models_config.py``
     - The classes' public import path, which the docs and plugins use, and
       ``Config``.
   * - ``models/registry.py``
     - The model types (every ``ModelConfig`` subclass, a plugin's included)
       and how a config is built from a YAML entry, command-line strings or
       the dashboard's form: ``model_types``, ``build_model(s)``,
       ``coerce_cli_args``, ``coerce_form_params``, ``click_options``,
       ``instantiate_model_config``, ``describe_types``.
   * - ``models/geometry.py``
     - ``ModelGeometry``: a model's shapes and voxel sizes, read once from its
       config, with the context and block shape derived.
   * - ``models/geometry_cache.py``
     - A model's geometry without building the model: from a running
       server's ``model_info``, else a cache file, else a build.
   * - ``models/hf_catalog.py``
     - The cellmap models on the Hugging Face Hub, for the Models tab.
   * - ``models/model_merger.py``
     - Merging several models' outputs in a blockwise run (``model_mode``).
   * - ``models/models.yaml``
     - The Models tab's catalog.
   * - ``models/model_registry.py``
     - A deprecated alias; see `Deprecated in 0.3.0`_.

Serving and jobs
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it is for
   * - ``server.py``
     - ``CellMapFlowServer``, the inference server: one model's output served
       as a zarr, plus ``/__control__/model_info`` and
       ``/__control__/restart``. It prints its address and writes the ready
       file.
   * - ``serving/launch.py``
     - The one place a server's command line is built (``server_argv``,
       ``server_argv_for``, ``server_command``).
   * - ``serving/protocol.py``
     - What a layer URL carries to a server: ``ARGS_KEY``,
       ``encode_to_str``, ``decode_to_json``, ``split_dataset_url``; and
       ``IP_PATTERN``, the markers a server prints its address between.
   * - ``serving/virtual_zarr.py``
     - The served zarr's metadata and chunk-key arithmetic, with no Flask.
   * - ``serving/client.py``
     - ``fetch_model_info``: ask a running server about its model.
   * - ``serving/probe.py``
     - What a model's output range says about the postprocessing it needs.
   * - ``serving/restart_token.py``
     - The secret a finetune job's server takes a restart with.
   * - ``jobs/``
     - Running jobs on LSF or on this machine, without importing ``globals``,
       Flask or torch at module level:

       - ``spec``: ``JobSpec``, ``Job``, ``JobStatus``, ``JobStartError``;
       - ``lsf``: ``bsub_argv`` (the one place a bsub line is built),
         ``submit``, ``LSFJob``, ``statuses`` (one bjobs call for many jobs);
       - ``local``: ``LocalJob``, for when there is no LSF;
       - ``queues``: which GPU queues are usable, and the order to try them;
       - ``site``: ``SiteProfile``, the cluster's numbers (queues, cores,
         walltime, timeouts); only ``JANELIA`` exists;
       - ``settings``: ``launcher_settings()``, the settings saved in
         ``~/.cellmap_flow/server_config.yaml`` (queue, charge group,
         walltime, ...), with their defaults;
       - ``ready``: the file a server writes once it knows its address;
       - ``launch``: ``start_hosts`` (the policy over the others),
         ``submit_bsub_job``, ``started_jobs``, ``install_cleanup_handlers``,
         ``SERVER_COMMAND`` and ``SERVER_LOG_DIR``.

The viewer and the dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it is for
   * - ``viewer/``
     - The neuroglancer viewer. It never imports or starts the dashboard;
       callers pass what a layer needs. (The raw layer is read through
       ``ImageDataInterface``, and so through the process's input chain,
       ``process_chain()``.)

       - ``bootstrap``: ``new_viewer()``;
       - ``layers``: ``prediction_layer``, ``prediction_source``,
         ``prediction_shader_for``, ``prediction_voxel_override`` and
         ``raw_layer``, which every path that shows a layer builds it with;
       - ``raw``: ``get_raw_layer`` (a pyramid as one ``ScalePyramid``), and
         the shaders.
   * - ``dashboard/app.py``
     - The Flask app: its blueprints, the log panel's handler, and
       ``create_and_run_app``.
   * - ``dashboard/state.py``
     - ``Session`` and ``get_session()``; see `Where state lives`_.
   * - ``dashboard/requests.py``
     - The typed bodies of the POST routes (pydantic), and ``parse``, which
       answers a bad body with a 400 ``{"success": false, "error"}`` before
       anything changes.
   * - ``dashboard/progress.py``
     - How far a long request has got (a YAML crop import, a resume), under
       the ``load_id`` the page polls with.
   * - ``dashboard/finetune_layers.py``
     - The job manager's listener: a finetuned model's viewer layer, and its
       ``FinetuneModelConfig`` among the session's models.
   * - ``dashboard/finetune_utils.py``
     - Hands the session's MinIO state and volume records to
       ``finetune.session``'s MinIO and sync functions.
   * - ``dashboard/services/``
     - ``launch``: start the Models tab's picks and show them.
       ``startup``: ``generate_neuroglancer_url``, the CLIs' last step (the
       viewer with their models, then the dashboard).
   * - ``dashboard/routes/``
     - One blueprint per area:

       - ``index_page``: ``/`` and ``/api/set-data``;
       - ``pipeline``: ``PUT /api/pipeline``, its two deprecated aliases,
         ``/api/blockwise-config``, ``/update/equivalences``;
       - ``pipeline_builder_page``: ``/pipeline-builder``;
       - ``models``: the model form, the catalog and Hugging Face picks, the
         saved settings, GPU queues, job logs, ``/api/export-config``;
       - ``model_advice``, ``blockwise``, ``bbx_generator`` (the box tool),
         ``logging_routes`` (the log stream), ``review_routes``;
       - ``finetune/``: ``training``, ``annotation_core``,
         ``annotation_sessions``, ``yaml_crops``, ``overlay``,
         ``good_regions``, ``instance_correction`` and ``layers``, all on the
         blueprint in ``finetune/blueprint.py``, with shared helpers in
         ``finetune/common.py``.
   * - ``dashboard/templates/``, ``dashboard/static/``
     - The two pages; below.

The pages are ES modules served as they are, with no bundler:

.. code-block:: text

   dashboard/templates/
     index.html               the dashboard page; includes _dashboard.html and the
                              _input_tab, _output_tab, _models_tab, _finetune_tab
                              and _review_tab partials
     pipeline_builder_v2.html the pipeline builder
     bbox_json_template.html
   dashboard/static/css/      tokens.css (the palette), dark.css, pipeline_builder.css
   dashboard/static/js/
     lib/                     api, dom, page-data, poll, sortable, sse: know no page
     shared/                  gpu-queues, op-chain, server-config: used by both pages
     dashboard/               main.js (the page's entry), connect, models-tab,
                              model-advice; finetune/ and review/, each an index.js
                              that wires one module per part of its tab
     pipeline-builder/        main.js (the entry), state, canvas, nodes, palette,
                              io, blockwise, bbx, dialogs, log-panel, messages,
                              model-config-modal, output-channels
     vendor/js-yaml.js

A template hands its scripts data as JSON, in
``<script type="application/json" id="page-data">``, which
``lib/page-data.js`` reads. Nothing is put on ``window`` and no markup has an
inline handler. Build DOM with ``lib/dom.js``'s ``h()`` rather than HTML
strings, so that names from a file or a server never run as script.

Blockwise and finetuning
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it is for
   * - ``blockwise/``
     - Running models over a whole volume, to disk, with daisy.
       ``blockwise_processor``: ``CellMapFlowBlockwiseProcessor`` (the master
       plans the blocks and creates the outputs; each worker runs blocks),
       ``precheck`` and ``spawn_worker``. ``cli``: ``cellmap_flow
       blockwise``, task YAMLs in turn as master or, with ``--client``, one
       as worker. ``multiple_cli``: the ``python -m`` path the dashboard's
       blockwise tab runs, the same command.
   * - ``finetune/finetune_cli.py``
     - The training job's entry point, ``python -m
       cellmap_flow.finetune.finetune_cli``. The job manager runs this path,
       and a running job outlives a dashboard upgrade, so its path, flags and
       output change only compatibly.
   * - ``finetune/cli.py``
     - The trainer's flags (``build_arg_parser``, ``parse_args``), the model
       and target they describe, and ``RESTARTABLE_ARGS`` with
       ``apply_restart_params``: what a restart may change.
   * - ``finetune/session_loop.py``
     - ``TrainingSession``, the job's loop: train, export, serve, wait for a
       restart, reset, again. ``RestartController``.
   * - ``finetune/run_outputs.py``
     - What each iteration leaves: ``iterations/NNN_<ts>/``, the symlink to
       the newest export, and the serving YAML in the session's ``models/``.
   * - ``finetune/markers.py``
     - The lines the trainer prints for the job manager to find in its log.
   * - ``finetune/adaptation.py``
     - ``LoraStrategy`` and ``FullStrategy``: everything that differs between
       a LoRA adapter and a full finetune (prepare, teacher, reset, restart,
       checkpoint, export, merge).
   * - ``finetune/losses.py``, ``finetune/target_transforms.py``
     - The masked losses; annotations (0 unannotated, 1 background, 2 and up
       objects) as training targets and masks.
   * - ``finetune/lora_trainer.py``, ``finetune/lora_wrapper.py``
     - ``LoRAFinetuner``, the training loop; wrapping a model's layers with
       PEFT adapters.
   * - ``finetune/model_loading.py``
     - ``load_trainable_model``, which the trainer and
       ``FinetuneModelConfig`` share, so a model is trained and served as the
       same module tree.
   * - ``finetune/data/``
     - What the trainer trains on: ``dataset`` (``VirtualPatchDataset``),
       ``sampler`` (the dense, sparse and rehearsal pools), ``reader``,
       ``augment``, ``loader``.
   * - ``finetune/session/``
     - A finetune session on disk, with no Flask, neuroglancer, torch or
       ``globals``; the dashboard passes its state in. ``manifest``
       (``_virtual_sources.json``, ``good_regions.json``), ``volume``
       (annotation volumes), ``store`` (``SessionStore``), ``minio``
       (``MinioServer``), ``sync``, ``instance`` (instance corrections).
   * - ``finetune/job_manager/``
     - Finetune jobs from the dashboard's side: ``manager``
       (``FinetuneJobManager``, what the routes call), ``state`` (a job's
       record and status transitions), ``submit`` (the command line, and
       launching it), ``persistence`` (``metadata.json``, and finding jobs
       again after a dashboard restart), ``monitor``, ``tailer``,
       ``listener``, ``restart``.
   * - ``finetune/crop_loader.py``, ``finetune/build_corrections.py``
     - The schema of a YAML crops manifest; building a corrections directory
       from one without a dashboard.
   * - ``finetune/export_merged.py``
     - Folding a finetune into the base weights and re-exporting it at a
       larger tile.
   * - ``finetune/finetuned_model_templates.py``
     - The YAML that serves a finetuned model.

Entry points and process-wide modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - What it is for
   * - ``cli/``
     - The ``cellmap_flow`` command (:doc:`cli`). ``main``: the group, its
       ``main()`` (loads the plugins, then runs it), ``dashboard``,
       ``models``, ``plugins`` and the ``finetune`` tools. ``infer``:
       ``infer <type>``, a command per model type built on request, and
       ``run``. ``yaml_cli`` (``yaml``, and ``run_multiple``),
       ``viewer_cli`` (``view``, a viewer and dashboard with no models),
       ``server_cli`` (``serve``, the inference server on its node, and
       ``cellmap_flow_server``), ``doctor``. ``common``: ``--log-level``,
       ``ModelTypeGroup`` and the deprecation notice; ``aliases``: the
       console scripts before 0.3.0. ``blockwise`` is in ``blockwise/``.
   * - ``config/yaml.py``
     - ``load_config`` and ``ConfigError`` for a ``cellmap_flow yaml`` or
       blockwise YAML, and ``resolve_data_path``, the one rule for a model's
       ``data_path`` and ``scale``.
   * - ``plugins.py``
     - Plugins in ``~/.cellmap_flow/plugins/``: registering them, and
       ``load_plugins``, which the commands, the dashboard
       (``create_and_run_app``) and the finetune job call when they start;
       ``import cellmap_flow`` does not. Also
       ``analyze_script``, the safety check a plugin or a model script
       passes.
   * - ``logging_setup.py``
     - ``configure_logging``: the one log format, applied with ``force=True``.
   * - ``globals.py``
     - Deprecated, and gone after 0.3.0: ``g`` (and its type ``Flow``), which
       forwards each name it had to that name's owner with a
       ``DeprecationWarning``. Importing it configures logging, as it always
       has, so nothing in the package imports it. See `Where state lives`_.
   * - ``review.py``, ``review_index.py``
     - The Review tab's SQLite index: reading it, and building one
       (``python -m cellmap_flow.review_index``). See :doc:`review`.
   * - ``utils/``
     - Two deprecated aliases and nothing else.


Request flows
-------------

A chunk served to Neuroglancer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. A prediction layer's source is
   ``zarr://<server>/<model>__CFLOW_ARGS__<blob>__CFLOW_ARGS__``
   (``viewer.layers.prediction_source``). The blob is the layer's chain,
   ``PipelineSpec.to_url_blob(...)``: minified JSON in URL-safe base64
   (``serving.protocol.encode_to_str``). Every request Neuroglancer makes for
   the layer carries it.
#. Neuroglancer asks the model's ``CellMapFlowServer`` for
   ``<dataset>/.zattrs``, ``<dataset>/s0/.zarray``, then chunks
   ``<dataset>/s0/<z>.<y>.<x>[.<c>]``. ``serving.virtual_zarr`` works out
   each answer: the OME attributes, the array's shape and dtype under the
   chain, and the world ROI of a chunk key.
#. ``CellMapFlowServer.chain_for`` finds the blob
   (``serving.protocol.split_dataset_url``), decodes it and builds its steps
   (``PipelineSpec.from_json_data(..., strict=True).build()``), once per URL;
   the last 32 chains are kept. A URL without a blob gets the process's chain,
   ``process_chain()``, which is empty in a server started from the command
   line; the log then warns that the model is fed raw voxel values.
#. ``Inferencer.process_chunk`` points the server's ``ImageDataInterface`` at
   the layer's normalizers (``with_input_norms``) and asks
   ``ModelRunner.predict`` for the chunk. The runner grows the ROI by the
   model's context and reads it with ``ImageDataInterface.to_ndarray_ts``:
   ``io.geometry.Grid.world_to_box`` turns the world ROI into voxels, and
   ``io.source.read_padded`` reads them through the input chain, padding
   what lies outside the array. A config with its own ``process_chunk``
   (TensorFlow, ONNX, cellpose, bioimage.io) runs that instead.
#. The forward runs inside a ``DeviceSlots`` slot: by default one chunk at a
   time on the GPU, first come first served (``CELLMAP_FLOW_GPU_SLOTS``). A
   chunk whose client hung up before its turn is answered 499 without being
   computed.
#. The ``(C, *spatial)`` output goes through the layer's postprocessors
   (``inferencer.apply_postprocess``), is put in the zarr's axis order, cast
   to the dtype the chain declares (``pipeline_spec.chain_output_dtype``) and
   encoded. A postprocessor that merges segments across chunks posts its
   equivalences to ``<dashboard_url>/update/equivalences``, at most every
   5 s.

Submit, or ``PUT /api/pipeline``, and the layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The dashboard page's Submit reads the ticked ops of the Input and
   Postprocess tabs (``static/js/shared/op-chain.js``) and sends
   ``PUT /api/pipeline`` with ``{input_norm, postprocess}``
   (``static/js/dashboard/main.js``). The pipeline builder sends the same,
   and its canvas as ``builder``, 2 s after its last edit
   (``static/js/pipeline-builder/state.js``).
#. ``dashboard/routes/pipeline.put_pipeline`` parses the body as a
   ``PipelineUpdate`` through ``dashboard.requests.parse`` (an unknown op is a
   400), then builds the steps (a value an op's class refuses is a 400).
   Nothing has changed yet.
#. ``_set_chain_and_redraw`` makes it the chain with
   ``Session.set_pipeline``, which sets the live steps and their config
   together, and keeps the canvas if one was sent. With no viewer yet (no
   dataset opened) it stops there.
#. The layers' source is named by the chain's digest,
   ``spec.to_url_blob(dashboard_url=..., digest=spec.digest())``. The same
   chain sent again gives the same URL, so Neuroglancer keeps the chunks it
   has, and each server reuses the chain it built.
#. It keeps the shaders the user set in the viewer, dropping a layer's
   contrast range when its chain changed. Then, in one ``viewer.txn()``, it
   rebuilds the raw layer (``viewer.layers.raw_layer``) and a prediction
   layer for each job that has a host (``viewer.layers.prediction_layer``,
   given the server's ``model_info`` from ``serving.client.fetch_model_info``).
   A chain that ends in labels gives a segmentation layer, and the server's
   ``effective_output_voxel_size`` places the layer.
#. The answer is ``{"success": true, "pipeline", "digest", "layers"}``; see
   :doc:`viewer_api`. The layers' next chunk requests carry the new blob.

Launching a model server
~~~~~~~~~~~~~~~~~~~~~~~~

#. A model config comes from ``cellmap_flow infer <type>`` (``cli/infer.py``:
   a command per type in ``models.registry.model_types()``, its options from
   ``registry.click_options``, its strings through ``coerce_cli_args``), or
   from a YAML (``cli/yaml_cli.py``: ``config.yaml.load_config``,
   ``registry.build_models``). The Models tab's catalog and Hugging Face
   picks (``POST /api/models``, ``dashboard/services/launch.py``) need no
   config for the command, only a type and its arguments.
#. ``serving.launch`` builds the command line: the words of
   ``jobs.launch.SERVER_COMMAND`` (``cellmap_flow serve`` by default), then
   ``--model`` and the model's entry as JSON (``ModelConfig.launch_entry``,
   ``to_dict()`` without None, in ``server_argv``; or the type and
   arguments, in ``server_argv_for``), then ``-d <data path>``.
#. ``jobs.launch.start_hosts`` submits it to LSF with one GPU
   (``submit_bsub_job``, then ``jobs.lsf.submit`` and ``bsub_argv``). When a
   queue does not start the job it tries the next of
   ``jobs.queues.candidates``. The caller gives the queue and charge group
   (the CLIs default to the saved settings); the walltime is the caller's,
   else the saved setting (``jobs.settings.launcher_settings()``), else
   ``jobs.site``'s. It runs the server
   on this machine (``jobs.local``) only when bsub is not installed; when
   bsub is there and every queue fails, it raises ``JobStartError``.
#. It waits for the server's address: from the ready file named in the
   job's environment (``jobs.ready``), else from the
   ``CELLMAP_FLOW_SERVER_IP(...)`` marker in the job's output, through
   bpeek. Then it adds the job to ``jobs.launch.started_jobs()``.
#. On the GPU node, ``cellmap_flow serve --model <entry> -d <data path>``
   (``cli/server_cli.py``) rebuilds the config with ``registry.build_model``,
   as a YAML's model entry is rebuilt, and starts ``CellMapFlowServer``. Its ``Inferencer`` loads the model, and the
   warmup forward checks the declared shapes, probes the output range (for
   ``model_info``) and keeps fp16 only if it agrees with fp32
   (``CELLMAP_FLOW_HALF_PRECISION``). ``run()`` prints the address marker,
   writes the ready file and serves.
#. The launcher adds the model's layer (``viewer.layers.prediction_layer``).
   The CLIs do it in ``dashboard.services.startup.generate_neuroglancer_url``,
   which then serves the dashboard and does not return.

A blockwise run
~~~~~~~~~~~~~~~

#. The pipeline builder's blockwise steps post to ``/api/blockwise/validate``,
   ``/generate``, ``/precheck`` and ``/submit`` (``dashboard/routes/blockwise.py``).
   Each answers 200, with ``valid`` or ``success`` false and an ``error`` when
   it cannot go on. Generate writes the task YAML (one per box when each box
   gets its own zarr) under ``blockwise_tasks_dir``; precheck is
   ``blockwise_processor.precheck``, which loads no model and writes
   nothing. Submit sends the master to LSF as a CPU job:
   ``python -m cellmap_flow.blockwise.multiple_cli <task YAMLs>``. From a
   shell, ``cellmap_flow blockwise <task.yaml>`` runs one master.
#. The master, ``CellMapFlowBlockwiseProcessor(yaml, create=True)``, reads
   the YAML (``config.yaml.load_config``, ``registry.build_models``,
   ``resolve_data_path``), builds no ``Inferencer`` (it needs the geometry
   only), and creates the output arrays, with OME attributes from
   ``io.ome.singlescale_attrs``, on a grid that starts at the corner of the
   raw level the model reads.
#. ``run()`` makes one daisy task per region (the whole extent, or each
   bounding box), with the id ``predict_<model>_<task>``. Daisy spawns the
   workers through ``spawn_worker``: each is an LSF job, submitted with
   ``jobs.launch.submit_bsub_job``, that runs
   ``cellmap_flow blockwise <task.yaml> --client``, with a walltime and its
   own log.
#. A worker builds an ``Inferencer`` per model, takes blocks from daisy and
   runs ``Inferencer.process_chunk`` on each, merging several models with
   ``models.model_merger``. With ``track_progress`` it writes a marker per
   finished block under ``tmp_dir``, and a resumed run skips the blocks that
   have one.
#. The master exits 1 if any block failed or never ran.

A finetune job
~~~~~~~~~~~~~~

#. The Finetune tab posts ``/api/finetune/submit``.
   ``dashboard/routes/finetune/training.py`` parses it
   (``requests.FinetuneSubmit``), brings the session's
   ``_virtual_sources.json`` up to date, settles the loss and target type
   (painted, sparse annotations change them), has the viewer's listener
   follow the manager (``finetune_layers.follow_jobs``), and calls
   ``FinetuneJobManager.submit_finetuning_job``.
#. The manager (``finetune/job_manager/manager.py``) works out what to run
   with ``submit`` (the model type, checkpoint, corrections, channels and
   voxel sizes), makes the run's directory ``<session>/runs/<model>_<ts>/``
   with its ``restart_token`` and ``metadata.json`` (``persistence``), and
   launches ``submit.build_command``'s line with ``submit.launch``: on LSF
   with one GPU, or here without bsub. The line runs
   ``python -m cellmap_flow.finetune.finetune_cli`` and pipes it through
   ``tee`` into the run's ``training_log.txt``. A monitor thread follows the
   job.
#. The trainer (``finetune_cli.main``) parses its flags (``finetune.cli``),
   loads the model (``model_loading.load_trainable_model``), prepares it
   (``adaptation``), and runs ``session_loop.TrainingSession``. Each
   iteration trains on ``finetune.data``'s patches with ``lora_trainer``,
   exports into ``iterations/NNN_<ts>/`` and writes a serving YAML
   (``run_outputs``), and prints ``FINETUNED_MODEL_YAML`` and
   ``TRAINING_ITERATION_COMPLETE`` (``markers.emit``). Without
   ``--auto-serve`` the job ends there. With it, the job starts a
   ``CellMapFlowServer`` on the model it trained, in the same process (its
   address marker goes into the same log), prints ``WAITING_FOR_RESTART``, and
   waits for ``/__control__/restart`` or a ``restart_signal.json``.
#. Every 3 s the monitor (``job_manager/monitor.py``) asks the scheduler
   about the job (``state.on_scheduler_status``) and reads the log's new
   lines (``tailer.LogTailer``): epochs and loss, status markers
   (``state.on_status_marker``), the server's address and each finished
   iteration. It records each change in ``metadata.json``. A job the
   scheduler says has completed is COMPLETED only once its export is found
   (``monitor.complete_job``), and FAILED if it is not.
#. The monitor tells the listeners ``on_server_ready`` and
   ``on_iteration_complete``. The dashboard's ``FinetuneLayerListener``
   (``dashboard/finetune_layers.py``) answers with the model's layer, served
   by the job's server, and a ``FinetuneModelConfig`` among the session's
   models, so the pipeline builder offers it.
#. The tab polls ``/api/finetune/job/<id>/status`` and streams
   ``.../logs/stream`` (server-sent events, ``static/js/lib/sse.js``), and
   stops at a final status. A restart (``.../restart``) syncs the
   annotations from MinIO, then ``job_manager/restart.py`` POSTs the new
   settings to the job's ``/__control__/restart`` with the run's token in
   ``X-Restart-Token``, or writes them to ``restart_signal.json`` when the
   server can't be reached. A dashboard started later finds a session's jobs
   again from their ``metadata.json`` and one bjobs call
   (``persistence.rehydrate``).

Annotation volumes and MinIO sync
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. New Volume posts ``/api/finetune/create-volume``
   (``routes/finetune/annotation_core.py``). The model's geometry comes from
   ``models.geometry_cache.resolve_model_geometry``: its running server,
   else the cache, else a build. ``finetune.session.volume.plan_volume`` sizes
   the volume over the whole raw dataset, one chunk per model output, and
   ``create_volume_zarr`` writes ``<id>.zarr`` into the session's
   ``corrections/``.
#. ``dashboard.finetune_utils.ensure_minio_serving`` calls
   ``finetune.session.minio.MinioServer.ensure_serving``. It starts
   ``minio server`` if none is running, with the bucket ``annotations`` open
   to anonymous access and the periodic sync thread. Then it pulls any
   chunks painted since the last sync, and mirrors the volume up with
   ``mc mirror``. The volume is registered in the session's
   ``annotation_volumes`` (``session.store.SessionStore``), and its
   ``_virtual_sources.json`` is written (``session.volume.build_manifest``,
   ``session.manifest.write_manifest``); the trainer finds the volume only
   through that file.
#. The page then posts ``/api/finetune/add-to-viewer``
   (``routes/finetune/overlay.py``): a writable segmentation layer on
   ``s3+<MinIO URL>``, selected, with the painting tools bound. The browser
   paints into MinIO directly.
#. ``finetune.session.sync`` brings the painted chunks back to the volume on
   disk: every 30 s, on Save (``/api/finetune/sync-annotations``), before a
   restart, and before each mirror. It lists the volume's chunks in the
   bucket with their ETags and copies those that changed (through ``.part``
   files), recording what it saw in the volume's ``chunk_sync_state``.
   Nothing on disk is deleted because MinIO lacks it.
#. The trainer reads the volume on disk, never MinIO. Volumes are also
   filled by a YAML crop import (``/api/finetune/load-crops``,
   ``session.volume.write_crop_into_volume``), resumed from an earlier
   session (``annotation_sessions``), seeded from a segmentation
   (``session.instance``), or built without a dashboard
   (``python -m cellmap_flow.finetune.build_corrections``).


Where state lives
-----------------

Its owners
~~~~~~~~~~

Four objects hold what a process shares. Each is read through a function, at
call time and never into a module global, because ``tests/conftest.py``
swaps in fresh ones for every test (``_fresh_process_state``).

- ``jobs.settings.launcher_settings()``: the launcher settings, one attribute
  per key of ``SERVER_CONFIG_DEFAULTS`` (``queue``, ``charge_group``,
  ``walltime``, ...), loaded from ``~/.cellmap_flow/server_config.yaml`` the
  first time the process asks, and ``save()``. The CLIs and
  ``/api/server-config`` write them; ``start_hosts``, blockwise and the
  dashboard's finetune submit read them.
- ``process_chain.process_chain()``: the process's chain. ``input_norms``
  and ``postprocess`` are the live steps, which can hold state;
  ``input_norm_config`` and ``postprocess_config`` the steps as the dashboard
  received them; ``spec`` the chain as data, from the configs or else the
  live steps; ``set(spec, built=None)`` the one way to change it. In the
  dashboard it is the chain the user submitted. In a server or a blockwise
  worker it is the fallback for a layer URL or model script that gives no
  chain.
- ``jobs.launch.started_jobs()``: the servers this process started, which
  ``start_hosts`` appends to, ``cleanup_handler`` kills, and
  ``serving.client.running_job_host`` looks a model's host up in.
- ``dashboard.state.get_session()``: the dashboard's own state (below).

The dashboard's ``Session``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dashboard code reads and writes its state through
``dashboard.state.get_session()``, a module registry rather than Flask's
``app.extensions``: the CLIs fill it before ``dashboard.app`` is imported,
and launch threads, job monitors, the MinIO sync thread and the log panel's
handler run without an app context. It holds:

- the dataset and the viewer: ``dataset_path``, ``viewer``, ``raw``,
  ``shaders``, ``shader_controls``, ``extra_layers``, ``neuroglancer_url``;
- the models: ``models_config``, ``model_catalog``;
- the pipeline builder's last apply: ``builder_state``,
  ``builder_model_configs``;
- blockwise's ``tmp_dir`` and ``blockwise_tasks_dir``, and ``tasks_dir()``;
- the finetune and review tabs: ``minio_state``, ``annotation_volumes``,
  ``output_sessions``, ``finetune_job_manager`` (made when first asked for),
  ``review``;
- the log panel's ``log_buffer`` and ``log_clients``, and the box tool's
  ``bbx_generator_state``.

It forwards the rest to the owners above: each setting, ``settings``,
``server_config`` and ``save_server_config()``; ``input_norms`` and
``postprocess`` (read-only), ``pipeline_spec`` and ``set_pipeline(spec)``;
and ``jobs``, whose assignment replaces the started list's contents. It has
``__slots__``, so a misspelt attribute raises instead of being stored.

The deprecated ``g``
~~~~~~~~~~~~~~~~~~~~

``cellmap_flow.globals.g`` stays for one release, for scripts and plugins.
Each name it had forwards to its owner with a ``DeprecationWarning`` that
names the replacement. The warnings are silent in the servers and the
dashboard, whose code is not ``__main__``, and show under pytest.

Nothing in the package imports ``globals``, and ``test_import_targets``
checks it. Importing ``globals`` also configures logging, so a module that
imported it reset a CLI's ``--log-level`` to INFO whenever the CLI imported
that module after parsing the flag, as ``cellmap_flow_server`` imports the
server. New code uses the owners:

- no ``import cellmap_flow.globals``, and no ``g.x``;
- dashboard code reads state through ``get_session()``;
- everything else takes what it needs as arguments, as ``finetune.session``
  does: its MinIO and sync functions are given the dashboard's
  ``minio_state`` and volume records, and keep none of their own.

On disk
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Where
     - What
   * - ``~/.cellmap_flow/server_config.yaml``
     - The saved job settings: read by ``launcher_settings()`` the first time
       a process asks, and by ``config.yaml.load_config``.
   * - ``~/.cellmap_flow/plugins/``
     - Registered plugins.
   * - ``~/.cellmap_flow/server_logs/``
     - Inference-server job logs and ready files (``jobs.launch.SERVER_LOG_DIR``).
   * - ``~/.cellmap_flow/model_geometry_cache.json``, ``~/.cellmap_flow/hugging_face/``
     - The geometry cache and the Hugging Face listing.
   * - ``~/.cellmap_flow/blockwise_tasks/``, ``~/.cellmap_flow/blockwise_tmp/``
     - The dashboard's blockwise task YAMLs and master logs, and its default
       ``tmp_dir`` for progress markers.
   * - ``~/.cellmap_flow/user_prefs.json``, ``~/.cellmap_flow/corrections/``
     - The Finetune tab's saved form, and the corrections directory used when
       no output path is given.
   * - ``<base>/<YYYYmmdd_HHMMSS>/``, a finetune session
     - ``corrections/`` (the volumes, ``_virtual_sources.json``, MinIO's
       ``.minio``), ``good_regions.json``, ``runs/<model>_<ts>/`` (per job:
       ``metadata.json``, ``training_log.txt``, ``restart_token``,
       ``iterations/``), and ``models/`` (the serving YAMLs).

The session layout, ``metadata.json``, the markers and the annotation volume
format are read by dashboards older and newer than whatever wrote them, so
they change only compatibly.


Conventions
-----------

An OME translation is voxel 0's centre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OME-NGFF, and Neuroglancer, put ``translation`` at the centre of voxel 0.
Everything inside cellmap-flow works with the lower corner of voxel 0
(``ArrayMeta.translation``, ``io.geometry.Grid``). Readers convert with
``io.ome.ome_corner`` and writers with ``io.ome.ome_translation`` (or
``multiscales_attrs`` and ``singlescale_attrs``, which use it); no other code
adds or subtracts half a voxel. Janelia pyramids have
``translation = scale/2 − 4`` nm, so every level's corner is at −4 nm. The
served zarr and the blockwise output start at the corner of the raw level the
model reads. The legacy
``resolution``/``offset`` and N5 ``transform`` attributes keep their old
meaning. An annotation volume's root attribute ``dataset_offset_nm`` is voxel
0's centre, and ``session.volume.volume_corner_nm`` gives the corner.

The chain format
~~~~~~~~~~~~~~~~

A chain is ``{"input_norm": [...], "postprocess": [...]}``, each an ordered
list of steps, each step a flat ``{"name": <op class>, **constructor
arguments}``. It is the same in layer URL blobs, exported YAMLs, blockwise
task YAMLs' ``json_data`` and finetune manifests, and it is read and written
through ``pipeline_spec.PipelineSpec``. A step is never nested as
``{"name", "params"}`` on the wire, because older servers pass every key but
``name`` to the constructor. Values are kept as given: the dashboard's forms
send strings and the constructors parse them. Files may still hold the older
``{Name: {params}}`` dict, which the readers accept (it can't hold the same
op twice); ``PUT /api/pipeline`` refuses it.

"Behaviour change:" commits
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A change to anything a user, a file or another version of cellmap-flow sees
(served data or metadata, files written, HTTP answers, the CLI, training
results) is its own commit, whose subject starts ``Behaviour change:``. It
carries a test that fails on its parent, and it is listed under
"Behaviour-change commits" in ``CHANGELOG.md``. A refactor leaves every guard
below unchanged, so that a behaviour change can be reviewed, reverted or
dropped on its own.

Tests
~~~~~

- **One test module per public module**, next to its subject in
  ``tests/utils/`` (most of the package, whatever its package),
  ``tests/cli/``, ``tests/finetune/`` or ``tests/blockwise/``, and tested
  through its public interface: nothing on private helpers, and no test that
  would fail only because of an internal rename.
- **Tables with readable row ids**: near-duplicates are one parametrized
  test, each ``pytest.param(..., id="...")`` named for its case, and a test's
  name says the behaviour it protects.
- **Shared fixtures** are in the conftest files. ``tests/conftest.py``
  points ``HOME`` at a temporary directory before anything imports
  cellmap_flow, gives every test fresh owners of the process's state
  (``_fresh_process_state``), restores the root logger after it, and has
  ``raw_zarr``, ``ome_pyramid``, ``write_array``, ``model_script``,
  ``fake_lsf``, ``viewer`` and ``dashboard``. ``tests/finetune/conftest.py``
  and ``tests/blockwise/conftest.py`` have their areas' fixtures.
- **Characterization snapshots are the guards.** Each pins, as literals in
  the test, something another party reads:

  - ``test_cli_surface``: the CLIs' commands and flags, each type's
    ``to_dict()`` and ``command``, and the model form;
  - ``test_url_map``: every dashboard URL and its methods;
  - ``test_served_metadata_snapshot``: what a server answers;
  - ``test_layer_sources_snapshot``: the layers every path puts in the
    viewer;
  - ``test_bsub_argv_snapshot``: every bsub and local command line;
  - ``test_blockwise_routes``: the task YAMLs generate writes;
  - ``test_training_snapshot``, ``test_dataset_draws_snapshot``,
    ``test_volume_snapshot``: training's numbers, markers and layout, the
    patches drawn, and the annotation volume format;
  - the rehydration snapshot in ``test_finetune_job_manager``;
  - ``test_deployability``, ``test_import_hygiene`` and
    ``test_import_targets`` (every ``from cellmap_flow... import`` in the
    package and the tests resolves).

  A refactor starts with the characterization test, which must pass on the
  code before it.
- **Patch where the name is used.** A test that monkeypatches a name patches
  it on the module that calls it, and asserts its fake was called, so a move
  can't turn it into a test of nothing, or of real bsub or ``mc``.
- **Markers**: ``finetune`` (needs peft, skipped without it), ``gpu``,
  ``lsf``, ``minio``, ``network`` (opt in with
  ``CELLMAP_FLOW_NETWORK_TESTS=1``) and ``slow``. CI runs
  ``pytest -m "not gpu and not lsf and not minio and not network"`` with and
  without the ``finetune`` extra.
- **Import hygiene**: what jobs, servers, finetune runs and the dashboard's
  request threads import must not drag in the dashboard or a model. No
  module imports ``globals`` (``test_import_targets``). A new module imports
  Flask, neuroglancer, huggingface_hub and peft only where it needs them,
  never at module level, and torch likewise outside ``finetune/`` and
  ``inference/``. ``test_import_hygiene`` checks it in a fresh interpreter
  per group: ``jobs/`` and the job manager, ``io/``, ``serving/`` (with
  ``models.geometry``, ``geometry_cache``, ``inference.runner``, the
  ``Inferencer`` and ``ImageDataInterface``), the registry (with
  ``config.yaml`` and ``serving.launch``), ``pipeline_spec`` and
  ``process_chain``, the state's owners, ``finetune/session/``, ``review``
  and ``viewer/``; that importing ``server.py`` or ``blockwise/`` leaves
  logging alone; and that ``import cellmap_flow`` writes nothing under
  ``HOME``.

The dashboard's JavaScript
~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository has no JavaScript test runner. CI checks that every file in
``static/js`` parses as an ES module (``node --input-type=module --check``;
plain ``node --check`` passes a module with a syntax error), and pytest
checks what it can without a browser: what the two pages hand their scripts
(``test_dashboard_pages``), that every element id the builder's and the
tabs' scripts look up is on the page (``test_builder_page``,
``test_dashboard_tab_scripts``), and that every static file ships and every
import resolves (``test_static_assets``).

What a page does was checked during the cleanup with a headless-Chrome
harness. It renders each page from the tree with Flask's test client and
loads it in ``chrome-headless-shell`` with the tree's own scripts and styles.
``fetch``, ``EventSource`` and the dialogs are stubbed, timers run on
virtual time, and scripted scenarios drive the page. Each scenario records
every request (method, URL, body, headers and the script that sent it),
every error and alert, and a signature of each rendered element;
``compare.py`` diffs a run on the base against one on the change, and a
refactor must show no difference. The harness is not in the repository: it
is kept with the cleanup's working notes, in
``cleanup_review/tools/fe_harness/`` (``harness.py``, ``compare.py``, the
``*_scenarios.py`` files and ``libcheck/``), with one machine's Chrome and
node paths written into it.

The deployability contract
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fileglancer runs cellmap-flow from this repository with pixi, and
``tests/utils/test_deployability.py`` pins what that needs:

- ``runnables.yaml`` runs ``pixi run <console script> [<subcommand>]`` for
  scripts in ``[project.scripts]``, with flags they accept (``-d`` for
  ``cellmap_flow view``, a positional config for ``cellmap_flow yaml``). A
  change to a command's name or flags changes the manifest in the same
  commit.
- The dashboard binds a free port with a threaded server, prints its URL, and
  writes it to the file named by ``SERVICE_URL_PATH`` when that is set.
- ``jobs.launch.SERVER_COMMAND`` comes from ``CELLMAP_FLOW_SERVER_COMMAND``,
  which pixi's activation sets to ``pixi run cellmap_flow serve``. It is
  read at import and used at call time; no other module keeps a copy of it.
  The value before 0.3.0, ``pixi run cellmap_flow_server``, still takes the
  ``--model`` the launchers pass.
- ``cellmap_flow view`` bills the models it launches to ``LSB_PROJECT_NAME``,
  the Fileglancer job's project, unless ``-P`` is given.
- CI runs ``pixi lock --check``, so a change to the environment is re-locked
  in the same commit.

See :doc:`install` for running from pixi.


Deprecated in 0.3.0
-------------------

Each of these still works in 0.3.0, warns, and goes in the next release.
``CHANGELOG.md`` has the details.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Deprecated
     - Use instead
   * - The separate console scripts (``cellmap_flow_yaml``,
       ``cellmap_flow_view``, ``cellmap_flow_server``, the blockwise scripts,
       ``cellmap_flow_app``), and ``cellmap_flow <type>``, ``run`` and the
       plugin commands
     - The subcommands of the one ``cellmap_flow`` command (K2, K3); servers
       start with ``cellmap_flow serve --model`` (K4). The old names stay
       for one release as aliases that say so. :doc:`cli` lists each one
       and its replacement.
   * - ``POST /api/process`` and ``POST /api/pipeline/apply``
     - ``PUT /api/pipeline`` (K11), which also redraws the layers. The old
       routes answer as they did, log a warning, and send a
       ``Deprecation`` header and a ``Link`` to the new route. See
       :doc:`viewer_api`.
   * - ``cellmap_flow.utils.bsub_utils.install_cleanup_handlers``
     - ``cellmap_flow.jobs.launch.install_cleanup_handlers``
   * - ``cellmap_flow.utils.serialize_config.Config``
     - ``cellmap_flow.models.models_config.Config``
   * - ``cellmap_flow.models.model_registry.list_huggingface_models`` and
       ``refresh_huggingface_models``
     - The same names in ``cellmap_flow.models.hf_catalog``
   * - ``ImageDataInterface(output_voxel_size=...)`` (K18)
     - Read at the dataset's voxel size and resample what is read.
   * - ``ImageDataInterface(custom_fill_value=...)`` (K18)
     - Read within the dataset's ROI and pad what is read.
       ``concurrency_limit`` is not deprecated: the inference server uses it.

Using a name through one of the three alias modules raises a
``DeprecationWarning`` that names its new path;
``tests/utils/test_deprecated_imports.py`` checks that each name still gives
the very same object. These went in 0.3.0 without an alias; nothing in the
docs or the examples used them:

- the rest of ``cellmap_flow.utils``, dissolved into ``io/``, ``jobs/``,
  ``serving/``, ``config/yaml.py``, ``models/``, ``norm/safe_expression``,
  ``dashboard/services/``, ``plugins`` and ``logging_setup``;
  ``python -m cellmap_flow.utils.doctor`` is
  ``python -m cellmap_flow.cli.doctor``;
- ``finetune/finetune_job_manager.py``, now ``finetune/job_manager/``, and
  ``finetune/virtual_dataset.py``, now ``finetune/data/``;
- ``lora_wrapper.merge_lora_into_base`` (K19), replaced by
  ``adaptation.LoraStrategy.merge``.
