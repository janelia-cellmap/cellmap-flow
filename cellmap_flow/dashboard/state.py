"""The dashboard's state, read and written through one object: ``get_session()``.

One Session per process, made at import and filled by the CLIs,
``services/startup`` and ``create_and_run_app``. It lives in this module
rather than in Flask's ``app.extensions`` because the CLIs fill it before
``dashboard.app`` is imported (they import it lazily so ``--help`` stays
fast); because launch threads, job-monitor listeners, the MinIO sync thread
and the log panel's handler run with no app context; and so that this module
imports without Flask.

The Session forwards three parts to their owners, which also serve processes
that have no dashboard:

- The launcher settings in ~/.cellmap_flow/server_config.yaml, one
  attribute per key of ``jobs.settings.SERVER_CONFIG_DEFAULTS`` (``queue``,
  ``charge_group``, ``walltime``, ...), and ``settings``, ``server_config``,
  ``server_config_cached`` and ``save_server_config()``:
  ``jobs.settings.launcher_settings()``.
- The chain: ``input_norms`` and ``postprocess`` (the live steps, read-only),
  ``pipeline_spec`` (the same as data), and ``set_pipeline()``, the one way
  to change them: ``process_chain.process_chain()``.
- ``jobs``, the running inference servers' jobs: ``jobs.launch.started_jobs()``.

The rest it stores itself, in ``__slots__``, so a misspelt attribute raises
rather than being stored beside the real one:

- The data and the viewer: ``dataset_path``, ``resample``, ``viewer``,
  ``raw``, ``neuroglancer_url``, ``dashboard_url``, ``shaders``,
  ``shader_controls``, ``extra_layers``, ``shown_channels``; and
  ``layer_blob()``, the chain as every prediction layer's URL carries it,
  and ``shown_channel()``, the channel a model's layer opens on.
- The models: ``models_config``, ``model_catalog``, and
  ``start_postprocess``, the postprocessing a model's start set.
- The pipeline builder's last apply: ``builder_state``,
  ``builder_model_configs``.
- Blockwise's ``tmp_dir`` and ``blockwise_tasks_dir``, and ``tasks_dir()``.
- The finetune tab's MinIO and volumes: ``minio_state``, ``label_undo``,
  ``annotation_volumes``, ``output_sessions``; the models its ranking was
  given by reference, ``rank_added``, and what it fitted each model's
  settings to, ``rank_fits``; its AI annotation's
  ``ai_annotate``; its training jobs' ``finetune_job_manager``, and the
  ``restart_key`` that restarts one from the viewer; the Review tab's
  ``review``.
- The log panel's ``log_buffer``, ``log_ids`` and ``log_clients``; the box
  tool's ``bbx_generator_state``.

Read the session through ``get_session()`` when it is needed, never into a
module global: tests/conftest.py installs a fresh Session for every test.
"""

import itertools
import os
import threading
import time
from collections import deque
from importlib.resources import files

import yaml

from cellmap_flow.jobs.launch import started_jobs
from cellmap_flow.jobs.settings import SERVER_CONFIG_KEYS, launcher_settings
from cellmap_flow.process_chain import process_chain

# The builder's node lists, as /api/pipeline/apply stores them.
_BUILDER_KEYS = ("inputs", "outputs", "edges", "normalizers", "models", "postprocessors")

LAYER_BOOKKEEPING = ("shaders", "shader_controls", "extra_layers")


class Session:
    """The dashboard's state; see the module docstring."""

    # _builder_state and _finetune_job_manager hold what the builder_state
    # and finetune_job_manager properties serve: a slot and a property of one
    # name cannot coexist.
    __slots__ = ("dataset_path", "resample", "viewer", "raw", "neuroglancer_url", "dashboard_url", "shaders",
                 "shader_controls", "extra_layers", "shown_channels",
                 "models_config", "model_catalog", "start_postprocess", "_builder_state", "builder_model_configs",
                 "tmp_dir", "blockwise_tasks_dir", "minio_state", "annotation_volumes", "output_sessions", "rank_added",
                 "rank_fits", "review",
                 "_finetune_job_manager", "restart_key", "label_undo", "ai_annotate", "log_buffer", "log_ids",
                 "log_clients", "bbx_generator_state")

    def __init__(self):
        # The data and the viewer
        # The raw data as the user gave it: a multiscale group or one level of it.
        self.dataset_path = None
        # Whether models launched from here, their finetunes and blockwise
        # runs read the data resampled to each model's input voxel size when
        # it has no level at that size (``--resample``, the default), rather
        # than reading the nearest level as if it were at that size. Set by
        # ``cellmap_flow yaml`` (the YAML's ``resample``), ``view
        # --no-resample`` and the Models tab's checkbox; servers already
        # running keep theirs.
        self.resample = True
        # The neuroglancer viewer; None until a dataset is opened.
        self.viewer = None
        # The raw data's layer, as last built.
        self.raw = None
        # The viewer's own address, which the index page embeds.
        self.neuroglancer_url = None
        # The dashboard's own address, by IP (dashboard.app.bind_dashboard
        # sets it); None until the dashboard is bound. The inference servers
        # post a merging postprocessor's merged ids to it (layer_blob).
        self.dashboard_url = None
        # {layer name: shader}: the user's, put back when a layer is rebuilt.
        self.shaders = {}
        # {layer name: shaderControls}, likewise.
        self.shader_controls = {}
        # {name: neuroglancer layer} that the viewer shows beside the raw
        # data: a YAML's extra_layers, built by `cellmap_flow yaml`. The
        # viewer-layer routes drop or rename their entries.
        self.extra_layers = {}
        # {model name: channel index}: the channel the Finetune tab picked to
        # show of a model of several (POST /api/finetune/channel: Use this
        # model's, the training form's Channel), which its layer, and its
        # finetunes', open on (shown_channel).
        self.shown_channels = {}

        # The models
        # The configured models (ModelConfig).
        self.models_config = []
        # {group: {model name: path}}, the Models tab's catalog.
        self.model_catalog = yaml.safe_load(files("cellmap_flow.models").joinpath("models.yaml").read_text()) or {}
        # {"model", "postprocess"}: the postprocessing a model's start set on
        # a chain with none, and the model (routes/pipeline.py
        # set_needed_postprocess), which GET /api/models/loaded says the
        # chain's came from while it is still that; None until a start sets
        # any.
        self.start_postprocess = None

        # The pipeline builder's last apply
        self._builder_state = {key: [] for key in _BUILDER_KEYS}
        # {model name: config}, from every model node the builder applied with one.
        self.builder_model_configs = {}

        # Blockwise
        # Where blockwise tasks keep their progress.
        self.tmp_dir = os.path.expanduser("~/.cellmap_flow/blockwise_tmp")
        # Where blockwise task YAMLs and master logs go.
        self.blockwise_tasks_dir = os.path.expanduser("~/.cellmap_flow/blockwise_tasks")

        # The finetune and review tabs. The MinIO sync thread keeps the
        # minio_state and annotation_volumes dicts it started with, so they
        # are changed in place and never replaced.
        # The dashboard's MinIO: its process, address, bucket and session directory.
        self.minio_state = {
            "process": None,
            "port": None,
            "ip": None,
            "bucket": "annotations",
            "output_base": None,
            "sync_thread": None,
        }
        # {volume id: its record}, the volumes served through MinIO.
        self.annotation_volumes = {}
        # {output base directory: its session directory}.
        self.output_sessions = {}
        # {model name: its candidate's fields}: the models the Finetune tab's
        # ranking was given by reference (POST /api/finetune/rank/add, Add a
        # model's resolve), which every listing of its candidates includes
        # after, until the dashboard stops; nothing is started for them.
        self.rank_added = {}
        # {model name: {"input_norm", "postprocess", "threshold"}}: the
        # settings the ranking's last score of each model fitted to the paint
        # (its row's input normalization, and the postprocessing its layer
        # needs, None when the chain needed none), which Use this model, Fit
        # to Paint and Use these settings set from, and its row's threshold.
        # The model advice does not suggest another input normalization over
        # them, and cuts an affinity model's chain into a segmentation at
        # the threshold; GET /api/models/loaded says the chain's
        # postprocessing was fitted to the paint while it is theirs
        # (routes/model_advice.py, routes/pipeline.py postprocess_set_by).
        self.rank_fits = {}
        # The Review tab's open index (review_routes.ReviewSession), or None
        # until /api/review/open.
        self.review = None
        # {volume id: deque of (lo, hi, before, after)}: what each one-click
        # label action (seed, background, split) replaced, newest last, for
        # the Finetune tab's Undo (routes/finetune/view_labels.py).
        self.label_undo = {}
        # The Finetune tab's AI annotation (routes/finetune/ai_annotate.py):
        # the provider, model, target and prompt the user chose (None until
        # they do); {dataset path: {provider id: the destination the user
        # acknowledged its images go to}}; the one job, running or staged
        # for review (None when there is none); and the lock its worker
        # thread and the routes change the job under. Never a key or other
        # secret: those are read when a call is made.
        self.ai_annotate = {
            "settings": None,
            "acknowledged": {},
            "job": None,
            "lock": threading.Lock(),
        }
        self._finetune_job_manager = None
        # Shift+R over the viewer (routes/finetune/restart_key.py): the job
        # the Finetune tab follows, the last whose status it asked for; {job
        # id: the presses that asked for its restart}, which its status
        # answer carries for the tab to act on; {job id: when the last press
        # that no restart has answered yet was made}; and the lock the key's
        # action, on neuroglancer's event loop, and the routes, in Flask's
        # threads, change these under.
        self.restart_key = {
            "followed": None,
            "requests": {},
            "asked": {},
            "lock": threading.Lock(),
        }

        # The log panel and the box tool
        # The last log lines, for a log panel that connects late, each as
        # (its number, the line).
        self.log_buffer = deque(maxlen=1000)
        # The numbers, which the log stream sends as event ids. They count on
        # from when the session was made, in milliseconds, so an id a page
        # kept from an earlier run of the dashboard (its EventSource
        # reconnects to this one by itself, sending that id) is below all of
        # this run's, and the page is sent this run's lines from the first
        # (unless that run logged more than a line a millisecond).
        self.log_ids = itertools.count(time.time_ns() // 1_000_000)
        # Each open log stream's queue.
        self.log_clients = []
        # The bounding-box tool's viewer and boxes.
        self.bbx_generator_state = {
            "dataset_path": None,
            "num_boxes": 0,
            "bounding_boxes": [],
            "viewer": None,
            "viewer_process": None,
            "viewer_url": None,
            "viewer_state": None,
        }

    # The data and the viewer
    def forget_layer(self, name):
        """Drop ``name``'s LAYER_BOOKKEEPING, so a later layer of that name starts fresh."""
        for attr in LAYER_BOOKKEEPING:
            getattr(self, attr).pop(name, None)

    def layer_blob(self, dashboard_url=None):
        """The args blob of a prediction layer's URL (viewer.layers.prediction_url):
        the chain (``pipeline_spec``), the dashboard's address and the chain's
        digest, as the browser app's layers carry them (web/src/backend/serving.ts).

        Every layer is built through this, whatever started its model (the
        CLI, Submit, Add, a finetune, Submit's redraw), so an inference server
        running a merging postprocessor knows where to post its merged ids
        (serving.chunkmirage_ops), and a redraw of the chain a layer already
        shows is told by its URL (routes.pipeline._shows). The address is
        the dashboard's own (``dashboard_url``, by IP, which a server on the
        cluster can reach), or else ``dashboard_url`` as given, the address
        a request reached the dashboard by; left out when neither is known.
        """
        spec = self.pipeline_spec
        url = self.dashboard_url or dashboard_url
        extras = {"dashboard_url": url} if url else {}
        return spec.to_url_blob(**extras, digest=spec.digest())

    def shown_channel(self, model_name):
        """The channel ``model_name``'s layer opens on (viewer.layers
        prediction_layer's ``channel``): the one picked for it
        (``shown_channels``), else the one picked for the model it is a
        finetune of, as finetune_layers names each iteration
        (``<base>_finetuned...``), so a finetune of mito's channel shows mito
        too; None when none was picked (the layer opens where its server
        says)."""
        if model_name in self.shown_channels:
            return self.shown_channels[model_name]
        bases = [base for base in self.shown_channels if (model_name or "").startswith(f"{base}_finetuned")]
        return self.shown_channels[max(bases, key=len)] if bases else None

    # The models
    @property
    def jobs(self):
        """The running inference servers' jobs, each with its model_name and
        host: jobs.launch.started_jobs(). Assigning replaces the list's
        contents, so start_hosts and cleanup_handler keep seeing it."""
        return started_jobs()

    @jobs.setter
    def jobs(self, jobs):
        started_jobs()[:] = jobs

    # The chain
    @property
    def input_norms(self):
        """The live input normalizers. Changed only by set_pipeline()."""
        return process_chain().input_norms

    @property
    def postprocess(self):
        """The live postprocessors. Changed only by set_pipeline()."""
        return process_chain().postprocess

    @property
    def pipeline_spec(self):
        """The chain as data (PipelineSpec), derived on every read."""
        return process_chain().spec

    def set_pipeline(self, spec, built=None):
        """Replace the chain; see ProcessChain.set."""
        process_chain().set(spec, built=built)

    # The pipeline builder's last apply
    @property
    def builder_state(self):
        """What the builder last applied, as it sent it: ``{inputs, outputs,
        edges, normalizers, models, postprocessors}``, each a list of its
        nodes, all empty until the first apply. A new dict on every read."""
        return dict(self._builder_state)

    @builder_state.setter
    def builder_state(self, state):
        self._builder_state = {key: state[key] for key in _BUILDER_KEYS}

    # The job settings (one attribute per key is added below the class)
    @property
    def settings(self):
        """The launcher settings themselves (jobs.settings.LauncherSettings)."""
        return launcher_settings()

    @property
    def server_config(self):
        """{key: value} of the saved settings."""
        return launcher_settings().as_dict()

    @property
    def server_config_cached(self):
        """Whether the settings came from, or were saved to, the file."""
        return launcher_settings().cached

    def save_server_config(self):
        """Write the settings to ~/.cellmap_flow/server_config.yaml."""
        launcher_settings().save()

    def tasks_dir(self) -> str:
        """``blockwise_tasks_dir``, made if it is missing."""
        tasks_dir = self.blockwise_tasks_dir or os.path.expanduser("~/.cellmap_flow/blockwise_tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        return tasks_dir

    # The finetune tab
    @property
    def finetune_job_manager(self):
        """The training jobs' FinetuneJobManager, made when first asked for."""
        if self._finetune_job_manager is None:
            from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager

            self._finetune_job_manager = FinetuneJobManager()
        return self._finetune_job_manager

    @finetune_job_manager.setter
    def finetune_job_manager(self, manager):
        self._finetune_job_manager = manager


def _setting(key):
    """A session attribute that is ``launcher_settings().<key>``."""
    def get(self):
        return getattr(launcher_settings(), key)

    def set(self, value):
        setattr(launcher_settings(), key, value)

    return property(get, set, doc=f"The saved setting {key!r}.")


# The saved settings, one attribute per key, so that the list cannot drift from
# the defaults (jobs.settings.SERVER_CONFIG_DEFAULTS is the one place a key is added).
for _key in SERVER_CONFIG_KEYS:
    setattr(Session, _key, _setting(_key))
del _key

_session = Session()


def get_session() -> Session:
    """The dashboard's session."""
    return _session
