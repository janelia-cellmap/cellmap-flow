"""The dashboard's state, read and written through one object: ``get_session()``.

Routes use the session's attributes rather than reaching into
``cellmap_flow.globals.g``. For now each attribute *is* an attribute of
``g``: the storage stays there because tests/conftest.py restores
``vars(g)`` after every test, and because the CLIs, the inference servers
and the finetune job manager still use ``g`` directly. When the storage
moves (Phase 4, K16), this module is what changes. A Session holds nothing
itself, and has no attributes but these (``__slots__``), so a misspelt one
raises rather than being stored beside ``g``.

- The data and the viewer: ``dataset_path``, ``viewer``, ``raw``,
  ``neuroglancer_url``, ``shaders``, ``shader_controls``, ``extra_layers``.
- The models: ``jobs``, ``models_config``, ``model_catalog``.
- The chain: ``input_norms`` and ``postprocess`` (the live steps),
  ``pipeline_spec`` (the same as data), and ``set_pipeline()``, the one way
  to change them.
- The pipeline builder's last apply: ``builder_state``,
  ``builder_model_configs``.
- The job settings in ~/.cellmap_flow/server_config.yaml, one attribute per
  key of ``globals.SERVER_CONFIG_DEFAULTS`` (``queue``, ``charge_group``,
  ``walltime``, ...), ``server_config``, ``server_config_cached`` and
  ``save_server_config()``; and blockwise's ``tmp_dir`` and
  ``blockwise_tasks_dir``.
- The finetune tab's MinIO and volumes: ``minio_state``,
  ``annotation_volumes``, ``output_sessions``; its training jobs'
  ``finetune_job_manager``; the Review tab's ``review``.
- The log panel's ``log_buffer`` and ``log_clients``; the box tool's
  ``bbx_generator_state``.
"""

from cellmap_flow.globals import SERVER_CONFIG_KEYS, g

# The builder's node lists, as /api/pipeline/apply stores them on g.
_BUILDER_KEYS = ("inputs", "outputs", "edges", "normalizers", "models", "postprocessors")


def _on_g(name, doc, writable=True):
    """A session attribute that is ``g.<name>``."""
    def get(self):
        return getattr(g, name)

    def set(self, value):
        setattr(g, name, value)

    return property(get, set if writable else None, doc=doc)


class Session:
    """The dashboard's state; see the module docstring."""

    __slots__ = ()

    # The data and the viewer
    dataset_path = _on_g("dataset_path", "The raw data as the user gave it: a multiscale group or one level of it.")
    viewer = _on_g("viewer", "The neuroglancer viewer; None until a dataset is opened.")
    raw = _on_g("raw", "The raw data's layer, as last built.")
    neuroglancer_url = _on_g("NEUROGLANCER_URL", "The viewer's own address, which the index page embeds.")
    shaders = _on_g("shaders", "{layer name: shader}: the user's, put back when a layer is rebuilt.")
    shader_controls = _on_g("shader_controls", "{layer name: shaderControls}, likewise.")
    extra_layers = _on_g("extra_layers", "{name: layer} shown beside the raw data: a YAML's extra_layers.")

    # The models
    jobs = _on_g("jobs", "The running inference servers' jobs, each with its model_name and host.")
    models_config = _on_g("models_config", "The configured models (ModelConfig).")
    model_catalog = _on_g("model_catalog", "{group: {model name: path}}, the Models tab's catalog.")

    # The chain
    input_norms = _on_g("input_norms", "The live input normalizers. Changed only by set_pipeline().", writable=False)
    postprocess = _on_g("postprocess", "The live postprocessors. Changed only by set_pipeline().", writable=False)

    @property
    def pipeline_spec(self):
        """The chain as data (PipelineSpec), derived on every read."""
        return g.pipeline_spec

    def set_pipeline(self, spec, built=None):
        """Replace the chain; see Flow.set_pipeline."""
        g.set_pipeline(spec, built=built)

    # The pipeline builder's last apply
    @property
    def builder_state(self):
        """What the builder last applied, as it sent it: ``{inputs, outputs,
        edges, normalizers, models, postprocessors}``, each a list of its
        nodes, all empty until the first apply."""
        return {key: getattr(g, f"pipeline_{key}") for key in _BUILDER_KEYS}

    @builder_state.setter
    def builder_state(self, state):
        for key in _BUILDER_KEYS:
            setattr(g, f"pipeline_{key}", state[key])

    @property
    def builder_model_configs(self):
        """{model name: config}, from every model node the builder applied with one."""
        if not hasattr(g, "pipeline_model_configs"):
            g.pipeline_model_configs = {}
        return g.pipeline_model_configs

    # The job settings (one attribute per key is added below the class)
    server_config_cached = _on_g("_server_config_cached", "Whether the settings came from, or were saved to, the file.",
                                 writable=False)
    tmp_dir = _on_g("tmp_dir", "Where blockwise tasks keep their progress.")
    blockwise_tasks_dir = _on_g("blockwise_tasks_dir", "Where blockwise task YAMLs and master logs go.")

    @property
    def server_config(self):
        """{key: value} of the saved settings."""
        return {key: getattr(g, key) for key in SERVER_CONFIG_KEYS}

    def save_server_config(self):
        """Write the settings to ~/.cellmap_flow/server_config.yaml."""
        g.save_server_config()

    # The finetune and review tabs
    minio_state = _on_g("minio_state", "The dashboard's MinIO: its process, address, bucket and session directory.")
    annotation_volumes = _on_g("annotation_volumes", "{volume id: its record}, the volumes served through MinIO.")
    output_sessions = _on_g("output_sessions", "{output base directory: its session directory}.")
    finetune_job_manager = _on_g("finetune_job_manager",
                                 "The training jobs' FinetuneJobManager, made when first asked for.")
    review = _on_g("review", "The Review tab's open index (review_routes.ReviewSession), or None.")

    # The log panel and the box tool
    log_buffer = _on_g("log_buffer", "The last log lines, for a log panel that connects late.")
    log_clients = _on_g("log_clients", "Each open log stream's queue.")
    bbx_generator_state = _on_g("bbx_generator_state", "The bounding-box tool's viewer and boxes.")


# The saved settings, one attribute per key, so that the list cannot drift from
# the defaults (globals.SERVER_CONFIG_DEFAULTS is the one place a key is added).
for _key in SERVER_CONFIG_KEYS:
    setattr(Session, _key, _on_g(_key, f"The saved setting {_key!r}."))
del _key

_session = Session()


def get_session() -> Session:
    """The dashboard's session."""
    return _session
