import os
import queue
import yaml
import logging
from collections import deque
from importlib.resources import files
from typing import Any, Dict, List, Optional

from cellmap_flow.pipeline_spec import (
    PipelineSpec,
    chain_output_dtype,
    normalize_steps,
)

logger = logging.getLogger(__name__)

# This is the basicConfig that actually takes effect in most processes,
# because globals is imported before any CLI gets to configure logging.
from cellmap_flow.utils.logging_setup import configure_logging

configure_logging()

SERVER_CONFIG_PATH = os.path.expanduser("~/.cellmap_flow/server_config.yaml")

SERVER_CONFIG_DEFAULTS = {
    "queue": "gpu_h100",
    "charge_group": "",
    # LSF's own default on the GPU queues is 120 minutes, which killed
    # inference servers two hours into a session. See DEFAULT_WALLTIME in
    # bsub_utils for why this matches the Fileglancer app's own 8 hours.
    "walltime": "08:00",
    # Try other GPU queues when the requested one is busy or closed. On by
    # default because a job that starts elsewhere beats one that never
    # starts; turn it off when the queue itself matters (a benchmark pinned
    # to one GPU model, a charge group valid on only one queue).
    "cycle_gpu_queues": True,
    "nb_cores_master": 4,
    "nb_cores_worker": 12,
    "nb_workers": 14,
}

SERVER_CONFIG_KEYS = list(SERVER_CONFIG_DEFAULTS.keys())


def load_server_config_cache() -> Optional[Dict[str, Any]]:
    """Load server config from cache file. Returns None if not found."""
    if os.path.exists(SERVER_CONFIG_PATH):
        with open(SERVER_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return None


def save_server_config_cache(config: Dict[str, Any]) -> None:
    """Save server config to cache file."""
    os.makedirs(os.path.dirname(SERVER_CONFIG_PATH), exist_ok=True)
    with open(SERVER_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class Flow:
    _instance: Optional["Flow"] = None
    
    # Class-level type annotations for all instance attributes
    jobs: List[Any]
    models_config: List[Any]
    raw: Optional[Any]
    input_norms: List[Any]
    postprocess: List[Any]
    input_norm_config: Any
    postprocess_config: Any
    viewer: Optional[Any]
    dataset_path: Optional[str]
    model_catalog: dict
    queue: str
    charge_group: str
    nb_cores_master: int
    nb_cores_worker: int
    nb_workers: int
    tmp_dir: Optional[str]
    blockwise_tasks_dir: Optional[str]
    pipeline_inputs: List[Any]
    pipeline_outputs: List[Any]
    pipeline_edges: List[Any]
    pipeline_normalizers: List[Any]
    pipeline_models: List[Any]
    pipeline_postprocessors: List[Any]
    shaders: dict
    shader_controls: dict
    _server_config_cached: bool

    # Dashboard state (moved from cellmap_flow.dashboard.state)
    log_buffer: deque
    log_clients: list
    NEUROGLANCER_URL: Optional[str]
    bbx_generator_state: dict
    finetune_job_manager: Any
    minio_state: dict
    annotation_volumes: dict
    output_sessions: dict
    review: Optional[Any]
    extra_layers: dict

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Flow, cls).__new__(cls)
            cls._instance.jobs = []
            cls._instance.models_config = []
            cls._instance.raw = None
            cls._instance.input_norms = []
            # The chain's steps as the dashboard received them, which the
            # finetune submit/restart flow hands the trainer so it normalizes
            # as inference does. Written only by set_pipeline(); read them
            # through pipeline_spec, which falls back to the live instances
            # when these are empty (yaml_cli and blockwise set input_norms
            # and postprocess from a YAML without touching them).
            cls._instance.input_norm_config = {}
            cls._instance.postprocess = []
            cls._instance.postprocess_config = {}
            cls._instance.viewer = None
            cls._instance.dataset_path = None
            # {name: neuroglancer layer} that the viewer shows beside the raw
            # data: a YAML's extra_layers, built by cellmap_flow_yaml.
            cls._instance.extra_layers = {}
            catalog = files("cellmap_flow.models").joinpath("models.yaml").read_text()
            cls._instance.model_catalog = yaml.safe_load(catalog) or {}

            # Load server config from cache or use defaults.
            #
            # Drive this from SERVER_CONFIG_DEFAULTS rather than naming each
            # key by hand. save_server_config() already iterates the same
            # dict, so a key listed there but missing from a hand-written
            # assignment raised AttributeError on save -- which is how adding
            # "walltime" killed every yaml run at startup.
            cached = load_server_config_cache() or {}
            for key, default in SERVER_CONFIG_DEFAULTS.items():
                setattr(cls._instance, key, cached.get(key, default))
            cls._instance._server_config_cached = bool(cached)
            cls._instance.tmp_dir = os.path.expanduser("~/.cellmap_flow/blockwise_tmp")
            cls._instance.blockwise_tasks_dir = os.path.expanduser("~/.cellmap_flow/blockwise_tasks")

            # Pipeline visual state storage
            cls._instance.pipeline_inputs = []
            cls._instance.pipeline_outputs = []
            cls._instance.pipeline_edges = []
            cls._instance.pipeline_normalizers = []
            cls._instance.pipeline_models = []
            cls._instance.pipeline_postprocessors = []

            # Shader state: key = layer name, value = shader string
            cls._instance.shaders = {}
            # ShaderControls state: key = layer name, value = shaderControls dict
            cls._instance.shader_controls = {}

            # Dashboard state (moved from cellmap_flow.dashboard.state)
            cls._instance.log_buffer = deque(maxlen=1000)
            cls._instance.log_clients = []
            cls._instance.NEUROGLANCER_URL = None
            cls._instance.bbx_generator_state = {
                "dataset_path": None,
                "num_boxes": 0,
                "bounding_boxes": [],
                "viewer": None,
                "viewer_process": None,
                "viewer_url": None,
                "viewer_state": None,
            }
            cls._instance.minio_state = {
                "process": None,
                "port": None,
                "ip": None,
                "bucket": "annotations",
                "output_base": None,
                "sync_thread": None,
            }
            cls._instance.annotation_volumes = {}
            cls._instance.output_sessions = {}
            # The Review tab's open index and its last pick, a
            # review_routes.ReviewSession; None until /api/review/open.
            cls._instance.review = None
            # Static layers added at startup (YAML extra_layers), by layer
            # name; the viewer-layer routes drop or rename their entries.
            cls._instance.extra_layers = {}
            cls._instance._finetune_job_manager = None

        return cls._instance

    @property
    def finetune_job_manager(self):
        if self._finetune_job_manager is None:
            from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager
            self._finetune_job_manager = FinetuneJobManager()
        return self._finetune_job_manager

    @finetune_job_manager.setter
    def finetune_job_manager(self, value):
        self._finetune_job_manager = value

    def __repr__(self):
        return f"Flow({self.__dict__})"

    def __str__(self):
        return f"Flow({self.__dict__})"

    def save_server_config(self):
        """Save current server config attributes to cache."""
        config = {k: getattr(self, k) for k in SERVER_CONFIG_KEYS}
        save_server_config_cache(config)
        self._server_config_cached = True

    @property
    def pipeline_spec(self) -> PipelineSpec:
        """The chain currently configured, as data. Derived on every read.

        Per chain: the ``*_config`` steps when set, otherwise the live
        instances' to_dict(). Never rebuilds the live instances, which can
        hold state (SimpleBlockwiseMerger's equivalences).
        """
        return PipelineSpec(
            _configured_steps(
                getattr(self, "input_norm_config", None),
                getattr(self, "input_norms", None),
            ),
            _configured_steps(
                getattr(self, "postprocess_config", None),
                getattr(self, "postprocess", None),
            ),
        )

    def set_pipeline(self, spec: PipelineSpec, built=None) -> None:
        """Replace the configured chain: both live chains and both configs.

        ``built`` is the ``(input_norms, postprocess)`` instances for
        ``spec`` when the caller already has them; otherwise they are built
        here. Everything is built before anything is assigned, so a chain
        that fails to build leaves the previous one in place.
        """
        input_norms, postprocess = spec.build() if built is None else built
        self.input_norms = list(input_norms)
        self.postprocess = list(postprocess)
        self.input_norm_config = list(spec.input_norm)
        self.postprocess_config = list(spec.postprocess)

    def get_output_dtype(self, model_output_dtype, postprocess=None):
        """The dtype a chain hands to the client.

        ``postprocess=None`` means ``self.postprocess``; the inference server
        passes the chain of the layer being served. The last step that
        declares a dtype decides, since the steps run in order: taking the
        first picked e.g. SigmoidPostprocessor's float32 ahead of a trailing
        AffinityPostprocessor's uint64, which both advertised the wrong dtype
        in the zarr metadata (neuroglancer: "Data type not compatible with
        segmentation layer") and cast uint64 label ids through float32,
        corrupting any id above 2**24.
        """
        if postprocess is None:
            postprocess = self.postprocess
        return chain_output_dtype(postprocess, model_output_dtype)


g = Flow()


# Custom handler to capture logs into Flow singleton
class LogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        g.log_buffer.append(log_entry)
        # Send to all connected clients
        for client_queue in g.log_clients:
            try:
                client_queue.put_nowait(log_entry)
            except queue.Full:
                pass


def _chain_config(steps) -> list:
    """``[{name, **params}]`` for a live chain, skipping steps that can't say."""
    derived = []
    for step in steps or []:
        try:
            d = dict(step.to_dict())
            d.setdefault("name", type(step).__name__)
            derived.append(d)
        except Exception:
            continue
    return derived


def _configured_steps(config, live):
    """The configured steps, or the live chain's when none are configured.

    The fallback matters because some startup paths (yaml_cli) populate the
    live chain from the YAML at server boot but never touch the config -- if
    the user submits training without first pressing Submit, the manifest
    would otherwise be written empty.
    """
    if config:
        return normalize_steps(config)
    return _chain_config(live)


def current_input_norm_config():
    """The dashboard's current input_norm as an ordered ``[{name, **params}]``."""
    return list(g.pipeline_spec.input_norm)


def current_postprocess_config():
    """The dashboard's current postprocess chain as an ordered ``[{name, **params}]``."""
    return list(g.pipeline_spec.postprocess)


def get_blockwise_tasks_dir():
    tasks_dir = g.blockwise_tasks_dir or os.path.expanduser(
        "~/.cellmap_flow/blockwise_tasks"
    )
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir
