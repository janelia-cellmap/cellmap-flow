import os
import queue
import yaml
import logging
from collections import deque
from importlib.resources import files
from typing import Any, Dict, List, Optional

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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Flow, cls).__new__(cls)
            cls._instance.jobs = []
            cls._instance.models_config = []
            cls._instance.raw = None
            cls._instance.input_norms = []
            # Raw JSON-serializable form of the dashboard's input_norm config.
            # Populated by /api/run from the request payload; used by the
            # finetune submit/restart flow so the trainer process applies the
            # same normalization the dashboard uses at inference.
            #
            # NOTE: prefer ``current_input_norm_config()`` over reading this
            # directly. Some startup paths (e.g. yaml_cli.py at server boot)
            # populate ``input_norms`` from a YAML's ``json_data.input_norm``
            # but never touch ``input_norm_config``. The helper falls back to
            # reconstructing the dict from the live normalizer instances.
            cls._instance.input_norm_config = {}
            cls._instance.postprocess = []
            cls._instance.postprocess_config = {}
            cls._instance.viewer = None
            cls._instance.dataset_path = None
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
                "minio_root": None,
                "output_base": None,
                "last_sync": {},
                "chunk_sync_state": {},
                "sync_thread": None,
            }
            cls._instance.annotation_volumes = {}
            cls._instance.output_sessions = {}
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

    def get_output_dtype(self, model_output_dtype, postprocess=None):
        """The dtype a chain hands to the client.

        ``postprocess=None`` means ``self.postprocess``; the inference server
        passes the chain of the layer being served.
        """
        dtype = model_output_dtype
        if postprocess is None:
            postprocess = self.postprocess

        if len(postprocess) > 0:
            # Postprocessors are applied in order (see Inferencer), so the dtype
            # that actually reaches the client is the one declared by the LAST
            # step that declares one. Scan in reverse, matching
            # is_output_segmentation(). Scanning forward picked e.g.
            # SigmoidPostprocessor's float32 ahead of a trailing
            # AffinityPostprocessor's uint64, which both advertised the wrong
            # dtype in the zarr metadata (neuroglancer: "Data type not
            # compatible with segmentation layer") and cast uint64 label ids
            # through float32, corrupting any id above 2**24.
            for step in postprocess[::-1]:
                if step.dtype:
                    logger.debug(
                        f"Setting output dtype to {step.dtype} from {step} - was {dtype}"
                    )
                    dtype = step.dtype
                    break

        return dtype


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


def current_input_norm_config():
    """Return the dashboard's current input_norm in a JSON-serializable form.

    Reads ``g.input_norm_config`` if populated (the ordered list the dashboard
    posts, or an older name-keyed dict); otherwise rebuilds the ordered
    ``[{name, **params}]`` list from the live ``g.input_norms`` instances.
    The fallback matters because some startup paths (yaml_cli) populate
    ``g.input_norms`` from the YAML at server boot but never touch
    ``input_norm_config`` -- if the user submits training without first
    hitting /api/run, the manifest would otherwise be written empty.
    """
    cfg = getattr(g, "input_norm_config", None) or {}
    if cfg:
        return cfg
    return _chain_config(getattr(g, "input_norms", None))


def current_postprocess_config():
    """Return the dashboard's current postprocess chain in a JSON-serializable form.

    Mirrors ``current_input_norm_config()``: reads ``g.postprocess_config`` if
    populated, otherwise rebuilds the ordered list from the live
    ``g.postprocess`` instances. The fallback matters for the same reason it
    does for input_norm -- e.g. a yaml booted with a ``json_data.postprocess``
    (like ``SigmoidPostprocessor``) populates ``g.postprocess`` but never
    touches ``postprocess_config``.
    """
    cfg = getattr(g, "postprocess_config", None) or {}
    if cfg:
        return cfg
    return _chain_config(getattr(g, "postprocess", None))


def get_blockwise_tasks_dir():
    tasks_dir = g.blockwise_tasks_dir or os.path.expanduser(
        "~/.cellmap_flow/blockwise_tasks"
    )
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir
