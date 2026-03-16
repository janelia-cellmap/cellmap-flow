from cellmap_flow.norm.input_normalize import MinMaxNormalizer, LambdaNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor, ThresholdPostprocessor

import os
import queue
import yaml
import logging
import threading
import numpy as np
from collections import deque
from typing import Any, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# input_norms = [MinMaxNormalizer(), LambdaNormalizer("x*2-1")]
# postprocess = [DefaultPostprocessor(), ThresholdPostprocessor(threshold=0.5)]

input_norms = []
postprocess = []
viewer = None


class Flow:
    _instance: Optional["Flow"] = None
    
    # Class-level type annotations for all instance attributes
    jobs: List[Any]
    models_config: List[Any]
    servers: List[Any]
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
    neuroglancer_thread: Optional[Any]
    pipeline_inputs: List[Any]
    pipeline_outputs: List[Any]
    pipeline_edges: List[Any]
    pipeline_normalizers: List[Any]
    pipeline_models: List[Any]
    pipeline_postprocessors: List[Any]
    shaders: dict
    shader_controls: dict

    # Dashboard state (moved from cellmap_flow.dashboard.state)
    log_buffer: deque
    log_clients: list
    NEUROGLANCER_URL: Optional[str]
    INFERENCE_SERVER: Optional[Any]
    CUSTOM_CODE_FOLDER: str
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
            cls._instance.servers = []
            cls._instance.raw = None
            cls._instance.input_norms = input_norms
            cls._instance.postprocess = postprocess
            cls._instance.viewer = None
            cls._instance.dataset_path = None
            cls._instance.model_catalog = {}
            # Uncomment and adjust if you want to load the model catalog:
            models_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__), os.pardir, "models", "models.yaml"
                )
            )
            with open(models_path, "r") as f:
                cls._instance.model_catalog = yaml.safe_load(f)

            cls._instance.queue = "gpu_h100"
            cls._instance.charge_group = "cellmap"
            cls._instance.nb_cores_master = 4
            cls._instance.nb_cores_worker = 12
            cls._instance.nb_workers = 14
            cls._instance.tmp_dir = os.path.expanduser("~/.cellmap_flow/blockwise_tmp")
            cls._instance.blockwise_tasks_dir = os.path.expanduser("~/.cellmap_flow/blockwise_tasks")
            cls._instance.neuroglancer_thread = None

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
            cls._instance.INFERENCE_SERVER = None
            cls._instance.CUSTOM_CODE_FOLDER = os.path.expanduser(
                os.environ.get(
                    "CUSTOM_CODE_FOLDER",
                    "~/Desktop/cellmap/cellmap-flow/example/example_norm",
                )
            )
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

    def to_dict(self):
        return self.__dict__.items()

    def __repr__(self):
        return f"Flow({self.__dict__})"

    def __str__(self):
        return f"Flow({self.__dict__})"

    def get_output_dtype(self, model_output_dtype):

        dtype = model_output_dtype

        if len(self.postprocess) > 0:
            for postprocess in self.postprocess:
                if postprocess.dtype:
                    logger.info(
                        f"Setting output dtype to {postprocess.dtype} from {postprocess} - was {dtype}"
                    )
                    dtype = postprocess.dtype
                    break

        return dtype

    @classmethod
    def run(
        cls,
        zarr_path,
        model_configs,
        queue="gpu_h100",
        charge_group="cellmap",
        input_normalizers=None,
        post_processors=None,
    ):

        from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
        from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url

        if input_normalizers is None:
            input_normalizers = []
        if post_processors is None:
            post_processors = []

        # Get the singleton instance (creates one if it doesn't exist)
        instance = cls()
        instance.queue = queue
        instance.charge_group = charge_group
        instance.dataset_path = zarr_path
        instance.input_norms = input_normalizers
        instance.postprocess = post_processors
        instance.models_config = model_configs
        instance.neuroglancer_thread = None

        threads = []

        for model_config in instance.models_config:
            model_command = model_config.command
            command = f"{SERVER_COMMAND} {model_command} -d {instance.dataset_path}"
            print(f"Starting server with command: {command}")
            thread = threading.Thread(
                target=start_hosts,
                args=(command, queue, charge_group, model_config.name),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        instance.neuroglancer_thread = threading.Thread(
            target=generate_neuroglancer_url, args=(instance.dataset_path,)
        )
        instance.neuroglancer_thread.start()
        # Optionally wait for the neuroglancer thread:
        # instance.neuroglancer_thread.join()

        print(f"*****Neuroglancer URL: {instance.dataset_path}")

    @classmethod
    def stop(cls):
        instance = cls()
        for job in instance.jobs:
            print(f"Killing job {job.job_id}")
            job.kill()
        if instance.neuroglancer_thread is not None:
            instance.neuroglancer_thread = None
        instance.jobs = []

    @classmethod
    def delete(cls):
        cls._instance = None


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


def get_blockwise_tasks_dir():
    tasks_dir = g.blockwise_tasks_dir or os.path.expanduser(
        "~/.cellmap_flow/blockwise_tasks"
    )
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir
