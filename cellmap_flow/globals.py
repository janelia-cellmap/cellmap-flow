from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor
from cellmap_flow.models.model_yaml import load_model_paths
import os
import threading
import numpy as np


class Flow:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Flow, cls).__new__(cls)
            cls._instance.jobs = []
            cls._instance.models_config = []
            cls._instance.servers = []
            cls._instance.raw = None
            cls._instance.input_norms = []  # or [MinMaxNormalizer(0, 255)]
            cls._instance.postprocess = []
            cls._instance.viewer = None
            cls._instance.dataset_path = None
            cls._instance.model_catalog = {}
            # Uncomment and adjust if you want to load the model catalog:
            # cls._instance.model_catalog = load_model_paths(
            #     os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, "models", "models.yaml"))
            # )
            cls._instance.queue = "gpu_h100"
            cls._instance.charge_group = "cellmap"
            cls._instance.neuroglancer_thread = None
        return cls._instance

    def to_dict(self):
        return self.__dict__.items()

    def __repr__(self):
        return f"Flow({self.__dict__})"

    def __str__(self):
        return f"Flow({self.__dict__})"

    def get_output_dtype(self):
        dtype = np.float32

        if len(self.input_norms) > 0:
            for norm in self.input_norms[::-1]:
                if norm.dtype:
                    dtype = norm.dtype
                    break

        if len(self.postprocess) > 0:
            for postprocess in self.postprocess[::-1]:
                if postprocess.dtype:
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
