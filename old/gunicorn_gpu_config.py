import os
from itertools import cycle
import torch

# List available GPUs (e.g., ['0', '1', '2', '3'])
num_gpus = torch.cuda.device_count()
AVAILABLE_GPUS = [
    str(i)
    for i in range(num_gpus)
    if torch.cuda.get_device_properties(i).total_memory > 0
]
gpu_cycle = cycle(AVAILABLE_GPUS)


def post_fork(server, worker):
    # Assign a GPU to the worker
    print(worker)
    # gpu_id = next(gpu_cycle)
    gpu_id = AVAILABLE_GPUS[worker.pid % len(AVAILABLE_GPUS)]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    server.log.info(f"Worker {worker.pid} assigned to GPU {gpu_id}")
