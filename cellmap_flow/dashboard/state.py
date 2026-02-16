import os
import logging
import queue
from collections import deque

logger = logging.getLogger(__name__)

# Global log buffer for streaming to frontend
log_buffer = deque(maxlen=1000)  # Keep last 1000 lines
log_clients = []  # List of queues for connected clients


# Custom handler to capture logs
class LogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_buffer.append(log_entry)
        # Send to all connected clients
        for client_queue in log_clients:
            try:
                client_queue.put_nowait(log_entry)
            except queue.Full:
                pass


NEUROGLANCER_URL = None
INFERENCE_SERVER = None

CUSTOM_CODE_FOLDER = os.path.expanduser(
    os.environ.get(
        "CUSTOM_CODE_FOLDER",
        "~/Desktop/cellmap/cellmap-flow/example/example_norm",
    )
)

# Blockwise task directory will be set from globals or use default
def get_blockwise_tasks_dir():
    from cellmap_flow.globals import g
    tasks_dir = getattr(g, 'blockwise_tasks_dir', None) or os.path.expanduser("~/.cellmap_flow/blockwise_tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir


# Global state for BBX generator
bbx_generator_state = {
    "dataset_path": None,
    "num_boxes": 0,
    "bounding_boxes": [],
    "viewer": None,
    "viewer_process": None,
    "viewer_url": None,
    "viewer_state": None
}

# --- Finetuning state ---
from datetime import datetime
from cellmap_flow.finetune.job_manager import FinetuneJobManager

# Global finetuning job manager
finetune_job_manager = FinetuneJobManager()

# Session management for timestamped output directories
# Maps base_output_path -> timestamped_session_path
output_sessions = {}


def get_or_create_session_path(base_output_path: str) -> str:
    """
    Get or create a timestamped session directory for the given base output path.

    If a session already exists for this base path, reuse it.
    Otherwise, create a new timestamped subdirectory.

    Args:
        base_output_path: Base output directory (e.g., "output/to/here")

    Returns:
        Timestamped session path (e.g., "output/to/here/20260213_123456")
    """
    base_output_path = os.path.expanduser(base_output_path)

    if base_output_path not in output_sessions:
        # Create new timestamped session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(base_output_path, timestamp)
        output_sessions[base_output_path] = session_path
        logger.info(f"Created new session path: {session_path}")

    return output_sessions[base_output_path]
