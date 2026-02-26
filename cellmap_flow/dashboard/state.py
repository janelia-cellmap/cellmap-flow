# Re-export all dashboard state from the globals singleton for backward compatibility.
# New code should import directly from cellmap_flow.globals.

from cellmap_flow.globals import g, LogHandler, get_blockwise_tasks_dir  # noqa: F401

log_buffer = g.log_buffer
log_clients = g.log_clients
NEUROGLANCER_URL = g.NEUROGLANCER_URL
INFERENCE_SERVER = g.INFERENCE_SERVER
CUSTOM_CODE_FOLDER = g.CUSTOM_CODE_FOLDER
bbx_generator_state = g.bbx_generator_state
finetune_job_manager = g.finetune_job_manager
minio_state = g.minio_state
annotation_volumes = g.annotation_volumes
output_sessions = g.output_sessions
