"""The lines the trainer prints for the job manager to find in its log.

The training job's loop (session_loop) and lora_trainer print these to
stdout, which tee copies into training_log.txt; the job manager reads them
back from there to follow the job. The two sides can be different
versions -- a training job outlives a dashboard upgrade -- so these only
ever change compatibly.

The patterns are the ones the job manager has always used, unchanged.
"""

import re

from cellmap_flow.finetune.finetuned_model_templates import (
    FINETUNED_MODEL_YAML_MARKER as FINETUNED_MODEL_YAML,
)
from cellmap_flow.utils.web_utils import IP_PATTERN

__all__ = [
    "TRAINING_ITERATION_COMPLETE",
    "FINETUNED_MODEL_YAML",
    "RESTART_FAILED",
    "INFERENCE_SERVER_FAILED",
    "TRAINING_DIVERGED",
    "RESTARTING_TRAINING",
    "WAITING_FOR_RESTART",
    "EPOCH_START_RE",
    "EPOCH_SUMMARY_RE",
    "ITERATION_COMPLETE_RE",
    "MODEL_YAML_RE",
    "STATUS_MARKER_RE",
    "SERVER_URL_RE",
    "emit",
]

# Followed by a value: "<marker> <value>".
TRAINING_ITERATION_COMPLETE = "TRAINING_ITERATION_COMPLETE:"  # the iteration's model name
RESTART_FAILED = "RESTART_FAILED:"  # why
INFERENCE_SERVER_FAILED = "INFERENCE_SERVER_FAILED:"  # why
# FINETUNED_MODEL_YAML, the iteration's serving YAML, is defined with the
# templates that write it and printed just before TRAINING_ITERATION_COMPLETE.

# On their own.
TRAINING_DIVERGED = "TRAINING_DIVERGED"
RESTARTING_TRAINING = "RESTARTING_TRAINING"
WAITING_FOR_RESTART = "WAITING_FOR_RESTART"

EPOCH_START_RE = re.compile(r"Starting\s+epoch\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
EPOCH_SUMMARY_RE = re.compile(r"Epoch\s+(\d+)/(\d+)\s*-\s*Loss:\s*([\d.]+)", re.IGNORECASE)
ITERATION_COMPLETE_RE = re.compile(r"TRAINING_ITERATION_COMPLETE:\s+(\S+)")
MODEL_YAML_RE = re.compile(r"^.*?FINETUNED_MODEL_YAML:\s*(.+?)\s*$", re.MULTILINE)
# Status markers, in the order they matter: the last one in a chunk of log
# decides.
STATUS_MARKER_RE = re.compile(r"TRAINING_DIVERGED|RESTARTING_TRAINING|WAITING_FOR_RESTART")
# The inference server's address, printed by CellMapFlowServer.run.
SERVER_URL_RE = re.compile(re.escape(IP_PATTERN[0]) + r"(.+?)" + re.escape(IP_PATTERN[1]))


def emit(marker: str, value=None) -> None:
    """Print ``marker`` (and its value) as a line of its own, at once.

    Flushed, because the job manager is reading the log while the job runs.
    """
    print(marker if value is None else f"{marker} {value}", flush=True)
