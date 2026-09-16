"""Ask a running inference server about the model it is serving.

The dashboard runs wherever the jobs were launched from, which is frequently a
node with no usable GPU -- or, on LSF, one holding a single GPU in
``exclusive_process`` mode that something else already has a context on. Building
the model there just to read its output shape is both wasteful and unreliable:
for a script-defined model it means re-downloading weights and re-running
torch.export, and it fails outright when a CUDA context cannot be created.

The inference server already loaded the model, on the node the job was actually
submitted to. Ask it instead.
"""

import logging
from types import SimpleNamespace

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 3

# ``model_info`` superseded ``output_probe`` when the geometry fields were
# added. Try the new name first, but fall back so a dashboard still works
# against a server that was started before the rename.
_PATHS = ("/__control__/model_info", "/__control__/output_probe")


def fetch_model_info(host: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Return the server's model info, or ``{"available": False, "reason": ...}``.

    Never raises: an unreachable server is an ordinary state here (the job may
    still be queued), not an error the caller should have to handle.
    """
    if not host:
        return {"available": False, "reason": "job has no host yet"}

    base = host.rstrip("/")
    last_reason = "no response"
    for path in _PATHS:
        try:
            r = requests.get(f"{base}{path}", timeout=timeout)
        except Exception as e:
            # Connection-level failure will repeat on the other path, so stop.
            return {"available": False, "reason": f"could not reach server ({e})"}
        if r.status_code == 404:
            last_reason = "server predates the model_info endpoint"
            continue
        if r.status_code != 200:
            return {"available": False, "reason": f"returned HTTP {r.status_code}"}
        try:
            return r.json()
        except Exception as e:
            return {"available": False, "reason": f"unreadable response ({e})"}
    return {"available": False, "reason": last_reason}


def model_geometry(info: dict):
    """Pull the finetune tab's geometry out of a model_info payload.

    Returns None when the payload has no geometry -- an older server, which
    reported only the output activation.
    """
    if not info or not info.get("write_shape"):
        return None
    try:
        return {
            "write_shape": [int(v) for v in info["write_shape"]],
            "output_voxel_size": [int(v) for v in info["output_voxel_size"]],
            "output_channels": int(info.get("output_channels", 1)),
        }
    except Exception as e:
        logger.debug(f"Malformed model geometry {info}: {e}")
        return None


def running_job_host(model_name):
    """The host serving ``model_name``, if a job for it is up."""
    from cellmap_flow.globals import g  # local: globals pulls in a lot

    for job in getattr(g, "jobs", []) or []:
        if getattr(job, "model_name", None) == model_name:
            return getattr(job, "host", None)
    return None


GEOMETRY_FIELDS = (
    "read_shape",
    "write_shape",
    "input_voxel_size",
    "output_voxel_size",
    "output_channels",
)


def model_geometry_config(model_name, timeout=DEFAULT_TIMEOUT_SECONDS):
    """A stand-in for ``ModelConfig.config`` carrying geometry and nothing else.

    Duck-types the real thing for callers that only read shapes and voxel
    sizes, so they do not have to build the model to get them. Returns None
    when no running server can answer, leaving the caller to fall back.
    """
    info = fetch_model_info(running_job_host(model_name), timeout)
    if not info:
        return None
    # Every field, or nothing. Callers use this as ``model_geometry_config(x)
    # or model_config.config``, and a SimpleNamespace missing one attribute is
    # still truthy -- so a partial answer would defeat the fallback and raise
    # AttributeError deep in the caller instead. An older server that cannot
    # report all of them should fall back cleanly.
    missing = [f for f in GEOMETRY_FIELDS if info.get(f) is None]
    if missing:
        logger.debug(
            f"Server geometry for {model_name} is missing {missing}; "
            "falling back to building the model locally."
        )
        return None
    return SimpleNamespace(**{f: info[f] for f in GEOMETRY_FIELDS})
