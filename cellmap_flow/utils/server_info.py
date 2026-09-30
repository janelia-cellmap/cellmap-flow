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

from cellmap_flow.models.geometry import number

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
            "write_shape": [number(v) for v in info["write_shape"]],
            "output_voxel_size": [number(v) for v in info["output_voxel_size"]],
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

# Reported when the server can, absent from servers predating it. Kept out of
# GEOMETRY_FIELDS so a server that cannot supply it still satisfies the
# all-or-nothing check below rather than forcing a local model build.
OPTIONAL_FIELDS = ("channels",)


def geometry_stand_in(fields):
    """A stand-in for ``ModelConfig.config`` with the geometry in ``fields``, or None.

    ``fields`` is a model_info payload or a geometry cache entry. The
    stand-in has the config's attribute names, so callers that only read
    shapes and voxel sizes, and ModelGeometry.from_config, take it as the
    config. Every GEOMETRY_FIELDS entry, or nothing: callers use it as
    ``stand_in or model_config.config``, and a SimpleNamespace missing one
    attribute is still truthy, so a partial answer would defeat the fallback
    and raise AttributeError deep in the caller instead.
    """
    if not isinstance(fields, dict) or any(fields.get(f) is None for f in GEOMETRY_FIELDS):
        return None
    present = {f: fields[f] for f in GEOMETRY_FIELDS}
    present.update({f: fields[f] for f in OPTIONAL_FIELDS if fields.get(f) is not None})
    return SimpleNamespace(**present)


def model_geometry_config(model_name, timeout=DEFAULT_TIMEOUT_SECONDS):
    """The geometry of ``model_name``'s running server, as a geometry_stand_in.

    None when no running server can answer, or an older one cannot report
    all of it, leaving the caller to fall back.
    """
    info = fetch_model_info(running_job_host(model_name), timeout)
    geometry = geometry_stand_in(info)
    if geometry is None and info.get("available", True):
        logger.debug(
            f"Server geometry for {model_name} is incomplete; "
            "falling back to building the model locally."
        )
    return geometry
