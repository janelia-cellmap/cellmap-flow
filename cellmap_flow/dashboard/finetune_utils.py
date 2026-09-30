"""The dashboard's side of cellmap_flow.finetune.session.

The session code takes its state as arguments. The dashboard's is ``g``'s
dicts, bound here at import -- ``minio_state``, ``annotation_volumes`` and
``output_sessions`` -- and the MinIO and sync entry points below pass them
on; serving a volume checks first that MinIO is installed. Tests replace the
three dicts here.
"""

import shutil

from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync as session_sync
from cellmap_flow.globals import g

minio_state = g.minio_state
annotation_volumes = g.annotation_volumes
output_sessions = g.output_sessions


# ---------------------------------------------------------------------------
# MinIO management
# ---------------------------------------------------------------------------

def _require_minio_binaries():
    """Fail early, with a fix, if the MinIO binaries are missing.

    Otherwise the missing binary surfaces as a bare
    ``FileNotFoundError: [Errno 2] ... 'minio'`` from subprocess, after the user
    has already picked an output path and created a session directory.

    MinIO no longer publishes prebuilt community server binaries (dl.min.io is
    410 Gone and the GitHub releases carry no assets), so conda-forge -- which
    still builds from source -- is the only practical way to install them.
    A pixi checkout gets them from pixi.toml's finetune feature.
    """
    missing = [name for name in ("minio", "mc") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"Required MinIO binaries not found on PATH: {', '.join(missing)}. "
            "Annotation volumes are served to Neuroglancer through a local MinIO "
            "server, so painting cannot start without them.\n\n"
            "In a pixi checkout they are part of the default environment:\n"
            "    pixi install\n"
            "otherwise:\n"
            "    mamba install minio-server minio-client -c conda-forge"
        )


# The mc alias for this dashboard's MinIO. It is defined per call through
# MC_HOST_<alias> (see _mc_env), not with `mc alias set`, which writes
# ~/.mc/config.json: that file is shared by every dashboard the user runs,
# so two of them repointed each other's alias and one's uploads went to the
# other's server.
MC_ALIAS = "myserver"


def ensure_minio_serving(zarr_path, crop_id, output_base_dir=None, mc_target_name=None):
    """Serve the volume at ``zarr_path`` through the dashboard's MinIO, starting
    it if needed; returns its URL. ``crop_id`` is the volume id, its bucket key
    without ".zarr". See ``session.minio.MinioServer.ensure_serving``."""
    _require_minio_binaries()
    server = session_minio.MinioServer(minio_state, volumes=annotation_volumes)
    return server.ensure_serving(zarr_path, crop_id, output_base_dir, mc_target_name=mc_target_name)


def sync_all_annotations_from_minio(force: bool = True):
    """``session.sync.sync_all`` for the dashboard's MinIO: how many volumes
    changed, or -1 if MinIO is not running."""
    return session_sync.sync_all(force, state=minio_state, volumes=annotation_volumes)


def sync_annotation_volume_from_minio(volume_id, force=False, zarr_path=None):
    """``session.sync.sync_volume`` for the dashboard's MinIO."""
    return session_sync.sync_volume(
        volume_id, force, zarr_path, state=minio_state, volumes=annotation_volumes
    )
