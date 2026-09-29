"""The dashboard's finetune helpers, over cellmap_flow.finetune.session.

The session code -- volumes, the session store, MinIO and the sync -- lives
in ``cellmap_flow.finetune.session`` and takes its state as arguments. This
module binds it to the dashboard's: ``minio_state``, ``annotation_volumes``
and ``output_sessions`` are ``g``'s dicts, bound at import, and every name
here passes them on. Routes and tests use these names, and tests patch the
three dicts here.
"""

import logging
import shutil

from cellmap_flow.finetune.session import instance as session_instance
from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync as session_sync
from cellmap_flow.finetune.session.store import SessionStore
from cellmap_flow.finetune.session.volume import VolumeGeometry, create_volume_zarr
from cellmap_flow.globals import g

minio_state = g.minio_state
annotation_volumes = g.annotation_volumes
output_sessions = g.output_sessions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def get_or_create_session_path(base_output_path: str) -> str:
    """This dashboard's session under ``base_output_path``; see ``SessionStore.get_or_create``."""
    return SessionStore(output_sessions).get_or_create(base_output_path)


def latest_session_on_disk(base_output_path: str):
    """The newest trainable session under ``base_output_path``; see ``SessionStore.latest_on_disk``."""
    return SessionStore(output_sessions).latest_on_disk(base_output_path)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

get_local_ip = session_minio.get_local_ip
find_available_port = session_minio.find_available_port


# ---------------------------------------------------------------------------
# Zarr creation
# ---------------------------------------------------------------------------

def create_annotation_volume_zarr(
    zarr_path,
    dataset_shape_voxels,
    output_voxel_size,
    dataset_offset_nm,
    chunk_size,
    dataset_path,
    model_name,
    input_size,
    input_voxel_size,
    claimed_output_voxel_size=None,
    claimed_input_voxel_size=None,
    input_norm_config=None,
    postprocess_config=None,
    annotation_dtype="uint8",
    annotation_type="annotation_volume",
):
    """``session.volume.create_volume_zarr``, with the geometry spelled out.

    ``dataset_offset_nm`` is voxel 0's centre (the OME translation).

    Returns:
        (success: bool, info: str): the zarr path, or the error.
    """
    try:
        geometry = VolumeGeometry(
            output_voxel_size=output_voxel_size,
            input_voxel_size=input_voxel_size,
            claimed_output_voxel_size=claimed_output_voxel_size,
            claimed_input_voxel_size=claimed_input_voxel_size,
            chunk_size=chunk_size,
            input_size=input_size,
            dataset_offset_nm=dataset_offset_nm,
            dataset_shape_voxels=dataset_shape_voxels,
        )
        return True, create_volume_zarr(
            zarr_path,
            geometry,
            dataset_path=dataset_path,
            model_name=model_name,
            input_norm=input_norm_config,
            postprocess=postprocess_config,
            annotation_dtype=annotation_dtype,
            annotation_type=annotation_type,
        )
    except Exception as e:
        logger.error(f"Error creating annotation volume zarr: {e}")
        return False, str(e)


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

MINIO_READY_TIMEOUT = session_minio.MINIO_READY_TIMEOUT
_minio_lock = session_minio._minio_lock
_wait_for_minio_ready = session_minio.wait_for_ready


def _mc_env(ip, port):
    return session_minio._mc_env(ip, port)


def ensure_minio_serving(zarr_path, crop_id, output_base_dir=None, mc_target_name=None):
    """
    Ensure MinIO is running and upload zarr file.

    Args:
        zarr_path: Path to zarr file to upload
        crop_id: Unique identifier for the crop
        output_base_dir: Base output directory (MinIO will use output_base_dir/.minio)
        mc_target_name: The bucket key, when it is not ``basename(zarr_path)``:
            instance corrections keep one key per ROI whichever snapshot on
            disk is served. ``crop_id`` must then be the key without
            ".zarr", since the sync finds a volume's chunks by its id.

    Returns:
        MinIO URL for the zarr file
    """
    _require_minio_binaries()
    server = session_minio.MinioServer(minio_state, volumes=annotation_volumes)
    return server.ensure_serving(zarr_path, crop_id, output_base_dir, mc_target_name=mc_target_name)


# ---------------------------------------------------------------------------
# Annotation sync, over session.sync
# ---------------------------------------------------------------------------

_sync_lock = session_sync._sync_lock  # the same lock: holding it here holds off every sync
_sync_failures = session_sync._sync_failures
SYNC_FAILURE_WARNING_INTERVAL = session_sync.SYNC_FAILURE_WARNING_INTERVAL
_sync_zarr_group_metadata = session_sync.sync_zarr_group_metadata
_diff_and_sync_chunks = session_sync.diff_and_sync_chunks


def _make_s3_filesystem():
    return session_minio.make_s3_filesystem(minio_state)


def _get_sync_worker_count() -> int:
    """How many threads chunk copies may use; see ``session.sync.worker_count``."""
    return session_sync.worker_count()


def _get_volume_metadata(volume_id, zarr_path=None):
    """The registered record of ``volume_id``, else rebuilt from ``zarr_path``'s attrs."""
    return session_sync.volume_record(volume_id, zarr_path, volumes=annotation_volumes)


def sync_all_annotations_from_minio(force: bool = True):
    """Sync every annotation volume in MinIO to local disk; see ``session.sync.sync_all``.

    Returns the number of volumes that had changed chunks, or -1 if MinIO is
    not initialized.
    """
    return session_sync.sync_all(force, state=minio_state, volumes=annotation_volumes)


def sync_annotation_volume_from_minio(volume_id, force=False, zarr_path=None):
    """Pull one volume's changed chunks to its zarr; see ``session.sync.sync_volume``."""
    return session_sync.sync_volume(
        volume_id, force, zarr_path, state=minio_state, volumes=annotation_volumes
    )


def _periodic_sync_once():
    session_sync.periodic_sync_once(state=minio_state, volumes=annotation_volumes)


def periodic_sync_annotations():
    """The periodic sync thread's loop; see ``session.sync.periodic_sync``."""
    session_sync.periodic_sync(state=minio_state, volumes=annotation_volumes)


def start_periodic_sync():
    """Start the periodic annotation sync thread if not already running."""
    session_sync.start_periodic_sync(minio_state, annotation_volumes)


# ---------------------------------------------------------------------------
# Instance corrections, over session.instance
# ---------------------------------------------------------------------------

def create_instance_annotation_volume_from_seg(*args, **kwargs):
    """``session.instance.seed_instance_volume``: returns (success, path or error)."""
    return session_instance.seed_instance_volume(*args, **kwargs)


def minio_backing_store_populated(output_dir, zarr_name):
    """Whether MinIO already holds painted chunks of ``zarr_name``; see
    ``session.instance.backing_store_populated``."""
    return session_instance.backing_store_populated(minio_state, output_dir, zarr_name)


def sync_instance_correction_from_minio(zarr_path, dst_path=None):
    """Copy an instance correction's MinIO object to disk; see
    ``session.instance.snapshot_from_minio``."""
    return session_instance.snapshot_from_minio(minio_state, zarr_path, dst_path)


def cc3d_relabel_instance_correction(zarr_path, target_label, snapshot_dir=None):
    """Split a label of an instance correction in MinIO; see ``session.instance.cc3d_relabel``."""
    return session_instance.cc3d_relabel(minio_state, zarr_path, target_label, snapshot_dir)
