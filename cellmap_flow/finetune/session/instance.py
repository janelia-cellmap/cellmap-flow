"""Instance corrections: annotation volumes seeded from an instance segmentation.

They are ordinary annotation volumes, only with instance-id labels (uint16
or uint32) in the ``AffinityTargetTransform`` scheme: 0 unannotated, 1 the
background shell around each instance, and instance id + 1 for each
instance. So the periodic sync, the pull before a mirror, session listing
and the overlay treat them like any other.

The routes in ``dashboard/routes/finetune/instance_correction`` document the
workflow; the functions here do the work for them and return
``(success, info or error)``.
"""

import gc
import logging
import os
import time

import numpy as np
import s3fs
import zarr

from cellmap_flow.finetune.session import minio
from cellmap_flow.finetune.session.volume import VolumeGeometry, create_volume_zarr

logger = logging.getLogger(__name__)


def seed_instance_volume(
    output_zarr_path,
    instance_zarr_path,
    dataset_path,
    model_name,
    input_size,
    input_voxel_size,
    dilation_radius_voxels=5,
    chunk_size=None,
    annotation_dtype="uint16",
    claimed_input_voxel_size=None,
    claimed_output_voxel_size=None,
    input_norm_config=None,
    postprocess_config=None,
):
    """Seed a paintable annotation volume at ``output_zarr_path`` from an instance zarr.

    ``instance_zarr_path`` is a zarr group whose ``s0`` holds the instance
    ids, with OME multiscales or ``resolution``/``offset`` attributes. The
    volume lies on that array's grid (shape, voxel size and position), so
    neuroglancer draws it over the segmentation it came from.

    Each instance gets a background shell ``dilation_radius_voxels`` thick,
    in the instance array's voxels (5 at 16 nm is an 80 nm shell).
    ``chunk_size`` defaults to the instance array's chunks; the dashboard
    passes the model's output shape, since each chunk is one training
    sample. ``annotation_dtype`` is "uint16" (up to 65534 instances) or
    "uint32". The model geometry (``input_size`` in voxels,
    ``input_voxel_size`` in nm, the claimed voxel sizes) and the chains are
    recorded in the root attrs, as for any volume.

    Returns ``(success, zarr path or error)``.
    """
    from scipy.ndimage import binary_dilation

    from cellmap_flow.io.metadata import read_array_meta
    from cellmap_flow.io.ome import ome_translation

    s0_path = os.path.join(instance_zarr_path, "s0")
    try:
        # OME multiscales or resolution/offset attributes alike; the
        # translation it gives is voxel 0's lower corner, in nm.
        meta = read_array_meta(s0_path).spatial()
        src_s0 = zarr.open(s0_path, mode="r")
    except Exception as e:
        return False, f"Failed to open instance zarr s0: {e}"
    source_voxel_size = [float(v) for v in meta.voxel_size]
    if all(v == 1.0 for v in source_voxel_size):
        # read_array_meta's fallback when the array says nothing.
        return False, f"{s0_path} has no voxel size in its metadata"
    # dataset_offset_nm is voxel 0's centre, the OME translation, as in every
    # volume (see volume.volume_corner_nm); meta.translation is the corner.
    source_offset_nm = ome_translation(meta.translation, source_voxel_size)
    source_shape = tuple(src_s0.shape)

    # The whole ROI fits in memory (~300 MB of uint32).
    instances = src_s0[:].astype(np.uint32)
    n_source_instances = int(instances.max())
    if n_source_instances + 1 > np.iinfo(np.dtype(annotation_dtype)).max:
        return False, (
            f"instance count {n_source_instances} + shell label 1 exceeds "
            f"{annotation_dtype} max {np.iinfo(np.dtype(annotation_dtype)).max}; "
            "use annotation_dtype='uint32'"
        )

    fg_mask = instances > 0
    # The shell: every instance grown by R face-connected steps, less the
    # instances themselves.
    shell_mask = binary_dilation(fg_mask, iterations=int(dilation_radius_voxels)) & ~fg_mask
    # Instance voxels get id + 1, keeping 1 for the shell (background); the
    # rest stays 0, unannotated.
    annotation = np.zeros(source_shape, dtype=annotation_dtype)
    annotation[shell_mask] = 1
    annotation[fg_mask] = (instances[fg_mask] + 1).astype(annotation_dtype)
    logger.info(
        f"Seeding {output_zarr_path} from {instance_zarr_path}: shape={source_shape}, "
        f"offset_nm={source_offset_nm}, voxel_nm={source_voxel_size}, "
        f"{n_source_instances} instances in {int(fg_mask.sum())} voxels, "
        f"{int(shell_mask.sum())} shell voxels (radius {dilation_radius_voxels})"
    )

    geometry = VolumeGeometry(
        output_voxel_size=source_voxel_size,
        input_voxel_size=list(input_voxel_size),
        claimed_output_voxel_size=claimed_output_voxel_size,
        claimed_input_voxel_size=claimed_input_voxel_size,
        chunk_size=list(src_s0.chunks if chunk_size is None else chunk_size),
        input_size=list(input_size),
        dataset_offset_nm=list(source_offset_nm),
        dataset_shape_voxels=list(source_shape),
    )
    try:
        create_volume_zarr(
            output_zarr_path,
            geometry,
            dataset_path=dataset_path,
            model_name=model_name,
            input_norm=input_norm_config,
            postprocess=postprocess_config,
            annotation_dtype=annotation_dtype,
        )
    except Exception as e:
        logger.error(f"Error creating annotation volume zarr: {e}")
        return False, str(e)

    try:
        # Chunks with no labels stay unwritten, as in any volume: the trainer
        # and the overlay count the chunks on disk as annotated.
        zarr.open_array(
            os.path.join(output_zarr_path, "annotation", "s0"),
            mode="r+",
            write_empty_chunks=False,
        )[:] = annotation
        # What the volume was seeded from and how, so it can be re-seeded
        # later without losing track of its source.
        root = zarr.open(output_zarr_path, mode="r+")
        root.attrs.update(
            seed_source_instance_zarr=str(instance_zarr_path),
            seed_dilation_radius_voxels=int(dilation_radius_voxels),
            seed_n_instances=n_source_instances,
        )
    except Exception as e:
        return False, f"Failed to write seeded annotation: {e}"

    # Free the intermediates before the next request allocates: the
    # dashboard shares its memory limit with inference servers.
    del instances, fg_mask, shell_mask, annotation
    gc.collect()
    return True, output_zarr_path


def populated_backing_store(state, output_dir, zarr_name):
    """Where MinIO's data directory holds painted chunks of ``zarr_name``, or None.

    The clobber guard of a fresh seed: re-seeding a zarr whose MinIO copy
    may hold edits not yet pulled would overwrite them with the seed on the
    first mirror. It looks at the files rather than asking MinIO because the
    decision comes before MinIO is started. A running MinIO keeps its data
    where it was first started, which need not be ``output_dir``, so the
    path is returned for the refusal to name.
    """
    process = state["process"]
    running = process is not None and process.poll() is None
    backing = minio.minio_root(state.get("output_base") if running else output_dir) / state["bucket"] / zarr_name
    s0_backing = backing / "annotation" / "s0"
    try:
        return backing if any(s0_backing.iterdir()) else None
    except OSError:
        return None


def bucket_key(volumes, zarr_path):
    """The MinIO key the volume at ``zarr_path`` is served under.

    Create serves every attach of a ROI under ``<roi_name>_annotation.zarr``,
    whatever the zarr on disk is called, and answers with the zarr's own
    path: a dated snapshot (``roi3_annotation_<ts>.zarr``) when one was
    reattached. So the key is that of the registered volume whose zarr this
    is (its id plus ".zarr"), and the basename only when none is.
    ``volumes`` is the volume registry, id -> record.
    """
    real = os.path.realpath(zarr_path)
    for volume_id, record in volumes.items():
        if record.get("zarr_path") and os.path.realpath(record["zarr_path"]) == real:
            return f"{volume_id}.zarr"
    return os.path.basename(os.path.normpath(zarr_path))


class _NotInMinio(Exception):
    pass


def _bucket_root(state, volumes, zarr_path):
    """``(s3, "<bucket>/<key>")``, or _NotInMinio if MinIO has no s0 for it."""
    s3 = minio.make_s3_filesystem(state)
    src_root = f"{state['bucket']}/{bucket_key(volumes, zarr_path)}"
    if not s3.exists(f"{src_root}/annotation/s0"):
        raise _NotInMinio(
            f"no MinIO bucket entry at {src_root}/annotation/s0 "
            "(was create-instance-correction ever POSTed for this zarr?)"
        )
    return s3, src_root


def snapshot_from_minio(state, volumes, zarr_path, dst_path=None):
    """Copy the MinIO state of a paintable instance-correction zarr to disk.

    Unlike ``sync.sync_volume``, which pulls changed chunks into the served
    volume, this copies the whole MinIO object, metadata included, so it can
    make a new zarr: a dated snapshot for rollback or audit, or a copy to
    train from.

    ``zarr_path`` names the MinIO bucket key (see :func:`bucket_key`; the
    registry ``volumes`` is read for it); the zarr itself is not opened.
    ``dst_path`` defaults to ``zarr_path``, but a fresh dated path
    (``.../roi3_annotation_<ts>.zarr``) is better: an in-place copy
    overwrites chunk files in their existing inodes, which corrupts any
    hardlinked copy of them (a ``cp -rl`` seed).

    Returns ``(True, {zarr_path, dst_path, keys_copied, keys_skipped,
    bytes_copied})`` or ``(False, error)``.
    """
    if not state["ip"] or not state["port"]:
        return False, "MinIO not running"
    dst_path = zarr_path if dst_path is None else dst_path
    try:
        s3, src_root = _bucket_root(state, volumes, zarr_path)
        # copy_store copies every key under the bucket root as it is: the
        # root and annotation group metadata, .zarray with its compressor, and
        # every chunk. The sync copies only annotation/, into a volume that
        # already exists, so a fresh destination would get no root .zgroup
        # and would not open as a group.
        os.makedirs(dst_path, exist_ok=True)
        n_copied, n_skipped, n_bytes = zarr.copy_store(
            s3fs.S3Map(root=src_root, s3=s3, check=False),
            zarr.DirectoryStore(str(dst_path)),
            if_exists="replace",
        )
        logger.info(
            f"Synced instance correction {src_root} from MinIO -> {dst_path}: "
            f"{n_copied} keys copied ({n_skipped} skipped, {n_bytes} bytes)"
        )
        return True, {
            "zarr_path": str(zarr_path),
            "dst_path": str(dst_path),
            "keys_copied": int(n_copied),
            "keys_skipped": int(n_skipped),
            "bytes_copied": int(n_bytes),
        }
    except _NotInMinio as e:
        return False, str(e)
    except Exception as e:
        logger.exception(f"Error syncing instance correction {zarr_path} -> {dst_path}")
        return False, str(e)


def cc3d_relabel(state, volumes, zarr_path, target_label, snapshot_dir=None):
    """Split one label of a paintable instance correction into its connected components.

    cc3d finds the 26-connected components of ``target_label``'s voxels; the
    largest keeps the label, and every other one gets a fresh id from
    ``max(existing) + 1``. ``target_label`` must be 2 or more: 0 is
    unannotated and 1 is background.

    The zarr is read and written in MinIO, which holds the in-progress brush
    edits; ``zarr_path`` only names the bucket key (see :func:`bucket_key`)
    and where the rollback snapshot goes. That snapshot of MinIO's state is taken
    before anything is written, to ``snapshot_dir`` (default
    ``<zarr dir>/snapshots``).

    Returns ``(True, {zarr_path, target_label, n_components, kept_voxels,
    splits: [{new_label, voxels}], snapshot_path})`` or ``(False, error)``.
    """
    if not state["ip"] or not state["port"]:
        return False, "MinIO not running"
    try:
        import cc3d  # connected-components-3d
    except ImportError as e:
        return False, f"cc3d not installed in the dashboard env: {e}"
    if int(target_label) < 2:
        return False, (
            f"target_label must be >= 2 (got {target_label}); "
            "label 0 is unannotated and label 1 is background"
        )

    try:
        s3, src_root = _bucket_root(state, volumes, zarr_path)
        # check=False skips S3Map's bucket probe: _bucket_root has just found s0.
        store = s3fs.S3Map(root=src_root, s3=s3, check=False)
        ann = zarr.open(store, mode="r+")["annotation/s0"]

        if snapshot_dir is None:
            snapshot_dir = os.path.join(os.path.dirname(os.path.normpath(zarr_path)), "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        name = os.path.basename(src_root).replace(".zarr", "")
        snapshot_path = os.path.join(
            snapshot_dir, f"{name}_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.zarr"
        )
        # Before any write, so there is a rollback point even if cc3d or the
        # write-back goes wrong. copy_store streams one chunk at a time, so
        # the volume is never held twice in memory.
        zarr.copy_store(store, zarr.DirectoryStore(snapshot_path), if_exists="replace")

        # The whole ROI in memory (~600 MB of uint16) rather than a GET per chunk per pass.
        arr = ann[:]
        mask = arr == int(target_label)
        n_target_voxels = int(mask.sum())
        if n_target_voxels == 0:
            return False, f"no voxels match label {target_label}"
        labeled, n_comp = cc3d.connected_components(mask, connectivity=26, return_N=True)

        # cc3d labels the components 1..n_comp; the largest keeps target_label.
        sizes = cc3d.statistics(labeled)["voxel_counts"]
        by_size = sorted(((i + 1, int(sizes[i + 1])) for i in range(n_comp)), key=lambda x: -x[1])
        splits = []
        for new_id, (component, voxels) in enumerate(by_size[1:], start=int(arr.max()) + 1):
            arr[labeled == component] = new_id
            splits.append({"new_label": int(new_id), "voxels": int(voxels)})
        info = {
            "zarr_path": str(zarr_path),
            "target_label": int(target_label),
            "n_components": int(n_comp),
            "kept_voxels": int(by_size[0][1]),
            "splits": splits,
            "snapshot_path": snapshot_path,
        }
        logger.info(f"[cc3d-relabel] {src_root}: label {target_label} has {n_comp} component(s); "
                    f"split off {splits}; snapshot at {snapshot_path}")
        if not splits:
            # Nothing on MinIO changes; the snapshot is still a backup.
            return True, {**info, "note": "single component, no split performed"}
        # Every chunk is PUT, not only the changed ones: acceptable at ROI
        # scale, ~15 s for 900 MB.
        ann[:] = arr
        return True, info
    except _NotInMinio as e:
        return False, str(e)
    except Exception as e:
        logger.exception(f"Error in cc3d_relabel of {zarr_path}")
        return False, str(e)
