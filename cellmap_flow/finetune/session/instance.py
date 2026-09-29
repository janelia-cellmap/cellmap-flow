"""Instance corrections: annotation volumes seeded from an instance segmentation.

They are ordinary annotation volumes, only with instance-id labels (uint16
or uint32) in the ``AffinityTargetTransform`` scheme: 0 unannotated, 1 the
background shell around each instance, and instance id + 1 for each
instance. So the periodic sync, the pull before a mirror, session listing
and the overlay treat them like any other.
"""

import gc
import logging
import os
import time
import traceback

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
    """Seed a paintable annotation volume from an existing instance zarr.

    The volume lies on the instance zarr's grid (shape, voxel size and
    position), so neuroglancer draws it over the segmentation it came from.

    Args:
        output_zarr_path: Where to write the new annotation zarr.
        instance_zarr_path: A zarr group whose ``s0`` holds the instance ids,
            with OME multiscales or ``resolution``/``offset`` attributes.
        dataset_path: The raw dataset the volume annotates.
        model_name: The model the volume is for.
        input_size: The model's input shape, in voxels.
        input_voxel_size: The model's input voxel size in nm.
        dilation_radius_voxels: Number of voxels to dilate each instance by
            to form the background shell. 5 @ 16nm output = 80 nm shell.
        chunk_size: Annotation chunks z,y,x; defaults to the instance
            array's own chunks. Each chunk is one training sample, so the
            dashboard passes the model's output shape.
        annotation_dtype: "uint16" (up to 65534 instances) or "uint32".
        claimed_*_voxel_size, input_norm_config, postprocess_config: recorded
            in the root attrs, as for any volume.

    Returns:
        (success: bool, zarr_path_or_error: str)
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
    # dataset_offset_nm is voxel 0's centre, the OME translation (see
    # volume.volume_corner_nm). The corner put the seeded labels half a
    # voxel off the segmentation they were seeded from.
    source_offset_nm = ome_translation(meta.translation, source_voxel_size)
    source_shape = tuple(src_s0.shape)
    if chunk_size is None:
        chunk_size = list(src_s0.chunks)
    logger.info(
        f"Seeding annotation volume from {instance_zarr_path}: "
        f"shape={source_shape}, offset_nm={source_offset_nm}, "
        f"voxel_nm={source_voxel_size}, dilation_r={dilation_radius_voxels} vox"
    )

    # Load the instance array fully (ROI-sized, fits in memory — ~300 MB uint32).
    instances = src_s0[:].astype(np.uint32)
    n_source_instances = int(instances.max())
    if n_source_instances + 1 > np.iinfo(np.dtype(annotation_dtype)).max:
        return False, (
            f"instance count {n_source_instances} + shell label 1 exceeds "
            f"{annotation_dtype} max {np.iinfo(np.dtype(annotation_dtype)).max}; "
            "use annotation_dtype='uint32'"
        )

    fg_mask = instances > 0
    # Dilation shell: grow each instance by R voxels and subtract the original.
    # iterate a face-connected 3D structuring element R times for a ball-ish shell.
    logger.info(
        f"Computing dilation shell (radius={dilation_radius_voxels} voxels)..."
    )
    dilated = binary_dilation(
        fg_mask, iterations=int(dilation_radius_voxels)
    )
    shell_mask = dilated & (~fg_mask)

    # Build annotation: shell=1, instance voxels=(id+1) to reserve label 1 for
    # background. Unannotated stays 0. Everything done in the target dtype.
    annotation = np.zeros(source_shape, dtype=annotation_dtype)
    annotation[shell_mask] = 1
    annotation[fg_mask] = (instances[fg_mask] + 1).astype(annotation_dtype)

    n_shell = int(shell_mask.sum())
    n_fg = int(fg_mask.sum())
    logger.info(
        f"annotation labels: {n_fg} fg voxels ({n_source_instances} instances), "
        f"{n_shell} shell (bg) voxels, "
        f"{int((annotation == 0).sum())} unannotated"
    )

    geometry = VolumeGeometry(
        output_voxel_size=list(source_voxel_size),
        input_voxel_size=list(input_voxel_size),
        claimed_output_voxel_size=claimed_output_voxel_size,
        claimed_input_voxel_size=claimed_input_voxel_size,
        chunk_size=list(chunk_size),
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

    # Write the seeded annotation into annotation/s0.
    try:
        root = zarr.open(output_zarr_path, mode="r+")
        # Chunks with no labels stay unwritten, as in any volume: the
        # trainer and the overlay count the chunks on disk as annotated.
        s0 = zarr.open_array(
            os.path.join(output_zarr_path, "annotation", "s0"),
            mode="r+",
            write_empty_chunks=False,
        )
        s0[:] = annotation
        # Record the seed source + parameters so we can re-seed later
        # without losing track of what this zarr was made from.
        root.attrs["seed_source_instance_zarr"] = str(instance_zarr_path)
        root.attrs["seed_dilation_radius_voxels"] = int(dilation_radius_voxels)
        root.attrs["seed_n_instances"] = n_source_instances
    except Exception as e:
        return False, f"Failed to write seeded annotation: {e}"

    # Drop the large intermediates before returning to the request handler:
    # under a tight memory limit shared with inference servers, overlapping
    # them with the next allocations matters.
    del instances, fg_mask, dilated, shell_mask, annotation
    gc.collect()

    return True, output_zarr_path


def backing_store_populated(state, output_dir, zarr_name):
    """Whether MinIO's data directory already holds painted chunks of ``zarr_name``.

    The clobber guard of a fresh seed: re-seeding a zarr whose MinIO copy
    may hold edits not yet pulled would overwrite them with the seed on the
    first mirror. It looks at the files rather than asking MinIO because the
    decision comes before MinIO is started. A running MinIO keeps its data
    where it was first started, which need not be ``output_dir``.
    """
    process = state["process"]
    running = process is not None and process.poll() is None
    root = minio.minio_root(state.get("output_base") if running else output_dir)
    s0_backing = root / state["bucket"] / zarr_name / "annotation" / "s0"
    if not s0_backing.exists():
        return False
    try:
        return any(s0_backing.iterdir())
    except Exception:
        return False


def snapshot_from_minio(state, zarr_path, dst_path=None):
    """Copy the MinIO state of a paintable instance-correction zarr to disk.

    Unlike ``sync.sync_volume``, which pulls changed chunks into the served
    volume, this copies the whole MinIO object, metadata included, so it can
    make a new zarr: a dated snapshot for rollback or audit, or a copy to
    train from.

    Args:
        zarr_path: The instance-correction zarr, e.g.
            `.../instance_corrections/roi3_annotation.zarr`. Its basename is
            the MinIO bucket key; the zarr itself is not opened.
        dst_path: Where to write the copy; defaults to `zarr_path`. Prefer a
            fresh dated path (e.g. `.../roi3_annotation_<ts>.zarr`): an
            in-place copy overwrites chunk files in their existing inodes,
            which corrupts any hardlinked copy of them (a `cp -rl` seed).
            The route only accepts a `.zarr` beside `zarr_path`.

    Returns:
        (success: bool, info_or_error: dict or str). On success, info is
        {"zarr_path", "dst_path", "keys_copied", "keys_skipped",
        "bytes_copied"}.
    """
    if not state["ip"] or not state["port"]:
        return False, "MinIO not running"

    if dst_path is None:
        dst_path = zarr_path

    try:
        zarr_name = os.path.basename(os.path.normpath(zarr_path))
        s3 = minio.make_s3_filesystem(state)
        src_root = f"{state['bucket']}/{zarr_name}"

        if not s3.exists(f"{src_root}/annotation/s0"):
            return False, (
                f"no MinIO bucket entry at {src_root}/annotation/s0 "
                "(was create-instance-correction ever POSTed for this zarr?)"
            )

        # `zarr.copy_store` copies every key under the bucket root as it
        # is: the root and annotation group metadata, `.zarray` with its
        # compressor, and every chunk. The sync only syncs `annotation/`
        # into a volume that already exists, so a fresh destination would
        # get no root `.zgroup` and would not open as a group.
        src_store = s3fs.S3Map(root=src_root, s3=s3, check=False)
        os.makedirs(dst_path, exist_ok=True)
        dst_store = zarr.DirectoryStore(str(dst_path))
        n_copied, n_skipped, n_bytes = zarr.copy_store(
            src_store, dst_store, if_exists="replace"
        )

        logger.info(
            f"Synced instance correction {zarr_name} from MinIO -> "
            f"{dst_path}: {n_copied} keys copied "
            f"({n_skipped} skipped, {n_bytes} bytes)"
        )

        return True, {
            "zarr_path": str(zarr_path),
            "dst_path": str(dst_path),
            "keys_copied": int(n_copied),
            "keys_skipped": int(n_skipped),
            "bytes_copied": int(n_bytes),
        }

    except Exception as e:
        logger.error(f"Error syncing instance correction {zarr_path} -> {dst_path}: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)


def cc3d_relabel(state, zarr_path, target_label, snapshot_dir=None):
    """Split a single label in a paintable instance-correction zarr by
    running 26-connectivity cc3d on its voxel mask and reassigning all
    components except the largest to fresh unused instance IDs.

    Typical workflow: the user erases a thin bridge between two fused
    mitochondria in NG's brush tool (still sharing `target_label`), then
    POSTs this route with that label. cc3d finds the now-separated
    components; we keep the largest as `target_label` and reassign the
    smaller components to `max(existing) + 1 ...`. A hard reload of the
    NG tab shows the split colors.

    Reads and writes the MinIO-backed zarr in place (MinIO is the source
    of truth for in-progress brush edits; the user-visible `zarr_path` is
    only used to derive the bucket object name and to place the rollback
    snapshot next to). Before writing, snapshots the current MinIO state
    to a local DirectoryStore for rollback safety.

    Args:
        zarr_path: Absolute path to the user-visible instance-correction
            zarr. Its basename is used as the MinIO object key.
        target_label: The instance ID to split (must be >= 2, since 0 is
            unannotated and 1 is background in the AffinityTargetTransform
            scheme).
        snapshot_dir: Where to drop the pre-split rollback snapshot.
            Defaults to `<parent_of_zarr_path>/snapshots/`.

    Returns:
        (success: bool, info_or_error: dict or str). On success, info is
        {"zarr_path", "target_label", "n_components", "kept_voxels",
         "splits": [{"new_label", "voxels"}, ...], "snapshot_path"}.
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
        zarr_name = os.path.basename(os.path.normpath(zarr_path))
        s3 = minio.make_s3_filesystem(state)
        src_root = f"{state['bucket']}/{zarr_name}"
        if not s3.exists(f"{src_root}/annotation/s0"):
            return False, (
                f"no MinIO bucket entry at {src_root}/annotation/s0 "
                "(was create-instance-correction ever POSTed for this zarr?)"
            )

        # Open the MinIO-backed zarr r+. S3Map with check=False skips the
        # initial bucket-probe round trip.
        store = s3fs.S3Map(root=src_root, s3=s3, check=False)
        root = zarr.open(store, mode="r+")
        ann = root["annotation/s0"]
        logger.info(
            f"[cc3d-relabel] opened {src_root}: shape={ann.shape} "
            f"dtype={ann.dtype} chunks={ann.chunks}"
        )

        # Snapshot to a local DirectoryStore before any writes so the user
        # has a guaranteed rollback point even if cc3d or the writeback
        # misbehaves. `zarr.copy_store` streams chunks one at a time without
        # materializing the full volume twice in memory.
        if snapshot_dir is None:
            snapshot_dir = os.path.join(
                os.path.dirname(os.path.normpath(zarr_path)), "snapshots"
            )
        os.makedirs(snapshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        snap_basename = zarr_name.replace(".zarr", "") + f"_snapshot_{ts}.zarr"
        snapshot_path = os.path.join(snapshot_dir, snap_basename)
        logger.info(f"[cc3d-relabel] snapshotting to {snapshot_path}")
        local_store = zarr.DirectoryStore(snapshot_path)
        zarr.copy_store(store, local_store, if_exists="replace")

        # Load the full annotation into memory — fits comfortably at ROI
        # scale (~600 MB uint16) and avoids per-chunk GET churn during the
        # cc3d + reassignment passes.
        logger.info("[cc3d-relabel] loading full annotation/s0 into memory")
        arr = ann[:]

        mask = arr == int(target_label)
        n_target_voxels = int(mask.sum())
        if n_target_voxels == 0:
            return False, f"no voxels match label {target_label}"
        logger.info(
            f"[cc3d-relabel] target mask: {n_target_voxels} voxels labeled "
            f"{target_label}"
        )

        labeled, n_comp = cc3d.connected_components(
            mask, connectivity=26, return_N=True
        )
        logger.info(f"[cc3d-relabel] cc3d found {n_comp} connected components")

        if n_comp == 1:
            logger.info(
                "[cc3d-relabel] only one connected component — nothing to "
                "split. (Erase may not have fully cut the bridge, or the "
                "label was already single-component.)"
            )
            # No writeback: the snapshot is still useful as a backup, but
            # nothing on MinIO changed.
            return True, {
                "zarr_path": str(zarr_path),
                "target_label": int(target_label),
                "n_components": int(n_comp),
                "kept_voxels": n_target_voxels,
                "splits": [],
                "snapshot_path": snapshot_path,
                "note": "single component, no split performed",
            }

        # Pick the largest component as the one to keep under target_label,
        # reassign every other component to fresh unused IDs. cc3d labels
        # background as 0 and foreground components as 1..n_comp.
        stats = cc3d.statistics(labeled)
        sizes = stats["voxel_counts"]
        fg_sizes = sorted(
            ((i + 1, int(sizes[i + 1])) for i in range(n_comp)),
            key=lambda x: -x[1],
        )
        biggest_comp, kept_voxels = fg_sizes[0]
        logger.info(
            f"[cc3d-relabel] keeping component {biggest_comp} "
            f"({kept_voxels} voxels) as label {target_label}"
        )

        existing_max = int(arr.max())
        next_id = existing_max + 1
        splits = []
        for comp_i, sz in fg_sizes[1:]:
            new_id = next_id
            next_id += 1
            arr[labeled == comp_i] = new_id
            splits.append({"new_label": int(new_id), "voxels": int(sz)})
            logger.info(
                f"[cc3d-relabel]   component {comp_i}: {sz} voxels "
                f"-> new label {new_id}"
            )

        # Write the full array back; zarr's S3 backend breaks this into
        # per-chunk PUTs. Every chunk is PUT (not just changed ones) —
        # acceptable at ROI scale, the prototype measured ~15 s for 900 MB.
        logger.info("[cc3d-relabel] writing back to MinIO")
        ann[:] = arr

        return True, {
            "zarr_path": str(zarr_path),
            "target_label": int(target_label),
            "n_components": int(n_comp),
            "kept_voxels": int(kept_voxels),
            "splits": splits,
            "snapshot_path": snapshot_path,
        }

    except Exception as e:
        logger.error(f"Error in cc3d_relabel_instance_correction: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)
