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

import numpy as np
import zarr

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
