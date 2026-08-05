"""
Zarr creation utilities for annotation crops and volumes.

Creates OME-NGFF v0.4 compliant zarr structures for Neuroglancer painting.
"""

import logging
from datetime import datetime

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def create_correction_zarr(
    zarr_path,
    raw_crop_shape,
    raw_voxel_size,
    raw_offset,
    annotation_crop_shape,
    annotation_voxel_size,
    annotation_offset,
    dataset_path,
    model_name,
    output_channels,
    raw_dtype="uint8",
    create_mask=False,
):
    """
    Create a correction zarr with OME-NGFF v0.4 metadata.

    Structure:
        crop_id.zarr/
            raw/s0/          (uint8, shape=raw_crop_shape)
            annotation/s0/   (uint8, shape=annotation_crop_shape, empty for manual annotation)
            mask/s0/         (optional, uint8, shape=annotation_crop_shape)
            .zattrs          (metadata)

    Args:
        zarr_path: Path to create zarr
        raw_crop_shape: Shape in voxels for raw [z, y, x]
        raw_voxel_size: Voxel size in nm for raw [z, y, x]
        raw_offset: Offset in voxels for raw [z, y, x]
        annotation_crop_shape: Shape in voxels for annotation [z, y, x]
        annotation_voxel_size: Voxel size in nm for annotation [z, y, x]
        annotation_offset: Offset in voxels for annotation [z, y, x]
        dataset_path: Source dataset path
        model_name: Model name for metadata
        output_channels: Number of output channels
        create_mask: Whether to create a mask group (default: False)

    Returns:
        (success: bool, info: str)
    """
    try:
        # Helper to add OME-NGFF metadata
        def add_ome_ngff_metadata(group, name, voxel_size, translation_offset=None):
            """Add OME-NGFF v0.4 metadata."""
            # Calculate physical translation
            if translation_offset is not None:
                physical_translation = [
                    float(o * v) for o, v in zip(translation_offset, voxel_size)
                ]
            else:
                physical_translation = [0.0, 0.0, 0.0]

            # Coordinate transformations
            transforms = [{"type": "scale", "scale": [float(v) for v in voxel_size]}]

            if translation_offset is not None:
                transforms.append(
                    {"type": "translation", "translation": physical_translation}
                )

            # OME-NGFF v0.4 metadata
            group.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "name": name,
                    "axes": [
                        {"name": "z", "type": "space", "unit": "nanometer"},
                        {"name": "y", "type": "space", "unit": "nanometer"},
                        {"name": "x", "type": "space", "unit": "nanometer"},
                    ],
                    "datasets": [
                        {"path": "s0", "coordinateTransformations": transforms}
                    ],
                }
            ]

        # Open zarr root
        root = zarr.open(zarr_path, mode="w")

        # Create raw group (will be filled by user later)
        raw_group = root.create_group("raw")
        raw_s0 = raw_group.create_dataset(
            "s0",
            shape=tuple(raw_crop_shape),
            chunks=(64, 64, 64),
            dtype=raw_dtype,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )
        add_ome_ngff_metadata(raw_group, "raw", raw_voxel_size, raw_offset)

        # Create annotation group (empty, will be filled by user annotations)
        annotation_group = root.create_group("annotation")
        annotation_s0 = annotation_group.create_dataset(
            "s0",
            shape=tuple(annotation_crop_shape),
            chunks=(64, 64, 64),
            dtype="uint8",
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )
        add_ome_ngff_metadata(
            annotation_group, "annotation", annotation_voxel_size, annotation_offset
        )

        # Optionally create mask group (will be filled by user annotations)
        if create_mask:
            mask_group = root.create_group("mask")
            mask_s0 = mask_group.create_dataset(
                "s0",
                shape=tuple(annotation_crop_shape),
                chunks=(64, 64, 64),
                dtype="uint8",
                compressor=zarr.Blosc(
                    cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE
                ),
                fill_value=0,
            )
            add_ome_ngff_metadata(
                mask_group, "mask", annotation_voxel_size, annotation_offset
            )

        # Add root metadata
        root.attrs["roi"] = {
            "raw_offset": (
                raw_offset.tolist()
                if hasattr(raw_offset, "tolist")
                else list(raw_offset)
            ),
            "raw_shape": (
                raw_crop_shape.tolist()
                if hasattr(raw_crop_shape, "tolist")
                else list(raw_crop_shape)
            ),
            "annotation_offset": (
                annotation_offset.tolist()
                if hasattr(annotation_offset, "tolist")
                else list(annotation_offset)
            ),
            "annotation_shape": (
                annotation_crop_shape.tolist()
                if hasattr(annotation_crop_shape, "tolist")
                else list(annotation_crop_shape)
            ),
        }
        root.attrs["raw_voxel_size"] = (
            raw_voxel_size.tolist()
            if hasattr(raw_voxel_size, "tolist")
            else list(raw_voxel_size)
        )
        root.attrs["annotation_voxel_size"] = (
            annotation_voxel_size.tolist()
            if hasattr(annotation_voxel_size, "tolist")
            else list(annotation_voxel_size)
        )
        root.attrs["model_name"] = model_name
        root.attrs["dataset_path"] = dataset_path
        root.attrs["created_at"] = datetime.now().isoformat()

        logger.info(f"Created correction zarr at {zarr_path}")
        logger.info(
            f"  Raw crop shape: {raw_crop_shape}, voxel size: {raw_voxel_size}, offset: {raw_offset}"
        )
        logger.info(
            f"  Annotation crop shape: {annotation_crop_shape}, voxel size: {annotation_voxel_size}, offset: {annotation_offset}"
        )
        logger.info(f"  Mask created: {create_mask}")

        return True, zarr_path

    except Exception as e:
        logger.error(f"Error creating zarr: {e}")
        return False, str(e)


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
):
    """
    Create a sparse annotation volume zarr covering the full dataset extent.

    The volume has chunk_size = model output_size so each chunk maps to one
    training sample. Only metadata files are created (no chunk data), so the
    zarr is tiny regardless of dataset size.

    Label scheme: 0=unannotated (ignored), 1=background, 2=foreground.

    Args:
        zarr_path: Path to create zarr
        dataset_shape_voxels: Full dataset shape in output voxels [z, y, x]
        output_voxel_size: nm per voxel for output [z, y, x]
        dataset_offset_nm: Dataset offset in nm [z, y, x]
        chunk_size: Chunk size in voxels = model output_size [z, y, x]
        dataset_path: Source dataset path
        model_name: Model name for metadata
        input_size: Model input size in voxels [z, y, x]
        input_voxel_size: nm per voxel for input [z, y, x]

    Returns:
        (success: bool, info: str)
    """
    try:
        root = zarr.open(zarr_path, mode="w")

        # Create annotation group with chunks = output_size
        annotation_group = root.create_group("annotation")
        annotation_group.create_dataset(
            "s0",
            shape=tuple(dataset_shape_voxels),
            chunks=tuple(chunk_size),
            dtype="uint8",
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )

        # OME-NGFF v0.4 metadata with translation for dataset offset
        physical_translation = [
            float(o) for o in dataset_offset_nm
        ]
        transforms = [
            {"type": "scale", "scale": [float(v) for v in output_voxel_size]},
            {"type": "translation", "translation": physical_translation},
        ]
        annotation_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "annotation",
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [
                    {"path": "s0", "coordinateTransformations": transforms}
                ],
            }
        ]

        # Root metadata marking this as an annotation volume
        root.attrs["type"] = "annotation_volume"
        root.attrs["model_name"] = model_name
        root.attrs["dataset_path"] = dataset_path
        root.attrs["chunk_size"] = (
            chunk_size.tolist() if hasattr(chunk_size, "tolist") else list(chunk_size)
        )
        root.attrs["output_voxel_size"] = (
            output_voxel_size.tolist()
            if hasattr(output_voxel_size, "tolist")
            else list(output_voxel_size)
        )
        root.attrs["input_size"] = (
            input_size.tolist() if hasattr(input_size, "tolist") else list(input_size)
        )
        root.attrs["input_voxel_size"] = (
            input_voxel_size.tolist()
            if hasattr(input_voxel_size, "tolist")
            else list(input_voxel_size)
        )
        root.attrs["dataset_offset_nm"] = (
            dataset_offset_nm.tolist()
            if hasattr(dataset_offset_nm, "tolist")
            else list(dataset_offset_nm)
        )
        root.attrs["dataset_shape_voxels"] = (
            dataset_shape_voxels.tolist()
            if hasattr(dataset_shape_voxels, "tolist")
            else list(dataset_shape_voxels)
        )
        root.attrs["created_at"] = datetime.now().isoformat()

        logger.info(f"Created annotation volume zarr at {zarr_path}")
        logger.info(
            f"  Shape: {dataset_shape_voxels}, chunks: {chunk_size}, "
            f"voxel size: {output_voxel_size}"
        )

        return True, zarr_path

    except Exception as e:
        logger.error(f"Error creating annotation volume zarr: {e}")
        return False, str(e)
