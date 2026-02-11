#!/usr/bin/env python3
"""
Generate sparse point-based corrections for partial annotation training.

This script:
1. Loads existing corrections with eroded mito
2. Samples random points from eroded regions (foreground) and background
3. Creates sparse annotations around each point
4. Saves with datetime stamps for tracking

Labels:
- 0: Unannotated (majority of volume)
- 1: Background (annotated background around mito)
- 2: Foreground (annotated mito in eroded regions)
"""

import numpy as np
import zarr
from pathlib import Path
from datetime import datetime
import uuid
from scipy.ndimage import distance_transform_edt


def sample_points_from_mask(
    mask: np.ndarray,
    num_points: int,
    min_distance: int = 5
) -> np.ndarray:
    """
    Sample random points from a binary mask with minimum distance constraint.

    Args:
        mask: Binary mask (True where valid to sample)
        num_points: Number of points to sample
        min_distance: Minimum distance between points in voxels

    Returns:
        Array of shape (N, 3) with point coordinates (z, y, x)
    """
    valid_coords = np.argwhere(mask)

    if len(valid_coords) == 0:
        return np.empty((0, 3), dtype=int)

    if len(valid_coords) <= num_points:
        return valid_coords

    # Sample with minimum distance constraint
    points = []
    available_mask = mask.copy()

    for _ in range(num_points):
        if not np.any(available_mask):
            break

        # Sample from available locations
        available_coords = np.argwhere(available_mask)
        idx = np.random.randint(len(available_coords))
        point = available_coords[idx]
        points.append(point)

        # Mark neighborhood as unavailable
        z, y, x = point
        z_min = max(0, z - min_distance)
        z_max = min(mask.shape[0], z + min_distance + 1)
        y_min = max(0, y - min_distance)
        y_max = min(mask.shape[1], y + min_distance + 1)
        x_min = max(0, x - min_distance)
        x_max = min(mask.shape[2], x + min_distance + 1)

        available_mask[z_min:z_max, y_min:y_max, x_min:x_max] = False

    return np.array(points)


def create_background_mask(
    mito_mask: np.ndarray,
    min_distance: int = 2,
    max_distance: int = 10
) -> np.ndarray:
    """
    Create a background sampling mask around mito regions.

    Args:
        mito_mask: Binary mito mask
        min_distance: Minimum distance from mito (in voxels)
        max_distance: Maximum distance from mito (in voxels)

    Returns:
        Binary mask for background sampling
    """
    # Compute distance transform from mito
    dist = distance_transform_edt(~mito_mask.astype(bool))

    # Background is within a shell around mito
    background_mask = (dist >= min_distance) & (dist <= max_distance)

    return background_mask


def create_sparse_annotation_mask(
    shape: tuple,
    foreground_points: np.ndarray,
    background_points: np.ndarray,
    annotation_radius: int = 3
) -> np.ndarray:
    """
    Create sparse annotation mask with labeled spheres around points.

    Args:
        shape: Shape of output mask (Z, Y, X)
        foreground_points: Foreground point coordinates (N, 3)
        background_points: Background point coordinates (M, 3)
        annotation_radius: Radius of annotation sphere around each point

    Returns:
        Sparse mask with: 0=unannotated, 1=background, 2=foreground
    """
    mask = np.zeros(shape, dtype=np.uint8)

    # Create a sphere kernel
    kernel_size = 2 * annotation_radius + 1
    center = annotation_radius
    z_grid, y_grid, x_grid = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    sphere = ((z_grid - center)**2 + (y_grid - center)**2 + (x_grid - center)**2) <= annotation_radius**2

    # Annotate background points (label = 1)
    for point in background_points:
        z, y, x = point
        z_min = max(0, z - annotation_radius)
        z_max = min(shape[0], z + annotation_radius + 1)
        y_min = max(0, y - annotation_radius)
        y_max = min(shape[1], y + annotation_radius + 1)
        x_min = max(0, x - annotation_radius)
        x_max = min(shape[2], x + annotation_radius + 1)

        # Get valid kernel region
        kz_min = annotation_radius - (z - z_min)
        kz_max = annotation_radius + (z_max - z)
        ky_min = annotation_radius - (y - y_min)
        ky_max = annotation_radius + (y_max - y)
        kx_min = annotation_radius - (x - x_min)
        kx_max = annotation_radius + (x_max - x)

        mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.where(
            sphere[kz_min:kz_max, ky_min:ky_max, kx_min:kx_max],
            1,
            mask[z_min:z_max, y_min:y_max, x_min:x_max]
        )

    # Annotate foreground points (label = 2, overwrites background if overlapping)
    for point in foreground_points:
        z, y, x = point
        z_min = max(0, z - annotation_radius)
        z_max = min(shape[0], z + annotation_radius + 1)
        y_min = max(0, y - annotation_radius)
        y_max = min(shape[1], y + annotation_radius + 1)
        x_min = max(0, x - annotation_radius)
        x_max = min(shape[2], x + annotation_radius + 1)

        # Get valid kernel region
        kz_min = annotation_radius - (z - z_min)
        kz_max = annotation_radius + (z_max - z)
        ky_min = annotation_radius - (y - y_min)
        ky_max = annotation_radius + (y_max - y)
        kx_min = annotation_radius - (x - x_min)
        kx_max = annotation_radius + (x_max - x)

        mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.where(
            sphere[kz_min:kz_max, ky_min:ky_max, kx_min:kx_max],
            2,
            mask[z_min:z_max, y_min:y_max, x_min:x_max]
        )

    return mask


def add_ome_ngff_metadata(group, name, voxel_size, translation_offset=None):
    """Add OME-NGFF v0.4 metadata to a zarr group."""
    transforms = []

    # Add scale first
    transforms.append({
        'type': 'scale',
        'scale': voxel_size.tolist()
    })

    # Then add translation if provided
    if translation_offset is not None:
        physical_translation = (np.array(translation_offset) * np.array(voxel_size)).tolist()
        transforms.append({
            'type': 'translation',
            'translation': physical_translation
        })

    group.attrs['multiscales'] = [{
        'version': '0.4',
        'name': name,
        'axes': [
            {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'x', 'type': 'space', 'unit': 'nanometer'}
        ],
        'datasets': [{
            'path': 's0',
            'coordinateTransformations': transforms
        }]
    }]


def main():
    # Configuration
    INPUT_CORRECTIONS = "/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections/mito_liver.zarr"
    OUTPUT_DIR = Path("/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections/sparse_points")

    NUM_FOREGROUND_POINTS = 1000
    NUM_BACKGROUND_POINTS = 1000
    ANNOTATION_RADIUS = 3  # Radius of annotation sphere around each point
    MIN_POINT_DISTANCE = 5  # Minimum distance between sampled points
    BACKGROUND_MIN_DIST = 2  # Min distance from mito for background sampling
    BACKGROUND_MAX_DIST = 10  # Max distance from mito for background sampling

    # Voxel size (from original data - 16nm isotropic)
    voxel_size = np.array([16, 16, 16])

    print("="*60)
    print("Sparse Point Correction Generator")
    print("="*60)
    print(f"Input corrections: {INPUT_CORRECTIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Foreground points: {NUM_FOREGROUND_POINTS}")
    print(f"Background points: {NUM_BACKGROUND_POINTS}")
    print(f"Annotation radius: {ANNOTATION_RADIUS} voxels")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load input corrections
    input_root = zarr.open(INPUT_CORRECTIONS, mode='r')
    correction_ids = [k for k in input_root.keys() if not k.startswith('.')]

    print(f"Found {len(correction_ids)} input corrections")
    print()

    # Process each correction
    for idx, corr_id in enumerate(correction_ids):
        print(f"[{idx+1}/{len(correction_ids)}] Processing correction {corr_id}...")

        # Load data
        corr_group = input_root[corr_id]
        raw = np.array(corr_group['raw/s0'])
        mito_mask = np.array(corr_group['mask/s0'])
        prediction = np.array(corr_group['prediction/s0'])

        print(f"  Raw shape: {raw.shape}")
        print(f"  Mask shape: {mito_mask.shape}")
        print(f"  Mito voxels: {np.sum(mito_mask > 0):,}")

        # Create binary mask for sampling
        mito_binary = mito_mask > 0

        # Sample foreground points (from eroded mito)
        foreground_points = sample_points_from_mask(
            mito_binary,
            NUM_FOREGROUND_POINTS,
            min_distance=MIN_POINT_DISTANCE
        )
        print(f"  Sampled {len(foreground_points)} foreground points")

        # Create background sampling mask
        background_sampling_mask = create_background_mask(
            mito_binary,
            min_distance=BACKGROUND_MIN_DIST,
            max_distance=BACKGROUND_MAX_DIST
        )
        background_voxels = np.sum(background_sampling_mask)
        print(f"  Background sampling region: {background_voxels:,} voxels")

        # Sample background points
        background_points = sample_points_from_mask(
            background_sampling_mask,
            NUM_BACKGROUND_POINTS,
            min_distance=MIN_POINT_DISTANCE
        )
        print(f"  Sampled {len(background_points)} background points")

        if len(foreground_points) == 0 or len(background_points) == 0:
            print(f"  ⚠ Skipping - insufficient points sampled")
            continue

        # Create sparse annotation mask
        # Labels: 0=unannotated, 1=background, 2=foreground
        sparse_mask = create_sparse_annotation_mask(
            mito_mask.shape,
            foreground_points,
            background_points,
            annotation_radius=ANNOTATION_RADIUS
        )

        annotated_voxels = np.sum(sparse_mask > 0)
        foreground_voxels = np.sum(sparse_mask == 2)
        background_voxels = np.sum(sparse_mask == 1)
        annotation_fraction = annotated_voxels / sparse_mask.size * 100

        print(f"  Annotated voxels: {annotated_voxels:,} ({annotation_fraction:.2f}%)")
        print(f"    - Foreground (2): {foreground_voxels:,}")
        print(f"    - Background (1): {background_voxels:,}")

        # Create output zarr with datetime stamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"sparse_points_{timestamp}_{idx:03d}.zarr"
        output_path = OUTPUT_DIR / output_name

        # Generate unique correction ID
        new_corr_id = str(uuid.uuid4())

        # Save to zarr
        output_root = zarr.open(str(output_path), mode='w')
        corr_output = output_root.create_group(new_corr_id)

        # Calculate offset for mask (centered in raw)
        offset_diff = (np.array(raw.shape) - np.array(mito_mask.shape)) // 2

        # Save raw (no translation)
        raw_group = corr_output.create_group('raw')
        raw_group.create_dataset(
            's0',
            data=raw,
            dtype=raw.dtype,
            compression='gzip',
            compression_opts=6,
            chunks=(64, 64, 64)
        )
        add_ome_ngff_metadata(raw_group, 'raw', voxel_size)

        # Save sparse mask (with translation offset)
        mask_group = corr_output.create_group('mask')
        mask_group.create_dataset(
            's0',
            data=sparse_mask,
            dtype=sparse_mask.dtype,
            compression='gzip',
            compression_opts=6,
            chunks=(56, 56, 56)
        )
        add_ome_ngff_metadata(mask_group, 'mask', voxel_size, translation_offset=offset_diff.tolist())

        # Save prediction (with translation offset)
        pred_group = corr_output.create_group('prediction')
        pred_group.create_dataset(
            's0',
            data=prediction,
            dtype=prediction.dtype,
            compression='gzip',
            compression_opts=6,
            chunks=(56, 56, 56)
        )
        add_ome_ngff_metadata(pred_group, 'prediction', voxel_size, translation_offset=offset_diff.tolist())

        # Save metadata
        corr_output.attrs.update({
            'correction_id': new_corr_id,
            'source_correction': corr_id,
            'timestamp': timestamp,
            'num_foreground_points': len(foreground_points),
            'num_background_points': len(background_points),
            'annotation_radius': ANNOTATION_RADIUS,
            'annotation_fraction': float(annotation_fraction),
            'voxel_size': voxel_size.tolist(),
            'label_scheme': '0=unannotated, 1=background, 2=foreground'
        })

        print(f"  ✓ Saved to: {output_name}")
        print()

    print("="*60)
    print(f"✓ Complete! Generated {len(correction_ids)} sparse corrections")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
