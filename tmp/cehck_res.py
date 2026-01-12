paths = ["/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s0"]
from cellmap_flow.utils.ds import get_ds_info,open_ds_tensorstore

for path in paths:
    voxel_size, chunk_shape, shape, roi, axes_names = get_ds_info(path)
    print(f"Path: {path}")
    print(f"Voxel size: {voxel_size}")
    print(f"Chunk shape: {chunk_shape}")
    print(f"Shape: {shape}")
    print(f"ROI: {roi}")
    print(f"Axes names: {axes_names}")
    print("-" * 40)