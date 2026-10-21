"""Reading datasets: paths, metadata, multiscale levels and OME attributes.

The reading itself is worldarray's (``worldarray.store``, ``.metadata``,
``Pyramid.voxel_sizes``, ``list_chunk_keys``, ``WorldArray.read_index`` and
``.read``); these modules keep cellmap-flow's names and conventions over it
(a corner-based ``ArrayMeta``, ``s<N>`` names for precomputed scales, the
corner ``Grid`` and its index ``Box``) and what worldarray lacks (the OME
writer in ``ome``).

The submodules import nothing from the rest of cellmap_flow -- in
particular not ``cellmap_flow.globals`` -- so they can be used without
configuring logging, loading a Flow or importing flask/neuroglancer/torch;
importing worldarray brings numpy and tensorstore. Import the submodule you
need; this package imports none of them itself.
"""
