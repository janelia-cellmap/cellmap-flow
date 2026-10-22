"""Reading datasets: paths, metadata, multiscale levels and OME attributes.

The reading itself is worldarray's (``worldarray.store``, ``.placement``,
``Pyramid.voxel_sizes`` and ``.metas``, ``list_chunk_keys``, ``WorldArray.read_index`` and
``.read``); these modules keep cellmap-flow's names and conventions over it
(a corner-based ``ArrayMeta``, ``s<N>`` names for precomputed scales) and
what worldarray lacks (the OME writer in ``ome``). ``raw`` opens the level a
model reads and reads it through the input chain (``open_raw``, ``RawInput``).

The submodules import nothing from the rest of cellmap_flow -- ``raw``
excepted, which reads the process's chain (``cellmap_flow.process_chain``)
-- and in particular not ``cellmap_flow.globals``, so they can be used
without configuring logging, loading a Flow or importing
flask/neuroglancer/torch; importing worldarray brings numpy and tensorstore.
Import the submodule you need; this package imports none of them itself.
"""
