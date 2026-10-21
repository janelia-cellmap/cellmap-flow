"""Reading datasets: paths, metadata, multiscale levels and OME attributes.

The reading itself is worldarray's (``worldarray.store``, ``.metadata``,
``.levels``, ``WorldArray.read_index`` and ``.read``); these modules keep
cellmap-flow's names and conventions over it (a corner-based ``ArrayMeta``,
``…/s<N>`` precomputed scales, the corner ``Grid``) and what worldarray lacks
(the OME writer in ``ome``, ``geometry``'s Box, Grid and chunk listing).

The submodules import nothing from the rest of cellmap_flow -- in
particular not ``cellmap_flow.globals`` -- so they can be used without
configuring logging, loading a Flow or importing flask/neuroglancer/torch;
importing worldarray brings numpy and tensorstore. Import the submodule you
need; this package imports none of them itself.
"""
