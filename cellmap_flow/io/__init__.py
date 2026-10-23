"""Reading datasets: what cellmap-flow adds over worldarray.

The reading is worldarray's: ``wa.open`` and ``wa.open_pyramid`` open arrays
and pyramids by path (zarr v2 and v3, N5, precomputed, on disk or at a URL),
``.meta`` and ``wa.metadata`` say what an array is and where it sits,
``list_chunk_keys`` lists the chunks that exist, ``read_index`` and ``read``
read voxel and world boxes. These modules keep what cellmap-flow needs on top:
``metadata`` opens a path the way every reader does (``open_level``) and
gives nanometers per unit; ``multiscale`` names the raw level a model reads
(``closest_raw_scale``); ``paths`` undoes a shell-escaped space; ``raw`` opens
the level a model reads and reads it through the input chain (``open_raw``,
``RawInput``). Writing is worldarray's too (``wa.create``, ``write``,
``write_index``, ``write_chunk``), at the writers themselves.

The submodules import nothing from the rest of cellmap_flow -- ``raw``
excepted, which reads the process's chain (``cellmap_flow.process_chain``)
-- and in particular not ``cellmap_flow.globals``, so they can be used
without configuring logging, loading a Flow or importing
flask/neuroglancer/torch; importing worldarray brings numpy and tensorstore.
Import the submodule you need; this package imports none of them itself.
"""

# The names of a channel axis wherever cellmap-flow names axes (a store's
# axes, a model's ``chunk_output_axes``): never space. worldarray writes an
# axis named "c" as an OME channel axis itself; "c^" and "channel" are
# cellmap-flow's own spellings. Defined here, where importing costs nothing.
CHANNEL_AXIS_NAMES = ("c", "c^", "channel")
