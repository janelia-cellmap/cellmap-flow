"""Reading datasets: paths, metadata, multiscale levels and OME attributes.

The submodules import nothing from the rest of cellmap_flow -- in
particular not ``cellmap_flow.globals`` -- so they can be used without
configuring logging, loading a Flow or importing flask/neuroglancer/torch.
Import the submodule you need; this package imports none of them itself.
"""
