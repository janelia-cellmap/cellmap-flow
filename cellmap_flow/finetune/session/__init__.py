"""A finetune session on disk: its annotation volumes, manifest and MinIO sync.

A session is ``<base>/<YYYYmmdd_HHMMSS>/``. Its ``corrections/`` holds the
annotation volumes and ``_virtual_sources.json``, the manifest the trainer
reads; ``good_regions.json`` sits beside ``corrections/``.

- ``manifest``: the manifest, good regions, and whether a volume is painted.
- ``volume``: planning, creating, reading and writing annotation volumes.
- ``store``: which session a base path means, and the volume registry.
- ``minio``: starting MinIO and serving volumes through it.
- ``sync``: pulling painted chunks back from MinIO, and the periodic sync.
- ``instance``: instance corrections, seeded from a segmentation.
- ``fill``: labelling a whole box of a volume at once, unpainted voxels only.

The submodules import neither flask, neuroglancer, torch nor
``cellmap_flow.globals``, so the CLI, the trainer and scripts can use them
without a dashboard. State such as the volume registry and MinIO's is passed
in (the dashboard passes its ``get_session()``'s), never held here. Import the submodule you need; this package imports
none of them itself.
"""
