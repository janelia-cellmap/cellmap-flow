"""The logic both modes run, written once.

Python importable with the standard library, numpy, scipy, pydantic and
Pillow alone, so that the server runs it in its process and the browser runs
the same files in chunkmirage's Pyodide workers (web/build/portable.ts bundles
them as text, ``Inference.python`` calls them; ``calls.py`` is what the
browser calls, and ``volume_calls.py`` for a volume's plan and its good
regions, ``volume.py`` and ``regions.py``). ``tests/portable/test_boundary.py``
holds every module here to that boundary. Nothing here reads a file, a zarr
or the network: the callers do, around it (``finetune/session/volume.py``'s
``carry_paint`` and ``plan_volume``,
``dashboard/routes/finetune/rank_routes.py``, ``web/src/backend/routes/rank.ts``).
"""
