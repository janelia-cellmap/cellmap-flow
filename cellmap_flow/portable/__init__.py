"""The logic both modes run, written once.

Python importable with the standard library, numpy, scipy, pydantic and
Pillow alone, so that the server runs it in its process and the browser runs
the same files in chunkmirage's Pyodide workers (web/build/portable.ts bundles
them as text, ``Inference.python`` calls them; ``calls.py`` is what the
browser calls, and ``volume_calls.py`` for a volume's plan and its good
regions, ``volume.py`` and ``regions.py``). ``label_it.py`` is Label it's
segmentation of a prediction, read by
``dashboard/routes/finetune/view_labels.py`` and, through
``label_it_calls.py``, by ``web/src/backend/routes/view-labels.ts``.
``tests/portable/test_boundary.py`` holds every module here to that
boundary. Nothing here reads a file, a zarr or the network: the callers do,
around it (``finetune/session/volume.py``'s ``carry_paint`` and
``plan_volume``,
``dashboard/routes/finetune/rank_routes.py``, ``web/src/backend/routes/rank.ts``).
A finetune's training targets are ``targets.py``, which the server's
``finetune/target_transforms.py`` wraps and the browser calls through
``targets_calls.py``.
AI-assisted annotation's plane, image, mask and labels are ``ai_annotate.py``
(the browser calls ``ai_annotate_calls.py``), and the label decisions under
it and the Finetune tab's box tools are ``fill.py`` (the browser's box tools
call ``fill_calls.py``).
A model published to Hugging Face, its card, its manifest's "published"
block and the app's links to it, is ``publish.py``, which the server's
publish route calls and the browser's through ``publish_calls.py``.
"""
