"""Finetune jobs, from the dashboard's side.

- ``state``: a job's record (``FinetuneJob``), its statuses, and what moves
  a job from one to another.
- ``tailer``: the trainer's log as the manager reads it.

Nothing here imports torch, Flask, neuroglancer or ``cellmap_flow.globals``
at module level. Import the submodule you need; this package imports none
of them itself.
"""
