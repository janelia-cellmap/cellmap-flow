"""Finetune jobs, from the dashboard's side.

- ``state``: a job's record (``FinetuneJob``), its statuses, and what moves
  a job from one to another.
- ``tailer``: the trainer's log as the manager reads it.
- ``listener``: what the manager tells its listeners (the dashboard's
  viewer) of each job, and how.
- ``submit``: what a submission runs and where: the model's type and
  settings, the trainer's command line, and launching it on LSF or here.

Nothing here imports torch, Flask, neuroglancer or ``cellmap_flow.globals``
at module level. Import the submodule you need; this package imports none
of them itself.
"""
