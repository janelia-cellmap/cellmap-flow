"""Finetune jobs, from the dashboard's side: submitting them, following them
until they end, restarting them, and finding them again after the dashboard
itself restarts.

- ``manager``: ``FinetuneJobManager``, which holds the jobs and is what the
  finetune routes call.
- ``state``: a job's record (``FinetuneJob``), its statuses, and what moves
  a job from one to another.
- ``submit``: what a submission runs and where: the model's type and
  settings, the trainer's command line, and launching it on LSF or here.
- ``persistence``: each run's ``metadata.json`` and export, and finding a
  session's jobs again from them (rehydration).
- ``monitor``: following a job until it ends, from its scheduler's answers
  and its log, and telling the listeners.
- ``tailer``: the trainer's log as the manager reads it.
- ``listener``: what the manager tells its listeners (the dashboard's
  viewer) of each job, and how.
- ``restart``: asking a job that waits for a restart to train again.

The trainer's side of the same protocol (its markers, its restart signal,
the layout of its run directory) is ``finetune.markers``,
``finetune.session_loop`` and ``finetune.run_outputs``.

Nothing here imports torch, Flask, neuroglancer or ``cellmap_flow.globals``
at module level. Import the submodule you need; this package imports none
of them itself.
"""
