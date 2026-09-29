"""Running cellmap-flow's jobs: on LSF, or as a process on this machine.

- ``spec``: JobSpec (what to run and with what), JobStatus, the Job base
  class, and the helpers both backends share.
- ``lsf``: submitting to LSF with bsub, LSFJob, and asking bjobs about jobs.
- ``local``: running a job here, LocalJob.
- ``queues``: which GPU queues are usable, and the order to try them in.
- ``site``: the site's numbers (queues, cores, walltime, timeouts).
- ``ready``: the file a server writes once it knows its address.

The policy of what to launch, and when (``start_hosts``, which reads the
dashboard's settings), stays in ``cellmap_flow.utils.bsub_utils``. Nothing
in this package imports ``cellmap_flow.globals``, Flask or torch at module
level, so launching a job does not pull in the dashboard or a model.
"""
