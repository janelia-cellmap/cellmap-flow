"""Running cellmap-flow's jobs: on LSF, or as a process on this machine.

- ``spec``: JobSpec (what to run and with what), JobStatus, the Job base
  class, and the helpers both backends share.
- ``lsf``: submitting to LSF with bsub, LSFJob, and asking bjobs about jobs.
- ``local``: running a job here, LocalJob.
- ``queues``: which GPU queues are usable, and the order to try them in.
- ``site``: the site's numbers (queues, cores, walltime, timeouts).
- ``ready``: the file a server writes once it knows its address.
- ``launch``: the policy over them. ``start_hosts`` starts an inference
  server, falling back through the GPU queues, and records it in the
  dashboard's ``g.jobs``; the deployment's ``SERVER_COMMAND`` and
  ``SERVER_LOG_DIR``.

Nothing in this package imports ``cellmap_flow.globals``, Flask or torch at
module level, so launching a job does not pull in the dashboard or a model.
"""
