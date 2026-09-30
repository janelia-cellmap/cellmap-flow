"""The numbers that are about the cluster, not about cellmap-flow.

Which GPU queues to offer, how many cores a job asks for, how long it may
run, and how long to wait on LSF: all of it is Janelia's, and it was spread
through the launch code as literals. It lives here so that the reasons sit
next to the numbers, and so that another site means another profile rather
than a search for "gpu_h100".

Only ``JANELIA`` exists. ``current_site()`` is the one place that will pick
between profiles once there is more than one.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SiteProfile:
    # (queue, label, LSF host group backing it). The only GPU types offered:
    # Janelia has others (gpu_l4, gpu_rtx8000), but they are not what anyone
    # wants for inference or finetuning here, and a shorter list is easier to
    # choose from than a complete one.
    gpu_queues: Tuple[Tuple[str, str, str], ...]
    default_queue: str = "gpu_h100"
    # Who a job is billed to when neither the request nor the dashboard's
    # settings say.
    default_charge_group: str = "cellmap"
    # LSF's own default run limit on the GPU queues is 120 minutes, and no -W
    # used to be passed, so every inference server was killed two hours in --
    # while the Fileglancer app job that spawns them asks for 8 hours, so the
    # dashboard outlived its own servers by six. Match the session: 8 hours,
    # overridable per-yaml, per-submission, or from the dashboard. The queues
    # allow up to 20160 minutes (14 days).
    default_walltime: str = "08:00"
    server_cpus: int = 4
    worker_cpus: int = 12
    # How long a job may sit PENDING before we give up on that queue and try
    # another. Long enough that a queue which is merely busy still gets used,
    # short enough that nobody watches a spinner while 9000 jobs clear ahead
    # of them on a queue that was never going to start.
    pending_fallback_seconds: int = 180
    # Once a job is running, how long it gets to report its host. Loading the
    # model dominates this -- weights off /nrs, a torch.export, sometimes a
    # HuggingFace download -- and none of it is a reason to try another queue.
    startup_timeout_seconds: int = 300
    # How long bsub may take to answer. It can block server-side, for example
    # while an esub delays an over-ratio request, and still create the job
    # after the client has given up; see lsf.BsubTimeoutError.
    bsub_timeout_seconds: int = 30


JANELIA = SiteProfile(
    gpu_queues=(
        ("gpu_h100", "H100", "h100s"),
        ("gpu_h200", "H200", "h200s"),
        ("gpu_a100", "A100", "a100s"),
    ),
)


def current_site() -> SiteProfile:
    """The profile of the cluster this process submits to."""
    return JANELIA
