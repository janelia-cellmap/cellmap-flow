"""GPU queues: which accept work (bqueues/bhosts, canned), and the order to try them."""

import pytest

from cellmap_flow.jobs import queues

HEADER = "QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP\n"


def _row(name):
    return f"{name:<15} 30  {'Open:Active':<15} -    -    -    -      8     3     5     0\n"


def _bqueues(known):
    """bqueues exits non-zero if any queue it is asked about does not exist."""
    def answer(argv, kwargs):
        if not set(argv[1:]) <= known:
            return (255, "", "No such queue")
        return HEADER + "".join(_row(n) for n in argv[1:])
    return answer


BHOSTS = {
    "-w": "HOST_NAME  STATUS  JL/U  MAX  NJOBS  RUN  SSUSP  USUSP  RSV\nh01 ok - 48 2 2 0 0 0\n",
    "-gpu": "HOST_NAME  GPU_ID  MODEL  MUSED  MRSV  NJOBS  RUN  SUSP  RSV\nh01  0  H100  0M  0M  0  0  0  0\n",
}


@pytest.mark.parametrize(
    "gpu_queues, known, asked, accepting",
    [
        # One retired queue made the whole panel say "LSF not reachable".
        ((("gpu_h100", "H100", "h100s"), ("gpu_h200", "H200", "h200s"), ("gpu_a100", "A100", "a100s")),
         {"gpu_h100", "gpu_a100"}, None, {"gpu_h100": True, "gpu_a100": True, "gpu_h200": False}),
        ((("gpu_h100", "H100", "h100s"),), {"gpu_h100"}, [["bqueues", "gpu_h100"]], {"gpu_h100": True}),
        (None, set(), None, None),  # no answer at all is still unreachable
    ],
    ids=["one-unknown", "one-call-when-it-works", "unreachable"],
)
def test_what_lsf_says_about_the_gpu_queues(fake_lsf, monkeypatch, gpu_queues, known, asked, accepting):
    if gpu_queues:
        monkeypatch.setattr(queues, "GPU_QUEUES", gpu_queues)
    fake_lsf.answers["bqueues"] = _bqueues(known)
    fake_lsf.answers["bhosts"] = lambda argv, kwargs: BHOSTS[argv[1]]
    payload = queues.gpu_queue_availability(force=True)
    assert payload["available"] is (accepting is not None)
    if accepting:
        assert {q["queue"]: q["accepting"] for q in payload["queues"]} == accepting
    if asked:
        assert fake_lsf.commands("bqueues") == asked
    assert fake_lsf.commands("bqueues"), "the fake answered"


def test_a_closed_hosts_idle_gpus_are_not_free(fake_lsf, monkeypatch):
    """gpu_h100 said 8 GPUs free, all on a node out for maintenance, so jobs
    were sent there to wait out the pending timeout."""
    monkeypatch.setattr(queues, "GPU_QUEUES", (("gpu_h100", "H100", "h100s"),))
    fake_lsf.answers["bqueues"] = _bqueues({"gpu_h100"})
    # A host's other GPUs are rows without its name.
    gpu = (
        "HOST_NAME  GPU_ID  MODEL  MUSED  MRSV  NJOBS  RUN  SUSP  RSV\n"
        "h01        0       H100   0M     0M    1      1    0     0\n"
        "           1       H100   0M     0M    0      0    0     0\n"
        "h02        0       H100   0M     0M    0      0    0     0\n"
        "           1       H100   0M     0M    0      0    0     0\n"
        "h03        0       H100   0M     0M    0      0    0     0\n"
    )
    hosts = {
        "-w": "HOST_NAME  STATUS  JL/U  MAX  NJOBS  RUN  SSUSP  USUSP  RSV\n"
              "h01 ok - 48 2 2 0 0 0\nh02 closed_Adm - 48 0 0 0 0 0\nh03 closed_Full - 48 48 48 0 0 0\n",
        "-gpu": gpu,
    }
    fake_lsf.answers["bhosts"] = lambda argv, kwargs: hosts[argv[1]]
    (h100,) = queues.gpu_queue_availability(force=True)["queues"]
    assert (h100["gpus_free"], h100["gpus_total"]) == (1, 5)


AVAILABILITY = {"available": True, "queues": [
    {"queue": "gpu_h100", "accepting": True, "gpus_free": 0, "pending": 40},
    {"queue": "gpu_a100", "accepting": True, "gpus_free": 12, "pending": 0},
    {"queue": "gpu_h200", "accepting": True, "gpus_free": 4, "pending": 2},
    {"queue": "gpu_rtx6000", "accepting": False, "gpus_free": 30, "pending": 0},
]}


@pytest.mark.parametrize("ticked, cycle, expected", [
    # The freest of the queues ticked first, then the next freest; one LSF
    # knows nothing of (a YAML's gpu_l4) after those, and a closed one last.
    # gpu_a100 has the most free GPUs, and is never tried: it is not ticked.
    pytest.param(["gpu_rtx6000", "gpu_l4", "gpu_h100", "gpu_h200"], {},
                 ["gpu_h200", "gpu_h100", "gpu_l4", "gpu_rtx6000"], id="freest-ticked-first"),
    # A YAML's one queue.
    pytest.param("gpu_h100", {}, ["gpu_h100"], id="one-queue"),
    # Work pinned to a queue on purpose waits for the first, and LSF isn't asked.
    pytest.param(["gpu_h200", "gpu_a100"], {"cycle": False}, ["gpu_h200"], id="pinned"),
])
def test_the_queues_a_job_is_tried_on(monkeypatch, ticked, cycle, expected):
    asked = []
    monkeypatch.setattr(queues, "gpu_queue_availability", lambda force=False: asked.append(1) or AVAILABILITY)
    assert queues.candidates(ticked, **cycle) == expected
    assert len(asked) == (0 if cycle.get("cycle") is False else 1)
