"""GPU queues: which accept work (bqueues/bhosts, canned), and the order to try them."""

import pytest

from cellmap_flow.globals import SERVER_CONFIG_DEFAULTS
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
        (None, {"gpu_h100", "gpu_a100"}, None, {"gpu_h100": True, "gpu_a100": True, "gpu_h200": False}),
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


AVAILABILITY = {"available": True, "queues": [
    {"queue": "gpu_h100", "accepting": True, "gpus_free": 0, "pending": 40},
    {"queue": "gpu_a100", "accepting": True, "gpus_free": 12, "pending": 0},
    {"queue": "gpu_h200", "accepting": True, "gpus_free": 4, "pending": 2},
]}


@pytest.mark.parametrize("cycle, expected", [
    ({}, ["gpu_h100", "gpu_a100", "gpu_h200"]),  # on by default: most free GPUs first after the one asked for
    ({"cycle": True}, ["gpu_h100", "gpu_a100", "gpu_h200"]),
    # Work pinned to a queue on purpose waits for it, and LSF isn't asked.
    ({"cycle": False}, ["gpu_h100"]),
])
def test_the_queues_a_job_is_tried_on(monkeypatch, cycle, expected):
    asked = []
    monkeypatch.setattr(queues, "gpu_queue_availability", lambda force=False: asked.append(1) or AVAILABILITY)
    assert queues.candidates("gpu_h100", **cycle) == expected
    assert len(asked) == (0 if cycle.get("cycle") is False else 1)
    assert SERVER_CONFIG_DEFAULTS["cycle_gpu_queues"] is True
