"""GPU queue availability from canned bqueues/bhosts output.

bqueues exits non-zero if any queue it is asked about does not exist, and
all the GPU queues were asked about in one call: one retired queue made the
whole queue panel say "LSF not reachable".
"""

import pytest

from cellmap_flow.jobs import queues as lsf_queues

HEADER = "QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP\n"


def _row(name, status="Open:Active", pend=3, run=5):
    return f"{name:<15} 30  {status:<15} -    -    -    -   {pend + run:>4} {pend:>5} {run:>5}     0\n"


BHOSTS_W = "HOST_NAME          STATUS          JL/U    MAX  NJOBS    RUN  SSUSP  USUSP    RSV\nh01 ok - 48 2 2 0 0 0\n"
BHOSTS_GPU = (
    "HOST_NAME  GPU_ID  MODEL  MUSED  MRSV  NJOBS  RUN  SUSP  RSV\n"
    "h01  0  H100  0M  0M  0  0  0  0\n"
    "     1  H100  0M  0M  1  1  0  0\n"
)


@pytest.fixture
def lsf(monkeypatch):
    """bqueues knows gpu_h100 and gpu_a100; gpu_h200 has been retired."""
    known = {"gpu_h100", "gpu_a100"}
    calls = []

    def run(args):
        calls.append(args)
        if args[0] == "bqueues":
            names = args[1:]
            if not set(names) <= known:
                return None  # exits 255: "gpu_h200: No such queue"
            return HEADER + "".join(_row(n) for n in names)
        if args[:2] == ["bhosts", "-w"]:
            return BHOSTS_W
        if args[:2] == ["bhosts", "-gpu"]:
            return BHOSTS_GPU
        raise AssertionError(args)

    monkeypatch.setattr(lsf_queues, "_run", run)
    return calls


def test_one_unknown_queue_does_not_hide_the_others(lsf):
    payload = lsf_queues._collect()

    assert payload["available"] is True
    by_name = {q["queue"]: q for q in payload["queues"]}
    assert by_name["gpu_h100"]["accepting"] is True
    assert by_name["gpu_h100"]["pending"] == 3
    assert by_name["gpu_a100"]["accepting"] is True
    assert by_name["gpu_h200"]["accepting"] is False
    assert by_name["gpu_h200"]["status"] == "unknown"
    assert [c for c in lsf if c[0] == "bqueues"], "the fake answered"


def test_the_single_call_is_used_when_it_works(lsf, monkeypatch):
    monkeypatch.setattr(lsf_queues, "GPU_QUEUES", (("gpu_h100", "H100", "h100s"),))
    payload = lsf_queues._collect()
    assert payload["available"] is True
    assert [c for c in lsf if c[0] == "bqueues"] == [["bqueues", "gpu_h100"]]


def test_no_answer_at_all_is_still_unreachable(monkeypatch):
    asked = []
    monkeypatch.setattr(lsf_queues, "_run", lambda args: asked.append(args))
    assert lsf_queues._collect()["available"] is False
    assert asked, "the fake answered"
