"""GPU queue cycling is on by default and can be turned off.

Cycling moves a job to whichever GPU queue has capacity when the requested
one is full or closed, which is what you want almost always. Almost: work
pinned to a queue on purpose -- a benchmark that must run on one GPU model,
a charge group valid on only one queue -- is better off waiting than landing
somewhere else silently.
"""

from unittest.mock import patch

from cellmap_flow.globals import SERVER_CONFIG_DEFAULTS
from cellmap_flow.utils.bsub_utils import gpu_queue_candidates

AVAILABILITY = {
    "available": True,
    "queues": [
        {"queue": "gpu_h100", "accepting": True, "gpus_free": 0, "pending": 40},
        {"queue": "gpu_a100", "accepting": True, "gpus_free": 12, "pending": 0},
        {"queue": "gpu_h200", "accepting": True, "gpus_free": 4, "pending": 2},
    ],
}


def test_cycling_is_the_default():
    assert SERVER_CONFIG_DEFAULTS["cycle_gpu_queues"] is True


def test_cycling_on_offers_fallbacks_most_free_first():
    with patch(
        "cellmap_flow.utils.lsf_queues.gpu_queue_availability", return_value=AVAILABILITY
    ):
        candidates = gpu_queue_candidates("gpu_h100", cycle=True)

    assert candidates[0] == "gpu_h100", "the requested queue is still tried first"
    assert candidates[1:] == ["gpu_a100", "gpu_h200"], "then most free GPUs first"


def test_cycling_off_pins_to_the_requested_queue():
    with patch(
        "cellmap_flow.utils.lsf_queues.gpu_queue_availability", return_value=AVAILABILITY
    ) as availability:
        candidates = gpu_queue_candidates("gpu_h100", cycle=False)

    assert candidates == ["gpu_h100"]
    # No reason to query LSF at all when the answer cannot change.
    availability.assert_not_called()


def test_cycling_defaults_to_on_when_the_argument_is_omitted():
    with patch(
        "cellmap_flow.utils.lsf_queues.gpu_queue_availability", return_value=AVAILABILITY
    ):
        assert len(gpu_queue_candidates("gpu_h100")) > 1
