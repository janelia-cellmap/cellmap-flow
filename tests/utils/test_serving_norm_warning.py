"""A layer served without normalization must say so.

get_process_dataset_url() returned ``None, [], []`` in silence when the
layer URL carried no args block. The model then received raw voxel values,
and for a model trained on normalized input the result looks like a badly
trained model rather than a misconfigured server -- which is exactly how a
serving mismatch gets mistaken for a training failure.
"""

import logging

from cellmap_flow.utils.serilization_utils import get_process_dataset_url
from cellmap_flow.utils.web_utils import ARGS_KEY, encode_to_str


def test_missing_args_block_warns_and_returns_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        dashboard_url, norms, posts = get_process_dataset_url(
            "http://host:1234/some/dataset.zarr"
        )

    assert (dashboard_url, norms, posts) == (None, [], [])
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "serving without normalization must warn"
    assert ARGS_KEY in warnings[0].getMessage()


def test_args_block_that_builds_no_normalizers_also_warns(caplog):
    """Decoding cleanly is not the same as producing normalizers."""
    encoded = encode_to_str({"input_norm": {}, "postprocess": {}})
    dataset = f"http://host:1234/d.zarr{ARGS_KEY}{encoded}{ARGS_KEY}"

    with caplog.at_level(logging.WARNING):
        _, norms, _ = get_process_dataset_url(dataset)

    assert norms == []
    assert any(
        "NO input normalizers" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_normal_case_reports_what_it_built(caplog):
    encoded = encode_to_str(
        {
            "input_norm": {
                "MinMaxNormalizer": {"min_value": 0, "max_value": 255, "invert": False},
                "LambdaNormalizer": {"expression": "x*2-1"},
            },
            "postprocess": {},
        }
    )
    dataset = f"http://host:1234/d.zarr{ARGS_KEY}{encoded}{ARGS_KEY}"

    with caplog.at_level(logging.INFO):
        _, norms, _ = get_process_dataset_url(dataset)

    assert [type(n).__name__ for n in norms] == [
        "MinMaxNormalizer",
        "LambdaNormalizer",
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("MinMaxNormalizer" in r.getMessage() for r in caplog.records)
