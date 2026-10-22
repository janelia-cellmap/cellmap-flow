"""The daily call limit: counted per day and refused at the limit."""

from datetime import date

import pytest

from cellmap_flow.ai_annotate import usage
from cellmap_flow.ai_annotate.errors import AIAnnotateError

DAY = date(2026, 10, 4)


def test_calls_are_counted_and_the_limit_raises(tmp_path):
    path = tmp_path / "usage.json"
    assert usage.calls_today(path=path, today=DAY) == 0
    assert [usage.check_and_count(3, path=path, today=DAY) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(AIAnnotateError) as caught:
        usage.check_and_count(3, path=path, today=DAY)
    assert (caught.value.category, caught.value.http_status) == ("limit", 429)
    assert "3" in caught.value.user_message
    assert usage.calls_today(path=path, today=DAY) == 3  # the refused call is not counted


def test_a_new_day_starts_from_zero(tmp_path):
    path = tmp_path / "usage.json"
    usage.check_and_count(1, path=path, today=DAY)
    with pytest.raises(AIAnnotateError):
        usage.check_and_count(1, path=path, today=DAY)
    assert usage.check_and_count(1, path=path, today=date(2026, 10, 5)) == 1
