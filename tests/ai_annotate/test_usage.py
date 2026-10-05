"""The daily call limit: counted per day, refused at the limit, written atomically."""

import json
import os
import threading
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


def test_the_default_file_is_in_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    usage.check_and_count(5)
    data = json.loads((tmp_path / ".cellmap_flow" / "ai_annotate_usage.json").read_text())
    assert data == {"date": date.today().isoformat(), "calls": 1}
    assert usage.calls_today() == 1


def test_an_unreadable_file_starts_afresh(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not json")
    assert usage.check_and_count(5, path=path, today=DAY) == 1


def test_a_failed_write_leaves_the_old_count_and_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "usage.json"
    usage.check_and_count(10, path=path, today=DAY)
    before = path.read_text()

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        usage.check_and_count(10, path=path, today=DAY)
    assert path.read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["usage.json", "usage.json.lock"]


def test_concurrent_calls_are_each_counted_once(tmp_path):
    path = tmp_path / "usage.json"
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(usage.check_and_count(1000, path=path, today=DAY)))
        for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == list(range(1, 21))
    assert usage.calls_today(path=path, today=DAY) == 20


def test_a_limit_of_zero_allows_no_calls(tmp_path):
    with pytest.raises(AIAnnotateError):
        usage.check_and_count(0, path=tmp_path / "usage.json", today=DAY)
