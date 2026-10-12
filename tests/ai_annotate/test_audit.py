"""The AI annotation audit log: one JSON line per event, nothing secret in it."""

import getpass
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from cellmap_flow.ai_annotate import audit, secrets


def _lines(corrections_dir):
    return [json.loads(line) for line in (Path(corrections_dir) / audit.LOG_NAME).read_text().splitlines()]


def test_each_event_is_one_line_with_time_user_and_the_fields_given(tmp_path):
    audit.record(tmp_path, "requested", annotate_id="a" * 32, provider_id="fake", model="fake-threshold",
                 point_nm=[1.0, 2.0, 3.0], plane="XY")
    audit.record(tmp_path, "accepted", annotate_id="a" * 32, filled_foreground=np.int64(12), overwrite=False,
                 zarr_path=tmp_path / "vol.zarr")
    first, second = _lines(tmp_path)
    assert set(first) == {"time", "user", "event", "annotate_id", "provider_id", "model", "point_nm", "plane"}
    assert first["event"] == "requested" and first["user"] == getpass.getuser()
    assert first["point_nm"] == [1.0, 2.0, 3.0]
    # Values that are not JSON are written as their str.
    assert second["filled_foreground"] == "12"
    assert second["zarr_path"] == str(tmp_path / "vol.zarr")
    assert second["time"] >= first["time"]


def test_the_log_is_made_where_it_belongs(tmp_path):
    audit.record(tmp_path / "new" / "dir", "rejected", annotate_id="b" * 32)
    assert audit.log_path(tmp_path / "new" / "dir").is_file()


def test_unknown_events_and_reserved_fields_are_refused(tmp_path):
    with pytest.raises(ValueError):
        audit.record(tmp_path, "deleted_everything")
    with pytest.raises(ValueError):
        audit.record(tmp_path, "staged", user="someone else")
    assert not audit.log_path(tmp_path).exists()


def test_a_registered_secret_is_redacted_even_if_a_caller_passed_it(tmp_path):
    key = "sk-live-0123456789abcdef"
    secrets.register_secret(key)
    try:
        audit.record(tmp_path, "failed", error=f"upstream said: bad key {key}")
    finally:
        secrets._clear_registered_secrets()
    text = audit.log_path(tmp_path).read_text()
    assert key not in text
    assert "[REDACTED]" in _lines(tmp_path)[0]["error"]


def test_lines_from_threads_do_not_interleave(tmp_path):
    def write(n):
        for i in range(25):
            audit.record(tmp_path, "staged", thread=n, i=i, prompt="x" * 2000)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = _lines(tmp_path)
    assert len(lines) == 100
    assert {(line["thread"], line["i"]) for line in lines} == {(n, i) for n in range(4) for i in range(25)}
