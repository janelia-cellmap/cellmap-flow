"""The AI annotation audit log: nothing secret in it, even when a caller passes it."""

import json
from pathlib import Path

from cellmap_flow.ai_annotate import audit, secrets


def _lines(corrections_dir):
    return [json.loads(line) for line in (Path(corrections_dir) / audit.LOG_NAME).read_text().splitlines()]


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
