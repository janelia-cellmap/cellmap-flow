"""Secrets: redacted from every part of a log record, and key files that others
can read are refused."""

import io
import json
import logging
import os

import pytest

from cellmap_flow.ai_annotate import secrets
from cellmap_flow.ai_annotate.errors import AIAnnotateError

KEY = "sk-test-0123456789abcdef"


@pytest.fixture(autouse=True)
def clean_registry():
    secrets._clear_registered_secrets()
    yield
    secrets._clear_registered_secrets()


@pytest.fixture
def captured():
    """A parent logger with a handler writing to a buffer, redaction
    installed on it, and a child logger that logs through it."""
    parent = logging.getLogger("redaction_test_parent")
    parent.handlers.clear()
    parent.filters.clear()
    parent.propagate = False
    parent.setLevel(logging.DEBUG)
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    parent.addHandler(handler)
    secrets.install_log_redaction(parent)
    yield logging.getLogger("redaction_test_parent.child"), buffer
    parent.handlers.clear()
    parent.filters.clear()


def test_redact_replaces_registered_values_and_ignores_short_ones():
    secrets.register_secret("short")
    secrets.register_secret(KEY)
    assert secrets.redact(f"key={KEY} and short") == "key=[REDACTED] and short"
    assert secrets.redact("nothing here") == "nothing here"


def test_the_url_quoted_form_is_redacted_too():
    secrets.register_secret("abc/def+ghi=jkl")
    assert secrets.redact("GET /v1?key=abc%2Fdef%2Bghi%3Djkl") == "GET /v1?key=[REDACTED]"


def test_a_child_loggers_message_args_and_traceback_are_redacted(captured):
    logger, buffer = captured
    secrets.register_secret(KEY)
    logger.info("calling with %s", KEY)
    logger.warning(f"inline {KEY}")
    try:
        raise RuntimeError(f"401 Unauthorized: key {KEY} is invalid")
    except RuntimeError:
        logger.exception("call failed: %r", {"key": KEY})
    text = buffer.getvalue()
    assert KEY not in text
    assert text.count("[REDACTED]") >= 4
    assert "RuntimeError" in text and "Traceback" in text


def test_a_secret_registered_after_install_is_still_redacted(captured):
    logger, buffer = captured
    logger.info("before any secret")
    secrets.register_secret(KEY)
    logger.info("after: %s", KEY)
    assert KEY not in buffer.getvalue()


def test_a_record_with_mismatched_args_is_kept_and_redacted(captured):
    logger, buffer = captured
    secrets.register_secret(KEY)
    logger.info("two %s %s", KEY)
    text = buffer.getvalue()
    assert "two" in text and KEY not in text


def test_install_is_idempotent():
    logger = logging.getLogger("redaction_idempotent")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    try:
        secrets.install_log_redaction(logger)
        secrets.install_log_redaction(logger)
        assert sum(isinstance(f, secrets.RedactingFilter) for f in logger.filters) == 1
        assert sum(isinstance(f, secrets.RedactingFilter) for f in handler.filters) == 1
    finally:
        logger.removeHandler(handler)
        logger.filters.clear()


def test_the_dashboard_installs_redaction_on_the_package_loggers_handlers():
    import cellmap_flow.dashboard.app  # noqa: F401  (installs it)
    from cellmap_flow.dashboard.routes.logging_routes import LogHandler

    package = logging.getLogger("cellmap_flow")
    handlers = [h for h in package.handlers if isinstance(h, LogHandler)]
    assert handlers
    for handler in handlers:
        assert any(isinstance(f, secrets.RedactingFilter) for f in handler.filters)


def test_resolve_api_key_reads_an_env_var_and_registers_it(monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", KEY)
    assert secrets.resolve_api_key({"api_key_env": "MY_PROVIDER_KEY"}) == KEY
    assert secrets.redact(KEY) == "[REDACTED]"


def test_resolve_api_key_without_key_options_is_none():
    assert secrets.resolve_api_key({"project": "p"}) is None


def test_a_missing_env_var_is_an_auth_error(monkeypatch):
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(AIAnnotateError) as caught:
        secrets.resolve_api_key({"api_key_env": "NOT_SET_ANYWHERE"})
    assert caught.value.category == "auth"


def test_an_inline_api_key_is_rejected():
    with pytest.raises(AIAnnotateError) as caught:
        secrets.resolve_api_key({"api_key": KEY})
    assert caught.value.category == "config" and KEY not in caught.value.user_message


def test_a_private_key_file_is_read_and_registered(tmp_path):
    path = tmp_path / "key"
    path.write_text(KEY + "\n")
    path.chmod(0o600)
    assert secrets.resolve_api_key({"api_key_file": str(path)}) == KEY
    assert secrets.redact(KEY) == "[REDACTED]"


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o644, 0o610])
def test_a_key_file_with_group_or_world_bits_is_rejected(tmp_path, mode):
    path = tmp_path / "key"
    path.write_text(KEY)
    path.chmod(mode)
    with pytest.raises(AIAnnotateError) as caught:
        secrets.resolve_api_key({"api_key_file": str(path)})
    assert caught.value.category == "config"
    assert "chmod 600" in caught.value.user_message
    assert KEY not in caught.value.user_message
    assert secrets.redact(KEY) == KEY  # never registered: it was never accepted


def test_a_key_file_owned_by_someone_else_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "key"
    path.write_text(KEY)
    path.chmod(0o600)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(path).st_uid + 1)
    with pytest.raises(AIAnnotateError, match="owned by you"):
        secrets.resolve_api_key({"api_key_file": str(path)})


def test_missing_empty_and_non_regular_key_files_are_rejected(tmp_path):
    with pytest.raises(AIAnnotateError, match="Could not open"):
        secrets.resolve_api_key({"api_key_file": str(tmp_path / "missing")})
    empty = tmp_path / "empty"
    empty.write_text("  \n")
    empty.chmod(0o600)
    with pytest.raises(AIAnnotateError, match="empty"):
        secrets.resolve_api_key({"api_key_file": str(empty)})
    directory = tmp_path / "dir"
    directory.mkdir(mode=0o700)
    with pytest.raises(AIAnnotateError):
        secrets.resolve_api_key({"api_key_file": str(directory)})


def test_env_and_file_together_are_ambiguous(tmp_path):
    with pytest.raises(AIAnnotateError, match="not both"):
        secrets.resolve_api_key({"api_key_env": "X", "api_key_file": str(tmp_path / "k")})


# What `gcloud auth application-default login` writes: a user's refresh token.
_ADC = {
    "type": "authorized_user",
    "client_id": "1234.apps.googleusercontent.com",
    "client_secret": "client-secret-value-1234",
    "refresh_token": "1//refresh-token-value-abcdefgh",
}


def _credentials_file(tmp_path, content=None, mode=0o600):
    path = tmp_path / "application_default_credentials.json"
    path.write_text(json.dumps(_ADC) if content is None else content)
    path.chmod(mode)
    return path


def test_a_private_google_credentials_file_is_loaded_and_its_secrets_registered(tmp_path):
    pytest.importorskip("google.auth")
    credentials = secrets.load_google_credentials(str(_credentials_file(tmp_path)))

    assert credentials.refresh_token == _ADC["refresh_token"]
    text = f"refresh {_ADC['refresh_token']} secret {_ADC['client_secret']} id {_ADC['client_id']}"
    assert secrets.redact(text) == f"refresh [REDACTED] secret [REDACTED] id {_ADC['client_id']}"


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660])
def test_a_google_credentials_file_others_can_read_is_rejected(tmp_path, mode):
    pytest.importorskip("google.auth")
    with pytest.raises(AIAnnotateError) as caught:
        secrets.load_google_credentials(str(_credentials_file(tmp_path, mode=mode)))
    assert caught.value.category == "config" and "chmod 600" in caught.value.user_message
    assert secrets.redact(_ADC["refresh_token"]) == _ADC["refresh_token"]  # never read, never registered


@pytest.mark.parametrize("content", ["not json", "[1, 2]", json.dumps({"type": "nonsense"})])
def test_a_file_that_is_not_google_credentials_is_rejected_without_quoting_it(tmp_path, content):
    pytest.importorskip("google.auth")
    with pytest.raises(AIAnnotateError) as caught:
        secrets.load_google_credentials(str(_credentials_file(tmp_path, content=content)))
    assert caught.value.category == "config" and "nonsense" not in caught.value.user_message
