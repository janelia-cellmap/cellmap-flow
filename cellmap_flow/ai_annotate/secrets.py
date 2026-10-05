"""Keeping provider secrets out of logs, and reading API keys safely.

The dashboard streams the ``cellmap_flow`` logger to the browser
(``/api/logs/stream``), so anything logged is as good as shown to whoever has
the page open. Every secret this process reads is registered here, and a
logging filter replaces registered values with ``[REDACTED]`` in each record
before a handler formats it: the message, its arguments, and the traceback
text, since an SDK exception's message is where a key is most likely to turn
up.

Vertex AI uses Application Default Credentials, so it has no key and
registers nothing. ``resolve_api_key`` is for key-based providers: the key
comes from an environment variable or from a file only the user can read,
never from the config file, and it is read at call time, not kept in session
state.
"""

import logging
import os
import re
import stat
import threading
import urllib.parse

from cellmap_flow.ai_annotate.errors import AIAnnotateError

REDACTED = "[REDACTED]"

# Shorter values are too likely to occur in ordinary text ("true", a port
# number) for replacing them to be anything but noise; no real key is this short.
MIN_SECRET_LENGTH = 8

# A key file bigger than this is not a key file; refusing it avoids reading
# something large (or a device) into memory by mistake.
_MAX_KEY_FILE_BYTES = 64 * 1024

_lock = threading.Lock()
_secrets = set()
_pattern = None


def register_secret(value):
    """Redact ``value`` from every log record from now on.

    Values shorter than eight characters are ignored. The URL-quoted form is
    registered too, as an HTTP library logging a request URL would quote a
    key passed as a query parameter.
    """
    global _pattern
    if not isinstance(value, str):
        return
    value = value.strip()
    if len(value) < MIN_SECRET_LENGTH:
        return
    with _lock:
        before = len(_secrets)
        _secrets.add(value)
        _secrets.add(urllib.parse.quote(value, safe=""))
        if len(_secrets) != before:
            # Longest first, so a secret that contains another is replaced whole.
            ordered = sorted(_secrets, key=len, reverse=True)
            _pattern = re.compile("|".join(re.escape(s) for s in ordered))


def redact(text):
    """``text`` with every registered secret replaced by ``[REDACTED]``."""
    pattern = _pattern
    if pattern is None or not text:
        return text
    return pattern.sub(REDACTED, text)


def _clear_registered_secrets():
    """Forget every registered secret. For tests only."""
    global _pattern
    with _lock:
        _secrets.clear()
        _pattern = None


class RedactingFilter(logging.Filter):
    """Rewrites a log record so no registered secret survives formatting.

    The message is formatted with its arguments first and the result redacted,
    because a secret is as likely to arrive as an argument (``"%s", exc``) as
    in the format string; the record then carries the finished text and no
    arguments. The traceback is formatted here too (``exc_text``), which
    ``logging.Formatter`` uses as is instead of formatting it again.

    The record is changed in place, so every handler after this one -- the
    terminal's included -- sees the redacted version.
    """

    def filter(self, record):
        if _pattern is None:
            return True
        try:
            message = record.getMessage()
        except Exception:
            # A format string that does not match its arguments: keep both,
            # redacted, rather than drop the record.
            message = f"{record.msg} {record.args!r}"
        record.msg = redact(str(message))
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


_filter = RedactingFilter()


def install_log_redaction(logger):
    """Add the redacting filter to ``logger`` and to each of its handlers.

    Both, because a logger's own filters see only records logged on that
    logger by name; a record from a child (``cellmap_flow.dashboard.x``)
    reaches the package logger's handlers without passing its filters. Calling
    this again adds nothing twice; a handler added later needs another call.
    """
    for target in [logger, *logger.handlers]:
        if not any(isinstance(f, RedactingFilter) for f in target.filters):
            target.addFilter(_filter)


def resolve_api_key(options):
    """A key-based provider's API key, from ``api_key_env`` or ``api_key_file``.

    None when the provider names neither (Vertex AI, which uses Application
    Default Credentials). The value is registered for redaction before it is
    returned, and is never logged. A key file must be a regular file owned by
    the current user with no group or world permission bits (``chmod 600``),
    the rule ssh applies to private keys: a key others can read is a key
    others can spend. An inline ``api_key`` is refused here as well as in the
    config loader, so a caller that skipped the loader cannot use one.
    """
    if "api_key" in options:
        raise AIAnnotateError(
            "config",
            "API keys may not be written in the AI-annotate config: use api_key_env or api_key_file.",
        )
    env_name = options.get("api_key_env")
    key_file = options.get("api_key_file")
    if env_name and key_file:
        raise AIAnnotateError("config", "Give either api_key_env or api_key_file, not both.")
    if env_name:
        value = os.environ.get(str(env_name), "").strip()
        if not value:
            raise AIAnnotateError("auth", f"The environment variable {env_name} that should hold the API key is not set.")
        register_secret(value)
        return value
    if key_file:
        value = _read_key_file(os.path.expanduser(str(key_file)))
        register_secret(value)
        return value
    return None


def _read_key_file(path):
    """The stripped contents of a key file, after checking who can read it.

    The checks run on the opened file (``fstat``), not the path, so the file
    cannot be swapped between the check and the read. ``O_NONBLOCK`` keeps a
    FIFO named by mistake from hanging the open; it is refused just after.
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        raise AIAnnotateError("config", f"Could not open the API key file {path}.") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AIAnnotateError("config", f"The API key file {path} is not a regular file.")
        if info.st_uid != os.getuid():
            raise AIAnnotateError("config", f"The API key file {path} must be owned by you.")
        if info.st_mode & 0o077:
            raise AIAnnotateError(
                "config",
                f"The API key file {path} can be read or written by other users: run `chmod 600 {path}`.",
            )
        if info.st_size > _MAX_KEY_FILE_BYTES:
            raise AIAnnotateError("config", f"The API key file {path} is too large to be a key.")
        with os.fdopen(fd, "r", closefd=False) as handle:
            value = handle.read().strip()
    finally:
        os.close(fd)
    if not value:
        raise AIAnnotateError("config", f"The API key file {path} is empty.")
    return value
