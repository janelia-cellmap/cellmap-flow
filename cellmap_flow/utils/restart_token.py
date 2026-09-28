"""The shared secret that authorizes restarting a finetune job over HTTP.

A job's embedded inference server accepts POST /__control__/restart, and a
restart can change what the job trains on, so it must not be callable by
anyone who can reach the node. The job manager writes a random token into the
job's output directory before submitting it, readable only by the user; the
training process reads it from there and the dashboard sends it with every
restart. It never goes on the command line, where `bjobs -l` would show it to
other users of the cluster.
"""

import hmac
import os
import secrets
from pathlib import Path
from typing import Optional

TOKEN_FILE = "restart_token"
TOKEN_HEADER = "X-Restart-Token"


def write_restart_token(output_dir) -> str:
    """Create a new token in ``output_dir`` (mode 0600) and return it."""
    token = secrets.token_urlsafe(32)
    path = Path(output_dir) / TOKEN_FILE
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.chmod(path, 0o600)  # O_CREAT's mode does not apply to an existing file
    return token


def read_restart_token(output_dir) -> Optional[str]:
    """The token in ``output_dir``, or None if there is none."""
    try:
        token = (Path(output_dir) / TOKEN_FILE).read_text().strip()
    except FileNotFoundError:
        return None
    return token or None


def read_or_create_restart_token(output_dir) -> str:
    """The job's token, creating one for runs the job manager did not launch."""
    return read_restart_token(output_dir) or write_restart_token(output_dir)


def tokens_match(expected: Optional[str], provided: Optional[str]) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode(), provided.encode())
