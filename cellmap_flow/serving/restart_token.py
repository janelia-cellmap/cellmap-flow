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
import tempfile
from pathlib import Path
from typing import Optional

TOKEN_FILE = "restart_token"
TOKEN_HEADER = "X-Restart-Token"


def write_restart_token(output_dir) -> str:
    """Create a new token in ``output_dir`` (mode 0600) and return it."""
    token = secrets.token_urlsafe(32)
    # Written to a new 0600 file (mkstemp's mode) and renamed over the old
    # one. Rewriting an existing token file in place would keep its mode
    # until a chmod afterwards, and anyone who had it open could read the
    # new token through their handle.
    fd, temp = tempfile.mkstemp(dir=output_dir, prefix=f".{TOKEN_FILE}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(token)
        os.replace(temp, Path(output_dir) / TOKEN_FILE)
    except BaseException:
        os.unlink(temp)
        raise
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
