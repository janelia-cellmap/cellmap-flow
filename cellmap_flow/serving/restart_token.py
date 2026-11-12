"""The shared secret that authorizes a server's control routes.

A finetune job's embedded inference server accepts POST /__control__/restart
and /__control__/stop, and a restart can change what the job trains on, so
neither may be callable by anyone who can reach the node. The job manager
writes a random token into the job's output directory before submitting it,
readable only by the user; the training process reads it from there and the
dashboard sends it with every restart and stop.

Every inference server the dashboard starts has a token too, for POST
/__control__/dataset (serving.protocol.DATASET_CONTROL_PATH), which opens
another dataset there: a route anyone could call would read any file the
user can. The launcher writes it to a file of its own before submitting
(``write_token_file``) and names the file in the job's environment
(``CONTROL_TOKEN_FILE_ENV``); the server reads it at startup. A server
started without one has the route turned off.

A token never goes on the command line, where `bjobs -l` would show it to
other users of the cluster.
"""

import hmac
import os
import secrets
import tempfile
from pathlib import Path
from typing import Optional

TOKEN_FILE = "restart_token"
# The name of the header every control request carries its token in, the
# dataset route's included: "Restart" only for where it started.
TOKEN_HEADER = "X-Restart-Token"
# The job's environment names the file an inference server reads its token
# from (write_token_file wrote it).
CONTROL_TOKEN_FILE_ENV = "CELLMAP_FLOW_CONTROL_TOKEN_FILE"


def write_token_file(path) -> str:
    """Create a new token at ``path`` (mode 0600) and return it."""
    path = Path(path)
    token = secrets.token_urlsafe(32)
    # Written to a new 0600 file (mkstemp's mode) and renamed over the old
    # one. Rewriting an existing token file in place would keep its mode
    # until a chmod afterwards, and anyone who had it open could read the
    # new token through their handle.
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(token)
        os.replace(temp, path)
    except BaseException:
        os.unlink(temp)
        raise
    return token


def read_token_file(path) -> Optional[str]:
    """The token at ``path``, or None if there is none."""
    try:
        token = Path(path).read_text().strip()
    except FileNotFoundError:
        return None
    return token or None


def write_restart_token(output_dir) -> str:
    """Create a new token in ``output_dir`` (mode 0600) and return it."""
    return write_token_file(Path(output_dir) / TOKEN_FILE)


def read_restart_token(output_dir) -> Optional[str]:
    """The token in ``output_dir``, or None if there is none."""
    return read_token_file(Path(output_dir) / TOKEN_FILE)


def read_or_create_restart_token(output_dir) -> str:
    """The job's token, creating one for runs the job manager did not launch."""
    return read_restart_token(output_dir) or write_restart_token(output_dir)


def tokens_match(expected: Optional[str], provided: Optional[str]) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode(), provided.encode())
