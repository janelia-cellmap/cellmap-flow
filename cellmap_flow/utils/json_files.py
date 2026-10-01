"""Writing a JSON file that another process, perhaps on another host, reads.

A finetune run's metadata.json is the case in point: the training job and the
dashboard's job monitor both update it, each from its own host. Written in
place, with open('w') and json.dump, the file can be read while it is empty
or cut short. Neither side imports the other, so the helper lives here.
"""

import json
import os
import uuid
from pathlib import Path


def write_json_atomically(path, data) -> None:
    """Replace ``path`` with ``data`` as JSON, whole.

    A reader sees the old file or the new one, never part of either. The
    JSON goes into a temporary file next to ``path``, named for this one
    write so that two writers never share one, and os.replace then moves it
    into place. If the write fails, the temporary file is removed and
    ``path`` is left as it was.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
