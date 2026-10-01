"""Writing a JSON file that another process, perhaps on another host, reads.

A finetune run's metadata.json is the case in point: the training job and the
dashboard's job monitor both update it, each from its own host. Written in
place, with open('w') and json.dump, the file can be read while it is empty
or cut short. Both import it from here: the training job cannot import the job manager,
which is the dashboard's.
"""

import json
import os
import uuid
from pathlib import Path


def write_json_atomically(path, data) -> None:
    """Write ``data`` to ``path`` as indented JSON, replacing the file whole.

    The JSON goes into a temporary file beside ``path``, named for this one
    write, which then replaces ``path``. So a reader finds the old file or
    the new one, never part of one: the trainer on its own host, another
    dashboard, or another thread of this one. And two writers never share a
    temporary file. Raises what serializing or writing raises (TypeError,
    OSError), with ``path`` left as it was and no temporary file behind.
    """
    path = Path(path)
    text = json.dumps(data, indent=2)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # still there only if the write or the replace failed
