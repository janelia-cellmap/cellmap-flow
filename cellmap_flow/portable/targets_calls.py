"""What the browser calls for training targets: the function
``web/src/finetune/target-call.ts`` runs through chunkmirage's
``Engine.call("cellmap_flow.portable.targets_calls:targets", args, arrays)``
(``calls.py`` says the shape). It decodes the chunk's labels, calls the maths
in ``targets.py`` and encodes the answer; the maths is not here.
"""

import numpy as np

from cellmap_flow.portable import targets as maths


def targets(args, arrays):
    """A painted chunk's target and mask (``targets.binary``, or
    ``targets.affinity`` when ``offsets`` is given). ``args``: ``shape``
    (z, y, x), ``labels_dtype``, ``channels`` (the model's output channels)
    and ``offsets``, null for a binary model, else one [dz, dy, dx] or null
    per channel from the first, as the caller found them; ``arrays``: the
    chunk's labels (z, y, x). Answers the channels with the target and the
    mask, each float32 (channels, z, y, x)."""
    labels = np.frombuffer(arrays[0], np.dtype(args["labels_dtype"])).reshape(tuple(args["shape"]))
    channels, offsets = int(args["channels"]), args.get("offsets")
    if offsets is None:
        target, mask = maths.binary(labels, channels)
    else:
        target, mask = maths.affinity(labels, offsets, channels)
    return {"channels": channels}, [target, mask]
