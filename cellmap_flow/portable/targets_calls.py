"""What the browser calls for training targets: the function
``web/src/finetune/target-call.ts`` runs through chunkmirage's
``Engine.call("cellmap_flow.portable.targets_calls:targets", args, arrays)``
(``calls.py`` says the shape). It decodes the chunk's labels, calls the maths
in ``targets.py`` and encodes the answer; the maths is not here.
"""

import numpy as np

from cellmap_flow.portable import targets as maths
from cellmap_flow.portable.label_it import affinity_offsets


def targets(args, arrays):
    """A painted chunk's target and mask (``targets.binary``, or
    ``targets.affinity`` for an affinity model). ``args``: ``shape``
    (z, y, x), ``labels_dtype``, ``channels`` (the model's output channels),
    and either ``offsets`` (null for a binary model, else one [dz, dy, dx]
    or null per channel from the first, as a caller that knows them gives
    them) or ``channel_names``, from which the server's rule decides
    (``label_it.affinity_offsets``: a model with a ``*_aff*`` channel is an
    affinity model, its first affinity channels along z, y and x), so the
    browser, which has only its manifest's names, trains on what the server
    would. ``arrays``: the chunk's labels (z, y, x). Answers the channels
    with the target and the mask, each float32 (channels, z, y, x)."""
    labels = np.frombuffer(arrays[0], np.dtype(args["labels_dtype"])).reshape(tuple(args["shape"]))
    channels, offsets = int(args["channels"]), args.get("offsets")
    if offsets is None and args.get("channel_names"):
        offsets = affinity_offsets(args["channel_names"])
    if offsets is None:
        target, mask = maths.binary(labels, channels)
    else:
        target, mask = maths.affinity(labels, offsets, channels)
    return {"channels": channels}, [target, mask]
