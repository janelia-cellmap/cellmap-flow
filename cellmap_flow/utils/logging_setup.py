"""One logging format, actually applied.

``logging.basicConfig`` does nothing once the root logger has a handler, and
``cellmap_flow.globals`` configures it at import time -- which nearly
everything imports early. So every later ``basicConfig`` call in a CLI was a
silent no-op: ``yaml_cli`` has been asking for a timestamped format for a
while, and the output has always come out as the bare default::

    INFO:cellmap_flow.cli.yaml_cli:Loading configuration from: ...

with no time on it, which is no help at all when the question is which step
took four minutes. ``finetune_cli`` is the one place that passed
``force=True``, which is why training logs have timestamps and nothing else
does.

Anything that wants to set the format has to pass ``force=True``, and should
use the same format while doing it. That is all this module is.
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level=logging.INFO):
    """Install the shared timestamped format, replacing any earlier config.

    ``force=True`` is the point: without it this is a no-op in every process
    that has already imported ``cellmap_flow.globals``.
    """
    logging.basicConfig(
        level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True
    )
