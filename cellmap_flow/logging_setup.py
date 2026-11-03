"""One logging format, actually applied.

``logging.basicConfig`` does nothing once the root logger has a handler, and
``cellmap_flow.globals`` configures it at import time -- and nearly
everything used to import that early. So every later ``basicConfig`` call in
a CLI was a silent no-op: ``yaml_cli`` has been asking for a timestamped format for a
while, and the output has always come out as the bare default::

    INFO:cellmap_flow.cli.yaml_cli:Loading configuration from: ...

with no time on it, which is no help at all when the question is which step
took four minutes. ``finetune_cli`` is the one place that passed
``force=True``, which is why training logs have timestamps and nothing else
does.

Anything that wants to set the format has to pass ``force=True``, and should
use the same format while doing it.

On a terminal the same lines are colored (``ConsoleFormatter``): the time
and logger dimmed, the level by its severity, so the message is what the
eye lands on and a warning stands out from the forty lines around it. A
job's log file, a pipe or a test's capture gets the plain format: the
escape codes would only litter them.
"""

import logging
import os
import re

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_RESET = "\033[0m"
_DIM = "\033[2m"
_URL_STYLE = "\033[1;4;36m"  # bold, underlined, cyan: the dashboard's and the hosts' addresses
_URL = re.compile(r"https?://[^\s'\"<>()\[\]]+")
_LEVEL_COLORS = {
    logging.DEBUG: _DIM,
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_PACKAGE_PREFIX = "cellmap_flow."


def wants_color(stream) -> bool:
    """Whether what goes to ``stream`` should be colored: it is a terminal,
    and neither NO_COLOR (no-color.org) nor TERM=dumb says otherwise."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class ConsoleFormatter(logging.Formatter):
    """LOG_FORMAT, colored for a terminal.

    The time and the logger's name are dimmed, the level is colored by its
    severity and padded so the messages line up, and a warning's or error's
    message takes the level's color too. A URL in a message is bold,
    underlined and cyan: the dashboard's address and the inference servers'
    are what the startup log is read for. The ``cellmap_flow.`` prefix is
    dropped from logger names: every line had it, so it said nothing.
    """

    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelno, "")
        name = record.name[len(_PACKAGE_PREFIX):] if record.name.startswith(_PACKAGE_PREFIX) else record.name
        message = record.getMessage()
        colored = record.levelno >= logging.WARNING
        # After a URL, back to the message's own color, if it has one.
        message = _URL.sub(lambda m: f"{_URL_STYLE}{m.group(0)}{_RESET}{color if colored else ''}", message)
        if colored:
            message = f"{color}{message}{_RESET}"
        tail = ""
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            tail += "\n" + record.exc_text
        if record.stack_info:
            tail += "\n" + self.formatStack(record.stack_info)
        return (
            f"{_DIM}{self.formatTime(record, self.datefmt)}{_RESET} "
            f"{color}{record.levelname:<7}{_RESET} {_DIM}{name}:{_RESET} {message}{tail}"
        )


def configure_logging(level=logging.INFO):
    """Install the shared timestamped format, replacing any earlier config;
    colored when the handler writes to a terminal.

    ``force=True`` is the point: without it this is a no-op in every process
    that has already imported ``cellmap_flow.globals``.
    """
    logging.basicConfig(
        level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True
    )
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and wants_color(getattr(handler, "stream", None)):
            handler.setFormatter(ConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
