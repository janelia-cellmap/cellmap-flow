"""Following a training log as it grows, a whole line at a time."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LogTailer:
    """The lines appended to a log since the last read, whole lines only.

    A read can end mid-line ("Epoch 7/10 - Lo", "TRAINING_ITERATION_COM");
    parsed as it stood, the epoch's loss or the marker would be lost or cut
    short. The incomplete end is held back until the rest of the line is
    written.

    tee writes one log for the job's whole life, restarts included, so it only
    grows. If it shrinks, something replaced it, and reading starts over.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.position = 0
        self._partial = ""

    def read(self) -> str:
        """The complete lines written since the last call, or "" if none.

        Raises OSError when the log cannot be read; nothing is consumed then,
        so the next call picks up from the same place.
        """
        size = self.path.stat().st_size
        if size < self.position:
            logger.info(f"Log file truncated (size {size} < position {self.position}), resetting")
            self.position = 0
            self._partial = ""

        with open(self.path, "r") as f:
            f.seek(self.position)
            new_content = f.read()
            self.position = f.tell()

        text = self._partial + new_content
        cut = text.rfind("\n") + 1
        text, self._partial = text[:cut], text[cut:]
        return text
