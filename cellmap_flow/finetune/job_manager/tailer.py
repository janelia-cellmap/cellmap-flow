"""The trainer's log, as the job manager reads it.

The trainer tees everything it prints into its run's ``training_log.txt``,
and the lines the manager reads are the markers in ``finetune.markers``. A
job outlives a dashboard upgrade, so both sides of those lines are a
protocol.

- ``LogTailer``: the lines appended since the last read, whole lines only;
  the monitor follows a running job's log with it.
- ``trainer_outputs_from_log``: the model name and serving YAML of the last
  iteration a log reports.
- ``finished_iterations``: how many iterations a log says finished, which
  is what rehydration has to go on for a job LSF no longer knows.
"""

import logging
from pathlib import Path

from cellmap_flow.finetune import markers

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


def trainer_outputs_from_log(log_text: str):
    """(model name, serving YAML path) of the last iteration the log reports.

    Either is None when the log has none. The YAML is only taken when it
    belongs to that iteration: the trainer prints it just before the
    iteration's completion marker, and skips it when it could not write one.
    """
    names = list(markers.ITERATION_COMPLETE_RE.finditer(log_text))
    if not names:
        return None, None
    last = names[-1]
    previous_end = names[-2].end() if len(names) > 1 else 0
    yamls = [
        m for m in markers.MODEL_YAML_RE.finditer(log_text, previous_end, last.start())
    ]
    return last.group(1), (yamls[-1].group(1) if yamls else None)


def finished_iterations(log_file):
    """(how many iterations the log says finished, the last one's model name).

    Read a line at a time, since the log is everything the run printed. A log
    that is missing or cannot be read is no evidence: (0, None).
    """
    count, last = 0, None
    try:
        with open(log_file, errors="replace") as f:
            for line in f:
                for name in markers.ITERATION_COMPLETE_RE.findall(line):
                    count, last = count + 1, name
    except OSError:
        return 0, None
    return count, last
