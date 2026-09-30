"""The trainer's log, as the job manager reads it.

The trainer tees everything it prints into its run's ``training_log.txt``,
and the lines the manager reads are the markers in ``finetune.markers``. A
job outlives a dashboard upgrade, so both sides of those lines are a
protocol.

- ``LogTailer``: the lines appended since the last read, whole lines only,
  and what all the lines read so far say of the finished iterations; the
  monitor follows a running job's log with it, reading each line once.
- ``Iterations``: what a log's lines, read in order, say of the iterations
  the trainer finished: how many, and the last one's model name and
  serving YAML.
- ``trainer_outputs_from_log``: that model name and YAML, for a whole log.
- ``finished_iterations``: how many iterations a log file says finished,
  which is what rehydration has to go on for a job LSF no longer knows.
"""

import logging
from pathlib import Path

from cellmap_flow.finetune import markers

logger = logging.getLogger(__name__)


class Iterations:
    """What a log's lines say of the iterations the trainer finished.

    ``feed`` takes the lines in the order they were printed, in as many
    calls as they come in. Then ``count`` is how many
    TRAINING_ITERATION_COMPLETE markers they held, ``name`` the last one's
    model name, and ``yaml_path`` its serving YAML: the trainer prints
    FINETUNED_MODEL_YAML just before the marker, and skips it when it could
    not write one, so a YAML is only the iteration's if it came after the
    iteration before. Both are None before the first.
    """

    def __init__(self):
        self.count = 0
        self.name = None
        self.yaml_path = None
        self._yaml = None  # printed since the last iteration finished

    def feed(self, lines) -> None:
        for line in lines:
            yaml = markers.MODEL_YAML_RE.search(line)
            if yaml:
                self._yaml = yaml.group(1)
            for name in markers.ITERATION_COMPLETE_RE.findall(line):
                self.count += 1
                self.name, self.yaml_path, self._yaml = name, self._yaml, None


class LogTailer:
    """The lines appended to a log since the last read, whole lines only.

    A read can end mid-line ("Epoch 7/10 - Lo", "TRAINING_ITERATION_COM");
    parsed as it stood, the epoch's loss or the marker would be lost or cut
    short. The incomplete end is held back until the rest of the line is
    written.

    ``iterations`` (Iterations) holds what every line handed out so far says
    of the finished iterations, so what depends on the whole log never needs
    it read again.

    tee writes one log for the job's whole life, restarts included, so it only
    grows. If it shrinks, something replaced it, and reading starts over,
    ``iterations`` with it.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._start_over()

    def _start_over(self):
        self.position = 0
        self._partial = ""
        self.iterations = Iterations()

    def read(self) -> str:
        """The complete lines written since the last call, or "" if none.

        Raises OSError when the log cannot be read; nothing is consumed then,
        so the next call picks up from the same place.
        """
        size = self.path.stat().st_size
        if size < self.position:
            logger.info(f"Log file truncated (size {size} < position {self.position}), resetting")
            self._start_over()

        with open(self.path, "r") as f:
            f.seek(self.position)
            new_content = f.read()
            self.position = f.tell()

        text = self._partial + new_content
        cut = text.rfind("\n") + 1
        text, self._partial = text[:cut], text[cut:]
        self.iterations.feed(text.splitlines())
        return text


def trainer_outputs_from_log(log_text: str):
    """(model name, serving YAML path) of the last iteration the log reports.

    Either is None when the log has none; see Iterations.
    """
    iterations = Iterations()
    iterations.feed(log_text.splitlines())
    return iterations.name, iterations.yaml_path


def finished_iterations(log_file):
    """(how many iterations the log says finished, the last one's model name).

    Read a line at a time, since the log is everything the run printed. A log
    that is missing or cannot be read is no evidence: (0, None).
    """
    iterations = Iterations()
    try:
        with open(log_file, errors="replace") as f:
            iterations.feed(f)
    except OSError:
        return 0, None
    return iterations.count, iterations.name
