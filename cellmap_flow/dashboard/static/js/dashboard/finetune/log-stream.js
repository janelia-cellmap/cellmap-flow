// The Training Logs card's text: what the job has logged, streamed live.
//
// The stream is lib/sse's: each event carries its byte offset in the log,
// so a reconnect continues where it left off instead of replaying the whole
// log, and a new stream can start part-way, after what the card already
// shows (a job restored after a reload).
//
// The card keeps the log's last MAX_LINES lines. Once earlier ones have
// gone, its first line says how many, and where the whole log is.
//
// onLine(line) gets every line the stream brings, one at a time; logFile()
// is the path of the job's log file, or null while that is not known.
import { openLogStream } from "../../lib/sse.js";

// Every change to a textarea's text lays all of it out again, so a log that
// only grows makes each new line slower to show than the last: in headless
// Chrome an append costs about 12 ms per 1,000 lines shown, and the stream
// sends up to ten a second. At 1,000 lines an append still fits in a frame,
// and that is the last ten epochs' batch lines at a hundred batches an
// epoch. The loss plot keeps every epoch, and the log file everything.
const MAX_LINES = 1000;

// The number of line breaks in text.
function countLines(text) {
  let count = 0;
  for (let i = text.indexOf("\n"); i !== -1; i = text.indexOf("\n", i + 1)) count += 1;
  return count;
}

export function createJobLog({ onLine, logFile }) {
  const area = document.getElementById("trainingLogs");
  const autoScroll = document.getElementById("autoScrollLogs");
  let stream = null;  // the open stream, or the last one
  let closeTimer = null;
  let connectedOnce = false;
  let restoredOffset = 0;  // where resume() starts when no stream has run
  // What the card shows, less the note about dropped lines: its placeholder
  // at first. lineCount is its line breaks; dropped, the lines taken off its
  // front since it was last cleared or replaced.
  let text = area.value;
  let lineCount = countLines(text);
  let dropped = 0;

  function dropOldest() {
    if (lineCount <= MAX_LINES) return;
    let cut = 0;
    for (let excess = lineCount - MAX_LINES; excess > 0; excess -= 1) {
      cut = text.indexOf("\n", cut) + 1;
    }
    text = text.slice(cut);
    dropped += lineCount - MAX_LINES;
    lineCount = MAX_LINES;
  }

  function render() {
    if (!dropped) {
      area.value = text;
      return;
    }
    const where = logFile() || "training_log.txt in the job's output directory";
    area.value = `[${dropped.toLocaleString()} earlier lines are not shown here; ` +
      `the whole log is ${where}]\n` + text;
  }

  function replace(value) {
    text = value;
    lineCount = countLines(value);
    dropped = 0;
    dropOldest();
    render();
  }

  function append(message) {
    text += message + "\n";
    lineCount += countLines(message) + 1;
    const droppedBefore = dropped;
    dropOldest();
    render();
    // With Auto-scroll off, someone is reading: keep their place. The lines
    // taken off the top (less the note, the first time it appears) would
    // move the text under the view up by as many. Every line is one line
    // high (white-space: pre), so a line's height is the text's over its
    // line count.
    const gone = dropped - droppedBefore - (droppedBefore ? 0 : 1);
    if (gone > 0 && !autoScroll.checked) {
      const linesShown = 1 + lineCount + 1;  // the note, then the text's lines
      area.scrollTop -= gone * (area.scrollHeight / linesShown);
    }
  }

  function follow(jobId, offset) {
    if (stream) stream.close();
    clearTimeout(closeTimer);
    closeTimer = null;
    stream = openLogStream(`/api/finetune/job/${jobId}/logs/stream`, {
      offset,
      onOpen() {
        if (!connectedOnce) {
          append("Connected to live log stream.");
          connectedOnce = true;
        }
      },
      // The server's last word: the job is finished and the log complete.
      onDone(status) {
        append(`=== Training ${status} ===`);
      },
      onLine(data) {
        append(data);
        // One event can carry several log lines: the server sends a block of
        // "data: " lines and the browser joins them into one newline-
        // separated event. Parsing the block as one line would only ever
        // match the first "Epoch N/M - Loss:" in it, dropping every other
        // epoch from the plot whenever epochs finish fast enough to batch.
        data.split("\n").forEach((line) => onLine(line));
        if (autoScroll.checked) {
          area.scrollTop = area.scrollHeight;
        }
      },
      onError(event) {
        console.error("Log streaming error:", event);
        // Not closed here: the browser reconnects by itself and resumes from
        // the last offset. Closing would stop this job's log for good.
      },
    });
  }

  return {
    append,
    clear() {
      replace("");
    },
    // The log so far, read in one piece (a job restored after a reload).
    show(log) {
      replace(log);
      area.scrollTop = area.scrollHeight;
    },
    // Stream a job's log into the card, from `offset` bytes in; a stream
    // already open is closed first.
    follow,
    // Close the stream once the job is over. Not at once: the server sends
    // the last lines and "done" within a moment of the job finishing, and
    // closing first would lose them. This is the backstop for a "done" that
    // never comes, e.g. a stream stuck reconnecting.
    closeSoon() {
      if (closeTimer || !stream || stream.closed) {
        return;  // already closed (by "done", as a rule), or closing
      }
      const closing = stream;
      closeTimer = setTimeout(() => {
        closeTimer = null;
        closing.close();
      }, 5000);
    },
    // Reopen the stream where it stopped, if it is not running.
    resume(jobId) {
      if (!stream || stream.closed) {
        follow(jobId, stream ? stream.offset : restoredOffset);
      }
    },
    // Where resume() starts when no stream has run: the end of the log the
    // card shows.
    resumeFrom(offset) {
      restoredOffset = offset;
    },
  };
}
