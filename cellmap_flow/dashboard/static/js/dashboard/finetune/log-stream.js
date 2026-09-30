// The Training Logs card's text: what the job has logged, streamed live.
//
// The stream is lib/sse's: each event carries its byte offset in the log,
// so a reconnect continues where it left off instead of replaying the whole
// log, and a new stream can start part-way, after what the card already
// shows (a job restored after a reload).
//
// onLine(line) gets every line the stream brings, one at a time.
import { openLogStream } from "../../lib/sse.js";

export function createJobLog({ onLine }) {
  const area = document.getElementById("trainingLogs");
  const autoScroll = document.getElementById("autoScrollLogs");
  let stream = null;  // the open stream, or the last one
  let closeTimer = null;
  let connectedOnce = false;
  let restoredOffset = 0;  // where resume() starts when no stream has run

  function append(message) {
    area.value += message + "\n";
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
      area.value = "";
    },
    // The log so far, read in one piece (a job restored after a reload).
    show(text) {
      area.value = text;
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
