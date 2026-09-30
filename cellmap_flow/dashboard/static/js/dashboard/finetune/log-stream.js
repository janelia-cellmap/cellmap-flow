// The Training Logs card's text: what the job has logged, streamed live.
//
// The stream is server-sent events. Every block the server sends carries its
// byte offset in the log as the event id; the browser sends the last one back
// when it reconnects, so a reconnect continues where it left off instead of
// replaying the whole log. `offset` starts a new stream part-way, after what
// the card already shows (a job restored after a reload).
//
// onLine(line) gets every line the stream brings, one at a time.
export function createJobLog({ onLine }) {
  const area = document.getElementById("trainingLogs");
  const autoScroll = document.getElementById("autoScrollLogs");
  let eventSource = null;
  let lastLogOffset = 0;  // byte offset in the job's log the page has shown up to
  let logStreamCloseTimer = null;
  let streamConnectedOnce = false;

  function append(message) {
    area.value += message + "\n";
  }

  function follow(jobId, offset) {
    // Close existing stream if any
    if (eventSource) {
      eventSource.close();
    }
    clearTimeout(logStreamCloseTimer);
    logStreamCloseTimer = null;
    lastLogOffset = offset || 0;

    const query = lastLogOffset ? `?offset=${lastLogOffset}` : '';
    eventSource = new EventSource(`/api/finetune/job/${jobId}/logs/stream${query}`);
    const stream = eventSource;

    eventSource.onopen = function() {
      if (!streamConnectedOnce) {
        append("Connected to live log stream.");
        streamConnectedOnce = true;
      }
    };

    // The server's last word: the job is finished and the log complete.
    eventSource.addEventListener("done", function(event) {
      stream.close();
      append(`=== Training ${event.data} ===`);
    });

    eventSource.onmessage = function(event) {
      const id = parseInt(event.lastEventId, 10);
      if (Number.isFinite(id)) {
        lastLogOffset = id;
      }
      append(event.data);
      // One SSE event can carry several log lines: the server emits a block of
      // "data: " lines and the browser concatenates them into a single
      // newline-separated event.data. Parsing the blob as one line would only
      // ever match the FIRST "Epoch N/M - Loss:" in it (String.match without
      // /g returns one match), silently dropping every other epoch from the
      // plot whenever epochs complete fast enough to batch.
      event.data.split("\n").forEach((line) => onLine(line));

      if (autoScroll.checked) {
        area.scrollTop = area.scrollHeight;
      }
    };

    eventSource.onerror = function(error) {
      console.error("Log streaming error:", error);
      // Do not close here - EventSource auto-reconnects by default, and
      // resumes from the last event id. Closing would permanently stop log
      // updates for this job.
    };
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
      if (logStreamCloseTimer || !eventSource || eventSource.readyState === EventSource.CLOSED) {
        return;  // already closed, or closing (the poller calls this every tick)
      }
      const stream = eventSource;
      logStreamCloseTimer = setTimeout(function() {
        logStreamCloseTimer = null;
        stream.close();
      }, 5000);
    },
    // Reopen the stream where it stopped, if it is not running.
    resume(jobId) {
      if (!eventSource || eventSource.readyState === EventSource.CLOSED) {
        follow(jobId, lastLogOffset);
      }
    },
    // Where resume() starts when no stream has run: the end of the log the
    // card shows.
    resumeFrom(offset) {
      lastLogOffset = offset;
    },
  };
}
