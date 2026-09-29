// A server-sent log stream that picks up where it stopped (the dashboard's
// log protocol, routes/finetune/training.py):
// - each event's id is the byte offset in the log just after its lines, and
//   the browser sends the last one back as Last-Event-ID when it reconnects
//   by itself after an error, so nothing is replayed;
// - `offset` starts a new stream part-way through (?offset=N), after what
//   the page already shows;
// - the server ends a finished log with a "done" event, whose data is the
//   job's final status; that closes the stream.
//
// onLine(data, event) gets each event's data, which may hold several
// newline-separated lines; onDone(status) runs after "done"; onError(event)
// on a connection error, after which the browser reconnects unless the
// stream was closed; onOpen(event) each time it connects.
// Returns { close(), offset, closed }.
export function openLogStream(url, { offset = 0, onLine, onDone, onError, onOpen } = {}) {
  let last = offset || 0;
  const source = new EventSource(last ? `${url}${url.includes("?") ? "&" : "?"}offset=${last}` : url);
  source.onopen = (event) => {
    if (onOpen) onOpen(event);
  };
  source.onmessage = (event) => {
    const id = parseInt(event.lastEventId, 10);
    if (Number.isFinite(id)) last = id;
    if (onLine) onLine(event.data, event);
  };
  source.addEventListener("done", (event) => {
    source.close();
    if (onDone) onDone(event.data, event);
  });
  source.onerror = (event) => {
    if (onError) onError(event);
  };
  return {
    close() {
      source.close();
    },
    get offset() {
      return last;
    },
    get closed() {
      return source.readyState === EventSource.CLOSED;
    },
  };
}
