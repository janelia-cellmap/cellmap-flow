// A repeating request that never overlaps itself.
//
// poll(fn, options) calls fn at once (unless immediate is false) and then
// every intervalMs. A tick that comes while the previous call's promise is
// still pending is skipped, so a slow server gets one request at a time.
// fn resolving to false stops the poller; after maxTicks calls it stops by
// itself, and that last call's answer still counts.
//
// fn gets { signal, tick, stale() }: signal is aborted, and stale() turns
// true, when stop() or restart() ends the run the call belongs to, so an
// answer that arrives afterwards can be dropped (pass signal to getJSON to
// cancel the request itself). fn should handle its own errors; a rejection
// is logged and the poller carries on.
//
// With pauseWhenHidden, ticks are skipped while the page is hidden, and one
// that was missed runs as soon as it is shown again.
//
// Returns { stop(), restart(), running }: restart() starts over, with a fresh
// tick count and, if immediate, a call at once; running says whether more
// calls are due.
export function poll(fn, { intervalMs, maxTicks = Infinity, pauseWhenHidden = false, immediate = true } = {}) {
  let timer = null;
  let run = null;  // { controller, calls, busy, missed } for the current run

  function tick() {
    const current = run;
    if (!current || current.busy) return;
    if (pauseWhenHidden && document.hidden) {
      current.missed = true;
      return;
    }
    current.missed = false;
    if (current.calls >= maxTicks) {
      finish();
      return;
    }
    current.calls += 1;
    if (current.calls >= maxTicks) finish();
    current.busy = true;
    const signal = current.controller.signal;
    let result;
    try {
      result = fn({ signal, tick: current.calls, stale: () => signal.aborted });
    } catch (err) {
      result = Promise.reject(err);
    }
    Promise.resolve(result)
      .then(
        (value) => { if (value === false && run === current) stop(); },
        (err) => { if (!signal.aborted) console.error("poll:", err); },
      )
      .then(() => { current.busy = false; });
  }

  function onVisibility() {
    if (!document.hidden && run && run.missed) tick();
  }

  // No more ticks; a call in flight still finishes and is still current.
  function finish() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    if (pauseWhenHidden) document.removeEventListener("visibilitychange", onVisibility);
  }

  function stop() {
    finish();
    if (run) run.controller.abort();
    run = null;
  }

  function start() {
    run = { controller: new AbortController(), calls: 0, busy: false, missed: false };
    timer = setInterval(tick, intervalMs);
    if (pauseWhenHidden) document.addEventListener("visibilitychange", onVisibility);
    if (immediate) tick();
  }

  start();
  return {
    stop,
    restart() {
      stop();
      start();
    },
    get running() {
      return timer !== null;
    },
  };
}
