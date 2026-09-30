// Requests to the finetune routes (dashboard/routes/finetune/).
//
// Those routes answer JSON with {success, error} at any HTTP status -- a 404
// for a job that is gone, a 400 for bad parameters -- and this tab shows a
// failed answer the same way whatever its status. So these resolve to the
// parsed answer for every status, where lib/api's getJSON and postJSON throw
// on a non-2xx one. A request that got no answer, or an answer that is not
// JSON, rejects as fetch and Response.json() do.
import { poll } from "../../lib/poll.js";

export function getAnswer(url) {
  return fetch(url).then((response) => response.json());
}

// The body is sent as JSON. With no body (undefined) the request has none,
// and no Content-Type either.
export function postAnswer(url, body) {
  const init = { method: "POST" };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return fetch(url, init).then((response) => response.json());
}

// A long POST that reports its progress as it runs. The page makes up a load
// id and sends it with the POST; the server publishes the POST's progress
// under it at progressUrl?load_id=<id>. This polls that every second, the
// first time at once, and passes each progress published to show(progress).
// A poll that fails is skipped; the next one tries again. While the page is
// hidden nobody sees the progress, so the polls wait, and one runs as soon as
// it is shown.
// Returns { loadId, stop() }; stop it once the POST has answered.
export function watchProgress(progressUrl, show) {
  const loadId = (crypto.randomUUID && crypto.randomUUID()) ||
    (Math.random().toString(36).slice(2) + Date.now());
  const url = `${progressUrl}?load_id=${encodeURIComponent(loadId)}`;
  const poller = poll(async ({ stale }) => {
    try {
      const answer = await getAnswer(url);
      if (stale() || !answer.success || !answer.progress) return;
      show(answer.progress);
    } catch (_) {
      // Not fatal: the next poll tries again.
    }
  }, { intervalMs: 1000, pauseWhenHidden: true });
  return { loadId, stop: poller.stop };
}
