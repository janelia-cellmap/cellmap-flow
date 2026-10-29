// Requests to the finetune routes (dashboard/routes/finetune/).
//
// Those routes answer JSON with {success, error} at any HTTP status -- a 404
// for a job that is gone, a 400 for bad parameters -- and this tab shows a
// failed answer the same way whatever its status. So these resolve to the
// parsed answer for every status, where lib/api's getJSON and postJSON throw
// on a non-2xx one. A request that got no answer, or an answer that is not
// JSON, rejects as fetch and Response.json() do.
import { poll } from "../../lib/poll.js";

// signal: an AbortSignal that cancels the request (a poll's).
export function getAnswer(url, { signal } = {}) {
  return fetch(url, { signal }).then((response) => response.json());
}

// getAnswer, except that a 404 resolves to null: the route saying that what
// was asked about does not exist (a job the dashboard does not know), or no
// such route at all. Its body is not read, as the second is not JSON.
export function getAnswerIfFound(url) {
  return fetch(url).then((response) => (response.status === 404 ? null : response.json()));
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

// What the New Volume button's click came to (crops.js): detail.ok, whether
// a volume was made.
export const VOLUME_CREATED = "cellmap-flow:volume-created";

// A refusal that needs an annotation volume first (the answer says
// needs_volume): says why, and offers to start one with the New Volume
// button, brought into view. Cancel leaves it to the user, e.g. to resume an
// existing volume instead. Once the volume is made, ``then`` runs (the action
// that needed it, again), so what was said, that there is no volume, gives
// way to what happens now; a New Volume that made none leaves no ``then``
// behind for a later one.
export function offerNewVolume(message, then) {
  const create = confirm(
    `${message}\n\nCreate a new annotation volume now? (Cancel to resume an existing one instead.)`
  );
  if (!create) return false;
  const button = document.getElementById("createVolumeBtn");
  if (!button) return false;
  button.scrollIntoView({ block: "center" });
  if (then) document.addEventListener(VOLUME_CREATED, (e) => e.detail.ok && then(), { once: true });
  button.click();
  return true;
}
