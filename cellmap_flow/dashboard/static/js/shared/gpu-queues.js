// The GPU queue pickers' data: which LSF GPU queues there are, which are
// open and how busy they are (/api/gpu-queues).
//
// LSF is the only thing that knows, and that changes without anyone here
// doing something (an admin takes nodes out, a thousand jobs land), so the
// answer is re-read every minute -- while the page is visible: a hidden page
// shows no picker, and asks once as soon as it is shown again. Every picker
// on the page shares this one poller, and the server caches its answer, so
// several open dashboards don't multiply into a query storm.
import { getJSON } from "../lib/api.js";
import { poll } from "../lib/poll.js";

const GPU_QUEUE_POLL_MS = 60000;

const subscribers = new Set();
let poller = null;
let latest;  // the last answer, for a subscriber that joins later

function refresh() {
  return getJSON("/api/gpu-queues")
    .then((data) => {
      latest = data;
      subscribers.forEach((render) => render(data));
    })
    .catch(() => {
      // Transient; the next poll retries. Leave the pickers as they are
      // rather than blanking them under the user.
    });
}

// Call render(data) with every answer, and at once with the last one if
// there is one. The first subscriber starts the poller, which asks at once;
// the unsubscribe function this returns stops it after the last one leaves.
export function subscribeGpuQueues(render) {
  subscribers.add(render);
  if (latest !== undefined) render(latest);
  if (!poller) poller = poll(refresh, { intervalMs: GPU_QUEUE_POLL_MS, pauseWhenHidden: true });
  return () => {
    subscribers.delete(render);
    if (!subscribers.size && poller) {
      poller.stop();
      poller = null;
    }
  };
}

// Fill a <select> from an answer: one option per queue, "label — description".
// A closed queue accepts a job and never starts it, which looks exactly like
// a slow one, so it is listed but disabled. `preferred` is then selected; if
// LSF doesn't list it (a YAML can name any queue), it is added as
// "<queue><missingSuffix>" rather than silently moving the user onto another
// GPU. With keepWhenEmpty, an answer with no queues leaves the options as
// they are.
export function fillQueueSelect(select, data, { preferred, missingSuffix = " (current)", keepWhenEmpty = false } = {}) {
  const queues = (data && data.queues) || [];
  if (!queues.length && keepWhenEmpty) return;
  select.replaceChildren(...queues.map((q) => {
    const opt = document.createElement("option");
    opt.value = q.queue;
    // textContent, not innerHTML: this string is assembled from LSF output.
    opt.textContent = q.label + (q.description ? " — " + q.description : "");
    opt.disabled = q.open === false;
    return opt;
  }));
  if (preferred && !queues.some((q) => q.queue === preferred)) {
    const opt = document.createElement("option");
    opt.value = preferred;
    opt.textContent = preferred + missingSuffix;
    select.appendChild(opt);
  }
  if (preferred) select.value = preferred;
}

// The note under a picker: why queue availability is unknown, or "".
export function queueHint(data) {
  return data && data.available === false ? data.reason || "Queue availability is unavailable." : "";
}
