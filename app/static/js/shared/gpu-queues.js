// The GPU queue pickers, and their data: which LSF GPU queues there are,
// which are open and how busy they are (/api/gpu-queues).
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

// A GPU queue picker, the one every chooser of GPU queues on the page is
// (the Models tab's two, the Finetune tab's, the first-run dialog's): a
// dropdown whose items are checkboxes, one per queue, "label — description",
// the description LSF's free GPUs and jobs waiting. A job may run on any of
// the queues ticked: it goes to the one with the most free GPUs, and moves
// to the next of them if it has not started in a few minutes (jobs/queues.py
// candidates), so each queue ticked is one more place a job can start.
//
// Every queue is ticked until a choice is made (setValue, or a box
// clicked), and the last one ticked cannot be unticked: a job needs
// somewhere to run. A closed queue is listed, and can be ticked: LSF takes a
// job there and never starts it, so it is tried last. A queue ticked that
// LSF doesn't list (a YAML can name any queue) is kept, as
// "<queue><missingSuffix>", rather than silently dropped.
//
// The dropdown is built in `root`, an empty element of the template, at the
// first answer that lists queues: until then, and where none is ever listed
// (the browser mode, whose rows are hidden), there is nothing to show, and
// an answer with no queues leaves the picker as it is. `onChange(queues)` is
// called when a box is clicked. Returns { value(), setValue(queues),
// render(answer) }; value() is the queues ticked, in the order listed, and
// [] before any answer while no choice has been made.
export function createQueuePicker(root, { small = false, missingSuffix = " (current)", onChange = () => {} } = {}) {
  let ticked = null;  // the queues chosen; null: every one listed
  let listed = [];  // the last answer's queues
  let button = null;
  let menu = null;

  const value = () => (ticked ? [...ticked] : listed.map((q) => q.queue));

  function build() {
    root.classList.add("dropdown", "gpu-queue-picker");
    button = document.createElement("button");
    button.type = "button";
    button.className = "form-select text-start" + (small ? " form-select-sm" : "");
    button.dataset.bsToggle = "dropdown";
    // Open until a click outside it: one click ticks one box, rather than
    // closing the list as well.
    button.dataset.bsAutoClose = "outside";
    button.setAttribute("aria-expanded", "false");
    button.title = "A job goes to whichever of the queues ticked has the most free GPUs.";
    menu = document.createElement("ul");
    menu.className = "dropdown-menu";
    menu.addEventListener("change", (event) => {
      const boxes = [...menu.querySelectorAll("input[type=checkbox]")];
      const chosen = boxes.filter((box) => box.checked).map((box) => box.value);
      if (!chosen.length) {
        event.target.checked = true;
        return;
      }
      ticked = chosen;
      button.textContent = summary();
      onChange(value());
    });
    root.replaceChildren(button, menu);
  }

  // The button's text: the labels of the queues ticked.
  function summary() {
    const label = (name) => (listed.find((q) => q.queue === name) || { label: name }).label;
    return value().map(label).join(", ");
  }

  function item(name, text, checked, closed) {
    const box = document.createElement("input");
    box.type = "checkbox";
    box.className = "form-check-input flex-shrink-0";
    box.value = name;
    box.checked = checked;
    const words = document.createElement("span");
    // textContent, not innerHTML: this string is assembled from LSF output.
    words.textContent = text;
    const label = document.createElement("label");
    label.className = "dropdown-item d-flex align-items-start gap-2" + (closed ? " text-muted" : "");
    label.append(box, words);
    const row = document.createElement("li");
    row.append(label);
    return row;
  }

  function draw() {
    if (!listed.length) return;
    if (!button) build();
    const chosen = new Set(value());
    const rows = listed.map((q) => [q.queue, q.label + (q.description ? " — " + q.description : ""), q.open === false]);
    for (const name of chosen) {
      if (!listed.some((q) => q.queue === name)) rows.push([name, name + missingSuffix, false]);
    }
    menu.replaceChildren(...rows.map(([name, text, closed]) => item(name, text, chosen.has(name), closed)));
    button.textContent = summary();
  }

  return {
    value,
    // Anything but a list of queues (none saved yet) is no choice.
    setValue(queues) {
      ticked = Array.isArray(queues) && queues.length ? [...queues] : null;
      draw();
    },
    render(data) {
      const queues = (data && data.queues) || [];
      if (!queues.length) return;
      listed = queues;
      draw();
    },
  };
}

// The note under a picker: why queue availability is unknown, or "".
export function queueHint(data) {
  return data && data.available === false ? data.reason || "Queue availability is unavailable." : "";
}
