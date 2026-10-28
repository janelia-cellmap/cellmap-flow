// The Processing tab's two lists, its input normalizers and its
// postprocessing (templates/_processing_tab.html), which the server renders:
// one row per op, the configured chain first in the order it runs (an op
// used twice has two rows), then every other op unticked. Each row has a
// drag handle, a checkbox whose value is the op's name, and its parameter
// fields.
//
// mountOpChain attaches their behaviour: dragging a row by its handle moves
// it, ticking it shows its parameters, and getChain() reads the ticked rows
// top to bottom, which is the order the steps run in.
//
// It also keeps the chain the backend runs, as this list last sent it or
// was shown it (markApplied), so pendingChanges() can say how far the list
// has moved from it: the tab's Apply bar (js/dashboard/main.js) shows the
// two lists' counts together, and nothing reaches the backend until Apply is
// pressed.
import { makeSortable } from "../lib/sortable.js";

// `held`: the op whose step apply() keeps, first, whatever it is given: the
// input list's ChannelSelector, the raw's channel (js/dashboard/raw-channel.js),
// which a model's suggested normalization says nothing about.
const KINDS = {
  input: { item: ".normalizer-item", checkbox: ".inputNormCheckbox", held: "ChannelSelector" },
  postprocess: { item: ".postprocessor-item", checkbox: ".postProcessCheckbox" },
};

// A step's identity for the comparison below: its name and its parameters,
// with the keys sorted so their order does not count. Both chains compared
// are getChain()'s, so every value is the text of a field.
function stepKey(step) {
  const keys = Object.keys(step).sort();
  return JSON.stringify(keys.map((key) => [key, step[key]]));
}

// How many edits turn the chain `from` into the chain `to` (each a list of
// {name, ...params} steps in the order they run): a step ticked or unticked
// is one edit, and so is a step whose parameters changed (the same op, other
// values); a step dragged to another place is two, since it leaves one place
// and arrives at another. This is the edit distance between the two lists,
// with "changed" allowed only between steps of the same op: unticking one op
// and ticking another is two edits, not one.
export function chainChanges(from, to) {
  const a = from || [];
  const b = to || [];
  // cost[i][j]: the edits that turn a's first i steps into b's first j.
  const cost = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) cost[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const removed = cost[i - 1][j] + 1;
      const added = cost[i][j - 1] + 1;
      let kept = Infinity;
      if (a[i - 1].name === b[j - 1].name) {
        kept = cost[i - 1][j - 1] + (stepKey(a[i - 1]) === stepKey(b[j - 1]) ? 0 : 1);
      }
      cost[i][j] = Math.min(removed, added, kept);
    }
  }
  return cost[a.length][b.length];
}

export function mountOpChain(listEl, { kind }) {
  const { item: itemSelector, checkbox: checkboxSelector, held } = KINDS[kind];
  const rows = () => Array.from(listEl.querySelectorAll(itemSelector));
  const editListeners = [];
  const edited = () => editListeners.forEach((listener) => listener());

  function showState(checkbox) {
    const row = checkbox.closest(itemSelector);
    if (!row) return;
    row.classList.toggle("item-checked", checkbox.checked);
    const params = row.querySelector(".step-params");
    if (params) params.style.display = checkbox.checked ? "block" : "none";
  }

  // [{name, ...params}] for the ticked rows, in order. Each row's own
  // fields: the same op can have two rows.
  function getChain() {
    const chain = [];
    rows().forEach((row) => {
      const checkbox = row.querySelector(checkboxSelector);
      if (!checkbox || !checkbox.checked) return;
      const step = { name: checkbox.value };
      row.querySelectorAll(".step-params input[data-param]").forEach((input) => {
        step[input.dataset.param] = input.value;
      });
      chain.push(step);
    });
    return chain;
  }

  // Make the chain exactly `steps` ([{name, ...params}], in order): tick
  // their rows, fill in each one's parameters from its step, move them to
  // the top, and untick every other row. The k-th step of an op takes the
  // k-th row for it, so two steps of the same op keep their own values; a
  // step with no row is skipped, and a parameter the row has no field for
  // is ignored.
  function show(steps) {
    const all = rows();
    const picked = [];
    steps.forEach((step) => {
      const row = all.find((r) => {
        const cb = r.querySelector(checkboxSelector);
        return cb && cb.value === step.name && !picked.some((p) => p.row === r);
      });
      if (row) picked.push({ row, step });
    });
    all.forEach((row) => {
      const cb = row.querySelector(checkboxSelector);
      const want = picked.some((p) => p.row === row);
      if (cb && cb.checked !== want) {
        cb.checked = want;
        cb.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    picked.forEach(({ row, step }) => {
      Object.keys(step).forEach((key) => {
        if (key === "name") return;
        const input = row.querySelector(`input[data-param="${key}"]`);
        if (input) input.value = step[key];
      });
    });
    picked.slice().reverse().forEach(({ row }) => listEl.insertBefore(row, listEl.firstChild));
    edited();
  }

  listEl.querySelectorAll(checkboxSelector).forEach((checkbox) => {
    showState(checkbox);
    checkbox.addEventListener("change", () => showState(checkbox));
  });
  makeSortable(listEl, { itemSelector, handleSelector: ".drag-handle", onReorder: edited });
  // A box ticked (change) or a value typed (input), anywhere in the list.
  listEl.addEventListener("change", edited);
  listEl.addEventListener("input", edited);

  // The chain the backend runs, as this list sends it. The page is rendered
  // (or, in the browser, shown with showChain) with the backend's chain, so
  // the list as it is now is that chain.
  let applied = getChain();

  return {
    getChain,
    show,

    // Make the chain exactly `names`, in that order, each step's parameters
    // from paramsByName[name] (show, with one set of values per op name),
    // after the list's `held` step, if it has one.
    apply(names, paramsByName) {
      const kept = getChain().filter((step) => step.name === held).slice(0, 1);
      show([...kept, ...names.filter((name) => name !== held)
        .map((name) => ({ ...((paramsByName || {})[name] || {}), name }))]);
    },

    // Record `chain` (by default the list as it is now) as the one the
    // backend runs: after Apply, the chain Apply sent; after the backend
    // changed its chain itself, the list once it shows it.
    markApplied(chain) {
      applied = (chain || getChain()).map((step) => ({ ...step }));
      edited();
    },

    // How many edits the list has that the backend does not run (chainChanges).
    pendingChanges() {
      return chainChanges(applied, getChain());
    },

    // Call `listener` after every edit to the list: a box ticked or
    // unticked, a value typed, a row dragged, show() or apply(), and after
    // markApplied (the count changes then too).
    onEdit(listener) {
      editListeners.push(listener);
    },
  };
}
