// The Input and Postprocess lists (templates/_input_tab.html and
// _output_tab.html), which the server renders: one row per op, the
// configured chain first in the order it runs (an op used twice has two
// rows), then every other op unticked. Each row has a drag handle, a
// checkbox whose value is the op's name, and its parameter fields.
//
// mountOpChain attaches their behaviour: dragging a row by its handle moves
// it, ticking it shows its parameters, and getChain() reads the ticked rows
// top to bottom, which is the order the steps run in.
import { makeSortable } from "../lib/sortable.js";

const KINDS = {
  input: { item: ".normalizer-item", checkbox: ".inputNormCheckbox" },
  postprocess: { item: ".postprocessor-item", checkbox: ".postProcessCheckbox" },
};

export function mountOpChain(listEl, { kind }) {
  const { item: itemSelector, checkbox: checkboxSelector } = KINDS[kind];
  const rows = () => Array.from(listEl.querySelectorAll(itemSelector));

  function showState(checkbox) {
    const row = checkbox.closest(itemSelector);
    if (!row) return;
    row.classList.toggle("item-checked", checkbox.checked);
    const params = row.querySelector(".step-params");
    if (params) params.style.display = checkbox.checked ? "block" : "none";
  }

  listEl.querySelectorAll(checkboxSelector).forEach((checkbox) => {
    showState(checkbox);
    checkbox.addEventListener("change", () => showState(checkbox));
  });
  makeSortable(listEl, { itemSelector, handleSelector: ".drag-handle" });

  return {
    // [{name, ...params}] for the ticked rows, in order. Each row's own
    // fields: the same op can have two rows.
    getChain() {
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
    },

    // Make the chain exactly `names`, in that order: tick those rows, fill
    // in their parameters from paramsByName[name], move them to the top,
    // and untick every other row. The k-th use of a name takes the k-th row
    // for it; a name with no row is skipped.
    apply(names, paramsByName) {
      const all = rows();
      const picked = [];
      names.forEach((name) => {
        const row = all.find((r) => {
          const cb = r.querySelector(checkboxSelector);
          return cb && cb.value === name && !picked.includes(r);
        });
        if (row) picked.push(row);
      });
      all.forEach((row) => {
        const cb = row.querySelector(checkboxSelector);
        const want = picked.includes(row);
        if (cb && cb.checked !== want) {
          cb.checked = want;
          cb.dispatchEvent(new Event("change", { bubbles: true }));
        }
      });
      picked.forEach((row) => {
        const name = row.querySelector(checkboxSelector).value;
        const values = (paramsByName || {})[name] || {};
        Object.keys(values).forEach((key) => {
          if (key === "name") return;
          const input = row.querySelector(`input[data-param="${key}"]`);
          if (input) input.value = values[key];
        });
      });
      picked.slice().reverse().forEach((row) => listEl.insertBefore(row, listEl.firstChild));
    },
  };
}
