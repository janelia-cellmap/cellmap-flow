// Kept in this browser: the Models tab's browser-only block
// (templates/_models_tab.html), the finetuned models and paint volumes this
// browser keeps from earlier tabs and visits, which this tab does not list
// (index.ts). One line while shut, "Kept in this browser from earlier: 2
// finetuned models, 1 paint volume", and Show; shown, a row for each with
// what tells it apart, Use in this tab (it is listed with the tab's own from
// then on: the Hugging Face list and the ranking take it in at once, the
// Finetune tab's sessions when they are listed next) and Delete, asked again
// on its row (the page shows no dialogs), and Delete all under them. The
// block is hidden while there is nothing from earlier.
//
// It is wired here, in the page, as the Save predictions tab is
// (run-region/panel.ts): it reads the stores itself rather than through
// /api/ routes, since nothing of it is the dashboard's, and the server mode
// has no such block.
import type { AppContext, FinetunedRecord } from "../contracts";
import { paintedChunkIndices, readVolume } from "../finetune/volume";
import { setHint } from "../shared/hint";
import { message } from "../shared/message";
import { keepInTab } from "../state";
import { deleteEarlier, deleteKept, type Earlier, earlierKept, type VolumeGroup } from "./index";

/** The block's elements, by the ids templates/_models_tab.html gives them. */
export const IDS = {
  block: "keptEarlier",
  summary: "keptEarlierSummary",
  toggle: "keptEarlierToggle",
  body: "keptEarlierBody",
  list: "keptEarlierList",
  deleteAll: "keptEarlierDeleteAll",
  status: "keptEarlierStatus",
} as const;

/** The Models tab's button (templates/_dashboard.html): the block reads the stores again each time the tab is shown. */
const MODELS_TAB_ID = "model-tab";

/** What the Hugging Face list (huggingface.js) and the ranking (rank.js)
 * list again on, lib/events.js MODEL_LIST_CHANGED_EVENT: a model came into
 * the tab without being served. */
export const MODEL_LIST_CHANGED_EVENT = "cellmap-flow:model-list-changed";

/** One row: its name, the line under it, and what its buttons do. */
interface Row {
  name: string;
  detail: string;
  use: () => Promise<void>;
  remove: () => Promise<void>;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const el = Object.assign(document.createElement(tag), props);
  el.append(...children);
  return el;
}

function button(label: string, className: string, onClick: () => void): HTMLButtonElement {
  const el = element("button", { type: "button", className: `btn btn-sm ${className}`, textContent: label });
  el.addEventListener("click", onClick);
  return el;
}

function plural(n: number, one: string): string {
  return `${n} ${one}${n === 1 ? "" : "s"}`;
}

/** The dataset of a path as a person names it: its .zarr or .n5 folder, else the path. */
function datasetName(path: string): string {
  return path.replace(/\/+$/, "").split("/").find((p) => p.endsWith(".zarr") || p.endsWith(".n5")) ?? path;
}

function when(iso: string): string {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? iso : new Date(t).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Where a finetuned model came from, in a few words. */
function origin(record: FinetunedRecord): string {
  if (record.from_file) return `opened from ${record.from_file.name}`;
  if (record.from_link) return `from a link, on ${record.base.name}`;
  return `from ${record.base.name}`;
}

/** The block, wired: does nothing on a page without it (the Flask
 * dashboard's). Call it once the dashboard's DOM is there. */
export function installKeptUi(ctx: AppContext): void {
  const byId = (id: string) => document.getElementById(id);
  if (!byId(IDS.block)) return;
  const [block, summary, toggle, body, list, deleteAll, status] = [IDS.block, IDS.summary, IDS.toggle, IDS.body, IDS.list, IDS.deleteAll, IDS.status].map((id) => byId(id)!);

  // Bumped by each read of the stores: an older read's answer is not drawn over a newer one's.
  let seq = 0;

  const say = (text: string, error = false) => {
    status.textContent = text;
    status.className = `small mt-1 ${error ? "text-danger" : "text-muted"}`;
  };

  /** `run`, its outcome said in the block and the log, then the stores read again. */
  async function act(run: () => Promise<void>, done: string, failed: string): Promise<void> {
    try {
      await run();
      say(done);
      ctx.log(`Kept in this browser: ${done}`);
    } catch (e) {
      say(`${failed}: ${message(e)}`, true);
      ctx.log(`Kept in this browser: ${failed}: ${message(e)}`);
    }
    await refresh();
  }

  function modelRow(record: FinetunedRecord, dataset: string | null): Row {
    const parts = [origin(record), dataset, when(record.created_at), `${record.entry.variant.file_mb} MB`];
    if (record.published) parts.push(`published to ${record.published.repo}/${record.published.folder}`);
    return {
      name: record.name,
      detail: parts.filter(Boolean).join(" · "),
      use: () => act(async () => {
        await keepInTab(ctx.session, { finetuned: [record.id] });
        document.dispatchEvent(new CustomEvent(MODEL_LIST_CHANGED_EVENT));
      }, `${record.name} is listed in this tab now, under Hugging Face`, `${record.name} could not be brought in`),
      remove: () => act(() => deleteKept(ctx, { finetuned: [record.id] }), `${record.name} deleted from this browser`, `${record.name} could not be deleted`),
    };
  }

  function volumeRow({ volume, ids }: VolumeGroup, chunks: number): Row {
    const parts = [volume.model_name ? `for ${volume.model_name}` : "painted before any model", datasetName(volume.dataset_path), when(volume.created_at), `${plural(chunks, "chunk")} painted`];
    return {
      name: volume.id,
      detail: parts.join(" · "),
      use: () => act(() => keepInTab(ctx.session, { volumes: ids }), `${volume.id} is listed in this tab now: Resume Existing Volume on the Finetune tab`, `${volume.id} could not be brought in`),
      remove: () => act(() => deleteKept(ctx, { volumes: ids }), `${volume.id} deleted from this browser`, `${volume.id} could not be deleted`),
    };
  }

  /** `actions` replaced, until answered, by `question` and a yes and a no:
   * a delete is asked again in place. */
  function askFirst(actions: HTMLElement, question: string, yes: string, onYes: () => Promise<void>): void {
    const before = [...actions.childNodes];
    const back = () => actions.replaceChildren(...before);
    actions.replaceChildren(
      element("span", { className: "small" }, question),
      button(yes, "btn-outline-danger", () => {
        actions.replaceChildren(element("span", { className: "small text-muted", textContent: "Deleting…" }));
        void onYes();
      }),
      button("Keep", "btn-outline-secondary", back),
    );
  }

  function itemOf(row: Row): HTMLLIElement {
    const actions = element("div", { className: "kept-item-actions" });
    actions.append(
      button("Use in this tab", "btn-outline-secondary", () => void row.use()),
      button("Delete", "btn-outline-danger", () => askFirst(actions, `Delete ${row.name} for good, in every tab?`, "Delete", row.remove)),
    );
    return element("li", { className: "kept-item" },
      element("div", { className: "loaded-model-name", textContent: row.name }),
      element("div", { className: "loaded-model-note", textContent: row.detail }),
      actions);
  }

  async function rowsOf(earlier: Earlier): Promise<Row[]> {
    const rows: Row[] = [];
    for (const record of earlier.finetuned) {
      const volume = record.volume ? await readVolume(ctx.paint, record.volume) : null;
      rows.push(modelRow(record, volume ? datasetName(volume.dataset_path) : null));
    }
    for (const group of earlier.volumes) rows.push(volumeRow(group, (await paintedChunkIndices(ctx.paint, group.volume.id)).length));
    return rows;
  }

  function render(earlier: Earlier, rows: Row[]): void {
    const counts = [
      ...(earlier.finetuned.length ? [plural(earlier.finetuned.length, "finetuned model")] : []),
      ...(earlier.volumes.length ? [plural(earlier.volumes.length, "paint volume")] : []),
    ];
    block.hidden = rows.length === 0;
    if (!rows.length) return;
    setHint(summary, `Kept in this browser from earlier: ${counts.join(", ")}`,
      "This tab lists only the finetuned models and paint it made, opened or brought in. These are from earlier tabs and visits: Use in this tab lists one here too; Delete removes it from this browser, in every tab.");
    list.replaceChildren(...rows.map(itemOf));
    const total = rows.length;
    deleteAll.replaceChildren(button("Delete all", "btn-outline-danger", () =>
      askFirst(deleteAll, `Delete all ${plural(total, "earlier item")} for good, in every tab?`, "Delete all",
        () => act(() => deleteEarlier(ctx), `${plural(total, "earlier item")} deleted from this browser`, "Not all could be deleted"))));
  }

  async function refresh(): Promise<void> {
    const mine = ++seq;
    let earlier: Earlier, rows: Row[];
    try {
      earlier = await earlierKept(ctx);
      rows = await rowsOf(earlier);
    } catch (e) {
      ctx.log(`Kept in this browser: could not read what this browser keeps (${message(e)})`);
      return;
    }
    if (mine === seq) render(earlier, rows);
  }

  toggle.addEventListener("click", () => {
    body.hidden = !body.hidden;
    toggle.textContent = body.hidden ? "Show" : "Hide";
    toggle.setAttribute("aria-expanded", String(!body.hidden));
    if (!body.hidden) void refresh();
  });
  // Read again when what is the tab's changes (a model or volume made, used
  // or brought in, here or by a route), and each time the Models tab is shown.
  let tabKept = JSON.stringify(ctx.session.state.tabKept);
  ctx.session.onChange((state) => {
    const now = JSON.stringify(state.tabKept);
    if (now === tabKept) return;
    tabKept = now;
    void refresh();
  });
  byId(MODELS_TAB_ID)?.addEventListener("shown.bs.tab", () => void refresh());
  void refresh();
}
