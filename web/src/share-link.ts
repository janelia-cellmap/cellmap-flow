// The link that opens elsewhere what this tab has open, for the header's
// Share button: the volume and the models in the query (src/link.ts reads
// them back), and the view in the `#!` hash, the viewer's own state
// (position, zoom, layout, each layer as shown with its shader), which
// viewer/index.ts restores; its model layers' sources carry the chain they
// were drawn through, which link.ts reads back too. Built when asked for,
// not kept in the address bar: a tab opened from the bare URL starts fresh,
// and Chrome's omnibox, which completes a typed address to one visited
// before, has no view to complete it to (which is how the old view came back
// in a new tab while the page's URL mirrored the viewer's).
//
// A model on Hugging Face goes in the link by its ref: a folder of the
// built-in catalog as it is, any other repo's as `<owner>/<repo>/<folder>`
// (its id), which the page that opens it looks up there. Two things of a
// finetuning session are in this browser alone. A model finetuned here
// that was published to Hugging Face goes in by where it was published to
// (FinetunedRecord.published), like any model there; one that was not goes
// in as what rebuilds it (link.ts finetunedRef: its base and its trained
// last layer's numbers), so it runs where the link opens; one whose whole
// network was trained (or that was finetuned from one), or that was opened
// from a file, cannot be rebuilt from a link, and stays here until it is
// published. The paint does not: an annotation volume is its
// chunks in the paint store, megabytes that no link carries, so the link
// keeps its layer, which opens only in this browser, and the page says so
// beside the button, with the Finetune tab's download that gives the paint
// as a file. The page that opens the link leaves out what it cannot show
// and says so in the same notice (showLinkNotice, called by main.ts).
import { message } from "./backend/router";
import { type Catalog, type FinetunedRecord, type ModelEntry, type SessionStore, type ViewerController } from "./contracts";
import { headOf } from "./finetune/onnx-head";
import { DATASET_PARAM, finetunedRef, MODEL_PARAM, paintVolumesIn } from "./link";
import { isFinetuned, lineage, publishedRef } from "./models/finetuned";
import { HF_REPO } from "./models/hf";
import { decodeState, isStateHash } from "./viewer/share";

/** The header's button (templates/index.html, browser mode only). */
export const SHARE_BUTTON_ID = "shareLinkBtn";

/** The notice under the header (showLinkNotice). */
export const LINK_NOTICE_ID = "linkNotice";

/** `page` (the page's URL, whatever query or hash it has) with `datasetUrl`
 * and the models' `refs` (linkRefs) in its query, in their order, and the
 * view in `viewerUrl`'s hash, or null when no volume is open. The dataset
 * URL is kept readable, not escaped as a form value would be: only what
 * would end the query or split it is. */
export function shareLink(page: string, datasetUrl: string | null, refs: readonly string[], viewerUrl: string | null): string | null {
  if (!datasetUrl) return null;
  const base = page.replace(/[?#].*$/, "");
  const params = [`${DATASET_PARAM}=${queryValue(datasetUrl)}`, ...refs.map((ref) => `${MODEL_PARAM}=${queryValue(ref)}`)];
  const hash = viewerUrl?.replace(/^[^#]*/, "") ?? "";
  return `${base}?${params.join("&")}${isStateHash(hash) ? hash : ""}`;
}

/** `value` as a query value: as it is, but for what would end the query (#),
 * split it (&) or be read back as a space (+); a space or a character
 * outside ASCII is escaped as in any URL. */
function queryValue(value: string): string {
  return encodeURI(value).replace(/[&#+]/g, (c) => encodeURIComponent(c));
}

/** The models' refs for a link, and what became of the finetuned ones. */
export interface LinkRefs {
  /** One per model, in the session's order. */
  refs: string[];
  /** The finetuned models whose last layer is in the link, each with the
   * model it is rebuilt from where the link opens. */
  rebuilt: { name: string; base: string }[];
  /** The finetuned models that could not be put in the link so, with why:
   * named as they are, they open only in this browser. */
  unlinked: { name: string; why: string }[];
}

/** A Hugging Face model's ref in a link: a folder of the built-in catalog by
 * its folder, as a link names one by hand, and any other repo's by its id,
 * `<owner>/<repo>/<folder>`. */
function hubLinkRef(entry: ModelEntry): string {
  return entry.repo === HF_REPO ? entry.folder : entry.id;
}

/** The refs of `models` for a link, in their order: a Hugging Face model's
 * hubLinkRef; a finetuned model published from here, where it was
 * published to; any other finetuned model's finetunedRef, its last layer
 * read from its bytes and its base traced back through the records to the
 * model a link can name at the root (a model finetuned from a finetuned one
 * is still the root's file with one other layer when only heads were
 * trained): the Hugging Face model it started from, or the first model on
 * the way there that was published. A finetuned model that cannot be put
 * so (its bytes gone, or the record of a model in its lineage; a whole
 * network trained, or a file opened, on the way) is named by its folder,
 * which opens it only in this browser, and said so. */
export async function linkRefs(models: readonly ModelEntry[], catalog: Pick<Catalog, "finetuned" | "modelBytes" | "resolve">): Promise<LinkRefs> {
  const linked: LinkRefs = { refs: [], rebuilt: [], unlinked: [] };
  let records: FinetunedRecord[] | null = null;
  for (const model of models) {
    if (!isFinetuned(model)) {
      linked.refs.push(hubLinkRef(model));
      continue;
    }
    try {
      records ??= await catalog.finetuned();
      const record = records.find((r) => r.id === model.id);
      if (!record) throw new Error("its record is not in this browser's catalog");
      if (record.published) {
        linked.refs.push(publishedRef(record.published));
        continue;
      }
      const { ancestry, base, published, kept } = lineage(record, records);
      if (!kept) throw new Error(`${base.name}, the model it was finetuned from, is no longer kept in this browser`);
      if (ancestry.some((r) => r.from_file)) {
        throw new Error("it was opened from a file in this browser, which a link does not carry; publish it to Hugging Face (the Finetune tab) to put it in a link, or send the file");
      }
      if (ancestry.some((r) => r.params.finetune_mode === "full")) {
        throw new Error("its whole network was trained in this browser, millions of weights where a link carries a last layer's few dozen; publish it to Hugging Face (the Finetune tab) to put it in a link, or send the .onnx file the Finetune tab's Download finetuned model saves");
      }
      const head = headOf(await catalog.modelBytes(model));
      linked.refs.push(finetunedRef({ name: model.name, base: published ?? (await hubBaseRef(base, catalog)), file: base.file, weight: head.weight, bias: head.bias }));
      linked.rebuilt.push({ name: model.name, base: base.name });
    } catch (e) {
      linked.refs.push(model.folder);
      linked.unlinked.push({ name: model.name, why: message(e) });
    }
  }
  return linked;
}

/** The ref in a link of `base`, the Hugging Face model a lineage ends at,
 * looked up in the catalog (which says when it is gone). */
async function hubBaseRef(base: FinetunedRecord["base"], catalog: Pick<Catalog, "resolve">): Promise<string> {
  const entry = await catalog.resolve(base.id);
  if (!entry) throw new Error(`${base.name}, the model it was finetuned from, is not in the model catalog`);
  return hubLinkRef(entry);
}

/** What the sender is told about a link beside its having been copied: a
 * sentence for each paint volume its view shows, which stays here, each
 * finetuned model rebuilt where it opens, and each one that could not be
 * put in it; none for a link that opens elsewhere as it is here. */
export function shareNotes(paint: readonly string[], refs: Pick<LinkRefs, "rebuilt" | "unlinked">): string[] {
  const notes = paint.map((volume) =>
    `The paint in ${volume} stays in this browser: someone else opening the link will not see it. The Finetune tab's Download paint (zarr) saves it as a .zarr.zip to send with the link.`);
  for (const { name, base } of refs.rebuilt) notes.push(`${name}'s last layer is in the link: it is rebuilt from ${base} where the link opens.`);
  for (const { name, why } of refs.unlinked) notes.push(`${name} could not be put in the link as a finetuned model (${why}): it opens only in this browser.`);
  return notes;
}

/** `notes` in a notice under the header, by the Share button, in the
 * dashboard's alert colours (browser.css), headed by `title`, until its
 * close button is pressed or the next notice takes its place; no notes
 * clears it. Where the page tells the sender what a link leaves behind and
 * the recipient what it could not open: the log goes to the console, and
 * the button's own word lasts two seconds, too short to read this by. */
export function showLinkNotice(title: string, notes: readonly string[], tone: "info" | "warning" = "info"): void {
  document.getElementById(LINK_NOTICE_ID)?.remove();
  if (!notes.length) return;
  const notice = document.createElement("div");
  notice.id = LINK_NOTICE_ID;
  notice.className = `link-notice ${tone}`;
  notice.setAttribute("role", tone === "warning" ? "alert" : "status");
  // Just below the header, wherever it ends: it wraps onto two rows on a phone.
  const header = document.querySelector(".app-header");
  notice.style.top = `${Math.round(header ? header.getBoundingClientRect().bottom + 8 : 16)}px`;
  const close = document.createElement("button");
  close.type = "button";
  close.className = "link-notice-close";
  close.setAttribute("aria-label", "Close");
  close.textContent = "×";
  close.addEventListener("click", () => notice.remove());
  const heading = document.createElement("strong");
  heading.textContent = title;
  notice.append(close, heading, ...notes.map((note) => {
    const line = document.createElement("p");
    line.textContent = note;
    return line;
  }));
  document.body.appendChild(notice);
}

/** The Share button: the link to the clipboard, a word on the button for a
 * moment to say so, and the notice when the link leaves something behind.
 * Where the clipboard cannot be written (a page the browser does not trust
 * with it), the link is shown to copy by hand. */
export function installShareUi(opts: {
  session: SessionStore;
  viewer: Pick<ViewerController, "viewerUrl">;
  catalog: Pick<Catalog, "finetuned" | "modelBytes" | "resolve">;
  log: (line: string) => void;
}): void {
  const button = document.getElementById(SHARE_BUTTON_ID) as HTMLButtonElement | null;
  if (!button) return;
  const label = button.textContent;
  let reset: ReturnType<typeof setTimeout> | undefined;
  const say = (text: string) => {
    button.textContent = text;
    clearTimeout(reset);
    reset = setTimeout(() => { button.textContent = label; }, 2000);
  };
  button.addEventListener("click", async () => {
    const { datasetUrl, models } = opts.session.state;
    const viewerUrl = opts.viewer.viewerUrl();
    if (!datasetUrl) {
      say("Nothing open yet");
      return;
    }
    // A finetuned model's head is read from its bytes in IndexedDB, a
    // moment's wait; the rest is at hand.
    const refs = await linkRefs(models, opts.catalog);
    const link = shareLink(location.href, datasetUrl, refs.refs, viewerUrl)!;
    const notes = shareNotes(paintVolumesIn(decodeState(viewerUrl?.replace(/^[^#]*/, "") ?? "")), refs);
    let copied = true;
    try {
      await navigator.clipboard.writeText(link);
      say("Link copied");
      opts.log(`a link to ${datasetUrl} with ${models.length} model(s) copied (${link.length} characters)`);
    } catch {
      copied = false;
      window.prompt("Copy this link:", link);
    }
    for (const note of notes) opts.log(note);
    showLinkNotice(copied ? "Link copied." : "Link shown to copy.", notes);
  });
}
