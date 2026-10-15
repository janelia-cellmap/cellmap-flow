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
import type { ModelEntry, SessionStore, ViewerController } from "./contracts";
import { DATASET_PARAM, MODEL_PARAM } from "./link";
import { isStateHash } from "./viewer/share";

/** The header's button (templates/index.html, browser mode only). */
export const SHARE_BUTTON_ID = "shareLinkBtn";

/** `page` (the page's URL, whatever query or hash it has) with `datasetUrl`
 * and `models` in its query and the view in `viewerUrl`'s hash, or null
 * when no volume is open. The dataset URL is kept readable, not escaped as
 * a form value would be: only what would end the query or split it is. */
export function shareLink(page: string, datasetUrl: string | null, models: readonly Pick<ModelEntry, "folder">[], viewerUrl: string | null): string | null {
  if (!datasetUrl) return null;
  const base = page.replace(/[?#].*$/, "");
  const params = [`${DATASET_PARAM}=${queryValue(datasetUrl)}`, ...models.map((m) => `${MODEL_PARAM}=${queryValue(m.folder)}`)];
  const hash = viewerUrl?.replace(/^[^#]*/, "") ?? "";
  return `${base}?${params.join("&")}${isStateHash(hash) ? hash : ""}`;
}

/** `value` as a query value: as it is, but for what would end the query (#),
 * split it (&) or be read back as a space (+); a space or a character
 * outside ASCII is escaped as in any URL. */
function queryValue(value: string): string {
  return encodeURI(value).replace(/[&#+]/g, (c) => encodeURIComponent(c));
}

/** The Share button: the link to the clipboard, and a word on the button
 * for a moment to say so. Where the clipboard cannot be written (a page
 * the browser does not trust with it), the link is shown to copy by hand. */
export function installShareUi(opts: { session: SessionStore; viewer: Pick<ViewerController, "viewerUrl">; log: (line: string) => void }): void {
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
    const link = shareLink(location.href, datasetUrl, models, opts.viewer.viewerUrl());
    if (!link) {
      say("Nothing open yet");
      return;
    }
    try {
      await navigator.clipboard.writeText(link);
      say("Link copied");
      opts.log(`a link to ${datasetUrl} with ${models.length} model(s) copied (${link.length} characters)`);
    } catch {
      window.prompt("Copy this link:", link);
    }
  });
}
