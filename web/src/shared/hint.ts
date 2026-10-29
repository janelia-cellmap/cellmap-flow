// Help text that shows one line and keeps the rest a click away, for the
// text the page's own scripts write: the markup the templates write and the
// dashboard's lib/dom.js hint() builds (a <details class="more"> after the
// line, its <summary> a small "i", the rest in a <span class="more-body">;
// css/dark.css details.more styles it), so it opens by click or tap with no
// script.

/** A hint's two parts: the line that shows, and the rest under the "i". */
export interface Hint {
  short: string;
  more?: string;
}

/** `short`, then `more` under an "i" beside it; no `more`, the line alone. */
export function hint(short: string, more?: string): (string | HTMLElement)[] {
  if (!more) return [short];
  const details = document.createElement("details");
  details.className = "more";
  const summary = document.createElement("summary");
  summary.setAttribute("aria-label", "More");
  summary.textContent = "i";
  const body = document.createElement("span");
  body.className = "more-body";
  body.textContent = more;
  details.append(summary, body);
  return [short, " ", details];
}

/** `el`'s content made hint()'s. */
export function setHint(el: Element, short: string, more?: string): void {
  el.replaceChildren(...hint(short, more));
}

/** A hint as one line of plain text, for the log. */
export function hintText({ short, more }: Hint): string {
  return more ? `${short} ${more}` : short;
}
