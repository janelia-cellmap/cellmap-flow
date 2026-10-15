// The shareable URL: the page's own URL carries the viewer's whole state in
// its `#!` hash, as Neuroglancer's does, so the address bar is a link to
// what is on screen (from Marwan Zouinkhi's prototype, web/legacy/index.html).
//
// Neuroglancer in the iframe writes its state to its own hash as
// "#!" + encodeURI(JSON) (plain JSON, double quotes, a few characters
// percent-encoded). The page mirrors that hash to its own URL as the user
// navigates (history.replaceState: no history entries, no reload), and on
// boot a page opened with a `#!` hash loads that state into the iframe.
//
// The pure parts (encoding, decoding, the URL arithmetic) are tested; the
// DOM parts (mirroring the iframe's hash) are kept small enough to read.
import type { NgState } from "./state";

/** What a hash that holds a state starts with. */
export const STATE_HASH_PREFIX = "#!";

/** Whether `hash` holds a whole state this module can decode: `#!{...}`,
 * encoded or not. Neuroglancer's two other forms, `#!+` (a diff against the
 * state before it) and `#!https://.../state.json` (a state to fetch), are
 * not states themselves and are left to Neuroglancer. */
export function isStateHash(hash: string): boolean {
  if (!hash.startsWith(STATE_HASH_PREFIX) || hash.length === STATE_HASH_PREFIX.length) return false;
  const body = hash.slice(STATE_HASH_PREFIX.length);
  return !body.startsWith("+") && !/^[a-z][a-z\d+\-.]*:\/\//i.test(body);
}

/** `state` as a hash for the viewer's URL. encodeURIComponent, which
 * Neuroglancer reads back with decodeURIComponent exactly as it reads its
 * own encoding. */
export function encodeState(state: object): string {
  return STATE_HASH_PREFIX + encodeURIComponent(JSON.stringify(state));
}

/** Neuroglancer's url-safe JSON, as its oldest links wrote it, back to JSON:
 * outside string literals `_` (and `&`) stand for commas, and strings may be
 * single-quoted (a `'` inside escaped as `\'`). Current Neuroglancer writes
 * plain JSON, which passes through unchanged. */
export function urlSafeToJson(text: string): string {
  let out = "";
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === '"' || ch === "'") {
      // A string literal: copied to its closing quote, single quotes made double.
      const quote = ch;
      let literal = '"';
      i += 1;
      while (i < text.length && text[i] !== quote) {
        if (text[i] === "\\" && i + 1 < text.length) {
          const next = text[i + 1];
          literal += next === "'" ? "'" : `\\${next}`;
          i += 2;
        } else {
          literal += quote === "'" && text[i] === '"' ? '\\"' : text[i];
          i += 1;
        }
      }
      out += `${literal}"`;
      i += 1; // the closing quote
    } else {
      out += ch === "_" || ch === "&" ? "," : ch;
      i += 1;
    }
  }
  return out;
}

/** The state a `#!` hash holds (the page's or the iframe's), or null when
 * it holds none or cannot be read. The hash may be percent-encoded (by
 * encodeState, by Neuroglancer, or not at all when typed) and in plain or
 * url-safe JSON. */
export function decodeState(hash: string): NgState | null {
  if (!isStateHash(hash)) return null;
  let text = hash.slice(STATE_HASH_PREFIX.length);
  try {
    text = decodeURIComponent(text);
  } catch {
    // Not percent-encoded (a stray %): read as it is.
  }
  for (const candidate of [text, urlSafeToJson(text)]) {
    try {
      const parsed: unknown = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as NgState;
    } catch {
      // Try the next form.
    }
  }
  return null;
}

/** `href` with `hash` as its fragment (`hash` starts with "#", or is empty
 * to drop the fragment). */
export function withHash(href: string, hash: string): string {
  const base = href.replace(/#.*$/, "");
  return hash ? base + hash : base;
}

// ---------------------------------------------------------------------------
// DOM: keeping the page's URL equal to the viewer's state
// ---------------------------------------------------------------------------

/** How often the iframe's hash is compared with the page's. Neuroglancer
 * rewrites its hash with history.replaceState, which fires no hashchange
 * event, so listening alone misses most changes. */
export const MIRROR_INTERVAL_MS = 1000;

/** Put `hash` on the page's URL, without a history entry, when it differs
 * from the one there. */
export function mirrorToPage(hash: string): void {
  if (!hash || hash === location.hash) return;
  history.replaceState(null, "", withHash(location.href, hash));
}

/** Mirror `frame`'s hash to the page's URL as it changes: on each load of the
 * frame, listen for hashchange in it and poll every MIRROR_INTERVAL_MS. The
 * frame is same-origin (the viewer must be, for the service worker to see
 * its chunk requests), so its window is readable; a frame that is not
 * (while it shows about:blank, say) is skipped until it loads. Returns a
 * function that stops the mirroring. */
export function mirrorFrameHash(frame: HTMLIFrameElement, onHash: (hash: string) => void = mirrorToPage): () => void {
  let timer: ReturnType<typeof setInterval> | null = null;
  let detach: (() => void) | null = null;

  const mirror = () => {
    try {
      const hash = frame.contentWindow?.location.hash ?? "";
      if (isStateHash(hash)) onHash(hash);
    } catch {
      // Cross-origin or gone: nothing to mirror.
    }
  };
  const attach = () => {
    detach?.();
    if (timer !== null) clearInterval(timer);
    const win = frame.contentWindow;
    if (!win) return;
    win.addEventListener("hashchange", mirror);
    detach = () => win.removeEventListener("hashchange", mirror);
    timer = setInterval(mirror, MIRROR_INTERVAL_MS);
  };

  frame.addEventListener("load", attach);
  return () => {
    frame.removeEventListener("load", attach);
    detach?.();
    if (timer !== null) clearInterval(timer);
  };
}
