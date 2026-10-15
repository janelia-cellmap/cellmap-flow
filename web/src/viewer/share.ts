// The viewer's state outside the viewer: Neuroglancer in the iframe writes
// its whole state to its own hash as "#!" + encodeURI(JSON) (plain JSON,
// double quotes, a few characters percent-encoded), and the page keeps a
// copy of that hash in the tab's sessionStorage as the user navigates, so
// the reload connect.js does after Connect (and any reload of the tab) puts
// the viewer back where it was. A page opened with a `#!` hash of its own,
// a link someone built or Neuroglancer's own URL pasted onto the page's,
// loads that state instead (viewer/index.ts stateFromPage).
//
// The page's own address bar is left alone. Marwan Zouinkhi's prototype
// mirrored the state into it, as Neuroglancer does, so the address bar was
// a link to what is on screen; but Chrome's address bar completes
// "localhost:5174" to the last URL it saw there, hash and all, so a new tab
// opened that way came back with the old dataset and layers when it should
// start fresh (the session is the tab's, state/index.ts). A link to share
// is a control to add, not the address bar.
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
// The tab's copy of the viewer's state
// ---------------------------------------------------------------------------

/** The viewer's hash in the tab's sessionStorage. */
export const TAB_STATE_KEY = "cellmap-flow:viewer";

/** Keep `hash` as the tab's viewer state (nothing without storage). */
export function rememberState(hash: string): void {
  if (!hash) return;
  try {
    sessionStorage.setItem(TAB_STATE_KEY, hash);
  } catch {
    // no storage: the reload starts from the session's dataset alone
  }
}

/** The tab's viewer state, a `#!` hash, or null. */
export function rememberedState(): string | null {
  try {
    const hash = sessionStorage.getItem(TAB_STATE_KEY);
    return hash && isStateHash(hash) ? hash : null;
  } catch {
    return null;
  }
}

/** Drop the tab's viewer state (a forgotten dataset). */
export function forgetState(): void {
  try {
    sessionStorage.removeItem(TAB_STATE_KEY);
  } catch {
    // nothing kept
  }
}

// ---------------------------------------------------------------------------
// DOM: following the iframe's hash
// ---------------------------------------------------------------------------

/** How often the iframe's hash is read. Neuroglancer rewrites its hash with
 * history.replaceState, which fires no hashchange event, so listening alone
 * misses most changes. */
export const MIRROR_INTERVAL_MS = 1000;

/** Follow `frame`'s hash as it changes, into the tab's state by default: on
 * each load of the frame, listen for hashchange in it and poll every
 * MIRROR_INTERVAL_MS. The frame is same-origin (the viewer must be, for the
 * service worker to see its chunk requests), so its window is readable; a
 * frame that is not (while it shows about:blank, say) is skipped until it
 * loads. Returns a function that stops the mirroring. */
export function mirrorFrameHash(frame: HTMLIFrameElement, onHash: (hash: string) => void = rememberState): () => void {
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
