// A volume on this machine: an OME-Zarr, zarr or N5 directory the user picks
// with the File System Access API (window.showDirectoryPicker) or drops on
// the connect panel from the file manager, read with no server and no https.
//
// The plumbing was there (Marwan Zouinkhi's prototype picked directories
// this way): the page stores the directory's handle in IndexedDB under a key
// (state/local-store.ts putHandle), the dataset URL is this page's own
// `local/<key>/`, and the service worker (src/sw.ts) answers every request
// under it with the file at that path, for the engine's reader worker and
// for Neuroglancer's iframe alike. This module is the user's side of it:
//
// - openLocalFolder(): the handle's read permission (asked for in the user's
//   click when it is not granted; a dropped folder comes with it), a key for
//   the handle, and the local URL put in the connect panel's field with
//   Connect pressed, so connect.js's own status messages, error path and
//   reload serve unchanged and /api/set-data opens the folder as it opens
//   any URL (the route takes this page's http(s) URL).
// - restoreLocalFolder(): the boot step before the viewer and the engine
//   open the session's dataset again. Chrome drops a stored handle's read
//   permission when the page reloads (unless the user chose "allow on every
//   visit" in its prompt), and asking for it again needs a user gesture, so
//   when the session's dataset is a local URL the connect panel shows a
//   one-click prompt first and boot waits for the click; a handle that is
//   gone (a link shared from another machine names its folder's key, which
//   this browser never stored) clears the dataset and the panel says so.
//
// The DOM is in panel.ts behind the LocalPanel interface, so the flows here
// run on fakes in the tests (test/unit/local.test.ts).
import type { SessionStore } from "../contracts";
import { getHandle, putHandle } from "../state/local-store";
import { connectPanel, type LocalPanel } from "./panel";

/** What this module needs of a FileSystemDirectoryHandle: its name and the
 * File System Access API's permission methods. lib.dom types only the
 * handle's standard half (the OPFS one), without these, so they are declared
 * here; a type, so the tests can hand in a fake. */
export interface PickedDirectory {
  readonly name: string;
  queryPermission(descriptor: { mode: "read" | "readwrite" }): Promise<PermissionState>;
  requestPermission(descriptor: { mode: "read" | "readwrite" }): Promise<PermissionState>;
}

/** Where picked handles are kept: local-store.ts's IndexedDB store, or a fake. */
export interface HandleStore {
  get(key: string): Promise<PickedDirectory | undefined>;
  put(key: string, handle: PickedDirectory): Promise<void>;
}

/** The IndexedDB store. The casts are the gap in lib.dom named above: what
 * it stores and gives back is a real FileSystemDirectoryHandle, which has
 * the permission methods in every browser that has the picker. */
const indexedDbHandles: HandleStore = {
  get: (key) => getHandle(key) as Promise<PickedDirectory | undefined>,
  put: (key, handle) => putHandle(key, handle as unknown as FileSystemDirectoryHandle),
};

interface LocalOptions {
  log: (line: string) => void;
  /** The connect panel's DOM; the real one when omitted. */
  panel?: LocalPanel;
  handles?: HandleStore;
  /** The page's URL, which the local URLs are under; document.baseURI when omitted. */
  base?: string;
}

// ------------------------------------------------------------------ the URL and its key

/** The directory under the page's own that the service worker serves picked
 * folders at (src/sw.ts matches it anywhere in the path). */
export const LOCAL_PREFIX = "local/";

/** The dataset URL for the folder stored under `key`: `<the page's directory>local/<key>/`. */
export function localUrl(key: string, base: string = document.baseURI): string {
  return new URL(`${LOCAL_PREFIX}${key}/`, base).href;
}

/** The key of a local URL of this page (`local/<key>/`, or any path under
 * it, as the service worker reads it), or null for every other URL: another
 * origin's, another path of this one, a bucket's, none. */
export function localKeyOf(url: string | null | undefined, base: string = document.baseURI): string | null {
  if (!url) return null;
  const root = new URL(LOCAL_PREFIX, base).href;
  if (!url.startsWith(root)) return null;
  const key = url.slice(root.length).split(/[/?#]/, 1)[0];
  return key || null;
}

/** How many random characters start a key. */
const RANDOM_LENGTH = 6;
/** The longest a key's readable part gets (a path segment of a URL that
 * also ends up in the viewer's layer source and the page's hash). */
const SLUG_LENGTH = 64;

/** A key for a picked folder: a few random characters, then the folder's
 * name made URL-safe, so the key is unique per pick (the same folder picked
 * twice is stored twice; neither pick's URL goes stale) and the URL still
 * reads as the folder. The name comes last, with its extension kept, so
 * `local/<key>/` ends in `.zarr/` or `.n5/` as the folder does, which is how
 * the viewer's raw layer tells an N5 from a zarr (viewer/layers.ts
 * rawSource). The letters a name loses (spaces, slashes, anything outside
 * [A-Za-z0-9._-]) become hyphens; accents are dropped. */
export function localKeyFor(name: string, random: string = randomChars(RANDOM_LENGTH)): string {
  let slug = name
    .normalize("NFKD")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^[-.]+|-+$/g, "");
  if (slug.length > SLUG_LENGTH) {
    const dot = slug.lastIndexOf(".");
    const extension = dot > 0 ? slug.slice(dot) : "";
    slug = slug.slice(0, SLUG_LENGTH - extension.length).replace(/-+$/, "") + extension;
  }
  return `${random}-${slug || "folder"}`;
}

/** The readable part of a key: the folder's name as localKeyFor kept it
 * (for a message about a folder whose handle is gone, when the name itself
 * is gone with it). */
export function folderNameOf(key: string): string {
  return key.slice(RANDOM_LENGTH + 1) || key;
}

const RANDOM_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789";
function randomChars(count: number): string {
  const bytes = crypto.getRandomValues(new Uint8Array(count));
  return Array.from(bytes, (b) => RANDOM_ALPHABET[b % RANDOM_ALPHABET.length]).join("");
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// ------------------------------------------------------------------ picking a folder

/** Open `handle`'s directory as the dataset: read permission (asked for if
 * it is not granted; the caller is in the user's gesture), the handle
 * stored under a fresh key, and its local URL put in the connect panel and
 * Connect pressed, from where connect.js and /api/set-data take it as they
 * take any URL. What goes wrong is said on the panel's status line, as
 * connect.js says its own failures. */
export async function openLocalFolder(handle: PickedDirectory, opts: LocalOptions): Promise<void> {
  const panel = opts.panel ?? connectPanel();
  const handles = opts.handles ?? indexedDbHandles;
  try {
    let permission = await handle.queryPermission({ mode: "read" });
    if (permission !== "granted") permission = await handle.requestPermission({ mode: "read" });
    if (permission !== "granted") {
      panel.status(`Reading ${handle.name} was not allowed, so it cannot be opened.`, "error");
      opts.log(`reading ${handle.name} on this machine was not allowed (${permission})`);
      return;
    }
    const key = localKeyFor(handle.name);
    await handles.put(key, handle);
    const url = localUrl(key, opts.base);
    opts.log(`${handle.name} on this machine is ${url}`);
    panel.connect(url);
  } catch (e) {
    // IndexedDB refusing the handle (a private window, say), or the
    // permission request thrown out for want of a user gesture.
    panel.status(`Could not open ${handle.name}: ${message(e)}`, "error");
    opts.log(`could not open ${handle.name} on this machine: ${message(e)}`);
  }
}

/** The picker as Chromium has it (not in lib.dom, so `window` is read as this). */
interface PickerWindow {
  showDirectoryPicker?(options?: { id?: string; mode?: "read" | "readwrite" }): Promise<PickedDirectory>;
}

/** Whether this browser can pick a folder: Chromium has the File System
 * Access API's picker; Firefox and Safari have only the API's OPFS half. */
export function supportsLocalFolders(win: PickerWindow = window as unknown as PickerWindow): boolean {
  return typeof win.showDirectoryPicker === "function";
}

/** What the connect panel says in place of the button where there is no picker. */
export const UNSUPPORTED_NOTE = "Opening a folder on this machine needs Chrome or Edge (the File System Access API).";

/** The connect panel's "Browse local folder…" button and its drop target,
 * wired; or, without the picker, both hidden behind a note. Call it once
 * the dashboard's own script listens on Connect, which the button presses.
 * `pick` is the picker (window.showDirectoryPicker, read-only); a test hands
 * in its own, and says `supported` rather than having a window. */
export function installLocalFolderUi(opts: LocalOptions & { pick?: () => Promise<PickedDirectory>; supported?: boolean }): void {
  const panel = opts.panel ?? connectPanel();
  if (!(opts.supported ?? supportsLocalFolders())) {
    panel.hideLocalControls(UNSUPPORTED_NOTE);
    return;
  }
  const pick = opts.pick ?? (() => (window as unknown as Required<PickerWindow>).showDirectoryPicker({ mode: "read" }));
  panel.onBrowse(async () => {
    let handle: PickedDirectory;
    try {
      handle = await pick();
    } catch (e) {
      // AbortError: the dialog was cancelled, which needs no message.
      if ((e as { name?: string })?.name !== "AbortError") panel.status(`Could not pick a folder: ${message(e)}`, "error");
      return;
    }
    await openLocalFolder(handle, opts);
  });
  panel.onDropDirectory((handle) => void openLocalFolder(handle, opts));
}

// ------------------------------------------------------------------ after a reload

/** The boot step for a session whose dataset is a folder on this machine,
 * before the viewer and the engine open it again: the folder's handle must
 * still be stored and readable. Does nothing for any other dataset.
 *
 * Permission "granted" (the user chose "allow on every visit", or the handle
 * was just stored): continue at once. "prompt", or "denied" (Chrome did not
 * grant it last time): the connect panel shows a one-click prompt and this
 * waits for the click, asks Chrome, and asks the user again if Chrome
 * refused, until it is granted or the user chooses to open something else,
 * which forgets the folder. Handle gone: the folder is forgotten and the
 * panel says so. Forgetting clears the dataset from the session and the
 * viewer state from the page's URL (a shared link's, or the one the reload
 * after Connect carries), or viewer.restore() would put it back. */
export async function restoreLocalFolder(opts: LocalOptions & { session: SessionStore }): Promise<void> {
  const { session, log } = opts;
  const key = localKeyOf(session.state.datasetUrl, opts.base);
  if (!key) return;
  const panel = opts.panel ?? connectPanel();
  const handles = opts.handles ?? indexedDbHandles;

  async function forget(): Promise<void> {
    panel.forgetPageState();
    await session.update({ datasetUrl: null });
    panel.show();
  }

  let handle: PickedDirectory | undefined;
  try {
    handle = await handles.get(key);
  } catch (e) {
    log(`could not look up the local folder ${folderNameOf(key)}: ${message(e)}`);
  }
  if (!handle) {
    const name = folderNameOf(key);
    await forget();
    panel.status(
      `${name} is a folder on a machine this browser has no access to (a link to a local folder works only in the browser that picked it). Pick the folder again, or open another volume.`,
      "error",
    );
    log(`the session's dataset was the local folder ${name}, whose handle is gone; forgetting it`);
    return;
  }

  let permission = await handle.queryPermission({ mode: "read" });
  let refused = permission === "denied";
  while (permission !== "granted") {
    panel.show();
    const allow = await panel.askAgain(handle.name, refused);
    if (!allow) {
      await forget();
      panel.status(`${handle.name} is no longer open. Pick a folder or enter a URL to open a volume.`, "info");
      log(`${handle.name} on this machine was not opened again, at the user's choice`);
      return;
    }
    try {
      permission = await handle.requestPermission({ mode: "read" });
    } catch (e) {
      log(`asking to read ${handle.name} failed: ${message(e)}`);
      permission = "denied";
    }
    refused = permission !== "granted";
  }
  log(`${handle.name} on this machine is readable again`);
}
