// The connect panel's DOM for a folder on this machine (src/local-folder/index.ts
// has the flows): the panel and its parts by the ids the dashboard's
// template gives them (templates/index.html; the Browse button and the drop
// hint are in its browser-only block), the one-click permission prompt put
// on the panel after a reload, and the panel as a drop target for a folder
// from the file manager. Behind the LocalPanel interface so the flows run on
// a fake in the tests; nothing here is tested without a browser.
//
// The dashboard's own connect.js owns the Connect button and the status line
// and is not changed: this fills the dataset field and clicks Connect, and
// writes the status line the way connect.js does (its colours too).
import { message } from "../shared/message";
import { forgetState, withHash } from "../viewer/share";
import type { PickedDirectory } from "./index";

/** The panel's ids in templates/index.html: the dashboard's own (connect.js
 * looks them up) and the browser-only block's. */
const PANEL_ID = "noNeuroglancerPanel";
const INPUT_ID = "datasetPathInput";
const CONNECT_ID = "setDataBtn";
const STATUS_ID = "setDataStatus";
const BROWSE_ID = "browseLocalBtn";
const HINT_ID = "localFolderHint";
/** On the panel while a folder from the file manager is over it (browser.css). */
export const DROP_CLASS = "local-drop-target";

/** The colours connect.js gives the status line, and the panel's own muted one. */
const STATUS_COLOR = { info: "var(--text-muted)", error: "#f87171" };

export interface LocalPanel {
  /** Show the connect panel: index.html's inline script hides it before the
   * first paint when the session has a dataset, for the viewer to take its
   * place, and a prompt or a message on it must be seen. */
  show(): void;
  /** The panel's status line (#setDataStatus, connect.js's). */
  status(message: string, tone: "info" | "error"): void;
  /** The one-click prompt for a folder's read permission: resolves true when
   * the user clicks Allow (the caller then asks Chrome, in that click), false
   * when they choose to open something else. `refused` words it for a
   * permission Chrome did not grant the last time it was asked. */
  askAgain(folderName: string, refused: boolean): Promise<boolean>;
  /** Drop the viewer state the tab keeps and any the page's URL carries (a
   * link's), so the viewer does not restore the forgotten folder from it. */
  forgetPageState(): void;
  /** Put `url` in the dataset field and press Connect; connect.js takes it from there. */
  connect(url: string): void;
  /** Hide the Browse button, with `note` in the hint's place (no picker here). */
  hideLocalControls(note: string): void;
  /** The Browse button's click. */
  onBrowse(handler: () => void): void;
  /** A folder dropped on the panel, as its handle. Anything else dropped (a
   * file, several things, a browser without handles for drops) is answered
   * on the status line. */
  onDropDirectory(handler: (handle: PickedDirectory) => void): void;
}

function byId<T extends HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}

function button(className: string, label: string): HTMLButtonElement {
  const el = document.createElement("button");
  el.type = "button";
  el.className = className;
  el.textContent = label;
  return el;
}

/** Whether a drag carries files (a folder is one to the drag data). */
export function carriesFiles(transfer: DataTransfer | null): transfer is DataTransfer {
  return !!transfer && Array.from(transfer.types).includes("Files");
}

/** The drop's handle method as Chromium has it (not in lib.dom). */
export type HandleItem = DataTransferItem & { getAsFileSystemHandle?: () => Promise<FileSystemHandle | null> };

export function connectPanel(): LocalPanel {
  const status: LocalPanel["status"] = (text, tone) => {
    const el = byId(STATUS_ID);
    if (!el) return;
    el.style.color = STATUS_COLOR[tone];
    el.textContent = text;
  };

  return {
    show() {
      delete document.documentElement.dataset.viewer;
    },

    status,

    askAgain(folderName, refused) {
      return new Promise<boolean>((resolve) => {
        const panel = byId(PANEL_ID);
        const box = document.createElement("div");
        box.className = "local-permission";
        const text = document.createElement("div");
        text.className = "local-permission-text";
        text.textContent = refused
          ? `Chrome did not grant access to ${folderName}. Try again (if it does not ask, allow file access for this site in the site settings behind the icon left of the address bar), or open another volume.`
          : `The open volume is the folder ${folderName} on this machine. The browser needs your permission to read it again.`;
        const actions = document.createElement("div");
        actions.className = "local-permission-actions";
        const allow = button("btn btn-primary", `Allow access to ${folderName}`);
        const other = button("btn btn-outline-secondary", "Open another volume");
        actions.append(allow, other);
        box.append(text, actions);
        // In place of the panel's description, URL field and Browse button
        // while the user chooses: the description is for a page with nothing
        // open, and the dashboard's script, which listens on Connect, loads
        // only after boot has this answer. Hidden through style: the
        // `hidden` attribute loses to the dashboard's display rules.
        const replaced = panel ? Array.from(panel.querySelectorAll<HTMLElement>(".no-data-desc, .no-data-input-group, #localFolderGroup")) : [];
        const displays = replaced.map((el) => el.style.display);
        for (const el of replaced) el.style.display = "none";
        if (replaced.length) replaced[0].before(box);
        else (panel ?? document.body).prepend(box);
        const choose = (allowed: boolean) => {
          box.remove();
          replaced.forEach((el, i) => (el.style.display = displays[i]));
          resolve(allowed);
        };
        allow.addEventListener("click", () => choose(true));
        other.addEventListener("click", () => choose(false));
        allow.focus();
      });
    },

    forgetPageState() {
      forgetState();
      if (location.hash) history.replaceState(null, "", withHash(location.href, ""));
    },

    connect(url) {
      const input = byId<HTMLInputElement>(INPUT_ID);
      const connect = byId<HTMLButtonElement>(CONNECT_ID);
      if (!input || !connect) return;
      input.value = url;
      connect.click();
    },

    hideLocalControls(note) {
      // Through style, not the `hidden` attribute, which Bootstrap's
      // display: inline-block on .btn would override.
      const browse = byId(BROWSE_ID);
      if (browse) browse.style.display = "none";
      const hint = byId(HINT_ID);
      if (hint) hint.textContent = note;
    },

    onBrowse(handler) {
      byId(BROWSE_ID)?.addEventListener("click", handler);
    },

    onDropDirectory(handler) {
      const panel = byId(PANEL_ID);
      if (!panel) return;
      // A folder dropped beside the panel would otherwise be navigated to,
      // and the page lost with it; the dashboard's own drags (the chain's
      // rows) carry no files and are left alone.
      for (const type of ["dragover", "drop"] as const) {
        document.addEventListener(type, (event) => {
          if (carriesFiles(event.dataTransfer)) event.preventDefault();
        });
      }
      panel.addEventListener("dragover", (event) => {
        if (!carriesFiles(event.dataTransfer)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "link";
        panel.classList.add(DROP_CLASS);
      });
      panel.addEventListener("dragleave", (event) => {
        // Leaving for one of the panel's own children is not leaving.
        if (!panel.contains(event.relatedTarget as Node | null)) panel.classList.remove(DROP_CLASS);
      });
      panel.addEventListener("drop", (event) => {
        panel.classList.remove(DROP_CLASS);
        if (!carriesFiles(event.dataTransfer)) return;
        event.preventDefault();
        const items = Array.from(event.dataTransfer.items).filter((item) => item.kind === "file") as HandleItem[];
        if (items.length !== 1) return status("Drop one folder: the volume's .zarr or .n5 directory.", "error");
        const item = items[0];
        if (!item.getAsFileSystemHandle) return status("This browser gives no handle for a dropped folder; use Chrome or Edge.", "error");
        // Asked for now, in the event: the drag's items are gone once it ends.
        item.getAsFileSystemHandle().then(
          (handle) => {
            if (!handle) return status("Nothing that can be opened was dropped.", "error");
            if (handle.kind !== "directory") return status(`${handle.name} is a file. Drop the volume's folder (the .zarr or .n5 directory).`, "error");
            handler(handle as unknown as PickedDirectory);
          },
          (e: unknown) => status(`Could not read what was dropped: ${message(e)}`, "error"),
        );
      });
    },
  };
}
