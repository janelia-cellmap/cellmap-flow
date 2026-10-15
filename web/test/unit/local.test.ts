// A folder on this machine (src/local/index.ts): the local URL and its key,
// the pick flow, and the boot-time permission check, on a fake connect panel,
// fake directory handles and the in-memory session. The DOM (panel.ts) is
// not run here.
import { describe, expect, it } from "vitest";

import type { SessionStore } from "../../src/contracts";
import {
  folderNameOf,
  type HandleStore,
  installLocalFolderUi,
  localKeyFor,
  localKeyOf,
  localUrl,
  openLocalFolder,
  type PickedDirectory,
  restoreLocalFolder,
  UNSUPPORTED_NOTE,
} from "../../src/local";
import type { LocalPanel } from "../../src/local/panel";
import { memorySession } from "../../src/state";
import { rawSource } from "../../src/viewer/layers";

const PAGE = "https://janelia-cellmap.github.io/cellmap-flow/app/index.html#!%7B%22layers%22%3A%5B%5D%7D";
const ROOT_PAGE = "http://localhost:5173/";

/** A handle whose permission starts as `permission` and becomes each of
 * `grants` in turn when asked. */
function fakeHandle(name: string, permission: PermissionState, grants: PermissionState[] = []) {
  const handle = {
    name,
    asked: 0,
    async queryPermission() {
      return permission;
    },
    async requestPermission() {
      handle.asked++;
      permission = grants.shift() ?? permission;
      return permission;
    },
  };
  return handle;
}

function fakeHandles(entries: Record<string, PickedDirectory> = {}, failPut?: string): HandleStore & { stored: Map<string, PickedDirectory>; gets: string[] } {
  const stored = new Map(Object.entries(entries));
  const gets: string[] = [];
  return {
    stored,
    gets,
    async get(key) {
      gets.push(key);
      return stored.get(key);
    },
    async put(key, handle) {
      if (failPut) throw new Error(failPut);
      stored.set(key, handle);
    },
  };
}

/** A panel that records what is done to it and answers each askAgain from `answers`. */
function fakePanel(answers: boolean[] = []) {
  const panel = {
    shown: 0,
    statuses: [] as { message: string; tone: string }[],
    asked: [] as { name: string; refused: boolean }[],
    forgotten: 0,
    connected: [] as string[],
    hidden: null as string | null,
    browse: null as (() => void) | null,
    drop: null as ((handle: PickedDirectory) => void) | null,
    show() {
      panel.shown++;
    },
    status(message: string, tone: "info" | "error") {
      panel.statuses.push({ message, tone });
    },
    async askAgain(name: string, refused: boolean) {
      panel.asked.push({ name, refused });
      const answer = answers.shift();
      if (answer === undefined) throw new Error("asked once more than the test allows");
      return answer;
    },
    forgetPageState() {
      panel.forgotten++;
    },
    connect(url: string) {
      panel.connected.push(url);
    },
    hideLocalControls(note: string) {
      panel.hidden = note;
    },
    onBrowse(handler: () => void) {
      panel.browse = handler;
    },
    onDropDirectory(handler: (handle: PickedDirectory) => void) {
      panel.drop = handler;
    },
  } satisfies LocalPanel & Record<string, unknown>;
  return panel;
}

function session(datasetUrl: string | null): SessionStore {
  return memorySession({ datasetUrl });
}

describe("the local URL and its key", () => {
  it("is local/<key>/ under the page's directory, whatever the page's file and hash", () => {
    expect(localUrl("ab12cd-jrc_hela-2.zarr", PAGE)).toBe("https://janelia-cellmap.github.io/cellmap-flow/app/local/ab12cd-jrc_hela-2.zarr/");
    expect(localUrl("k", ROOT_PAGE)).toBe("http://localhost:5173/local/k/");
  });

  it("gives the key back from the URL, or from any path under it", () => {
    expect(localKeyOf(localUrl("ab12cd-jrc_hela-2.zarr", PAGE), PAGE)).toBe("ab12cd-jrc_hela-2.zarr");
    expect(localKeyOf("http://localhost:5173/local/k/", ROOT_PAGE)).toBe("k");
    expect(localKeyOf("http://localhost:5173/local/k", ROOT_PAGE)).toBe("k");
    expect(localKeyOf("http://localhost:5173/local/k/recon-1/em/s0/zarr.json", ROOT_PAGE)).toBe("k");
  });

  it("knows every other URL for what it is", () => {
    for (const url of [
      "https://janelia-cellmap.github.io/local/k/", // the origin's root, not the page's directory
      "https://other.example.org/cellmap-flow/app/local/k/", // another origin
      "https://janelia-cellmap.github.io/cellmap-flow/app/data.zarr", // the page's directory, not local/
      "https://janelia-cellmap.github.io/cellmap-flow/app/local/", // no key
      "s3://bucket/local/k/",
      "",
      null,
      undefined,
    ]) {
      expect(localKeyOf(url, PAGE), String(url)).toBeNull();
    }
  });
});

describe("localKeyFor", () => {
  it("is a few random characters, then the folder's name made URL-safe, with the name readable back", () => {
    const key = localKeyFor("jrc_hela-2.zarr");
    expect(key).toMatch(/^[a-z0-9]{6}-jrc_hela-2\.zarr$/);
    expect(folderNameOf(key)).toBe("jrc_hela-2.zarr");
    expect(localKeyFor("jrc_hela-2.zarr")).not.toBe(key); // unique per pick
    expect(localKeyFor("x", "fixed0")).toBe("fixed0-x");
  });

  it("turns what a URL path segment cannot carry into hyphens and keeps a long name's extension", () => {
    expect(localKeyFor("my volume (é) copy.n5", "fixed0")).toBe("fixed0-my-volume-e-copy.n5");
    expect(localKeyFor("a/b\\c", "fixed0")).toBe("fixed0-a-b-c");
    expect(localKeyFor(".hidden", "fixed0")).toBe("fixed0-hidden");
    expect(localKeyFor("   ", "fixed0")).toBe("fixed0-folder");
    const long = localKeyFor(`${"n".repeat(100)}.zarr`, "fixed0");
    expect(long.length).toBeLessThanOrEqual(7 + 64);
    expect(long).toMatch(/^fixed0-n+\.zarr$/);
  });

  it("ends the local URL as the folder does, so the viewer's raw layer tells an N5 from a zarr", () => {
    expect(rawSource(localUrl(localKeyFor("volume.n5", "fixed0"), ROOT_PAGE))).toBe("n5://http://localhost:5173/local/fixed0-volume.n5/");
    expect(rawSource(localUrl(localKeyFor("volume.zarr", "fixed0"), ROOT_PAGE))).toBe("zarr://http://localhost:5173/local/fixed0-volume.zarr/");
    expect(rawSource(localUrl(localKeyFor("volume", "fixed0"), ROOT_PAGE))).toBe("zarr://http://localhost:5173/local/fixed0-volume/");
  });
});

describe("restoreLocalFolder", () => {
  const KEY = "ab12cd-jrc_hela-2.zarr";
  const LOCAL = localUrl(KEY, PAGE);

  it("does nothing for a dataset that is not a folder on this machine", async () => {
    const panel = fakePanel();
    const handles = fakeHandles();
    for (const url of [null, "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8"]) {
      const store = session(url);
      await restoreLocalFolder({ session: store, log: () => {}, panel, handles, base: PAGE });
      expect(store.state.datasetUrl).toBe(url);
    }
    expect(handles.gets).toEqual([]);
    expect(panel.shown).toBe(0);
    expect(panel.asked).toEqual([]);
  });

  it("goes straight on when the folder is still readable", async () => {
    const handle = fakeHandle("jrc_hela-2.zarr", "granted");
    const panel = fakePanel();
    const store = session(LOCAL);
    const lines: string[] = [];
    await restoreLocalFolder({ session: store, log: (l) => lines.push(l), panel, handles: fakeHandles({ [KEY]: handle }), base: PAGE });
    expect(store.state.datasetUrl).toBe(LOCAL);
    expect(handle.asked).toBe(0);
    expect(panel.asked).toEqual([]);
    expect(panel.shown).toBe(0);
    expect(panel.forgotten).toBe(0);
    expect(lines).toEqual(["jrc_hela-2.zarr on this machine is readable again"]);
  });

  it("shows the panel and waits for the click when the permission is to be asked for, then asks and goes on", async () => {
    const handle = fakeHandle("jrc_hela-2.zarr", "prompt", ["granted"]);
    const panel = fakePanel([true]);
    const store = session(LOCAL);
    let done = false;
    const restoring = restoreLocalFolder({ session: store, log: () => {}, panel, handles: fakeHandles({ [KEY]: handle }), base: PAGE }).then(() => {
      done = true;
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(done).toBe(false); // waiting on the user
    await restoring;
    expect(panel.shown).toBe(1);
    expect(panel.asked).toEqual([{ name: "jrc_hela-2.zarr", refused: false }]);
    expect(handle.asked).toBe(1);
    expect(store.state.datasetUrl).toBe(LOCAL);
    expect(panel.forgotten).toBe(0);
  });

  it("asks again when Chrome refuses, and forgets the folder when the user opens something else", async () => {
    const handle = fakeHandle("jrc_hela-2.zarr", "prompt", ["denied"]);
    const panel = fakePanel([true, false]);
    const store = session(LOCAL);
    await restoreLocalFolder({ session: store, log: () => {}, panel, handles: fakeHandles({ [KEY]: handle }), base: PAGE });
    expect(panel.asked).toEqual([
      { name: "jrc_hela-2.zarr", refused: false },
      { name: "jrc_hela-2.zarr", refused: true },
    ]);
    expect(handle.asked).toBe(1);
    expect(store.state.datasetUrl).toBeNull();
    expect(panel.forgotten).toBe(1);
    expect(panel.statuses).toEqual([{ message: "jrc_hela-2.zarr is no longer open. Pick a folder or enter a URL to open a volume.", tone: "info" }]);
  });

  it("words the first prompt for a permission Chrome denied before", async () => {
    const handle = fakeHandle("v.zarr", "denied", ["granted"]);
    const panel = fakePanel([true]);
    await restoreLocalFolder({ session: session(LOCAL), log: () => {}, panel, handles: fakeHandles({ [KEY]: handle }), base: PAGE });
    expect(panel.asked).toEqual([{ name: "v.zarr", refused: true }]);
  });

  it("forgets a folder whose handle is gone, and says which on the panel", async () => {
    const panel = fakePanel();
    const store = session(LOCAL);
    const lines: string[] = [];
    await restoreLocalFolder({ session: store, log: (l) => lines.push(l), panel, handles: fakeHandles(), base: PAGE });
    expect(store.state.datasetUrl).toBeNull();
    expect(panel.forgotten).toBe(1);
    expect(panel.shown).toBe(1);
    expect(panel.asked).toEqual([]);
    expect(panel.statuses).toHaveLength(1);
    expect(panel.statuses[0].tone).toBe("error");
    expect(panel.statuses[0].message).toMatch(/^jrc_hela-2\.zarr is a folder on a machine this browser has no access to/);
    expect(lines).toEqual(["the session's dataset was the local folder jrc_hela-2.zarr, whose handle is gone; forgetting it"]);
  });

  it("forgets the folder when the handles cannot be read either", async () => {
    const panel = fakePanel();
    const store = session(LOCAL);
    const broken: HandleStore = {
      async get() {
        throw new Error("IndexedDB is closed");
      },
      async put() {},
    };
    const lines: string[] = [];
    await restoreLocalFolder({ session: store, log: (l) => lines.push(l), panel, handles: broken, base: PAGE });
    expect(store.state.datasetUrl).toBeNull();
    expect(lines[0]).toBe("could not look up the local folder jrc_hela-2.zarr: IndexedDB is closed");
  });
});

describe("openLocalFolder", () => {
  it("stores the handle under a fresh key and presses Connect with its local URL", async () => {
    const handle = fakeHandle("volume.zarr", "granted");
    const panel = fakePanel();
    const handles = fakeHandles();
    await openLocalFolder(handle, { log: () => {}, panel, handles, base: ROOT_PAGE });
    expect(handles.stored.size).toBe(1);
    const [key, stored] = [...handles.stored.entries()][0];
    expect(stored).toBe(handle);
    expect(key).toMatch(/^[a-z0-9]{6}-volume\.zarr$/);
    expect(panel.connected).toEqual([`http://localhost:5173/local/${key}/`]);
    expect(localKeyOf(panel.connected[0], ROOT_PAGE)).toBe(key);
    expect(panel.statuses).toEqual([]);
  });

  it("asks for the permission first when it is not granted yet", async () => {
    const handle = fakeHandle("volume.zarr", "prompt", ["granted"]);
    const panel = fakePanel();
    await openLocalFolder(handle, { log: () => {}, panel, handles: fakeHandles(), base: ROOT_PAGE });
    expect(handle.asked).toBe(1);
    expect(panel.connected).toHaveLength(1);
  });

  it("opens nothing when reading is not allowed, and says so", async () => {
    const handle = fakeHandle("volume.zarr", "prompt", ["denied"]);
    const panel = fakePanel();
    const handles = fakeHandles();
    await openLocalFolder(handle, { log: () => {}, panel, handles, base: ROOT_PAGE });
    expect(handles.stored.size).toBe(0);
    expect(panel.connected).toEqual([]);
    expect(panel.statuses).toEqual([{ message: "Reading volume.zarr was not allowed, so it cannot be opened.", tone: "error" }]);
  });

  it("says when the handle cannot be stored", async () => {
    const panel = fakePanel();
    await openLocalFolder(fakeHandle("volume.zarr", "granted"), { log: () => {}, panel, handles: fakeHandles({}, "IndexedDB refused"), base: ROOT_PAGE });
    expect(panel.connected).toEqual([]);
    expect(panel.statuses).toEqual([{ message: "Could not open volume.zarr: IndexedDB refused", tone: "error" }]);
  });
});

describe("installLocalFolderUi", () => {
  it("hides the controls behind a note where there is no picker", () => {
    const panel = fakePanel();
    installLocalFolderUi({ log: () => {}, panel, handles: fakeHandles(), supported: false });
    expect(panel.hidden).toBe(UNSUPPORTED_NOTE);
    expect(panel.browse).toBeNull();
    expect(panel.drop).toBeNull();
  });

  it("opens the picked or dropped folder, and says nothing of a cancelled dialog", async () => {
    const panel = fakePanel();
    const handles = fakeHandles();
    let picked: () => Promise<PickedDirectory> = async () => fakeHandle("picked.zarr", "granted");
    installLocalFolderUi({ log: () => {}, panel, handles, base: ROOT_PAGE, supported: true, pick: () => picked() });
    await panel.browse!();
    expect(panel.connected).toHaveLength(1);
    expect(panel.connected[0]).toMatch(/\/local\/[a-z0-9]{6}-picked\.zarr\/$/);

    panel.drop!(fakeHandle("dropped.n5", "granted"));
    await new Promise((r) => setTimeout(r, 0));
    expect(panel.connected).toHaveLength(2);
    expect(panel.connected[1]).toMatch(/\/local\/[a-z0-9]{6}-dropped\.n5\/$/);

    picked = async () => {
      throw Object.assign(new Error("The user aborted a request."), { name: "AbortError" });
    };
    await panel.browse!();
    expect(panel.connected).toHaveLength(2);
    expect(panel.statuses).toEqual([]);

    picked = async () => {
      throw new Error("no gesture");
    };
    await panel.browse!();
    expect(panel.statuses).toEqual([{ message: "Could not pick a folder: no gesture", tone: "error" }]);
  });
});
