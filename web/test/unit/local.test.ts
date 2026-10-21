// A folder on this machine (src/local-folder/index.ts): the local URL and its key,
// the pick flow, and the boot-time permission check, on a fake connect panel,
// fake directory handles and the in-memory session. The DOM (panel.ts) is
// not run here.
import { describe, expect, it } from "vitest";

import type { SessionStore } from "../../src/contracts";
import { type HandleStore, localKeyFor, localKeyOf, localUrl, openLocalFolder, type PickedDirectory, restoreLocalFolder } from "../../src/local-folder";
import type { LocalPanel } from "../../src/local-folder/panel";
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

function fakeHandles(entries: Record<string, PickedDirectory> = {}): HandleStore & { stored: Map<string, PickedDirectory> } {
  const stored = new Map(Object.entries(entries));
  return {
    stored,
    async get(key) {
      return stored.get(key);
    },
    async put(key, handle) {
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

describe("the local URL", () => {
  it("ends the local URL as the folder does, so the viewer's raw layer tells an N5 from a zarr", () => {
    expect(rawSource(localUrl(localKeyFor("volume.n5", "fixed0"), ROOT_PAGE))).toBe("n5://http://localhost:5173/local/fixed0-volume.n5/");
    expect(rawSource(localUrl(localKeyFor("volume.zarr", "fixed0"), ROOT_PAGE))).toBe("zarr://http://localhost:5173/local/fixed0-volume.zarr/");
    expect(rawSource(localUrl(localKeyFor("volume", "fixed0"), ROOT_PAGE))).toBe("zarr://http://localhost:5173/local/fixed0-volume/");
  });
});

describe("restoreLocalFolder", () => {
  const KEY = "ab12cd-jrc_hela-2.zarr";
  const LOCAL = localUrl(KEY, PAGE);

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

  it("forgets a folder whose handle is gone, and says which on the panel", async () => {
    const panel = fakePanel();
    const store = session(LOCAL);
    await restoreLocalFolder({ session: store, log: () => {}, panel, handles: fakeHandles(), base: PAGE });
    expect(store.state.datasetUrl).toBeNull();
    expect(panel.forgotten).toBe(1);
    expect(panel.shown).toBe(1);
    expect(panel.asked).toEqual([]);
    expect(panel.statuses).toHaveLength(1);
    expect(panel.statuses[0].tone).toBe("error");
    expect(panel.statuses[0].message).toContain("jrc_hela-2.zarr");
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
    expect(handle.asked).toBe(0);
    expect(key).toMatch(/^[a-z0-9]{6}-volume\.zarr$/);
    expect(panel.connected).toEqual([`http://localhost:5173/local/${key}/`]);
    expect(localKeyOf(panel.connected[0], ROOT_PAGE)).toBe(key);
    expect(panel.statuses).toEqual([]);
  });
});
