// Run on a region: where the predictions go. A folder on this machine, with
// the File System Access API (Chrome and Edge): the user picks it once
// (showDirectoryPicker in "readwrite" mode), its handle is kept in IndexedDB
// as the connect panel keeps a picked volume's (state/local-store.ts), and
// each model is written to `<folder>/<model>.zarr` through chunkmirage's
// directoryStore. Chrome drops a stored handle's permission when the page
// reloads, and asking for it again needs a user gesture, so it is asked for
// in the Run click (writable). A model's zarr that is there already is said
// before anything is written (existingOutputs), and removed whole once the
// user agrees (clearOutputs): empty chunks are skipped, not written, so a
// chunk of the old zarr where the new one is empty would survive in it, and
// the two runs' chunks would mix.
//
// Without the API (Firefox, Safari) a small run is kept in memory, one
// store per model, and offered as a zip of the zarrs (zipRun, with the
// paint's own zip writer, finetune/zip.ts), up to ZIP_LIMIT_BYTES of
// predictions before compression: what a page can hold twice over (the
// stores, then the archive) without straining a laptop.
//
// After a run, a folder's zarrs open as layers in the viewer through the
// service worker's local/ route (src/sw.ts), the one a picked volume is
// read through: the folder's handle stored under a fresh key (a new key per
// opening, so neither the worker's cached directory handles nor the
// viewer's cached chunks outlive a run written over them), and each zarr at
// `local/<key>/<model>.zarr/`.
import type { Chain, LayerSpec, RegionStore, ServedModel } from "../contracts";
import { type ZipEntry, zipStore } from "../finetune/zip";
import type { PickedDirectory } from "../local-folder";
import { message } from "../shared/message";
import { type KeptShader, predictionLayer } from "../viewer/layers";

/** What a run needs of a picked folder: its name and permission methods
 * (local-folder's PickedDirectory), an entry looked up by name, and an
 * entry removed. lib.dom types the handle's OPFS half only, so these are
 * declared here; a type, so the tests hand in a fake. */
export interface OutputFolder extends PickedDirectory {
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<unknown>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
}

/** Where the folder's handle is kept: local-store's IndexedDB store, or a fake. */
export interface FolderStore {
  get(key: string): Promise<OutputFolder | undefined>;
  put(key: string, folder: OutputFolder): Promise<void>;
}

/** The key the output folder's handle is kept under, beside the picked
 * volumes' (whose keys are `<6 random>-<name>`, so they never meet it). One
 * per browser: the folder last chosen is offered again on the next visit. */
export const OUTPUT_FOLDER_KEY = "run-region:output";

/** The picker's options: under its own id, so Chrome opens the dialog where
 * the predictions went last time and not where a volume was opened from. */
export const OUTPUT_PICKER_OPTIONS = { id: "cellmap-flow-predictions", mode: "readwrite", startIn: "documents" } as const;

/** A model's zarr in the folder: `<model>.zarr`, as the server's blockwise
 * runs name theirs after the dataset they write. Model names are URL path
 * segments (ModelEntry.name), so they are folder names too. */
export function zarrName(model: string): string {
  return `${model}.zarr`;
}

/** Whether the page may write into `folder`, asking for it when it may not
 * yet (the caller is in the user's click). */
export async function writable(folder: Pick<PickedDirectory, "queryPermission" | "requestPermission">): Promise<boolean> {
  let permission = await folder.queryPermission({ mode: "readwrite" });
  if (permission !== "granted") permission = await folder.requestPermission({ mode: "readwrite" });
  return permission === "granted";
}

/** The zarrs of `models` that `folder` has already, as entries named
 * zarrName(model): a directory (an earlier run's), or a file of that name,
 * which would be in the way as much. */
export async function existingOutputs(folder: Pick<OutputFolder, "getDirectoryHandle">, models: readonly string[]): Promise<string[]> {
  const found: string[] = [];
  for (const model of models) {
    try {
      await folder.getDirectoryHandle(zarrName(model));
      found.push(zarrName(model));
    } catch (e) {
      const name = (e as { name?: string })?.name;
      // TypeMismatchError: a file by that name. NotFoundError: nothing.
      if (name === "TypeMismatchError") found.push(zarrName(model));
      else if (name !== "NotFoundError") throw e;
    }
  }
  return found;
}

/** Remove each of `entries` (zarrName()s existingOutputs found) from `folder`, whole. */
export async function clearOutputs(folder: Pick<OutputFolder, "removeEntry">, entries: readonly string[]): Promise<void> {
  for (const entry of entries) await folder.removeEntry(entry, { recursive: true });
}

/** The question asked before writing over `entries` in `folderName`. */
export function overwriteQuestion(folderName: string, entries: readonly string[]): string {
  const list = entries.map((e) => `  ${folderName}/${e}`).join("\n");
  return `${entries.length === 1 ? "This is" : "These are"} in the folder already:\n${list}\n\nReplace ${entries.length === 1 ? "it" : "them"}? ${entries.length === 1 ? "It is" : "They are"} deleted first, so the old chunks do not mix with the new ones.`;
}

/** `folder` made ready for a run of `models`, in the Run click: writing
 * allowed (asked for when Chrome has dropped it), and each model's zarr
 * that is there already removed once `ask` (window.confirm, with
 * overwriteQuestion) says yes. Either the zarrs removed, or why nothing may
 * be written, `refused` when the user said no to a question (nothing is
 * wrong then) rather than the browser or the folder failing. */
export async function prepareFolder(
  folder: OutputFolder,
  models: readonly string[],
  ask: (question: string) => boolean,
): Promise<{ ready: true; removed: string[] } | { ready: false; reason: string; refused: boolean }> {
  try {
    if (!(await writable(folder))) {
      return { ready: false, refused: true, reason: `Writing to ${folder.name} was not allowed, so nothing was run. Click Run to be asked again, or choose another folder.` };
    }
  } catch (e) {
    return { ready: false, refused: false, reason: `Could not ask to write to ${folder.name}: ${message(e)}` };
  }
  try {
    const existing = await existingOutputs(folder, models);
    if (existing.length && !ask(overwriteQuestion(folder.name, existing))) {
      return { ready: false, refused: true, reason: `Nothing was run; ${folder.name} is as it was.` };
    }
    await clearOutputs(folder, existing);
    return { ready: true, removed: existing };
  } catch (e) {
    return { ready: false, refused: false, reason: `Could not clear the way in ${folder.name}: ${message(e)}` };
  }
}

// ---------------------------------------------------------------------------
// Without a folder: a zip of the zarrs, from memory
// ---------------------------------------------------------------------------

/** The most a run kept in memory may hold, in predictions before
 * compression (ModelPlan.bytes summed): 256 MB. */
export const ZIP_LIMIT_BYTES = 256e6;

/** A store that keeps a zarr's files in memory: chunkmirage's memoryStore,
 * or anything of its shape. */
export type MemoryStore = RegionStore & { files: Map<string, Uint8Array> };

/** The zip of a run kept in memory: each model's files under
 * `<model>.zarr/`, so the archive unpacks to the folders a folder run
 * writes. Throws as zipStore does past its limits (65,535 files, 4 GB),
 * which ZIP_LIMIT_BYTES keeps a run well inside. */
export function zipRun(stores: ReadonlyMap<string, Pick<MemoryStore, "files">>, now: Date = new Date()): Uint8Array {
  const entries: ZipEntry[] = [];
  for (const [model, store] of stores) {
    for (const [path, bytes] of [...store.files].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
      entries.push({ name: `${zarrName(model)}/${path}`, bytes });
    }
  }
  return zipStore(entries, now);
}

/** The zip's file name: `predictions-<YYYYmmdd-HHMMSS>.zip`, local time, so
 * runs saved one after the other do not overwrite each other in Downloads. */
export function zipFileName(now: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `predictions-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.zip`;
}

// ---------------------------------------------------------------------------
// The zarrs written, back in the viewer
// ---------------------------------------------------------------------------

/** The layer a model's saved zarr is shown as, beside the model's own. */
export function savedLayerName(model: string): string {
  return `${model} (saved)`;
}

/** A model's saved zarr as a layer, `folderUrl` the folder's local/ URL
 * (local-folder's localUrl, with its trailing slash): the model's own layer
 * (viewer/layers.ts predictionLayer: its shader over the chain's range, or
 * a segmentation layer for labels, the channel it opens on, the transform
 * for a model read at another voxel size) on the zarr's URL, `zarr2://`
 * since the run writes zarr v2 (infer/engine.ts materialize), so it looks
 * like the live layer and lies where it does: the zarr declares the served
 * level's scale and the box's translation, which Neuroglancer places it by. `keep`
 * is the live layer's shader as the user left it, when there is one. */
export function savedLayer(served: ServedModel, chain: Chain, folderUrl: string, color: string, keep?: KeptShader): LayerSpec {
  const base = folderUrl.endsWith("/") ? folderUrl : `${folderUrl}/`;
  const url = `zarr2://${base}${zarrName(served.entry.name)}/`;
  return { ...predictionLayer({ ...served, url }, chain, color, keep), name: savedLayerName(served.entry.name) };
}
