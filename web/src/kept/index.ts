// What this browser keeps of finetuning, by tab. IndexedDB holds the
// finetuned models (models/finetuned.ts) and the paint volumes
// (state/paint-store.ts) for every tab and every visit, since a model or a
// paint is work to make again; the server keeps its runs on disk the same
// way. But a server's dashboard starts with its process and knows of no
// earlier run's models, and a browser tab used to list every model and
// volume the browser had ever made, an old experiment's beside the new.
// So the session says which are the tab's (SessionState.tabKept, kept with
// the tab in sessionStorage), and the dashboard's lists show those alone:
// the Hugging Face list and the ranking's candidates (listedInTab), and the
// Finetune tab's sessions to resume (volumeGroups, inTab). The rest are
// "kept from earlier", which the Models tab's Kept in this browser panel
// (panel.ts) lists, to bring one into the tab (state/index.ts keepInTab) or
// delete it. A share link still opens a finetuned model kept here that it
// names: resolving a model by its ref looks at everything kept, and serving
// it makes it the tab's.
//
// Nothing here touches the DOM: the backend's routes use it, in node too.
import type { AppContext, FinetunedRecord, ModelEntry, PaintStore, PaintVolume, SessionState, TabKept } from "../contracts";
import { readVolume } from "../finetune/volume";
import { isFinetuned } from "../models/finetuned";
import { message } from "../shared/message";

/** Whether the dashboard lists `entry` in this tab: a model of Hugging Face
 * always, one finetuned or opened from a file in this browser only when it
 * is the tab's. */
export function listedInTab(state: SessionState, entry: ModelEntry): boolean {
  return !isFinetuned(entry) || state.tabKept.finetuned.includes(entry.id);
}

/** A paint volume as the Finetune tab resumes it: the current one, and the
 * ids of it and of the volumes it took the place of (Use this model's
 * supersedeVolume), which are the same paint on other grids. */
export interface VolumeGroup {
  volume: PaintVolume;
  ids: string[];
}

/** The store's annotation volumes, each current one with those it took the
 * place of, newest first. One whose successor was deleted since is current,
 * so every volume the store holds is in a group; files that are no
 * annotation volume's are left out. */
export async function volumeGroups(paint: PaintStore): Promise<VolumeGroup[]> {
  const volumes = new Map<string, PaintVolume>();
  for (const id of await paint.volumes()) {
    const volume = await readVolume(paint, id);
    if (volume) volumes.set(id, volume);
  }
  const groups = new Map<string, VolumeGroup>();
  for (const volume of volumes.values()) {
    // Followed as currentVolume follows it, each id once, so a ring ends.
    let current = volume;
    const followed = new Set([volume.id]);
    while (current.superseded_by && volumes.has(current.superseded_by) && !followed.has(current.superseded_by)) {
      current = volumes.get(current.superseded_by)!;
      followed.add(current.id);
    }
    const group = groups.get(current.id) ?? { volume: current, ids: [] };
    group.ids.push(volume.id);
    groups.set(current.id, group);
  }
  return [...groups.values()].sort((a, b) => b.volume.created_at.localeCompare(a.volume.created_at) || b.volume.id.localeCompare(a.volume.id));
}

/** Whether `group` is the tab's: any of its volumes is (a volume carried to
 * another grid, in another tab, stays this one's). */
export function inTab(state: SessionState, group: VolumeGroup): boolean {
  return group.ids.some((id) => state.tabKept.volumes.includes(id));
}

/** What the browser keeps that is not this tab's: the finetuned models,
 * newest first, and the volumes as groups (volumeGroups), newest first. */
export interface Earlier {
  finetuned: FinetunedRecord[];
  volumes: VolumeGroup[];
}

export async function earlierKept(ctx: Pick<AppContext, "session" | "catalog" | "paint">): Promise<Earlier> {
  const { state } = ctx.session;
  const finetuned = (await ctx.catalog.finetuned()).filter((r) => !state.tabKept.finetuned.includes(r.id));
  const volumes = (await volumeGroups(ctx.paint)).filter((g) => !inTab(state, g));
  return { finetuned, volumes };
}

/** Forget kept things for good, in every tab: the finetuned models' records
 * and bytes, and every file of the volumes. */
export async function deleteKept(ctx: Pick<AppContext, "catalog" | "paint">, kept: Partial<TabKept>): Promise<void> {
  for (const id of kept.finetuned ?? []) await ctx.catalog.removeFinetuned(id);
  for (const id of kept.volumes ?? []) await ctx.paint.removeVolume(id);
}

/** Every earlier thing (earlierKept), the volumes' superseded copies with
 * them; the tab's own stay. */
export async function deleteEarlier(ctx: Pick<AppContext, "session" | "catalog" | "paint">): Promise<void> {
  const earlier = await earlierKept(ctx);
  await deleteKept(ctx, { finetuned: earlier.finetuned.map((r) => r.id), volumes: earlier.volumes.flatMap((g) => g.ids) });
}

/** What navigator.storage offers to keep this origin's storage. */
type StorageManagerLike = Pick<StorageManager, "persist">;

/** Ask the browser to keep this origin's storage (navigator.storage.persist):
 * without it the browser may clear IndexedDB, the finetuned models and the
 * paint with it, when the disk runs short. The browser decides, by how the
 * site is used or by asking; the answer goes to the log, once. Nothing
 * where there is no such call (an older browser, node). */
export async function askToKeepStorage(log: (line: string) => void, storage: StorageManagerLike | undefined = globalThis.navigator?.storage): Promise<void> {
  if (!storage?.persist) return;
  try {
    log((await storage.persist())
      ? "storage: the browser keeps the finetuned models and paint until they are deleted (persistent storage granted)"
      : "storage: the browser may clear the finetuned models and paint kept here when it needs the space (persistent storage not granted)");
  } catch (e) {
    log(`storage: could not ask the browser to keep the finetuned models and paint (${message(e)})`);
  }
}
