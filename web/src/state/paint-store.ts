// The paint store: an annotation volume's files (its zarr v2 metadata and
// the chunks the viewer paints), by volume id and path within the volume,
// as the PaintStore contract (src/contracts.ts) has them. The page writes a
// new volume's metadata here and reads the paint for training; the service
// worker (src/sw.ts, through src/state/paint-route.ts) answers the viewer's
// GET, PUT and DELETE of the files from it, the way MinIO answers the
// Neuroglancer fork's writes in server mode. IndexedDB holds the bytes
// (db.ts's "paint" store: both the page and the worker can open it, and it
// outlives both); MemoryPaintStore stands in where there is none (tests, a
// browser with storage blocked), and is the reference for the semantics.
//
// The service worker imports this file: nothing of the DOM here, and no
// dynamic import().
import type { PaintStore } from "../contracts";
import { idbDelete, idbDeleteRange, idbGet, idbKeys, idbPut } from "./db";

/** The key a volume's file is stored under: `<volume>/<path>`. */
export function paintKey(volume: string, path: string): string {
  return `${volume}/${path.replace(/^\/+/, "")}`;
}

/** The first string past every key that starts with `prefix`: U+FFFF is the
 * last code unit, so `[prefix, prefix + "￿")` is the prefix's range in
 * code-unit order (IndexedDB's for strings, and Array.prototype.sort's). */
const past = (prefix: string): string => `${prefix}￿`;

/** A store that lasts as long as the page. */
export class MemoryPaintStore implements PaintStore {
  private readonly files = new Map<string, Uint8Array>();

  async get(volume: string, path: string): Promise<Uint8Array | undefined> {
    return this.files.get(paintKey(volume, path));
  }

  async put(volume: string, path: string, bytes: Uint8Array): Promise<void> {
    this.files.set(paintKey(volume, path), bytes);
  }

  async delete(volume: string, path: string): Promise<void> {
    this.files.delete(paintKey(volume, path));
  }

  async list(volume: string, prefix = ""): Promise<string[]> {
    const under = paintKey(volume, prefix);
    const paths: string[] = [];
    for (const key of this.files.keys()) if (key.startsWith(under)) paths.push(key.slice(volume.length + 1));
    return paths.sort();
  }

  async volumes(): Promise<string[]> {
    const ids = new Set<string>();
    for (const key of this.files.keys()) ids.add(key.slice(0, key.indexOf("/")));
    return [...ids].sort();
  }

  async removeVolume(volume: string): Promise<void> {
    for (const key of [...this.files.keys()]) if (key.startsWith(`${volume}/`)) this.files.delete(key);
  }
}

/** The store on IndexedDB (db.ts, object store "paint"), with
 * MemoryPaintStore's semantics: one record per file under its paintKey, a
 * volume's files the key range of its `<volume>/` prefix. The store is
 * small (one volume is its metadata and the chunks painted, a few hundred
 * at most), so a listing reads the keys of a range whole, and volumes()
 * reads every key. */
export class IndexedDbPaintStore implements PaintStore {
  async get(volume: string, path: string): Promise<Uint8Array | undefined> {
    const value = await idbGet<Uint8Array | ArrayBuffer>("paint", paintKey(volume, path));
    if (value === undefined || value instanceof Uint8Array) return value;
    return new Uint8Array(value);  // written as an ArrayBuffer by another hand: the same bytes
  }

  async put(volume: string, path: string, bytes: Uint8Array): Promise<void> {
    // A structured clone of a view keeps its whole buffer: a slice of a
    // larger one (a chunk cut from a block) would store the block.
    const own = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes : bytes.slice();
    await idbPut("paint", paintKey(volume, path), own);
  }

  delete(volume: string, path: string): Promise<void> {
    return idbDelete("paint", paintKey(volume, path));
  }

  async list(volume: string, prefix = ""): Promise<string[]> {
    const under = paintKey(volume, prefix);
    const keys = await idbKeys("paint", under, past(under));
    return keys.map((key) => String(key).slice(volume.length + 1));  // in key order, which is sorted
  }

  async volumes(): Promise<string[]> {
    const ids = new Set<string>();
    for (const key of await idbKeys("paint")) {
      const k = String(key);
      ids.add(k.slice(0, k.indexOf("/")));
    }
    return [...ids].sort();
  }

  removeVolume(volume: string): Promise<void> {
    const under = `${volume}/`;
    return idbDeleteRange("paint", under, past(under));
  }
}

/** The store for this runtime: IndexedDB when there is one, else memory
 * (with a note, since the paint is then lost when the page closes). */
export function openPaintStore(log: (line: string) => void = (line) => console.warn(line)): PaintStore {
  if (globalThis.indexedDB) return new IndexedDbPaintStore();
  log("paint store: no IndexedDB here, the paint lasts only as long as the page");
  return new MemoryPaintStore();
}
