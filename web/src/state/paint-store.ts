// The paint store: an annotation volume's files (its zarr v2 metadata and
// the chunks the viewer paints), by volume id and path within the volume,
// as the PaintStore contract (src/contracts.ts) has them. The page writes a
// new volume's metadata here and reads the paint for training; the service
// worker (src/sw.ts) answers the viewer's GET, PUT and DELETE of the files
// from it, the way MinIO answers the Neuroglancer fork's writes in server
// mode. IndexedDB holds the bytes (db.ts's "paint" store: both the page and
// the worker can open it, and it outlives both); MemoryPaintStore stands in
// where there is none (tests).
import type { PaintStore } from "../contracts";

/** The key a volume's file is stored under: `<volume>/<path>`. */
export function paintKey(volume: string, path: string): string {
  return `${volume}/${path.replace(/^\/+/, "")}`;
}

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

/** The store for this runtime: IndexedDB when there is one, else memory. */
export function openPaintStore(): PaintStore {
  // The IndexedDB store is to come (IndexedDbPaintStore, on db.ts).
  return new MemoryPaintStore();
}
