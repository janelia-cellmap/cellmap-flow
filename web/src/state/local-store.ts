// Zarr directories picked on this machine with the File System Access API
// (showDirectoryPicker), and raw bytes the page shares with the service
// worker. From Marwan Zouinkhi's prototype (web/legacy/local-store.js), on
// the shared database of db.ts.
//
// The page stores a picked directory's handle under a key; the dataset URL
// it then opens is `<page base>local/<key>/`, and the service worker
// (src/sw.ts) answers each request under it with the file at that path, read
// through FileSystemStore. The handle survives reloads and a recycled
// service worker, but read permission does not: the page must ask for it
// again from a user gesture (handle.requestPermission({mode: "read"})).
//
// No zarrita here: the service worker imports this file, and service workers
// forbid the dynamic import() zarrita's codecs use.
import { idbGet, idbPut } from "./db";

export const putHandle = (key: string, handle: FileSystemDirectoryHandle): Promise<void> => idbPut("handles", key, handle);
export const getHandle = (key: string): Promise<FileSystemDirectoryHandle | undefined> => idbGet("handles", key);

export const putBlob = (key: string, data: ArrayBuffer | Uint8Array): Promise<void> => idbPut("blobs", key, data);
export const getBlob = (key: string): Promise<ArrayBuffer | Uint8Array | undefined> => idbGet("blobs", key);

/** The least a directory handle must do for FileSystemStore (the File System
 * Access API's FileSystemDirectoryHandle does); a type so tests can hand it
 * a fake. */
export interface DirectoryLike {
  getDirectoryHandle(name: string): Promise<DirectoryLike>;
  getFileHandle(name: string): Promise<{ getFile(): Promise<FileLike> }>;
}

/** What a directory's file gives: its bytes, or (a real File, a Blob) a
 * body a Response can stream without the bytes passing through here. */
export interface FileLike {
  arrayBuffer(): Promise<ArrayBuffer>;
}

/** Reads files of a picked directory by their path within it, the way a
 * zarr store is read: `get("s0/zarr.json")`, `get("s0/c/0/0/0")`. A path
 * that is not there is undefined (a missing chunk means the fill value), so
 * is an empty path.
 *
 * Directory handles are kept once looked up: every File System Access call
 * is a round trip to the browser process, and a viewer asks for hundreds of
 * chunks of one directory in a row, each of which walked the path from the
 * root again (a scale's chunks nested two or three directories deep cost
 * three or four calls before the file was reached), which is what made a
 * local folder slower than a bucket. A lookup that fails is not kept, so a
 * directory made later (a label store) or a passing error is tried again. */
export class FileSystemStore {
  private readonly dirs = new Map<string, Promise<DirectoryLike>>();

  constructor(private readonly dir: DirectoryLike) {}

  /** The directory at `path` (its segments), the root for none. */
  private directory(path: string[]): Promise<DirectoryLike> {
    if (path.length === 0) return Promise.resolve(this.dir);
    const key = path.join("/");
    let pending = this.dirs.get(key);
    if (!pending) {
      pending = this.directory(path.slice(0, -1)).then((parent) => parent.getDirectoryHandle(path[path.length - 1]));
      pending.catch(() => this.dirs.delete(key));
      this.dirs.set(key, pending);
    }
    return pending;
  }

  async get(key: string): Promise<Uint8Array | undefined> {
    const file = await this.file(key);
    return file && new Uint8Array(await file.arrayBuffer());
  }

  /** The file at `key` itself (a File in a browser: a Response streams it
   * without its bytes passing through the caller), or undefined as get(). */
  async file(key: string): Promise<FileLike | undefined> {
    const parts = key.replace(/^\//, "").split("/").filter(Boolean);
    if (parts.length === 0) return undefined;
    try {
      const dir = await this.directory(parts.slice(0, -1));
      return await (await dir.getFileHandle(parts[parts.length - 1])).getFile();
    } catch (e) {
      const name = (e as { name?: string })?.name;
      // NotFoundError: no such file or directory. TypeMismatchError: a file
      // where a directory was expected, or the other way round.
      if (name === "NotFoundError" || name === "TypeMismatchError") return undefined;
      throw e;
    }
  }
}
