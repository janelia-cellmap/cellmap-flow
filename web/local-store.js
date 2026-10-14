// local-store.js — tiny IndexedDB layer shared by the page and the service worker.
//
// Two object stores:
//   "models"  : modelName -> meta (geometry, onnxUrl, source descriptor, norms/post)
//   "handles" : key -> FileSystemDirectoryHandle (structured-cloneable, survives SW restarts)
//
// Handles are stored so a locally-picked Zarr directory remains readable after
// the page reloads or the service worker is recycled. Read permission must be
// re-granted from a user gesture on the page (see index.html).

const DB_NAME = "cellmap-flow-serverless";
const DB_VERSION = 2;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("models")) db.createObjectStore("models");
      if (!db.objectStoreNames.contains("handles")) db.createObjectStore("handles");
      if (!db.objectStoreNames.contains("blobs")) db.createObjectStore("blobs");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(store, mode, fn) {
  return openDB().then(
    (db) =>
      new Promise((resolve, reject) => {
        const t = db.transaction(store, mode);
        const os = t.objectStore(store);
        const result = fn(os);
        t.oncomplete = () => resolve(result && result.__req ? result.__req.result : result);
        t.onerror = () => reject(t.error);
      })
  );
}

const wrap = (req) => ({ __req: req });

export const putModel = (name, meta) =>
  tx("models", "readwrite", (os) => wrap(os.put(meta, name)));
export const getModel = (name) =>
  tx("models", "readonly", (os) => wrap(os.get(name)));
export const listModels = () =>
  tx("models", "readonly", (os) => wrap(os.getAllKeys()));
export const deleteModel = (name) =>
  tx("models", "readwrite", (os) => wrap(os.delete(name)));

export const putHandle = (key, handle) =>
  tx("handles", "readwrite", (os) => wrap(os.put(handle, key)));
export const getHandle = (key) =>
  tx("handles", "readonly", (os) => wrap(os.get(key)));

// A zarrita Readable store backed by a File System Access directory handle, so
// locally-picked Zarr directories can be read in a worker or the service worker.
// Lives here (no zarrita dependency) so the SW can use it for /local passthrough
// without pulling zarrita — whose codecs use dynamic import(), forbidden in SWs.
export class FileSystemStore {
  constructor(dirHandle) {
    this.dir = dirHandle;
  }
  async get(key) {
    const parts = key.replace(/^\//, "").split("/").filter(Boolean);
    if (parts.length === 0) return undefined;
    try {
      let handle = this.dir;
      for (let i = 0; i < parts.length - 1; i++) {
        handle = await handle.getDirectoryHandle(parts[i]);
      }
      const fileHandle = await handle.getFileHandle(parts[parts.length - 1]);
      const file = await fileHandle.getFile();
      return new Uint8Array(await file.arrayBuffer());
    } catch (e) {
      if (e && (e.name === "NotFoundError" || e.name === "TypeMismatchError")) {
        return undefined; // missing chunk -> zarrita treats as fill_value
      }
      throw e;
    }
  }
}

// Raw bytes (e.g. a local ONNX model) shared from the page to the SW.
export const putBlob = (key, data) =>
  tx("blobs", "readwrite", (os) => wrap(os.put(data, key)));
export const getBlob = (key) =>
  tx("blobs", "readonly", (os) => wrap(os.get(key)));
