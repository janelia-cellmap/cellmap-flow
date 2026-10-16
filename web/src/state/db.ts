// The browser mode's IndexedDB database, for the local-directory store
// (local-store.ts) and the paint store (paint-store.ts), shared by the page
// and the service worker (IndexedDB is per origin, so both see it). The
// session itself is not here: it is the tab's (index.ts, sessionStorage).
//
// One database with every object store in it: two modules each opening the
// same database with their own version and stores would race on upgrades.
// Add a store here, bump the version, and the next open creates it.
//
// Object stores:
//   "handles": key -> FileSystemDirectoryHandle of a zarr directory the user
//              picked; structured-cloneable, so it survives reloads and a
//              recycled service worker (permission must be granted again from
//              a user gesture on the page)
//   "blobs":   key -> raw bytes shared from the page to the service worker
//   "paint":   "<volume>/<path>" -> the bytes of an annotation volume's file
//              (paint-store.ts): the page writes a new volume's metadata and
//              reads its paint for training, the service worker reads and
//              writes the files on the viewer's behalf (src/sw.ts, the paint
//              route)
//
// Version 2 dropped the "session" store (the session moved to
// sessionStorage); a database from version 1 loses it on upgrade. Version 3
// added "paint".

export const DB_NAME = "cellmap-flow-browser";
export const DB_VERSION = 3;
export const STORES = ["handles", "blobs", "paint"] as const;
const DROPPED_STORES = ["session"];
export type StoreName = (typeof STORES)[number];

let opening: Promise<IDBDatabase> | null = null;

/** The database, opened once and kept; stores that do not exist yet are created. */
export function openDb(): Promise<IDBDatabase> {
  if (opening) return opening;
  opening = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      for (const name of STORES) if (!db.objectStoreNames.contains(name)) db.createObjectStore(name);
      for (const name of DROPPED_STORES) if (db.objectStoreNames.contains(name)) db.deleteObjectStore(name);
    };
    req.onsuccess = () => {
      const db = req.result;
      // Another tab upgrading the database asks this one to close; the next
      // call opens it again at the new version.
      db.onversionchange = () => {
        db.close();
        opening = null;
      };
      resolve(db);
    };
    req.onerror = () => reject(req.error ?? new Error(`could not open the ${DB_NAME} database`));
    req.onblocked = () => reject(new Error(`the ${DB_NAME} database is open in another tab at an older version`));
  });
  opening.catch(() => {
    opening = null;
  });
  return opening;
}

/** One request in a transaction on `store`, resolved with its result once the transaction completes. */
async function request<T>(store: StoreName, mode: IDBTransactionMode, make: (os: IDBObjectStore) => IDBRequest): Promise<T> {
  const db = await openDb();
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(store, mode);
    const req = make(tx.objectStore(store));
    tx.oncomplete = () => resolve(req.result as T);
    tx.onerror = () => reject(tx.error ?? new Error(`${mode} on ${store} failed`));
    tx.onabort = () => reject(tx.error ?? new Error(`${mode} on ${store} was aborted`));
  });
}

export function idbGet<T>(store: StoreName, key: IDBValidKey): Promise<T | undefined> {
  return request<T | undefined>(store, "readonly", (os) => os.get(key));
}

export function idbPut(store: StoreName, key: IDBValidKey, value: unknown): Promise<void> {
  return request<unknown>(store, "readwrite", (os) => os.put(value, key)).then(() => undefined);
}

export function idbDelete(store: StoreName, key: IDBValidKey): Promise<void> {
  return request<unknown>(store, "readwrite", (os) => os.delete(key)).then(() => undefined);
}

/** The keys of `store` from `lower` up to but not including `upper`, every
 * key when no bounds are given, in IndexedDB's key order (for strings, by
 * UTF-16 code unit, the order Array.prototype.sort gives). Strings bound a
 * prefix: `[prefix, prefix + "￿")` is every key that starts with it. */
export function idbKeys(store: StoreName, lower?: string, upper?: string): Promise<IDBValidKey[]> {
  const range = lower === undefined || upper === undefined ? undefined : IDBKeyRange.bound(lower, upper, false, true);
  return request<IDBValidKey[]>(store, "readonly", (os) => os.getAllKeys(range));
}

/** Delete the keys of `store` from `lower` up to but not including `upper`, in one transaction. */
export function idbDeleteRange(store: StoreName, lower: string, upper: string): Promise<void> {
  return request<unknown>(store, "readwrite", (os) => os.delete(IDBKeyRange.bound(lower, upper, false, true))).then(() => undefined);
}
