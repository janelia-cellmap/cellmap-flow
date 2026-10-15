// Where the downloaded .onnx bytes live between visits: IndexedDB, database
// "cellmap-flow-models", one object store with the file's URL as the key and
// its ArrayBuffer as the value. A model is 11–43 MB and Hugging Face
// rate-limits anonymous downloads, so a page that re-downloaded on every load
// would soon be told to wait.
//
// ByteCache is the small interface the catalog uses; IndexedDbByteCache is
// the browser's implementation and MemoryByteCache stands in where there is
// no IndexedDB (tests, or a browser with storage blocked).

export interface ByteCache {
  get(key: string): Promise<ArrayBuffer | undefined>;
  put(key: string, bytes: ArrayBuffer): Promise<void>;
  delete(key: string): Promise<void>;
}

export const MODEL_DB_NAME = "cellmap-flow-models";
const STORE = "bytes";
const DB_VERSION = 1;

/** A cache that lasts as long as the page. */
export class MemoryByteCache implements ByteCache {
  private readonly entries = new Map<string, ArrayBuffer>();

  async get(key: string): Promise<ArrayBuffer | undefined> {
    return this.entries.get(key);
  }

  async put(key: string, bytes: ArrayBuffer): Promise<void> {
    this.entries.set(key, bytes);
  }

  async delete(key: string): Promise<void> {
    this.entries.delete(key);
  }
}

/** The IndexedDB cache. The database is opened on first use and kept open;
 * when another tab upgrades it the connection is closed and reopened on the
 * next call. */
export class IndexedDbByteCache implements ByteCache {
  private db: Promise<IDBDatabase> | null = null;

  constructor(
    private readonly factory: IDBFactory,
    private readonly name: string = MODEL_DB_NAME,
  ) {}

  get(key: string): Promise<ArrayBuffer | undefined> {
    return this.run("readonly", (store) => store.get(key) as IDBRequest<ArrayBuffer | undefined>);
  }

  async put(key: string, bytes: ArrayBuffer): Promise<void> {
    await this.run("readwrite", (store) => store.put(bytes, key));
  }

  async delete(key: string): Promise<void> {
    await this.run("readwrite", (store) => store.delete(key));
  }

  private open(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    const opening = new Promise<IDBDatabase>((resolve, reject) => {
      const request = this.factory.open(this.name, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
      };
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => {
          db.close();
          this.db = null;
        };
        resolve(db);
      };
      request.onerror = () => reject(request.error ?? new Error(`Could not open IndexedDB ${this.name}`));
      request.onblocked = () => reject(new Error(`IndexedDB ${this.name} is open in another tab at an older version`));
    });
    this.db = opening.catch((error: unknown) => {
      this.db = null; // so the next call tries again
      throw error;
    });
    return this.db;
  }

  // One request in its own transaction; the result once the transaction has
  // committed, so a put is on disk when the promise resolves.
  private async run<T>(mode: IDBTransactionMode, request: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.open();
    return new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = request(tx.objectStore(STORE));
      const fail = () => reject(tx.error ?? req.error ?? new Error(`IndexedDB ${mode} transaction on ${this.name} failed`));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = fail;
      tx.onabort = fail;
    });
  }
}

/** The cache for this runtime: IndexedDB when there is one, else memory (with a log line, since models will then be downloaded on every visit). */
export function openByteCache(factory: IDBFactory | undefined, log: (line: string) => void): ByteCache {
  if (factory) return new IndexedDbByteCache(factory);
  log("model cache: no IndexedDB here, model files will be downloaded on every visit");
  return new MemoryByteCache();
}
