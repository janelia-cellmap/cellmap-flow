// Where the models kept in this browser live between visits: IndexedDB,
// database "cellmap-flow-models", two object stores.
//
//   "bytes":     key -> ArrayBuffer of a .onnx file. For a Hugging Face model
//                the key is the file's URL and the bytes are a cache: a model
//                is 11–43 MB and Hugging Face rate-limits anonymous downloads,
//                so a page that re-downloaded on every load would soon be told
//                to wait. For a model finetuned in this browser the key is
//                `finetuned:<id>` (manifest.ts finetunedBytesKey) and the
//                bytes are the only copy there is.
//   "finetuned": id -> FinetunedRecord (finetuned.ts): what a finetuned
//                model is, so it is listed again on the next visit.
//
// One database for both, with one opener (ModelDatabase) that creates every
// store on upgrade, for the reason state/db.ts gives: two modules opening the
// same database with their own version and stores would race on upgrades.
// The records sit beside the bytes they describe because they are two halves
// of one thing, a model that exists only in this browser; clearing the
// database forgets both halves together. (state/db.ts's database is the
// service worker's too, for the local-directory store; this one is the
// page's own.) Version 2 added the "finetuned" store.
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
/** Version 2 added the "finetuned" store; a database from version 1 gains it on upgrade. */
export const MODEL_DB_VERSION = 2;
export const BYTES_STORE = "bytes";
export const FINETUNED_STORE = "finetuned";
const STORES = [BYTES_STORE, FINETUNED_STORE] as const;
export type ModelStore = (typeof STORES)[number];

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

/** One connection to the models database, opened on first use and kept;
 * when another tab upgrades it the connection is closed and reopened on the
 * next call. Every store is created on upgrade, whichever module opens the
 * database first. */
export class ModelDatabase {
  private db: Promise<IDBDatabase> | null = null;

  constructor(
    private readonly factory: IDBFactory,
    readonly name: string = MODEL_DB_NAME,
  ) {}

  /** One request in its own transaction on `store`; the result once the
   * transaction has committed, so a put is on disk when the promise resolves. */
  async run<T>(store: ModelStore, mode: IDBTransactionMode, request: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.open();
    return new Promise<T>((resolve, reject) => {
      const tx = db.transaction(store, mode);
      const req = request(tx.objectStore(store));
      const fail = () => reject(tx.error ?? req.error ?? new Error(`IndexedDB ${mode} transaction on ${this.name} failed`));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = fail;
      tx.onabort = fail;
    });
  }

  private open(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    const opening = new Promise<IDBDatabase>((resolve, reject) => {
      const request = this.factory.open(this.name, MODEL_DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        for (const store of STORES) if (!db.objectStoreNames.contains(store)) db.createObjectStore(store);
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
}

/** The IndexedDB cache: the "bytes" store of the models database. */
export class IndexedDbByteCache implements ByteCache {
  private readonly db: ModelDatabase;

  constructor(factory: IDBFactory, name: string = MODEL_DB_NAME) {
    this.db = new ModelDatabase(factory, name);
  }

  get(key: string): Promise<ArrayBuffer | undefined> {
    return this.db.run("bytes", "readonly", (store) => store.get(key) as IDBRequest<ArrayBuffer | undefined>);
  }

  async put(key: string, bytes: ArrayBuffer): Promise<void> {
    await this.db.run("bytes", "readwrite", (store) => store.put(bytes, key));
  }

  async delete(key: string): Promise<void> {
    await this.db.run("bytes", "readwrite", (store) => store.delete(key));
  }
}

/** The cache for this runtime: IndexedDB when there is one, else memory, with
 * a log line that says what a missing models database costs (the one note
 * for the database: finetuned.ts's opener is silent). */
export function openByteCache(factory: IDBFactory | undefined, log: (line: string) => void): ByteCache {
  if (factory) return new IndexedDbByteCache(factory);
  log("model cache: no IndexedDB here; model files will be downloaded on every visit, and a model finetuned in this browser is forgotten when the page closes");
  return new MemoryByteCache();
}
