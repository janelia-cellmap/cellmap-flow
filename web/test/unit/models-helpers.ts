// Shared by the models-*.test.ts files: the real manifests and tree listings
// saved from cellmap/onnx-web-models on 2026-10-14 (fixtures/onnx-web-models),
// a stand-in for fetch that answers as the Hub does, so the catalog can be
// exercised without a network, and a tiny in-memory IndexedDB for the models
// database.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { GpuLimits, ModelEntry } from "../../src/contracts";
import type { FetchLike } from "../../src/models/hf";
import { entryFor, parseManifest } from "../../src/models/manifest";
import { chooseVariant } from "../../src/models/variant";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "onnx-web-models");

/** The folders the saved listing has (every model folder of the repo). */
export const FOLDERS = ["setup_53", "setup_54", "setup_55", "setup_56", "setup_61", "setup_67"];

export function readFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(FIXTURES, name), "utf8"));
}

/** `folder`'s manifest.json as fetched (unknown: not yet parsed). */
export function rawManifest(folder: string): unknown {
  return readFixture(`${folder}.manifest.json`);
}

/** The repo's top-level tree listing (the folders, conv3d/ among them, README, bench). */
export function treeListing(): unknown {
  return readFixture("tree.main.json");
}

/** A ModelEntry for `folder`: with `limits`, the variant chooseVariant picks
 * for them; without, the fp32 middle block (132³ for a ResUNet, 140³ for a
 * Unet3dIso), a fixed point for the tests' numbers whatever the chooser's
 * default for unknown limits is. */
export function entryFixture(folder = "setup_54", limits?: GpuLimits): ModelEntry {
  const manifest = parseManifest(rawManifest(folder), `${folder}/manifest.json`);
  if (limits) return entryFor(folder, manifest, chooseVariant(manifest, limits));
  const fp32 = manifest.variants.filter((v) => v.variant === "fp32").sort((a, b) => a.input_size[0]! - b.input_size[0]!);
  return entryFor(folder, manifest, fp32[Math.floor((fp32.length - 1) / 2)]!);
}

export function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}

export function statusResponse(status: number, body = ""): Response {
  return new Response(body, { status });
}

export interface HubOptions {
  /** The status to answer the tree listing with, instead of 200 and the fixture. */
  treeStatus?: number;
  /** Folders whose manifest answers with this status instead of the fixture. */
  manifestStatus?: Record<string, number>;
  /** Bytes to answer a .onnx URL with (by file name), streamed in chunks of `chunkSize`. */
  onnx?: Record<string, Uint8Array>;
  onnxStatus?: number;
  chunkSize?: number;
  /** Leave out Content-Length on the .onnx answer. */
  noContentLength?: boolean;
}

export interface FakeHub {
  fetch: FetchLike;
  /** Every URL asked for, in order. */
  calls: string[];
}

/** A fetch that answers like the Hub, from the fixtures. */
export function fakeHub(options: HubOptions = {}): FakeHub {
  const calls: string[] = [];
  const fetch: FetchLike = async (url) => {
    calls.push(url);
    if (/\/api\/models\/[^/]+\/[^/]+\/tree\/main$/.test(url)) {
      return options.treeStatus && options.treeStatus !== 200 ? statusResponse(options.treeStatus) : jsonResponse(treeListing());
    }
    const manifest = /\/raw\/main\/([^/]+)\/manifest\.json$/.exec(url);
    if (manifest) {
      const folder = manifest[1]!;
      const status = options.manifestStatus?.[folder];
      if (status && status !== 200) return statusResponse(status);
      if (!FOLDERS.includes(folder)) return statusResponse(404, "Entry not found");
      return jsonResponse(rawManifest(folder));
    }
    const onnx = /\/resolve\/main\/[^/]+\/([^/]+\.onnx)$/.exec(url);
    if (onnx) {
      if (options.onnxStatus && options.onnxStatus !== 200) return statusResponse(options.onnxStatus);
      const bytes = options.onnx?.[onnx[1]!];
      if (!bytes) return statusResponse(404, "Entry not found");
      return streamedResponse(bytes, options.chunkSize ?? 7, !options.noContentLength);
    }
    return statusResponse(404, `unexpected URL ${url}`);
  };
  return { fetch, calls };
}

/** `bytes` as a response whose body streams in `chunkSize` pieces. */
export function streamedResponse(bytes: Uint8Array, chunkSize: number, withLength: boolean): Response {
  let offset = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.byteLength) {
        controller.close();
        return;
      }
      controller.enqueue(bytes.slice(offset, offset + chunkSize));
      offset += chunkSize;
    },
  });
  const headers: Record<string, string> = { "content-type": "application/octet-stream" };
  if (withLength) headers["content-length"] = String(bytes.byteLength);
  return new Response(body, { status: 200, headers });
}

/** A log that keeps its lines. */
export function fakeLog(): { log: (line: string) => void; lines: string[] } {
  const lines: string[] = [];
  return { log: (line) => lines.push(line), lines };
}

// ---------------------------------------------------------------------------
// A tiny IndexedDB: open/upgrade, one object store per name, get/put/delete/
// getAll in transactions, callbacks fired asynchronously as the real one
// does. Just enough for ModelDatabase (the byte cache and the finetuned
// registry); not a general implementation. Versions are not modelled: the
// upgrade runs once, when a database is first opened.
// ---------------------------------------------------------------------------

type Callback = ((event: unknown) => void) | null;

class FakeRequest<T> {
  result!: T;
  error: Error | null = null;
  onsuccess: Callback = null;
  onerror: Callback = null;
  onupgradeneeded: Callback = null;
  onblocked: Callback = null;
}

class FakeStore {
  constructor(readonly data: Map<string, unknown>) {}
  get(key: string) {
    return this.request(() => this.data.get(key));
  }
  getAll() {
    return this.request(() => [...this.data.values()]);
  }
  put(value: unknown, key: string) {
    return this.request(() => {
      this.data.set(key, value);
      return key;
    });
  }
  delete(key: string) {
    return this.request(() => {
      this.data.delete(key);
      return undefined;
    });
  }
  private request<T>(run: () => T) {
    const req = new FakeRequest<T>();
    queueMicrotask(() => {
      req.result = run();
      req.onsuccess?.({ target: req });
    });
    return req;
  }
}

class FakeTransaction {
  oncomplete: Callback = null;
  onerror: Callback = null;
  onabort: Callback = null;
  error: Error | null = null;
  constructor(private readonly stores: Map<string, Map<string, unknown>>) {
    // Completes after the requests made in this tick have answered.
    queueMicrotask(() => queueMicrotask(() => this.oncomplete?.({ target: this })));
  }
  objectStore(name: string) {
    const data = this.stores.get(name);
    if (!data) throw new Error(`no object store ${name}`);
    return new FakeStore(data);
  }
}

// One connection; the data (`stores`) belongs to the database and outlives it.
class FakeDatabase {
  onversionchange: Callback = null;
  closed = false;
  constructor(
    readonly name: string,
    readonly stores: Map<string, Map<string, unknown>>,
  ) {}
  get objectStoreNames() {
    return { contains: (name: string) => this.stores.has(name) };
  }
  createObjectStore(name: string) {
    this.stores.set(name, new Map());
  }
  transaction(_name: string) {
    if (this.closed) throw new Error("database is closed");
    return new FakeTransaction(this.stores);
  }
  close() {
    this.closed = true;
  }
}

export interface FakeIndexedDb {
  factory: IDBFactory;
  /** The data of each database by name: store name -> key -> value. */
  databases: Map<string, Map<string, Map<string, unknown>>>;
  /** Every connection handed out, in order. */
  connections: FakeDatabase[];
  opens: () => number;
}

export function fakeIndexedDb(options: { failOpen?: boolean } = {}): FakeIndexedDb {
  const databases = new Map<string, Map<string, Map<string, unknown>>>();
  const connections: FakeDatabase[] = [];
  let opens = 0;
  const factory = {
    open(name: string) {
      opens++;
      const req = new FakeRequest<FakeDatabase>();
      queueMicrotask(() => {
        if (options.failOpen) {
          req.error = new Error("open refused");
          req.onerror?.({ target: req });
          return;
        }
        let stores = databases.get(name);
        const fresh = !stores;
        if (!stores) {
          stores = new Map();
          databases.set(name, stores);
        }
        const db = new FakeDatabase(name, stores);
        connections.push(db);
        req.result = db;
        if (fresh) req.onupgradeneeded?.({ target: req });
        req.onsuccess?.({ target: req });
      });
      return req;
    },
  };
  return { factory: factory as unknown as IDBFactory, databases, connections, opens: () => opens };
}
