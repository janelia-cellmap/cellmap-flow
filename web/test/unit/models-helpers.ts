// Shared by the models-*.test.ts files: the real manifests and tree listings
// saved from cellmap/onnx-web-models on 2026-10-14 and -15 (fixtures/onnx-web-models),
// a stand-in for fetch that answers as the Hub does (other repos too, from
// the manifests a test gives), so the catalog can be exercised without a
// network, and a tiny in-memory IndexedDB for the models database.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { GpuLimits, ModelEntry } from "../../src/contracts";
import { finetunedEntry } from "../../src/models/finetuned";
import type { FetchLike } from "../../src/models/hf";
import { entryFor, parseManifest } from "../../src/models/manifest";
import { chooseVariant } from "../../src/models/variant";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "onnx-web-models");

/** The folders the saved listing has (every model folder of the repo). */
export const FOLDERS = ["setup_53", "setup_53_2d", "setup_54", "setup_54_2d", "setup_55", "setup_56", "setup_61", "setup_67"];

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

/** The manifest a model finetuned from `base` as `name` is published with
 * (the spec's layout): its entry's manifest, one variant `<name>.onnx`
 * whose source is the base's file, and a published block; as JSON, as the
 * Hub serves it. */
export function publishedManifest(base: ModelEntry, name: string): unknown {
  const { manifest } = finetunedEntry(base, name);
  const published = { name, finetuned_from: base.id, finetune_mode: "head", epochs: 20, final_loss: 0.031, published_at: "2026-10-26T12:00:00.000Z" };
  return JSON.parse(JSON.stringify({ ...manifest, published }));
}

/** Another repo than cellmap/onnx-web-models as the fake Hub serves it: a
 * folder per manifest, `dirs` more folders without one, a README and
 * `files` more files at its root; or `status` for every request of it
 * (401: there is none, or it is private). */
export interface FakeRepo {
  manifests?: Record<string, unknown>;
  dirs?: string[];
  files?: string[];
  status?: number;
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
  /** The status every .onnx request answers with, HEAD and GET alike (the file is gone, or rate-limited). */
  onnxStatus?: number;
  /** The status a GET of a .onnx file answers with; its HEAD still answers 200 (the file is there, the download refused). */
  getStatus?: number;
  chunkSize?: number;
  /** Leave out Content-Length on the .onnx answer. */
  noContentLength?: boolean;
  /** The ETag a .onnx file's answers (HEAD and GET) carry, by file name, instead of fakeEtag of its bytes; null for none. */
  etag?: Record<string, string | null>;
  /** Leave the ETag off every .onnx answer: a server that exposes none. */
  noEtag?: boolean;
  /** Leave the ETag off the GET answers only; the HEAD's still names the file. */
  noGetEtag?: boolean;
  /** Every HEAD fails as fetch does offline (a TypeError); GETs are unaffected. */
  headFails?: boolean;
  /** Other repos, by id; their .onnx files are `onnx`'s, by name, like the built-in repo's. */
  repos?: Record<string, FakeRepo>;
}

export interface FakeHub {
  fetch: FetchLike;
  /** Every request, in order: a GET as its URL, any other method as "<METHOD> <url>" ("HEAD https://…"). */
  calls: string[];
}

/** A fetch that answers like the Hub, from the fixtures. A .onnx URL answers
 * a HEAD with the file's length and ETag and a GET with its bytes, the ETag
 * on both, as Hugging Face's CDN does (hf.ts fetchEtag). */
export function fakeHub(options: HubOptions = {}): FakeHub {
  const calls: string[] = [];
  const fetch: FetchLike = async (url, init) => {
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push(method === "GET" ? url : `${method} ${url}`);
    if (method === "HEAD" && options.headFails) throw new TypeError("Failed to fetch");
    const repo = options.repos?.[/^https:\/\/huggingface\.co\/(?:api\/models\/)?([^/]+\/[^/]+)\//.exec(url)?.[1] ?? ""];
    if (repo?.status) return statusResponse(repo.status);
    if (repo && url.endsWith("/tree/main")) {
      const folders = [...Object.keys(repo.manifests ?? {}), ...(repo.dirs ?? [])];
      const files = ["README.md", ...(repo.files ?? [])];
      return jsonResponse([...folders.map((path) => ({ type: "directory", path })), ...files.map((path) => ({ type: "file", path, size: 900 }))]);
    }
    const repoManifest = repo && /\/raw\/main\/([^/]+)\/manifest\.json$/.exec(url);
    if (repoManifest) return repoManifest[1]! in (repo.manifests ?? {}) ? jsonResponse(repo.manifests![repoManifest[1]!]) : statusResponse(404, "Entry not found");
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
      const file = onnx[1]!;
      const bytes = options.onnx?.[file];
      if (!bytes) return statusResponse(404, "Entry not found");
      const etag = options.noEtag ? null : options.etag && file in options.etag ? options.etag[file]! : fakeEtag(bytes);
      if (method === "HEAD") return headResponse(bytes.byteLength, etag);
      if (options.getStatus && options.getStatus !== 200) return statusResponse(options.getStatus);
      return streamedResponse(bytes, options.chunkSize ?? 7, !options.noContentLength, options.noGetEtag ? null : etag);
    }
    return statusResponse(404, `unexpected URL ${url}`);
  };
  return { fetch, calls };
}

/** A content hash of `bytes` standing in for the CDN's ETag, 16 hex digits
 * (two FNV-1a passes): the same bytes give the same tag, a changed byte
 * another, so a test that changes a model's bytes changes its ETag with
 * them, as a re-export on the Hub does. */
export function fakeEtag(bytes: Uint8Array): string {
  const fnv = (seed: number) => {
    let h = seed >>> 0;
    for (const b of bytes) h = Math.imul(h ^ b, 16777619) >>> 0;
    return h.toString(16).padStart(8, "0");
  };
  return fnv(2166136261) + fnv(0x9747b28c);
}

/** A HEAD's answer: no body, the length, and the ETag quoted as servers send it. */
export function headResponse(byteLength: number, etag: string | null): Response {
  const headers: Record<string, string> = { "content-type": "application/octet-stream", "content-length": String(byteLength) };
  if (etag !== null) headers.etag = `"${etag}"`;
  return new Response(null, { status: 200, headers });
}

/** `bytes` as a response whose body streams in `chunkSize` pieces, with the ETag (quoted) when given. */
export function streamedResponse(bytes: Uint8Array, chunkSize: number, withLength: boolean, etag: string | null = null): Response {
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
  if (etag !== null) headers.etag = `"${etag}"`;
  return new Response(body, { status: 200, headers });
}

/** A log that keeps its lines. */
export function fakeLog(): { log: (line: string) => void; lines: string[] } {
  const lines: string[] = [];
  return { log: (line) => lines.push(line), lines };
}

// ---------------------------------------------------------------------------
// A tiny IndexedDB: open/upgrade, one object store per name, get/put/delete/
// getAll/getAllKeys in transactions, callbacks fired asynchronously as the
// real one does. Just enough for ModelDatabase (the byte cache and the
// finetuned registry); not a general implementation. Versions are not
// modelled: the upgrade runs once, when a database is first opened. vitest
// runs in node, which has no IDBKeyRange: fakeIndexedDb() installs a
// stand-in with the one method the models database uses, `bound`.
// ---------------------------------------------------------------------------

interface FakeKeyRange {
  lower: string;
  upper: string;
  lowerOpen: boolean;
  upperOpen: boolean;
}

function installKeyRange(): void {
  if (typeof globalThis.IDBKeyRange !== "undefined") return;
  (globalThis as { IDBKeyRange?: unknown }).IDBKeyRange = {
    bound: (lower: string, upper: string, lowerOpen = false, upperOpen = false): FakeKeyRange => ({ lower, upper, lowerOpen, upperOpen }),
  };
}

// Strings compare by UTF-16 code unit, IndexedDB's order for them.
function inRange(key: string, range?: FakeKeyRange): boolean {
  if (!range) return true;
  const aboveLower = range.lowerOpen ? key > range.lower : key >= range.lower;
  const belowUpper = range.upperOpen ? key < range.upper : key <= range.upper;
  return aboveLower && belowUpper;
}

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
  getAllKeys(range?: FakeKeyRange) {
    return this.request(() => [...this.data.keys()].sort().filter((key) => inRange(key, range)));
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
  installKeyRange();
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
