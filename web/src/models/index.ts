// The model catalog: the models the browser can run, from the Hugging Face
// repo cellmap/onnx-web-models, one folder per model with a manifest.json
// and its .onnx files (three block sizes times four precisions for a 3D
// model, times two for a 2D one), and after them the models finetuned in
// this browser (finetuned.ts).
//
// list() reads the repo's folder list from the Hub API and each folder's
// manifest, keeps them in localStorage for a day (listing-cache.ts), and
// falls back to the folders known at the time of writing when the Hub will
// not answer (it rate-limits anonymous requests). Each manifest becomes one
// ModelEntry with a default variant (fp32, the smallest block: the choice
// for a GPU nobody has asked yet). Whoever knows the GPU, the inference
// side, calls chooseVariant again with its limits and swaps the variant in.
// Then come the finetuned models' entries, newest first, from their records
// in IndexedDB (the FinetunedRegistry); addFinetuned() and removeFinetuned()
// keep that part of the list in step without another load.
//
// modelBytes() downloads the chosen file with progress and keeps the bytes
// in IndexedDB (byte-cache.ts) under the file's URL and ETag, so the next
// visit does not download again; it asks the Hub for the file's ETag first
// (one HEAD), so a model re-exported under the same name is downloaded
// again, and serves what it has when the Hub cannot be asked. A finetuned
// model's bytes are in that same store or nowhere: there is no download to
// fall back on, and the error says to train it again.
import type { Catalog, FinetunedRecord, GpuLimits, ModelEntry, ModelManifest, ModelVariant } from "../contracts";

import { type ByteCache, modelBytesKey, openByteCache } from "./byte-cache";
import { type FinetunedRegistry, isFinetuned, newestFirst, openFinetunedRegistry } from "./finetuned";
import { type FetchLike, fetchEtag, fetchFile, fetchManifest, fetchTree, HF_REPO, HF_REVISION, RateLimitError } from "./hf";
import { defaultKeyValueStore, type KeyValueStore, ListingCache } from "./listing-cache";
import { chooseVariant, UNKNOWN_LIMITS } from "./variant";
import { entryFor, FALLBACK_FOLDERS, finetunedBytesKey, modelFolders, parseManifest } from "./manifest";

export { chooseVariant, estimatePeakBytes, isOneSlice, PREFER_SMALL_DOWNLOAD, PREFERRED_2D_SIDE, UNKNOWN_LIMITS } from "./variant";
export { entryFor, entryName, FALLBACK_FOLDERS, finetunedBytesKey, ManifestError, parseManifest } from "./manifest";
export { FINETUNED_AUTHOR, metadataFor, MODELS_AUTHOR } from "./metadata";
export { fetchEtag, HF_REPO, HF_REVISION, RateLimitError, resolveUrl } from "./hf";
export { IndexedDbByteCache, MemoryByteCache, MODEL_DB_NAME, MODEL_DB_VERSION, ModelDatabase, modelBytesKey, type ByteCache } from "./byte-cache";
export { type KeyValueStore, LISTING_TTL_MS, MemoryKeyValueStore } from "./listing-cache";
export {
  finetunedEntry,
  type FinetunedRegistry,
  IndexedDbFinetunedRegistry,
  isFinetuned,
  MemoryFinetunedRegistry,
  newestFirst,
  nextFinetunedName,
  openFinetunedRegistry,
} from "./finetuned";

import { metadataFor } from "./metadata";

/** The localStorage key of the cached listing. */
export const LISTING_KEY = `cellmap-flow:models:${HF_REPO}@${HF_REVISION}:v1`;

export interface CatalogOptions {
  log: (line: string) => void;
  // The rest are seams for tests; the browser's own are used when absent.
  fetch?: FetchLike;
  /** Where the listing is cached (localStorage). */
  storage?: KeyValueStore;
  /** Where the .onnx bytes are cached (IndexedDB). */
  bytes?: ByteCache;
  /** Where the records of the models finetuned in this browser are kept (IndexedDB). */
  finetuned?: FinetunedRegistry;
  now?: () => number;
}

/** The Catalog of contracts.ts, plus a way to refetch the listing (the Models tab's Refresh button). */
export interface HfCatalog extends Catalog {
  /** Fetch the listing again, ignoring the cache, and make it the one list() answers. */
  refresh(): Promise<ModelEntry[]>;
}

export function createCatalog(opts: CatalogOptions): HfCatalog {
  const log = opts.log;
  const fetchImpl: FetchLike = opts.fetch ?? ((input, init) => globalThis.fetch(input, init));
  const listing = new ListingCache(opts.storage ?? defaultKeyValueStore(), LISTING_KEY);
  const bytes = opts.bytes ?? openByteCache(globalThis.indexedDB, log);
  const registry = opts.finetuned ?? openFinetunedRegistry(globalThis.indexedDB);
  const now = opts.now ?? Date.now;

  // The list as last built, the Hub's entries then the finetuned ones, for
  // the page's lifetime; null until a load has succeeded.
  let listed: ModelEntry[] | null = null;
  // A load under way, shared by everyone asking meanwhile; dropped when it
  // settles, so after a failure the next list() tries again (the Hub may
  // have stopped rate-limiting).
  let loading: Promise<ModelEntry[]> | null = null;
  // The registry's records as last read (null until then), newest first,
  // kept in step by addFinetuned and removeFinetuned: the finetuned part of
  // the list, and what metadataFor describes a finetuned entry from.
  let records: FinetunedRecord[] | null = null;
  // Downloads under way, by URL, so two callers of one model share a download.
  const downloads = new Map<string, Promise<ArrayBuffer>>();

  function list(): Promise<ModelEntry[]> {
    if (listed) return Promise.resolve(listed);
    loading ??= track(load(false));
    return loading;
  }

  function refresh(): Promise<ModelEntry[]> {
    const refreshed = track(load(true));
    // list() answers the old list until the new one is in, and keeps the old
    // one when the refresh fails; with no list yet, it shares this load.
    if (!listed) loading = refreshed;
    return refreshed;
  }

  // A load's result becomes the list; the load itself is forgotten either way.
  function track(result: Promise<ModelEntry[]>): Promise<ModelEntry[]> {
    const tracked: Promise<ModelEntry[]> = result
      .then((list) => {
        listed = list;
        return list;
      })
      .finally(() => {
        if (loading === tracked) loading = null;
      });
    return tracked;
  }

  // The whole list: the Hub's models, then the finetuned ones. A Hub listing
  // that fails outright fails the list, as before, so the Models tab shows
  // why; a registry that cannot be read only costs the finetuned entries.
  async function load(force: boolean): Promise<ModelEntry[]> {
    const hub = await loadHub(force);
    const kept = await readFinetuned();
    return [...hub, ...kept.map((r) => r.entry)];
  }

  // The Hub listing: from the cache while fresh, else from the Hub, with the
  // cache (even stale) and the built-in folder list as fallbacks.
  async function loadHub(force: boolean): Promise<ModelEntry[]> {
    const cached = listing.read();
    if (!force && cached && listing.isFresh(cached, now())) return build(cached.manifests, "the browser's cache");

    let folders: string[];
    try {
      folders = modelFolders(await fetchTree(fetchImpl));
      if (folders.length === 0) throw new Error("the listing names no model folder");
    } catch (error) {
      folders = FALLBACK_FOLDERS.slice();
      log(`model catalog: could not list ${HF_REPO} (${message(error)}); using the built-in list of ${folders.length} folders`);
    }

    const results = await Promise.allSettled(folders.map((folder) => fetchManifest(fetchImpl, folder)));
    const manifests: Record<string, unknown> = {};
    const failures: string[] = [];
    results.forEach((result, i) => {
      const folder = folders[i]!;
      if (result.status === "fulfilled") manifests[folder] = result.value;
      else failures.push(`${folder}: ${message(result.reason)}`);
    });
    for (const failure of failures) log(`model catalog: ${failure}`);

    if (Object.keys(manifests).length === 0) {
      if (cached) {
        log(`model catalog: no manifest could be fetched; showing the list cached on ${new Date(cached.fetchedAt).toISOString()}`);
        return build(cached.manifests, "the browser's cache (stale)");
      }
      const rateLimited = results.some((r) => r.status === "rejected" && r.reason instanceof RateLimitError);
      throw new Error(
        rateLimited
          ? `Hugging Face rate-limited the anonymous requests for ${HF_REPO}'s manifests (HTTP 429). Please retry in a minute.`
          : `Could not fetch any model manifest from ${HF_REPO}: ${failures[0] ?? "no folders"}`,
      );
    }
    // A folder whose manifest failed keeps its cached one; the listing is
    // cached as fresh only when every folder answered, so a gap heals on the
    // next load rather than in a day.
    if (failures.length > 0 && cached) {
      for (const [folder, manifest] of Object.entries(cached.manifests)) manifests[folder] ??= manifest;
    }
    if (failures.length === 0 && !listing.write({ fetchedAt: now(), manifests })) {
      log("model catalog: could not cache the model list (storage blocked or full)");
    }
    return build(manifests, "Hugging Face");
  }

  // Entries from raw manifests, folder order; a manifest the parser rejects is logged and left out.
  function build(manifests: Record<string, unknown>, source: string): ModelEntry[] {
    const built: ModelEntry[] = [];
    for (const folder of Object.keys(manifests).sort()) {
      try {
        const manifest = parseManifest(manifests[folder], `${folder}/manifest.json`);
        built.push(entryFor(folder, manifest, chooseVariant(manifest, UNKNOWN_LIMITS)));
      } catch (error) {
        log(`model catalog: skipping ${folder}: ${message(error)}`);
      }
    }
    log(`model catalog: ${built.length} models from ${source}`);
    return built;
  }

  // The registry's records into `records`, newest first. A registry that
  // cannot be read keeps the last listing (none the first time) and says
  // so; the Hub's models are still there to serve, and a refresh reads again.
  async function readFinetuned(): Promise<FinetunedRecord[]> {
    try {
      records = await registry.list();
      if (records.length > 0) log(`model catalog: ${records.length} ${records.length === 1 ? "model" : "models"} finetuned in this browser`);
    } catch (error) {
      log(`model catalog: could not read the models finetuned in this browser (${message(error)})`);
      records ??= [];
    }
    return records;
  }

  // Waits for a load under way, so that a change to the records lands after
  // its result rather than being buried by it.
  async function settled(): Promise<void> {
    if (loading) await loading.catch(() => {});
  }

  // `list` with its finetuned part replaced by `records`' entries.
  function withFinetuned(list: ModelEntry[], kept: FinetunedRecord[]): ModelEntry[] {
    return [...list.filter((e) => !isFinetuned(e)), ...kept.map((r) => r.entry)];
  }

  /** The entry `ref` names: its id ("cellmap/onnx-web-models/setup_54",
   * which is also repo/folder, or "finetuned/mito_aff_setup54_16nm-ft1"),
   * its folder ("setup_54") or its name ("mito_aff_setup54_16nm"). */
  async function resolve(ref: string): Promise<ModelEntry | null> {
    const key = ref.trim().replace(/\/+$/, "");
    if (!key) return null;
    const all = await list();
    return all.find((e) => e.id === key || e.folder === key || e.name === key) ?? null;
  }

  function modelBytes(entry: ModelEntry, onProgress?: Progress): Promise<ArrayBuffer> {
    if (isFinetuned(entry)) return finetunedBytes(entry, onProgress);
    const url = entry.onnxUrl;
    const underway = downloads.get(url);
    // A second caller gets the same bytes; the first caller's progress callback is the one that runs.
    if (underway) return underway;
    const download = hubBytes(entry, onProgress).finally(() => downloads.delete(url));
    downloads.set(url, download);
    return download;
  }

  // A Hugging Face model's bytes: the cached copy when it is the file as the
  // Hub has it now, else a download.
  //
  // The Hub is asked for the file's ETag first (fetchEtag, one HEAD) and the
  // cache is read under `<url>#<etag>` (modelBytesKey): a model is
  // re-exported under the same file name when it is retrained, so a copy
  // keyed by URL alone could be the old weights. A miss downloads, keeps the
  // bytes under the ETag of the download (the HEAD's when the GET exposes
  // none) and deletes the URL's other copies: an older ETag's, and the bare
  // URL's from before the ETag was in the key. When the Hub cannot be asked
  // (offline, or a server that exposes no ETag) the copy the cache holds for
  // the URL is served, whichever key it is under, with a line saying it was
  // not checked; with none, the download goes ahead as it always did.
  async function hubBytes(entry: ModelEntry, onProgress?: Progress): Promise<ArrayBuffer> {
    const url = entry.onnxUrl;
    const file = entry.variant.file;
    const etag = await fetchEtag(fetchImpl, url);
    const wanted = modelBytesKey(url, etag);
    // The cache's copies of this file: under `<url>#<etag>` keys, and the bare URL.
    let keyed: string[] = [];
    let bare = false;
    let cached: ArrayBuffer | undefined;
    try {
      const held = await bytes.keys(url);
      keyed = held.filter((key) => key.startsWith(`${url}#`));
      bare = held.includes(url);
      const serve = etag !== null ? (keyed.includes(wanted) ? wanted : null) : (keyed[0] ?? (bare ? url : null));
      if (serve !== null) cached = await bytes.get(serve);
    } catch (error) {
      log(`${entry.name}: could not read the model cache (${message(error)})`);
    }
    if (cached) {
      const unchecked = etag === null ? " (could not check Hugging Face for a newer file)" : "";
      log(`${entry.name}: ${mb(cached.byteLength)} MB from the browser's cache${unchecked}`);
      onProgress?.(cached.byteLength, cached.byteLength);
      return cached;
    }
    const size = `${entry.variant.file_mb.toFixed(1)} MB`;
    if (etag !== null && keyed.length > 0) log(`${entry.name}: ${file} updated on Hugging Face (new ${etag.slice(0, 8)}…); downloading again`);
    else if (etag !== null && bare) log(`${entry.name}: the cached ${file} predates the version check; downloading it again (${size}) from Hugging Face`);
    else log(`${entry.name}: downloading ${file} (${size}) from Hugging Face`);
    const downloaded = await fetchFile(fetchImpl, url, onProgress);
    log(`${entry.name}: downloaded ${mb(downloaded.bytes.byteLength)} MB`);
    const key = modelBytesKey(url, downloaded.etag ?? etag);
    try {
      await bytes.put(key, downloaded.bytes);
      await bytes.deleteWhere(`${url}#`, key);
      if (key !== url) await bytes.delete(url);
    } catch (error) {
      log(`${entry.name}: could not cache the model (${message(error)}); it will be downloaded again next time`);
    }
    return downloaded.bytes;
  }

  // A finetuned model's bytes: in the byte cache under its onnxUrl, or gone
  // (site data cleared, or this is another browser or profile than the one
  // that trained it). Never a download: nothing on the network has them.
  async function finetunedBytes(entry: ModelEntry, onProgress?: Progress): Promise<ArrayBuffer> {
    let kept: ArrayBuffer | undefined;
    try {
      kept = await bytes.get(entry.onnxUrl);
    } catch (error) {
      throw new Error(`${entry.name} was finetuned in this browser and its storage could not be read (${message(error)}).`);
    }
    if (!kept) {
      throw new Error(
        `${entry.name} was finetuned in this browser and its bytes are no longer in the browser's storage (site data cleared, or another browser or profile). Train it again to serve it.`,
      );
    }
    log(`${entry.name}: ${mb(kept.byteLength)} MB from the browser's storage`);
    onProgress?.(kept.byteLength, kept.byteLength);
    return kept;
  }

  function chooseFor(manifest: ModelManifest, limits: GpuLimits): ModelVariant {
    return chooseVariant(manifest, limits);
  }

  /** The metadata of `entry`, a finetuned entry's with its record (where it came from). */
  function metadataOf(entry: ModelEntry): Record<string, unknown> {
    return metadataFor(entry, isFinetuned(entry) ? records?.find((r) => r.id === entry.id) : undefined);
  }

  /** Keep a finetuned model: its bytes first, then its record, so a record
   * never points at bytes that were not written, and a record that cannot
   * be written takes the bytes with it. Then the list shows it at once,
   * newest first among the finetuned entries. */
  async function addFinetuned(record: FinetunedRecord, data: ArrayBuffer): Promise<void> {
    const entry = record.entry;
    if (!isFinetuned(entry) || entry.id !== record.id) {
      throw new Error(`${record.name}: the record's entry is not a model finetuned in this browser (repo ${entry.repo}, id ${entry.id})`);
    }
    await bytes.put(entry.onnxUrl, data);
    try {
      await registry.put(record);
    } catch (error) {
      await bytes.delete(entry.onnxUrl).catch(() => {});
      throw error;
    }
    await settled();
    const current = records ?? (await readFinetuned());
    records = [record, ...current.filter((r) => r.id !== record.id)].sort(newestFirst);
    if (listed) listed = withFinetuned(listed, records);
  }

  /** The finetuned models kept, newest first (read from the registry the first time). */
  async function finetuned(): Promise<FinetunedRecord[]> {
    return (records ?? (await readFinetuned())).slice();
  }

  /** Forget a finetuned model: its record first, so it leaves the list even
   * if its bytes cannot be deleted (then they are logged and left), then its
   * bytes. An id no finetuned model has is nothing to forget. */
  async function removeFinetuned(id: string): Promise<void> {
    const key = records?.find((r) => r.id === id)?.entry.onnxUrl ?? finetunedBytesKey(id);
    await registry.remove(id);
    try {
      await bytes.delete(key);
    } catch (error) {
      log(`model catalog: could not delete the bytes of ${id} (${message(error)})`);
    }
    await settled();
    if (records) records = records.filter((r) => r.id !== id);
    if (listed) listed = listed.filter((e) => e.id !== id);
  }

  return { list, refresh, resolve, metadataFor: metadataOf, chooseVariant: chooseFor, modelBytes, addFinetuned, finetuned, removeFinetuned };
}

/** modelBytes's progress callback: bytes so far, and the total when the server said (null otherwise). */
type Progress = (loaded: number, total: number | null) => void;

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function mb(byteLength: number): string {
  return (byteLength / 1e6).toFixed(1);
}
