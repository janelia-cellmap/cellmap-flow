// The model catalog: the models the browser can run, from the Hugging Face
// repo cellmap/onnx-web-models, one folder per model with a manifest.json
// and its .onnx files (three block sizes times four precisions for a 3D
// model, times two for a 2D one), then from the repos added beside it (any
// public model repo laid out the same way, such as one a model was
// published to from a browser), and after them the models finetuned in this
// browser (finetuned.ts).
//
// list() reads the repo's folder list from the Hub API and each folder's
// manifest, keeps them in localStorage for a day (listing-cache.ts), and
// falls back to the folders known at the time of writing when the Hub will
// not answer (it rate-limits anonymous requests). Each manifest becomes one
// ModelEntry with a default variant (fp32, the smallest block: the choice
// for a GPU nobody has asked yet). Whoever knows the GPU, the inference
// side, calls chooseVariant again with its limits and swaps the variant in.
// The added repos are listed the same way, each cached for a day under its
// own key, without the built-in folder list to fall back on: one that cannot
// be listed costs only its own entries. A repo is added by resolving a ref
// of it (the Models tab's Add a model, a link): resolve() lists it on the
// spot and remembers it (AddedRepos), so the next visit lists it too.
// Then come the finetuned models' entries, newest first, from their records
// in IndexedDB (the FinetunedRegistry); addFinetuned() and removeFinetuned()
// keep that part of the list in step without another load.
//
// The added repos come after the built-in catalog and before the finetuned
// models: the list reads as the models on Hugging Face, cellmap's first and
// then the others in the order they were added, and then the ones that
// exist only in this browser, the part addFinetuned and removeFinetuned
// change. Every name in it is unique (withUniqueNames): a name is a layer's
// and a served view's, and the dashboard and links name models by it.
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
import { bareRef, type FetchLike, fetchEtag, fetchFile, fetchManifest, fetchTree, HF_REPO, HF_REVISION, hubRef, isMissing, RateLimitError } from "./hf";
import { AddedRepos, defaultKeyValueStore, type KeyValueStore, ListingCache } from "./listing-cache";
import { chooseVariant, UNKNOWN_LIMITS } from "./variant";
import { entryFor, FALLBACK_FOLDERS, finetunedBytesKey, freeName, hubName, modelFolders, parseManifest } from "./manifest";

export { chooseVariant, estimatePeakBytes, isOneSlice, PREFER_SMALL_DOWNLOAD, PREFERRED_2D_SIDE, UNKNOWN_LIMITS } from "./variant";
export { entryFor, entryName, FALLBACK_FOLDERS, finetunedBytesKey, hubName, isLayerName, ManifestError, parseManifest, withVariant } from "./manifest";
export { FINETUNED_AUTHOR, metadataFor, MODELS_AUTHOR } from "./metadata";
export { fetchEtag, HF_REPO, HF_REVISION, hubRef, RateLimitError, resolveUrl } from "./hf";
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

/** The localStorage key of `repo`'s cached listing. v2 since the manifests
 * gained their finetune block (2026-10-16): a v1 listing, cached for a day,
 * holds manifests from before it, which would offer the old training only. */
export function listingKey(repo: string): string {
  return `cellmap-flow:models:${repo}@${HF_REVISION}:v2`;
}

/** The built-in catalog's. */
export const LISTING_KEY = listingKey(HF_REPO);

/** The localStorage key of the repos added beside the built-in catalog (AddedRepos). */
export const ADDED_REPOS_KEY = "cellmap-flow:models:added-repos:v1";

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
  const storage = opts.storage ?? defaultKeyValueStore();
  const listing = new ListingCache(storage, LISTING_KEY);
  const added = new AddedRepos(storage, ADDED_REPOS_KEY);
  const bytes = opts.bytes ?? openByteCache(globalThis.indexedDB, log);
  const registry = opts.finetuned ?? openFinetunedRegistry(globalThis.indexedDB);
  const now = opts.now ?? Date.now;

  // The list as last built, the Hub's entries (the built-in catalog's, then
  // the added repos') then the finetuned ones, for the page's lifetime; null
  // until a load has succeeded.
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
  // Repos being listed for resolve(), by id, so two refs of one repo share a listing.
  const adding = new Map<string, Promise<ModelEntry[]>>();

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
  // why; an added repo that cannot be listed, or a registry that cannot be
  // read, only costs its own entries.
  async function load(force: boolean): Promise<ModelEntry[]> {
    const [hub, others] = await Promise.all([loadHub(force), loadAdded(force)]);
    const kept = await readFinetuned();
    return arranged([...hub, ...others], kept);
  }

  // The Hub listing: from the cache while fresh, else from the Hub, with the
  // cache (even stale) and the built-in folder list as fallbacks.
  async function loadHub(force: boolean): Promise<ModelEntry[]> {
    const cached = listing.read();
    if (!force && cached && listing.isFresh(cached, now())) return build(HF_REPO, cached.manifests, "the browser's cache");

    let folders: string[];
    try {
      folders = modelFolders(await fetchTree(fetchImpl, HF_REPO));
      if (folders.length === 0) throw new Error("the listing names no model folder");
    } catch (error) {
      folders = FALLBACK_FOLDERS.slice();
      log(`model catalog: could not list ${HF_REPO} (${message(error)}); using the built-in list of ${folders.length} folders`);
    }

    const results = await Promise.allSettled(folders.map((folder) => fetchManifest(fetchImpl, HF_REPO, folder)));
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
        return build(HF_REPO, cached.manifests, "the browser's cache (stale)");
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
    return build(HF_REPO, manifests, "Hugging Face");
  }

  // The added repos' entries, in the order they were added. A repo that
  // cannot be listed is logged and tried again on the next load; one the
  // Hub says is not there to read (deleted, or made private) is forgotten.
  async function loadAdded(force: boolean): Promise<ModelEntry[]> {
    const lists = await Promise.all(
      added.list().map(async (repo) => {
        try {
          return await loadRepo(repo, force);
        } catch (error) {
          if (isMissing(error)) {
            added.remove(repo);
            new ListingCache(storage, listingKey(repo)).clear();
            log(`model catalog: ${message(error)}; ${repo} is no longer listed`);
          } else {
            log(`model catalog: could not list ${repo} (${message(error)})`);
          }
          return [];
        }
      }),
    );
    return lists.flat();
  }

  // An added repo's entries: from its cache while fresh, else from the Hub,
  // with the cache (even stale) to fall back on unless the repo is gone.
  // Only a folder holding a manifest.json is a model's (a 404 for it is a
  // folder of something else); any other failure fails the repo's listing,
  // which is cached only whole. A repo with no model the browser can run is
  // an error that says why.
  async function loadRepo(repo: string, force: boolean): Promise<ModelEntry[]> {
    const cache = new ListingCache(storage, listingKey(repo));
    const cached = cache.read();
    let manifests: Record<string, unknown>;
    let source: string;
    if (!force && cached && cache.isFresh(cached, now())) {
      manifests = cached.manifests;
      source = "the browser's cache";
    } else {
      try {
        manifests = await fetchRepo(repo);
        source = "Hugging Face";
        if (!cache.write({ fetchedAt: now(), manifests })) log(`model catalog: could not cache the model list of ${repo} (storage blocked or full)`);
      } catch (error) {
        if (!cached || isMissing(error)) throw error;
        log(`model catalog: could not list ${repo} (${message(error)}); showing the list cached on ${new Date(cached.fetchedAt).toISOString()}`);
        manifests = cached.manifests;
        source = "the browser's cache (stale)";
      }
    }
    const skipped: string[] = [];
    const built = build(repo, manifests, source, skipped);
    if (built.length === 0) {
      throw new Error(skipped.length ? `${repo} has no model the browser can run: ${skipped.join("; ")}` : `${repo} has no model folder (a top-level folder with a manifest.json and the .onnx files it names)`);
    }
    return built;
  }

  // `repo`'s raw manifests by folder, from the Hub.
  async function fetchRepo(repo: string): Promise<Record<string, unknown>> {
    const folders = modelFolders(await fetchTree(fetchImpl, repo));
    const results = await Promise.allSettled(folders.map((folder) => fetchManifest(fetchImpl, repo, folder)));
    const manifests: Record<string, unknown> = {};
    results.forEach((result, i) => {
      if (result.status === "fulfilled") manifests[folders[i]!] = result.value;
      else if (!isMissing(result.reason)) throw result.reason;
    });
    return manifests;
  }

  // Entries from raw manifests, folder order; a manifest the parser rejects
  // is logged, left out and said in `skipped`.
  function build(repo: string, manifests: Record<string, unknown>, source: string, skipped: string[] = []): ModelEntry[] {
    const built: ModelEntry[] = [];
    for (const folder of Object.keys(manifests).sort()) {
      try {
        const manifest = parseManifest(manifests[folder], `${folder}/manifest.json`);
        built.push(entryFor(folder, manifest, chooseVariant(manifest, UNKNOWN_LIMITS), repo));
      } catch (error) {
        log(`model catalog: skipping ${repo === HF_REPO ? "" : `${repo}/`}${folder}: ${message(error)}`);
        skipped.push(message(error));
      }
    }
    log(`model catalog: ${built.length} models ${repo === HF_REPO ? "" : `of ${repo} `}from ${source}`);
    return built;
  }

  // `repo` listed afresh for resolve() and remembered, its entries put in
  // the list where its own were, or after the other repos' (added last):
  // the first ref of a repo adds it, and a ref of a folder the list does
  // not have looks again, in case the folder is newer than the cached
  // listing (a second model published to the repo).
  function addRepo(repo: string): Promise<ModelEntry[]> {
    let pending = adding.get(repo);
    if (!pending) {
      pending = (async () => {
        const entries = await loadRepo(repo, true);
        if (!added.add(repo)) log(`model catalog: could not remember ${repo} (storage blocked or full); it is listed until the page is closed`);
        await settled();
        const hub = (listed ?? []).filter((e) => !isFinetuned(e));
        const at = hub.findIndex((e) => e.repo === repo);
        const others = hub.filter((e) => e.repo !== repo);
        others.splice(at === -1 ? others.length : at, 0, ...entries);
        listed = arranged(others, records ?? []);
        return ofRepo(listed, repo);
      })().finally(() => adding.delete(repo));
      adding.set(repo, pending);
    }
    return pending;
  }

  // `list`'s entries of `repo`: the Hub's, and those of the models kept here
  // that were published to it, which stand for theirs (arranged).
  function ofRepo(list: ModelEntry[], repo: string): ModelEntry[] {
    return list.filter((e) => e.repo === repo || records?.some((r) => r.id === e.id && r.published?.repo === repo));
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
    return arranged(list.filter((e) => !isFinetuned(e)), kept);
  }

  /** The entry `ref` names: its id ("cellmap/onnx-web-models/setup_54",
   * which is also repo/folder, or "finetuned/mito_aff_setup54_16nm-ft1"),
   * a folder of the built-in catalog ("setup_54"), its name
   * ("mito_aff_setup54_16nm"), or any other form hubRef reads (a prefix, a
   * huggingface.co URL). A ref of a repo the list does not have adds the
   * repo (addRepo); `<owner>/<repo>` alone names the repo's one model, and
   * is refused with the refs of its models when it has several. The place
   * a model kept here was published to names that model, whose bytes are
   * here already. */
  async function resolve(ref: string): Promise<ModelEntry | null> {
    const key = bareRef(ref);
    if (!key) return null;
    const all = await list();
    const target = hubRef(key);
    const id = target?.folder ? `${target.repo}/${target.folder}` : key;
    const known = all.find((e) => e.id === id || e.name === key || (e.repo === HF_REPO && e.folder === key));
    if (known) return known;
    if (!target) return null;
    const published = records?.find((r) => r.published?.repo === target.repo && r.published.folder === target.folder);
    if (published) return published.entry;
    let entries = ofRepo(all, target.repo);
    const missing = target.folder !== null && !entries.some((e) => e.folder === target.folder);
    if (target.repo !== HF_REPO && (entries.length === 0 || missing)) entries = await addRepo(target.repo);
    const refs = entries.map((e) => e.id).join(", ");
    if (target.folder === null) {
      if (entries.length === 1) return entries[0]!;
      throw new Error(`${target.repo} holds ${entries.length} models, one a folder: add one as ${target.repo}/<folder> (${refs})`);
    }
    const entry = entries.find((e) => e.folder === target.folder);
    if (!entry) throw new Error(`${target.repo} has no model folder ${target.folder} (a folder with a manifest.json); it has ${refs}`);
    return entry;
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
    // The cache's copies of this file: under `<url>#<etag>` keys, and the bare URL.
    let keyed: string[] = [];
    let bare = false;
    try {
      const held = await bytes.keys(url);
      keyed = held.filter((key) => key.startsWith(`${url}#`));
      bare = held.includes(url);
    } catch (error) {
      log(`${entry.name}: could not read the model cache (${message(error)})`);
    }
    // The ETag is asked for only when there is a copy to check it against:
    // a first download learns it from the GET, and Hugging Face rate-limits
    // anonymous requests, so a fresh browser serving eight models makes eight
    // requests, not sixteen.
    const etag = keyed.length > 0 || bare ? await fetchEtag(fetchImpl, url) : null;
    const wanted = modelBytesKey(url, etag);
    let cached: ArrayBuffer | undefined;
    try {
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
    if (listed) listed = withFinetuned(listed, records ?? []);
  }

  return { list, refresh, resolve, metadataFor: metadataOf, chooseVariant: chooseFor, modelBytes, addFinetuned, finetuned, removeFinetuned };
}

/** The list of `hub`, the Hub's entries in their order (the built-in
 * catalog's, then the added repos'), and `kept`'s, the finetuned models',
 * with every name unique (withUniqueNames). A Hub entry that a model kept
 * here was published as is left out: the kept model, the same bytes under
 * its own name, stands for it (resolve). */
function arranged(hub: ModelEntry[], kept: FinetunedRecord[]): ModelEntry[] {
  const publishedHere = new Set(kept.flatMap((r) => (r.published ? [`${r.published.repo}/${r.published.folder}`] : [])));
  return withUniqueNames([...hub.filter((e) => !publishedHere.has(e.id)), ...kept.map((r) => r.entry)]);
}

/** `list` with no two names alike. The built-in catalog's names and the
 * finetuned models' are kept as they are: links and the server name the
 * one, and the other's id, folder and byte-cache key are made of the name.
 * An added repo's entry takes its hubName, or when that is taken the first
 * free `<name>_<n>` (manifest.ts freeName), in list order. The names are
 * worked out afresh each time, so one taken by a model finetuned since
 * moves its holder on, and one freed again gives it back. */
function withUniqueNames(list: ModelEntry[]): ModelEntry[] {
  const kept = (e: ModelEntry) => e.repo === HF_REPO || isFinetuned(e);
  const taken = new Set(list.filter(kept).map((e) => e.name));
  return list.map((e) => {
    if (kept(e)) return e;
    const name = freeName(hubName(e.manifest), taken);
    taken.add(name);
    return name === e.name ? e : { ...e, name };
  });
}

/** modelBytes's progress callback: bytes so far, and the total when the server said (null otherwise). */
type Progress = (loaded: number, total: number | null) => void;

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function mb(byteLength: number): string {
  return (byteLength / 1e6).toFixed(1);
}
