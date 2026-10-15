// The model catalog: the models the browser can run, from the Hugging Face
// repo cellmap/onnx-web-models, one folder per model with a manifest.json
// and twelve .onnx files (three block sizes, four precisions).
//
// list() reads the repo's folder list from the Hub API and each folder's
// manifest, keeps them in localStorage for a day (listing-cache.ts), and
// falls back to the folders known at the time of writing when the Hub will
// not answer (it rate-limits anonymous requests). Each manifest becomes one
// ModelEntry with a default variant (fp32, the middle block: the choice for
// a GPU nobody has asked yet). Whoever knows the GPU, the inference side,
// calls chooseVariant again with its limits and swaps the variant in.
//
// modelBytes() downloads the chosen file with progress and keeps the bytes
// in IndexedDB (byte-cache.ts), so the next visit does not download again.
import type { Catalog, GpuLimits, ModelEntry, ModelManifest, ModelVariant } from "../contracts";

import { type ByteCache, openByteCache } from "./byte-cache";
import { type FetchLike, fetchBytes, fetchManifest, fetchTree, HF_REPO, HF_REVISION, RateLimitError } from "./hf";
import { defaultKeyValueStore, type KeyValueStore, ListingCache } from "./listing-cache";
import { chooseVariant, UNKNOWN_LIMITS } from "./variant";
import { entryFor, FALLBACK_FOLDERS, modelFolders, parseManifest } from "./manifest";

export { chooseVariant, estimatePeakBytes, PREFER_SMALL_DOWNLOAD, UNKNOWN_LIMITS } from "./variant";
export { entryFor, entryName, FALLBACK_FOLDERS, ManifestError, parseManifest } from "./manifest";
export { metadataFor } from "./metadata";
export { HF_REPO, HF_REVISION, RateLimitError, resolveUrl } from "./hf";
export { IndexedDbByteCache, MemoryByteCache, MODEL_DB_NAME, type ByteCache } from "./byte-cache";
export { type KeyValueStore, LISTING_TTL_MS, MemoryKeyValueStore } from "./listing-cache";

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
  const now = opts.now ?? Date.now;

  // The listing, once asked for, for the page's lifetime; dropped on failure
  // so the next call tries again (the Hub may have stopped rate-limiting).
  let entries: Promise<ModelEntry[]> | null = null;
  // Downloads under way, by URL, so two callers of one model share a download.
  const downloads = new Map<string, Promise<ArrayBuffer>>();

  function list(): Promise<ModelEntry[]> {
    entries ??= load(false).catch((error: unknown) => {
      entries = null;
      throw error;
    });
    return entries;
  }

  function refresh(): Promise<ModelEntry[]> {
    const previous = entries;
    const refreshed = load(true);
    // list() answers the old list until the new one is in. A failed refresh
    // keeps the old list when the page had one; with none, the next list()
    // asks the Hub again.
    const replacement: Promise<ModelEntry[]> = refreshed.catch(async (error: unknown) => {
      if (previous) {
        try {
          return await previous;
        } catch {
          // the old list had failed too
        }
      }
      if (entries === replacement) entries = null;
      throw error;
    });
    replacement.catch(() => {}); // the caller hears of the failure through `refreshed`
    entries = replacement;
    return refreshed;
  }

  // The listing: from the cache while fresh, else from the Hub, with the
  // cache (even stale) and the built-in folder list as fallbacks.
  async function load(force: boolean): Promise<ModelEntry[]> {
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

  /** The entry `ref` names: its id ("cellmap/onnx-web-models/setup_54", which is also repo/folder), its folder ("setup_54") or its name ("mito_aff_setup54_16nm"). */
  async function resolve(ref: string): Promise<ModelEntry | null> {
    const key = ref.trim().replace(/\/+$/, "");
    if (!key) return null;
    const all = await list();
    return all.find((e) => e.id === key || e.folder === key || e.name === key) ?? null;
  }

  function modelBytes(entry: ModelEntry, onProgress?: (loaded: number, total: number | null) => void): Promise<ArrayBuffer> {
    const url = entry.onnxUrl;
    const underway = downloads.get(url);
    // A second caller gets the same bytes; the first caller's progress callback is the one that runs.
    if (underway) return underway;
    const download = (async () => {
      let cached: ArrayBuffer | undefined;
      try {
        cached = await bytes.get(url);
      } catch (error) {
        log(`${entry.name}: could not read the model cache (${message(error)})`);
      }
      if (cached) {
        log(`${entry.name}: ${mb(cached.byteLength)} MB from the browser's cache`);
        onProgress?.(cached.byteLength, cached.byteLength);
        return cached;
      }
      log(`${entry.name}: downloading ${entry.variant.file} (${entry.variant.file_mb.toFixed(1)} MB) from Hugging Face`);
      const downloaded = await fetchBytes(fetchImpl, url, onProgress);
      log(`${entry.name}: downloaded ${mb(downloaded.byteLength)} MB`);
      try {
        await bytes.put(url, downloaded);
      } catch (error) {
        log(`${entry.name}: could not cache the model (${message(error)}); it will be downloaded again next time`);
      }
      return downloaded;
    })().finally(() => downloads.delete(url));
    downloads.set(url, download);
    return download;
  }

  function chooseFor(manifest: ModelManifest, limits: GpuLimits): ModelVariant {
    return chooseVariant(manifest, limits);
  }

  // Finetuned models (finetuned.ts): to come.
  async function addFinetuned(): Promise<void> {
    throw new Error("addFinetuned is not implemented yet");
  }
  async function finetuned() {
    return [];
  }
  async function removeFinetuned(): Promise<void> {}

  return { list, refresh, resolve, metadataFor, chooseVariant: chooseFor, modelBytes, addFinetuned, finetuned, removeFinetuned };
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function mb(byteLength: number): string {
  return (byteLength / 1e6).toFixed(1);
}
