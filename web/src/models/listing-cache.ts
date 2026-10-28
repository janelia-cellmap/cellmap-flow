// The model list between visits: the raw manifest.json of every folder, with
// the time they were fetched, in localStorage, one listing per repo (the
// built-in catalog's, and each repo added beside it). Six manifests are 30
// KB, so they fit beside the dashboard's own keys; a day's time-to-live
// means a returning visitor asks the Hub nothing until tomorrow (the Python
// side keeps its Hugging Face listing until the Refresh button, for the same
// reason: opening the dashboard should not query the Hub each time). Which
// repos were added is kept beside them (AddedRepos).
//
// The manifests are kept as fetched, not parsed: a fix to parseManifest then
// applies to cached data too, and a cached manifest the parser rejects is
// skipped the same way a fresh one is.

/** What the cache needs of localStorage, so a test can hand in a Map. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export const LISTING_TTL_MS = 24 * 60 * 60 * 1000;

export interface CachedListing {
  /** Date.now() when the manifests were fetched. */
  fetchedAt: number;
  /** The raw manifest.json per folder. */
  manifests: Record<string, unknown>;
}

interface Stored extends CachedListing {
  v: 1;
}

/** A store that lasts as long as the page. */
export class MemoryKeyValueStore implements KeyValueStore {
  private readonly entries = new Map<string, string>();

  getItem(key: string): string | null {
    return this.entries.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.entries.set(key, value);
  }

  removeItem(key: string): void {
    this.entries.delete(key);
  }
}

export class ListingCache {
  constructor(
    private readonly store: KeyValueStore,
    private readonly key: string,
  ) {}

  /** The cached listing, fresh or stale, or null when there is none or it cannot be read. */
  read(): CachedListing | null {
    let text: string | null;
    try {
      text = this.store.getItem(this.key);
    } catch {
      return null; // storage blocked (private mode, a strict site setting)
    }
    if (!text) return null;
    try {
      const stored = JSON.parse(text) as Partial<Stored>;
      if (stored.v !== 1 || typeof stored.fetchedAt !== "number" || !stored.manifests || typeof stored.manifests !== "object") return null;
      return { fetchedAt: stored.fetchedAt, manifests: stored.manifests };
    } catch {
      return null;
    }
  }

  /** Is `listing` younger than the time-to-live, as of `now`? */
  isFresh(listing: CachedListing, now: number): boolean {
    return now - listing.fetchedAt < LISTING_TTL_MS;
  }

  /** Keep `listing`; false when storage refused it (quota, private mode), which is not an error for the caller. */
  write(listing: CachedListing): boolean {
    const stored: Stored = { v: 1, ...listing };
    try {
      this.store.setItem(this.key, JSON.stringify(stored));
      return true;
    } catch {
      return false;
    }
  }

  clear(): void {
    try {
      this.store.removeItem(this.key);
    } catch {
      // nothing to clear, or storage blocked
    }
  }
}

/** The Hugging Face repos added to the catalog beside the built-in one (a
 * ref of one resolved: models/index.ts), in the order they were added, so
 * that the next visit lists their models too. A JSON list of repo ids,
 * read and written whole; one that cannot be read is no repos. */
export class AddedRepos {
  constructor(
    private readonly store: KeyValueStore,
    private readonly key: string,
  ) {}

  list(): string[] {
    try {
      const repos: unknown = JSON.parse(this.store.getItem(this.key) ?? "[]");
      return Array.isArray(repos) ? repos.filter((r): r is string => typeof r === "string") : [];
    } catch {
      return [];
    }
  }

  /** Add `repo` after the others; false when storage refused it (it is then listed for this page only). */
  add(repo: string): boolean {
    const repos = this.list();
    return repos.includes(repo) || this.write([...repos, repo]);
  }

  remove(repo: string): void {
    this.write(this.list().filter((r) => r !== repo));
  }

  private write(repos: string[]): boolean {
    try {
      this.store.setItem(this.key, JSON.stringify(repos));
      return true;
    } catch {
      return false;
    }
  }
}

/** localStorage when the browser lets the page use it, else a page-lifetime store. */
export function defaultKeyValueStore(): KeyValueStore {
  try {
    const storage = globalThis.localStorage;
    if (storage) {
      storage.getItem("cellmap-flow:storage-probe"); // throws where storage is blocked
      return storage;
    }
  } catch {
    // fall through
  }
  return new MemoryKeyValueStore();
}
