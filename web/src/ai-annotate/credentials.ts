// The user's credentials for the AI providers, kept in the page. The server
// reads its keys from the environment or a file named in its config
// (cellmap_flow/ai_annotate/config.py, secrets.py); in the browser there is
// no server, so the user types them into the panel and the page holds them:
// in memory for the page's life, and in this browser's localStorage only for
// a provider the user ticked *Remember in this browser* for. *Forget* clears
// both.
//
// Where they never go: the session state (which share links and the state
// store carry), a log line, an answer to the page beyond `configured:
// true/false` per provider, or anything but the provider's own request. The
// routes pass what get() returns straight to Provider.segment and
// Provider.listModels and nowhere else.
//
// Storage is a KeyValueStore (localStorage's three methods), so a test hands
// in a MemoryKeyValueStore; every storage access is in try/catch, since a
// browser may block storage (a private window, a strict site setting) or
// refuse a write (quota), and neither should stop the panel from working for
// the page's life.
import type { KeyValueStore } from "../models/listing-cache";
import type { CredentialField, Credentials } from "./types";

/** The localStorage key the remembered credentials live under. */
export const CREDENTIALS_KEY = "cellmap-flow:ai-annotate:credentials";

/** The fields a Credentials may carry; anything else stored is dropped on load. */
const FIELDS: readonly (keyof Credentials)[] = ["api_key", "project", "access_token", "location"];

interface Stored {
  v: 1;
  /** The remembered credentials by provider id. */
  providers: Record<string, Credentials>;
}

export interface CredentialStore {
  /** Keep `credentials` for `provider`, and in storage too when `remember`
   * (a provider remembered before and set again without `remember` is
   * taken out of storage: the user asked for it not to be kept). */
  set(provider: string, credentials: Credentials, remember: boolean): void;
  /** The provider's credentials, or null when none were entered. */
  get(provider: string): Credentials | null;
  /** Drop the provider's credentials from the page and from storage. */
  forget(provider: string): void;
  /** Whether the provider can be called: with `fields` (what its
   * ProviderInfo.credentials lists), every one of them is present and
   * non-blank, so a provider that needs none is configured; without, any
   * field is. */
  configured(provider: string, fields?: readonly CredentialField[]): boolean;
  /** Whether the provider's credentials are the ones storage keeps. */
  remembered(provider: string): boolean;
}

export function createCredentialStore(storage: KeyValueStore | null): CredentialStore {
  const inMemory = new Map<string, Credentials>();
  const stored: Stored = load(storage);
  for (const [provider, credentials] of Object.entries(stored.providers)) inMemory.set(provider, credentials);

  function save(): void {
    if (!storage) return;
    try {
      if (Object.keys(stored.providers).length) storage.setItem(CREDENTIALS_KEY, JSON.stringify(stored));
      else storage.removeItem(CREDENTIALS_KEY);
    } catch {
      // storage refused the write (quota, private mode): the page still has them
    }
  }

  return {
    set(provider, credentials, remember) {
      const clean = cleanCredentials(credentials);
      inMemory.set(provider, clean);
      if (remember) stored.providers[provider] = clean;
      else delete stored.providers[provider];
      save();
    },
    get(provider) {
      const credentials = inMemory.get(provider);
      return credentials ? { ...credentials } : null;
    },
    forget(provider) {
      inMemory.delete(provider);
      delete stored.providers[provider];
      save();
    },
    configured(provider, fields) {
      const credentials = inMemory.get(provider);
      if (fields) return fields.every((field) => Boolean(credentials?.[field]));
      return Boolean(credentials && FIELDS.some((field) => credentials[field]));
    },
    remembered(provider) {
      return provider in stored.providers;
    },
  };
}

/** Only the known fields, trimmed, and only those with text. */
function cleanCredentials(credentials: Credentials): Credentials {
  const clean: Credentials = {};
  for (const field of FIELDS) {
    const value = credentials[field];
    if (typeof value === "string" && value.trim()) clean[field] = value.trim();
  }
  return clean;
}

/** What storage remembers, or an empty record when there is nothing, it
 * cannot be read, or it is not in this module's shape. */
function load(storage: KeyValueStore | null): Stored {
  const empty: Stored = { v: 1, providers: {} };
  if (!storage) return empty;
  let text: string | null;
  try {
    text = storage.getItem(CREDENTIALS_KEY);
  } catch {
    return empty;
  }
  if (!text) return empty;
  try {
    const parsed = JSON.parse(text) as Partial<Stored>;
    if (parsed.v !== 1 || !parsed.providers || typeof parsed.providers !== "object") return empty;
    const providers: Record<string, Credentials> = {};
    for (const [provider, credentials] of Object.entries(parsed.providers)) {
      if (!credentials || typeof credentials !== "object") continue;
      const clean = cleanCredentials(credentials as Credentials);
      if (Object.keys(clean).length) providers[provider] = clean;
    }
    return { v: 1, providers };
  } catch {
    return empty;
  }
}
