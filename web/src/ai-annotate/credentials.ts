// The user's credentials for the AI providers, kept in the page's memory and
// nowhere else. The server reads its keys from the environment or a file
// named in its config (cellmap_flow/ai_annotate/config.py, secrets.py); in
// the browser there is no server, so the user pastes them into the panel,
// which hands them to the credentials route as they are typed.
//
// Why memory only: a page has no secret place to keep a key. Written to
// localStorage it would sit there unencrypted, readable by any script on the
// site and by anyone with a copy of the browser profile, and encrypting it in
// the page would not help, since the key to decrypt it would be in the same
// page. So the credentials live as long as the page does, and a reload asks
// for them again. A build of this panel once offered to remember them in
// localStorage; the copy it wrote, if any, is deleted when the store is made
// (purgeRemembered).
//
// Where they never go: the session state (which share links and the state
// store carry), any storage, a log line, an answer to the page beyond
// `configured: true/false` per provider, or anything but the provider's own
// request. The routes pass what get() returns straight to Provider.segment
// and Provider.listModels and nowhere else.
import type { KeyValueStore } from "../models/listing-cache";
import type { CredentialField, Credentials } from "./types";

/** Where an earlier build of the panel remembered credentials, when asked:
 * deleted on sight, so a key once remembered does not stay on the disk. */
export const CREDENTIALS_KEY = "cellmap-flow:ai-annotate:credentials";

/** The fields a Credentials may carry; anything else is dropped. */
const FIELDS: readonly (keyof Credentials)[] = ["api_key", "project", "access_token", "location"];

export interface CredentialStore {
  /** Keep `credentials` for `provider` in the page, in place of any it had. */
  set(provider: string, credentials: Credentials): void;
  /** The provider's credentials, or null when none were entered. */
  get(provider: string): Credentials | null;
  /** Drop the provider's credentials. */
  forget(provider: string): void;
  /** Whether the provider can be called: with `fields` (what its
   * ProviderInfo.credentials lists), every one of them is present and
   * non-blank, so a provider that needs none is configured; without, any
   * field is. */
  configured(provider: string, fields?: readonly CredentialField[]): boolean;
}

/** A store in the page's memory. `storage` is only cleared of what an earlier
 * build remembered there (purgeRemembered); nothing is ever written to it. */
export function createCredentialStore(storage: KeyValueStore | null = null): CredentialStore {
  purgeRemembered(storage);
  const inMemory = new Map<string, Credentials>();
  return {
    set(provider, credentials) {
      inMemory.set(provider, cleanCredentials(credentials));
    },
    get(provider) {
      const credentials = inMemory.get(provider);
      return credentials ? { ...credentials } : null;
    },
    forget(provider) {
      inMemory.delete(provider);
    },
    configured(provider, fields) {
      const credentials = inMemory.get(provider);
      if (fields) return fields.every((field) => Boolean(credentials?.[field]));
      return Boolean(credentials && FIELDS.some((field) => credentials[field]));
    },
  };
}

/** Delete the credentials an earlier build of the panel remembered in
 * `storage` (localStorage), if any. A storage that refuses is left alone:
 * nothing more can be done from the page. */
export function purgeRemembered(storage: KeyValueStore | null): void {
  if (!storage) return;
  try {
    if (storage.getItem(CREDENTIALS_KEY) !== null) storage.removeItem(CREDENTIALS_KEY);
  } catch {
    // blocked storage (a private window, a strict site setting)
  }
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
