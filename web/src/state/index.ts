// The session: what dashboard/state.py keeps between requests in server mode
// (the dataset, the chain, the models being served, the user's preferences).
// The server's session lives as long as the server process; the browser's
// equivalent is the tab, so the session is kept in sessionStorage: it
// survives the page reload connect.js does after /api/set-data (and the one
// that makes the service worker the page's controller) and ends with the
// tab. A new tab, or a later visit, starts at the connect panel as a freshly
// started server does, rather than reopening the last volume and serving its
// models again unasked; a shared link carries its dataset and layers in its
// hash. (The model bytes are another matter: models/byte-cache.ts keeps them
// in IndexedDB across visits, since Hugging Face rate-limits the downloads.)
//
// openSession() reads it (or starts a fresh one with the defaults below) and
// gives a SessionStore (contracts.ts): `state` is the current value, update()
// merges a change in, persists it and tells the listeners. Without
// sessionStorage (a test, or a browser that blocks storage) the session
// lives in memory for the page's life and says so in the console.
import type { Chain, SessionState, SessionStore } from "../contracts";
import { PYTHON } from "../generated/python-data";

/** The session's key in the tab's sessionStorage. */
export const SESSION_KEY = "cellmap-flow:session";

/** The chain a fresh session starts with, a fresh copy each call. Every
 * model the browser runs (cellmap/onnx-web-models) was trained on uint8
 * scaled to [-1, 1] and outputs logits, so this is what makes their
 * predictions right; the dashboard's Processing tab shows it
 * ticked: it is web/build/export_data.py's DEFAULT_CHAIN, the one
 * build/render.py ticks on the page. */
export function defaultChain(): Chain {
  return structuredClone(PYTHON.default_chain);
}

export function defaultState(): SessionState {
  return { datasetUrl: null, chain: defaultChain(), models: [], resample: false, prefs: {}, paintVolume: null, shownChannels: {} };
}

/** A stored state made whole: a session written by an older build may lack
 * a field, and a field of the wrong shape (a chain that is not two lists)
 * falls back to its default rather than breaking every route. */
export function completeState(stored: unknown): SessionState {
  const base = defaultState();
  if (!stored || typeof stored !== "object") return base;
  const s = stored as Partial<Record<keyof SessionState, unknown>>;
  const chain = s.chain as Partial<Chain> | undefined;
  const chainOk = chain && Array.isArray(chain.input_norm) && Array.isArray(chain.postprocess);
  return {
    datasetUrl: typeof s.datasetUrl === "string" ? s.datasetUrl : null,
    chain: chainOk ? { input_norm: chain.input_norm!, postprocess: chain.postprocess! } : base.chain,
    models: Array.isArray(s.models) ? (s.models as SessionState["models"]) : [],
    resample: typeof s.resample === "boolean" ? s.resample : false,
    prefs: s.prefs && typeof s.prefs === "object" && !Array.isArray(s.prefs) ? (s.prefs as SessionState["prefs"]) : {},
    paintVolume: typeof s.paintVolume === "string" ? s.paintVolume : null,
    shownChannels: shownChannelsOf(s.shownChannels),
  };
}

/** A stored `shownChannels` made whole: the entries that are a channel's
 * index (a whole number, 0 and up), since each becomes a layer's position
 * on its channel axis; {} for anything that is not an object of them. */
function shownChannelsOf(stored: unknown): Record<string, number> {
  if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {};
  return Object.fromEntries(Object.entries(stored).filter(([, c]) => Number.isInteger(c) && (c as number) >= 0)) as Record<string, number>;
}

/** A SessionStore over `initial`, persisted through `save` after each update. */
function makeStore(initial: SessionState, save: (state: SessionState) => Promise<void>): SessionStore {
  let state = initial;
  noteDataset(state);  // a session saved before the flag existed gets it now
  const listeners = new Set<(state: SessionState) => void>();
  return {
    get state() {
      return state;
    },
    async update(change) {
      state = { ...state, ...change };
      noteDataset(state);
      await save(state);
      for (const listener of listeners) listener(state);
    },
    onChange(listener) {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
  };
}

/** Whether the session has a dataset, as a flag of its own in the tab's
 * sessionStorage, which index.html's inline script reads before the first
 * paint (one key, nothing to parse): it hides the connect panel on it, so
 * the reload connect.js does after Connect shows the viewer's place, not the
 * panel, until the viewer is restored. Storage can be unavailable; then the
 * panel just flashes. */
export const DATASET_FLAG = "cellmap-flow:dataset";
function noteDataset(state: SessionState): void {
  try {
    if (state.datasetUrl) sessionStorage.setItem(DATASET_FLAG, "1");
    else sessionStorage.removeItem(DATASET_FLAG);
  } catch {
    // no storage: nothing to note
  }
}

/** A session that lives in memory only (tests; the fallback without sessionStorage). */
export function memorySession(initial: Partial<SessionState> = {}): SessionStore {
  return makeStore({ ...defaultState(), ...initial }, async () => {});
}

/** The tab's sessionStorage, or null where there is none (node) or the
 * browser blocks storage, which throws on the access itself. */
function tabStorage(): Storage | null {
  try {
    const storage = globalThis.sessionStorage as Storage | undefined;
    if (!storage) return null;
    storage.getItem(SESSION_KEY);
    return storage;
  } catch {
    return null;
  }
}

export async function openSession(): Promise<SessionStore> {
  const storage = tabStorage();
  if (!storage) {
    console.warn("[cellmap-flow] no sessionStorage here: the session will not survive a reload");
    return memorySession();
  }
  let stored: unknown;
  try {
    const raw = storage.getItem(SESSION_KEY);
    stored = raw ? JSON.parse(raw) : undefined;
  } catch (e) {
    console.warn("[cellmap-flow] the tab's stored session could not be read; starting afresh:", e);
  }
  return makeStore(completeState(stored), async (state) => {
    try {
      storage.setItem(SESSION_KEY, JSON.stringify(state));
    } catch (e) {
      // Storage full or blocked since: the session goes on in memory for this page.
      console.warn("[cellmap-flow] the session could not be saved; it lives in memory for this page:", e);
    }
  });
}
