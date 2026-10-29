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
// What IndexedDB keeps of a finetuning session, the finetuned models and the
// paint volumes, outlives the tab too, but the session says which of them
// are the tab's (tabKept), and only those are listed: a new tab starts with
// none, as a freshly started server knows of no earlier run's.
//
// openSession() reads it (or starts a fresh one with the defaults below) and
// gives a SessionStore (contracts.ts): `state` is the current value, update()
// merges a change in, persists it and tells the listeners. Without
// sessionStorage (a test, or a browser that blocks storage) the session
// lives in memory for the page's life and says so in the console.
import type { Chain, RankFit, SessionState, SessionStore, Step, TabKept } from "../contracts";
import { FINETUNED_REPO } from "../contracts";
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
  return { datasetUrl: null, chain: defaultChain(), models: [], resample: false, prefs: {}, paintVolume: null, shownChannels: {}, rankAdded: [], rankFits: {}, startPostprocess: null, tabKept: { finetuned: [], volumes: [] } };
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
    rankAdded: Array.isArray(s.rankAdded) ? s.rankAdded.filter((id): id is string => typeof id === "string") : [],
    rankFits: rankFitsOf(s.rankFits),
    startPostprocess: startPostprocessOf(s.startPostprocess),
    tabKept: tabKeptOf(s.tabKept),
  };
}

/** A stored `rankFits` made whole: the entries that are a fit (steps a
 * list, the postprocessing a list or null), each threshold a number or
 * null; {} for anything else, which only forgets what the ranking fitted
 * until the next score. */
function rankFitsOf(stored: unknown): Record<string, RankFit> {
  if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {};
  return Object.fromEntries(Object.entries(stored as Record<string, Partial<RankFit> | null>).flatMap(([name, f]) =>
    f && Array.isArray(f.input_norm) && (f.postprocess === null || Array.isArray(f.postprocess))
      ? [[name, { input_norm: f.input_norm, postprocess: f.postprocess, threshold: typeof f.threshold === "number" ? f.threshold : null }]]
      : []));
}

/** A stored `startPostprocess` made whole, or null when it is not one. */
function startPostprocessOf(stored: unknown): SessionState["startPostprocess"] {
  const s = stored as { model?: unknown; postprocess?: unknown } | null;
  return s && typeof s === "object" && typeof s.model === "string" && Array.isArray(s.postprocess)
    ? { model: s.model, postprocess: s.postprocess as Step[] } : null;
}

/** A stored `shownChannels` made whole: the entries that are a channel's
 * index (a whole number, 0 and up), since each becomes a layer's position
 * on its channel axis; {} for anything that is not an object of them. */
function shownChannelsOf(stored: unknown): Record<string, number> {
  if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {};
  return Object.fromEntries(Object.entries(stored).filter(([, c]) => Number.isInteger(c) && (c as number) >= 0)) as Record<string, number>;
}

/** A stored `tabKept` made whole: each kind's ids that are strings, none for
 * a kind that is not a list. */
function tabKeptOf(stored: unknown): TabKept {
  const s = stored && typeof stored === "object" ? (stored as Partial<Record<keyof TabKept, unknown>>) : {};
  const ids = (value: unknown) => (Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : []);
  return { finetuned: ids(s.finetuned), volumes: ids(s.volumes) };
}

/** `kept` with `ids` added after it, each id once; `kept` itself when it has them all. */
function withIds(kept: string[], ids: readonly string[]): string[] {
  const added = ids.filter((id, i) => !kept.includes(id) && ids.indexOf(id) === i);
  return added.length ? [...kept, ...added] : kept;
}

/** `state` with what it uses made the tab's (tabKept): the finetuned models
 * it serves or added to the ranking, and the volume it paints into. Applied
 * to every state the store holds, so whichever route serves a model or
 * makes, resumes or carries a volume, the model or volume stays listed in
 * the tab after it is stopped or another volume is made, and after a
 * reload. */
function withTabUse(state: SessionState): SessionState {
  const served = state.models.filter((m) => m.repo === FINETUNED_REPO).map((m) => m.id);
  const ranked = state.rankAdded.filter((id) => id.startsWith(`${FINETUNED_REPO}/`));
  const finetuned = withIds(state.tabKept.finetuned, [...served, ...ranked]);
  const volumes = withIds(state.tabKept.volumes, state.paintVolume ? [state.paintVolume] : []);
  return finetuned === state.tabKept.finetuned && volumes === state.tabKept.volumes ? state : { ...state, tabKept: { finetuned, volumes } };
}

/** Make `kept`'s finetuned models and paint volumes the tab's (tabKept), to
 * be listed from now on: one finetuned or opened from a file here, a link's
 * paint, and what Kept in this browser brings back (src/kept). */
export function keepInTab(session: SessionStore, kept: Partial<TabKept>): Promise<void> {
  const { finetuned, volumes } = session.state.tabKept;
  return session.update({ tabKept: { finetuned: withIds(finetuned, kept.finetuned ?? []), volumes: withIds(volumes, kept.volumes ?? []) } });
}

/** A SessionStore over `initial`, persisted through `save` after each update. */
function makeStore(initial: SessionState, save: (state: SessionState) => Promise<void>): SessionStore {
  let state = withTabUse(initial);
  noteDataset(state);  // a session saved before the flag existed gets it now
  const listeners = new Set<(state: SessionState) => void>();
  return {
    get state() {
      return state;
    },
    async update(change) {
      state = withTabUse({ ...state, ...change });
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
