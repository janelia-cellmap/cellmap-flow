// The session: what dashboard/state.py keeps between requests in server mode
// (the dataset, the chain, the models being served, the user's preferences),
// kept here in IndexedDB so it survives the page reload connect.js does after
// /api/set-data, and a browser restart.
//
// openSession() reads it (or starts a fresh one with the defaults below) and
// gives a SessionStore (contracts.ts): `state` is the current value, update()
// merges a change in, persists it and tells the listeners. Without IndexedDB
// (a test, or a browser that refuses it in a private window) the session
// lives in memory for the page's life and says so in the console.
import type { Chain, SessionState, SessionStore } from "../contracts";
import { idbGet, idbPut } from "./db";

const STATE_KEY = "state";

/** The chain a fresh session starts with. Every model the browser runs
 * (cellmap/onnx-web-models) was trained on uint8 scaled to [-1, 1] and
 * outputs logits, so this is what makes their predictions right; the
 * dashboard's Input and Postprocess tabs show it ticked. */
export function defaultChain(): Chain {
  return {
    input_norm: [
      { name: "MinMaxNormalizer", min_value: 0, max_value: 255 },
      { name: "LambdaNormalizer", expression: "2*x-1" },
    ],
    postprocess: [{ name: "SigmoidPostprocessor" }],
  };
}

export function defaultState(): SessionState {
  return { datasetUrl: null, chain: defaultChain(), models: [], resample: false, prefs: {} };
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
  };
}

/** A SessionStore over `initial`, persisted through `save` after each update. */
function makeStore(initial: SessionState, save: (state: SessionState) => Promise<void>): SessionStore {
  let state = initial;
  const listeners = new Set<(state: SessionState) => void>();
  return {
    get state() {
      return state;
    },
    async update(change) {
      state = { ...state, ...change };
      await save(state);
      for (const listener of listeners) listener(state);
    },
    onChange(listener) {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
  };
}

/** A session that lives in memory only (tests; the fallback without IndexedDB). */
export function memorySession(initial: Partial<SessionState> = {}): SessionStore {
  return makeStore({ ...defaultState(), ...initial }, async () => {});
}

export async function openSession(): Promise<SessionStore> {
  if (typeof indexedDB === "undefined") {
    console.warn("[cellmap-flow] no IndexedDB here: the session will not survive a reload");
    return memorySession();
  }
  try {
    const stored = await idbGet<unknown>("session", STATE_KEY);
    return makeStore(completeState(stored), (state) => idbPut("session", STATE_KEY, state));
  } catch (e) {
    console.warn("[cellmap-flow] IndexedDB refused the session; keeping it in memory for this page:", e);
    return memorySession();
  }
}
