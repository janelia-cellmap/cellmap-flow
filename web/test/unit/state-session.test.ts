// The session store: its defaults, update() and the listeners, the tab's
// sessionStorage as its home (and the dataset flag beside it), the memory
// fallback without storage (which is what node has), and how a stored state
// from an older build is made whole.
import { afterEach, describe, expect, it, vi } from "vitest";

import { completeState, DATASET_FLAG, defaultChain, defaultState, memorySession, openSession, SESSION_KEY } from "../../src/state";

describe("the defaults", () => {
  it("start with the chain every browser model expects and nothing open", () => {
    expect(defaultState()).toEqual({
      datasetUrl: null,
      chain: {
        input_norm: [
          { name: "MinMaxNormalizer", min_value: 0, max_value: 255 },
          { name: "LambdaNormalizer", expression: "2*x-1" },
        ],
        postprocess: [{ name: "SigmoidPostprocessor" }],
      },
      models: [],
      resample: false,
      prefs: {},
    });
    // a fresh object each time, so a caller changing one does not change the default
    expect(defaultChain()).not.toBe(defaultChain());
  });
});

describe("completeState", () => {
  it("fills in what a stored state lacks and drops what is not of the right shape", () => {
    expect(completeState(undefined)).toEqual(defaultState());
    expect(completeState("junk")).toEqual(defaultState());
    const partial = completeState({ datasetUrl: "https://x/y.zarr", chain: { input_norm: [] }, models: "no", resample: "yes", prefs: [1] });
    expect(partial.datasetUrl).toBe("https://x/y.zarr");
    expect(partial.chain).toEqual(defaultChain()); // not two lists
    expect(partial.models).toEqual([]);
    expect(partial.resample).toBe(false);
    expect(partial.prefs).toEqual({});
    const whole = completeState({ datasetUrl: null, chain: { input_norm: [], postprocess: [] }, models: [], resample: true, prefs: { a: 1 } });
    expect(whole).toEqual({ datasetUrl: null, chain: { input_norm: [], postprocess: [] }, models: [], resample: true, prefs: { a: 1 } });
  });
});

describe("the store", () => {
  it("merges an update into the state and tells the listeners, until they unsubscribe", async () => {
    const store = memorySession({ resample: true });
    expect(store.state.resample).toBe(true);
    const seen: string[] = [];
    const stop = store.onChange((s) => void seen.push(String(s.datasetUrl)));
    await store.update({ datasetUrl: "s3://b/k" });
    expect(store.state).toMatchObject({ datasetUrl: "s3://b/k", resample: true, chain: defaultChain() });
    stop();
    await store.update({ datasetUrl: null });
    expect(seen).toEqual(["s3://b/k"]);
  });

  it("opens in memory where there is no sessionStorage (here), and says so", async () => {
    expect(typeof sessionStorage).toBe("undefined");
    const store = await openSession();
    expect(store.state).toEqual(defaultState());
    await store.update({ resample: true });
    expect(store.state.resample).toBe(true);
  });
});

describe("the session in the tab", () => {
  afterEach(() => vi.unstubAllGlobals());

  function stubStorage(): Map<string, string> {
    const items = new Map<string, string>();
    vi.stubGlobal("sessionStorage", {
      getItem: (k: string) => items.get(k) ?? null,
      setItem: (k: string, v: string) => void items.set(k, v),
      removeItem: (k: string) => void items.delete(k),
    });
    return items;
  }

  it("is kept in sessionStorage across a reload of the same tab, with the dataset flag beside it", async () => {
    const items = stubStorage();
    const first = await openSession();
    expect(first.state).toEqual(defaultState());
    await first.update({ datasetUrl: "s3://b/k", resample: true });
    expect(items.get(DATASET_FLAG)).toBe("1");
    const again = await openSession();  // the page reloaded: the same tab
    expect(again.state).toMatchObject({ datasetUrl: "s3://b/k", resample: true, chain: defaultChain() });
    await again.update({ datasetUrl: null });
    expect(items.has(DATASET_FLAG)).toBe(false);
    expect(JSON.parse(items.get(SESSION_KEY)!)).toMatchObject({ datasetUrl: null, resample: true });
  });

  it("starts afresh from a stored session it cannot read", async () => {
    const items = stubStorage();
    items.set(SESSION_KEY, "{not json");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect((await openSession()).state).toEqual(defaultState());
    expect(warn).toHaveBeenCalledOnce();
  });
});
