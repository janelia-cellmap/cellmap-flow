// The session store: its defaults, update() and the listeners, the memory
// fallback without IndexedDB (which is what node has), and how a stored
// state from an older build is made whole.
import { describe, expect, it } from "vitest";

import { completeState, defaultChain, defaultState, memorySession, openSession } from "../../src/state";

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

  it("opens in memory where there is no IndexedDB (here), and says so once", async () => {
    expect(typeof indexedDB).toBe("undefined");
    const store = await openSession();
    expect(store.state).toEqual(defaultState());
    await store.update({ resample: true });
    expect(store.state.resample).toBe(true);
  });
});
