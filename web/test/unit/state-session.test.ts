// The session store in the tab's sessionStorage, its home across a reload
// (and the dataset flag beside it), and a stored session it cannot read.
import { afterEach, describe, expect, it, vi } from "vitest";

import { DATASET_FLAG, defaultChain, defaultState, openSession, SESSION_KEY } from "../../src/state";

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
