// A segmentation layer's equivalences in the viewer's state
// (src/viewer/equivalences.ts), and the controller's setEquivalences before
// it has an iframe (src/viewer/index.ts). What happens once the iframe
// exists goes through the same state read/restore path as setLayers and is
// checked in a browser.
import { describe, expect, it } from "vitest";

import type { SessionState, SessionStore } from "../../src/contracts";
import { equivalencesOf, withEquivalences } from "../../src/viewer/equivalences";
import { createViewer } from "../../src/viewer/index";
import { layersOf, type NgState } from "../../src/viewer/state";

const raw = { type: "image", name: "data", source: "zarr://https://x/em.zarr/s0", shader: "void main(){}" };
const mito = { type: "segmentation", name: "mito", source: "zarr3://http://h/mito__CFLOW_ARGS__b__CFLOW_ARGS__/", visible: false };
const er = { type: "image", name: "er", source: "zarr3://http://h/er/" };
const state: NgState = { dimensions: { z: [8e-9, "m"] }, position: [1, 2, 3], layers: [raw, mito, er], layout: "xy" };

describe("withEquivalences", () => {
  it("puts the sets, as strings, on the segmentation layer of that name and leaves the rest", () => {
    const next = withEquivalences(state, "mito", [["1", "1001", "4003"], ["2", "2001"]]);
    expect(next).not.toBeNull();
    expect(layersOf(next!)).toEqual([raw, { ...mito, equivalences: [["1", "1001", "4003"], ["2", "2001"]] }, er]);
    expect(next!.position).toEqual([1, 2, 3]);
    expect(next!.layout).toBe("xy");
    // The input is untouched.
    expect(state.layers).toEqual([raw, mito, er]);
    expect((state.layers as unknown[])[1]).toBe(mito);
  });
  it("replaces what the layer had", () => {
    const before = withEquivalences(state, "mito", [["1", "2"]])!;
    const after = withEquivalences(before, "mito", [["5", "6"]])!;
    expect(equivalencesOf(after, "mito")).toEqual([["5", "6"]]);
    expect(equivalencesOf(withEquivalences(before, "mito", [])!, "mito")).toEqual([]);
  });
  it("is null when the state has no segmentation layer of that name", () => {
    expect(withEquivalences(state, "nope", [["1", "2"]])).toBeNull();
    expect(withEquivalences(state, "er", [["1", "2"]]), "an image layer").toBeNull();
    expect(withEquivalences({}, "mito", [["1", "2"]])).toBeNull();
  });
  it("writes ids as strings whatever it was given, since a string is exact", () => {
    const next = withEquivalences(state, "mito", [[1, 18446744073709551615n] as unknown as string[]])!;
    expect(equivalencesOf(next, "mito")).toEqual([["1", "18446744073709551615"]]);
  });
  it("reads the oldest links' object form of layers too", () => {
    const old: NgState = { layers: { data: { type: "image", source: "zarr://a" }, mito: { type: "segmentation", source: "zarr3://m" } } };
    const next = withEquivalences(old, "mito", [["3", "4"]])!;
    expect(layersOf(next)).toEqual([
      { name: "data", type: "image", source: "zarr://a" },
      { name: "mito", type: "segmentation", source: "zarr3://m", equivalences: [["3", "4"]] },
    ]);
  });
});

describe("equivalencesOf", () => {
  it("is the layer's equivalences, or null without any", () => {
    expect(equivalencesOf(state, "mito")).toBeNull();
    expect(equivalencesOf(null, "mito")).toBeNull();
    expect(equivalencesOf({ layers: [{ name: "mito", type: "segmentation", equivalences: [[1, 2], ["3", "4"]] }] }, "mito")).toEqual([[1, 2], ["3", "4"]]);
    expect(equivalencesOf({ layers: [{ name: "mito", type: "segmentation", equivalences: "junk" }] }, "mito")).toBeNull();
  });
});

function fakeSession(): SessionStore {
  const current: SessionState = { datasetUrl: null, chain: { input_norm: [], postprocess: [] }, models: [], resample: false, prefs: {}, paintVolume: null };
  return { get state() { return current; }, async update(change) { Object.assign(current, change); }, onChange() { return () => {}; } };
}

describe("createViewer().setEquivalences before the iframe exists", () => {
  it("takes equivalences for a queued layer, or for no layer, without touching the page", () => {
    const lines: string[] = [];
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    viewer.setLayers({ mito: { type: "segmentation", source: "zarr3://m" } });
    expect(() => viewer.setEquivalences("mito", [["1", "2"]])).not.toThrow();
    expect(() => viewer.setEquivalences("unknown", [["1", "2"]])).not.toThrow();
    expect(lines).toEqual([]);
  });
});
