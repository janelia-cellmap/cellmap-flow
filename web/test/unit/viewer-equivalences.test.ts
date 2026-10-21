// A segmentation layer's equivalences in the viewer's state
// (src/viewer/equivalences.ts): the merged ids put on the layer, exactly.
// What happens once the iframe exists goes through the same state
// read/restore path as setLayers and is checked in a browser.
import { describe, expect, it } from "vitest";

import { equivalencesOf, withEquivalences } from "../../src/viewer/equivalences";
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
  it("is null when the state has no segmentation layer of that name", () => {
    expect(withEquivalences(state, "nope", [["1", "2"]])).toBeNull();
    expect(withEquivalences(state, "er", [["1", "2"]]), "an image layer").toBeNull();
    expect(withEquivalences({}, "mito", [["1", "2"]])).toBeNull();
  });
  it("writes ids as strings whatever it was given, since a string is exact", () => {
    const next = withEquivalences(state, "mito", [[1, 18446744073709551615n] as unknown as string[]])!;
    expect(equivalencesOf(next, "mito")).toEqual([["1", "18446744073709551615"]]);
  });
});
