// The pure functions over a Neuroglancer state's JSON (src/viewer/state.ts):
// replacing layers as the Python dashboard's _replace_layer does, the
// position in nm, a layer's kept shader, the dataset a state shows.
import { describe, expect, it } from "vitest";

import {
  datasetUrlFromState, layerShaderOf, layersOf, type NgState, positionNm, replaceLayers, sourceUrl, withoutLayer,
} from "../../src/viewer/state";

const raw = { type: "image", name: "data", source: "zarr://https://x/dataset.zarr/recon-1/em/fibsem-uint8", shader: "void main(){}" };
const model = { type: "image", name: "model", source: "zarr3://http://h/m/", shader: "old", visible: false };
const state: NgState = {
  dimensions: { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] },
  position: [10.5, 20.5, 30.5],
  layers: [raw, model],
  layout: "xy",
};

describe("layersOf", () => {
  it("reads the array form and the oldest links' object form alike", () => {
    expect(layersOf(state)).toEqual([raw, model]);
    expect(layersOf({ layers: { data: { type: "image", source: "zarr://a" } } })).toEqual([{ name: "data", type: "image", source: "zarr://a" }]);
    expect(layersOf({})).toEqual([]);
    expect(layersOf(null)).toEqual([]);
  });
});

describe("replaceLayers (_replace_layer)", () => {
  it("replaces a layer in place and keeps it hidden where it was hidden", () => {
    const next = replaceLayers(state, { model: { type: "image", source: "zarr3://http://h/m2/", shader: "new" } });
    expect(next.layers).toEqual([raw, { type: "image", name: "model", source: "zarr3://http://h/m2/", shader: "new", visible: false }]);
    // The rest of the state, and the input, are untouched.
    expect(next.position).toEqual([10.5, 20.5, 30.5]);
    expect(next.layout).toBe("xy");
    expect(state.layers).toEqual([raw, model]);
  });
  it("shows a replaced layer that was shown, whatever the new spec says", () => {
    const next = replaceLayers(state, { data: { type: "image", source: "zarr://https://y/", shader: "s" } });
    expect(layersOf(next)[0]).toEqual({ type: "image", name: "data", source: "zarr://https://y/", shader: "s" });
  });
  it("appends layers that were not there, in the given order, after the existing ones", () => {
    const next = replaceLayers(state, { b: { type: "segmentation", source: "zarr3://b" }, a: { type: "image", source: "zarr3://a" } });
    expect(layersOf(next).map((l) => l.name)).toEqual(["data", "model", "b", "a"]);
  });
  it("names each layer by its key, whatever name the spec carried", () => {
    const next = replaceLayers({ layers: [] }, { mine: { type: "image", name: "other", source: "zarr3://a" } });
    expect(layersOf(next)).toEqual([{ type: "image", name: "mine", source: "zarr3://a" }]);
  });
});

describe("withoutLayer", () => {
  it("drops the named layer and nothing else", () => {
    expect(layersOf(withoutLayer(state, "model"))).toEqual([raw]);
    expect(layersOf(withoutLayer(state, "nope"))).toEqual([raw, model]);
  });
});

describe("positionNm", () => {
  it("multiplies the voxel position by the dimensions' scales, in nm, z y x", () => {
    expect(positionNm(state)).toEqual([84, 164, 244]);
  });
  it("matches axes by name, so x y z dimensions still give z y x", () => {
    const xyz: NgState = { dimensions: { x: [4e-9, "m"], y: [4e-9, "m"], z: [8e-9, "m"] }, position: [100, 200, 300] };
    expect(positionNm(xyz)).toEqual([2400, 800, 400]);
  });
  it("reads nm and um units too", () => {
    expect(positionNm({ dimensions: { z: [8, "nm"], y: [8, "nm"], x: [8, "nm"] }, position: [1, 2, 3] })).toEqual([8, 16, 24]);
    expect(positionNm({ dimensions: { z: [1, "um"], y: [1, "µm"], x: [0.5, "um"] }, position: [1, 2, 3] })).toEqual([1000, 2000, 1500]);
  });
  it("falls back to the first three dimensions when they are not named z y x", () => {
    expect(positionNm({ dimensions: { a: [1e-9, "m"], b: [2e-9, "m"], c: [3e-9, "m"] }, position: [1, 1, 1] })).toEqual([1, 2, 3]);
  });
  it("is null without a position or dimensions, or when a dimension is not a length", () => {
    expect(positionNm({ dimensions: state.dimensions })).toBeNull();
    expect(positionNm({ position: [1, 2, 3] })).toBeNull();
    expect(positionNm({ dimensions: state.dimensions, position: [1, 2] })).toBeNull();
    expect(positionNm({ dimensions: { z: [1, ""], y: [8e-9, "m"], x: [8e-9, "m"] }, position: [1, 2, 3] })).toBeNull();
    expect(positionNm(null)).toBeNull();
  });
});

describe("layerShaderOf (_save_shaders_from_viewer)", () => {
  it("gives the layer's shader and controls, whichever it has", () => {
    const withControls: NgState = { layers: [{ name: "data", shader: "s", shaderControls: { normalized: { range: [0, 1] } } }] };
    expect(layerShaderOf(withControls, "data")).toEqual({ shader: "s", shaderControls: { normalized: { range: [0, 1] } } });
    expect(layerShaderOf(state, "model")).toEqual({ shader: "old" });
    expect(layerShaderOf({ layers: [{ name: "seg", type: "segmentation" }] }, "seg")).toEqual({});
  });
  it("treats the Python API's \"None\" and empty controls as unset", () => {
    expect(layerShaderOf({ layers: [{ name: "data", shader: "None", shaderControls: {} }] }, "data")).toEqual({});
  });
  it("is null without the layer", () => {
    expect(layerShaderOf(state, "nope")).toBeNull();
    expect(layerShaderOf(null, "data")).toBeNull();
  });
});

describe("sourceUrl and datasetUrlFromState", () => {
  it("reads every form a source takes", () => {
    expect(sourceUrl("zarr://a")).toBe("zarr://a");
    expect(sourceUrl({ url: "zarr3://b", transform: {} })).toBe("zarr3://b");
    expect(sourceUrl([{ url: "zarr://c" }])).toBe("zarr://c"); // the Python API's
    expect(sourceUrl(["zarr://d"])).toBe("zarr://d");
    expect(sourceUrl([])).toBeNull();
    expect(sourceUrl(undefined)).toBeNull();
    expect(sourceUrl({ nope: 1 })).toBeNull();
  });
  it("gives the raw layer's URL without its format", () => {
    expect(datasetUrlFromState(state)).toBe("https://x/dataset.zarr/recon-1/em/fibsem-uint8");
    expect(datasetUrlFromState({ layers: [{ name: "data", type: "image", source: [{ url: "n5://s3://bucket/x.n5/em" }] }] })).toBe("s3://bucket/x.n5/em");
    expect(datasetUrlFromState({ layers: [{ name: "data", type: "image", source: "precomputed://gs://bucket/vol" }] })).toBe("gs://bucket/vol");
  });
  it("falls back to the first image layer when none is named data", () => {
    expect(datasetUrlFromState({ layers: [{ name: "seg", type: "segmentation", source: "zarr://s" }, { name: "em", type: "image", source: "zarr://https://e/" }] })).toBe("https://e/");
  });
  it("is null without a raw layer", () => {
    expect(datasetUrlFromState({ layers: [] })).toBeNull();
    expect(datasetUrlFromState({ layers: [{ name: "seg", type: "segmentation", source: "zarr://s" }] })).toBeNull();
    expect(datasetUrlFromState(null)).toBeNull();
  });
});
