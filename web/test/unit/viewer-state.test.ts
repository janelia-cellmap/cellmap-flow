// The pure functions over a Neuroglancer state's JSON (src/viewer/state.ts):
// replacing layers as the Python dashboard's replace_layer does, keeping the
// user's settings on them (carriedLayer), the
// position (and the pointer's) in nm, the data axis the layout looks along
// (cellmap_flow/ai_annotate/geometry.py's depth_axis_from_layout), a
// layer's kept shader, the dataset a state shows.
import { describe, expect, it } from "vitest";

import {
  carriedLayer, datasetUrlFromState, depthAxisFromLayout, displayDimensionsOf, layerShaderOf, layersOf, type NgState, pointerNm, positionNm, replaceLayers, sourceUrl,
  sourceUrls, viewDepthAxisOf, withoutLayer,
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

describe("replaceLayers (replace_layer)", () => {
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

describe("carriedLayer: what the user set on a layer survives its redraw", () => {
  // A model layer as the user left it: additive, half opaque, finer
  // resolution, channel 2 shown, its contrast moved.
  const shown = {
    type: "image", name: "mito", source: "zarr3://http://h/mito__CFLOW_ARGS__a__CFLOW_ARGS__/", shader: "user's", shaderControls: { normalized: { range: [0, 9] } },
    blend: "additive", opacity: 0.5, crossSectionRenderScale: 0.5, localDimensions: { "c^": [1, ""] }, localPosition: [2.5], tab: "rendering",
  };

  it("keeps every setting the new spec does not make, as when another model is added and this one is drawn again", () => {
    const next = carriedLayer(shown, { type: "image", source: shown.source, shader: "user's" }, "mito");
    expect(next).toEqual(shown);
  });

  it("lets the new spec's own properties win", () => {
    const next = carriedLayer(shown, { type: "image", source: shown.source, shader: "the app's", opacity: 1 }, "mito");
    expect(next).toMatchObject({ shader: "the app's", opacity: 1, blend: "additive", crossSectionRenderScale: 0.5 });
  });

  it("drops what belonged to the old source's data when the source is another (a new chain)", () => {
    const next = carriedLayer(shown, { type: "image", source: "zarr3://http://h/mito__CFLOW_ARGS__b__CFLOW_ARGS__/", shader: "new range" }, "mito");
    expect(next).toEqual({
      type: "image", name: "mito", source: "zarr3://http://h/mito__CFLOW_ARGS__b__CFLOW_ARGS__/", shader: "new range",
      blend: "additive", opacity: 0.5, crossSectionRenderScale: 0.5, tab: "rendering",
    });
  });

  it("keeps a segmentation's colouring, and its segments and equivalences only while the source is the same", () => {
    const seg = { type: "segmentation", name: "m", source: "zarr3://http://h/a/", selectedAlpha: 0.8, colorSeed: 7, segments: ["5"], equivalences: [["5", "6"]] };
    expect(carriedLayer(seg, { type: "segmentation", source: "zarr3://http://h/a/" }, "m")).toEqual(seg);
    expect(carriedLayer(seg, { type: "segmentation", source: "zarr3://http://h/b/" }, "m")).toEqual({
      type: "segmentation", name: "m", source: "zarr3://http://h/b/", selectedAlpha: 0.8, colorSeed: 7,
    });
  });

  it("compares sources by their URLs, whatever form Neuroglancer wrote them back in", () => {
    const written = { ...shown, source: [{ url: shown.source, transform: { outputDimensions: {} } }] };
    expect(carriedLayer(written, { type: "image", source: shown.source, shader: "user's" }, "mito")).toMatchObject({ localPosition: [2.5] });
  });

  it("keeps only being hidden across a change of type (an image layer become a segmentation)", () => {
    const hidden = { ...shown, visible: false };
    expect(carriedLayer(hidden, { type: "segmentation", source: "zarr3://http://h/labels/" }, "mito")).toEqual({
      type: "segmentation", name: "mito", source: "zarr3://http://h/labels/", visible: false,
    });
  });

  it("is the spec, named, when there was no layer", () => {
    expect(carriedLayer(undefined, { type: "image", source: "zarr3://a", name: "x" }, "mine")).toEqual({ type: "image", source: "zarr3://a", name: "mine" });
  });
});

describe("replaceLayers with a layer handed over", () => {
  const ft1 = { type: "image", name: "m-ft1", source: "zarr3://http://h/m-ft1/", shader: "s", blend: "additive" };
  const base = { type: "image", name: "m", source: "zarr3://http://h/m/", shader: "s" };

  it("puts the new layer in the old one's place, with its settings, and drops the old one", () => {
    const next = replaceLayers({ layers: [raw, base, ft1, model] }, { "m-ft2": { type: "image", source: "zarr3://http://h/m-ft2/", shader: "s" } }, { "m-ft2": "m-ft1" });
    expect(layersOf(next).map((l) => l.name)).toEqual(["data", "m", "m-ft2", "model"]);
    expect(layersOf(next)[2]).toEqual({ type: "image", name: "m-ft2", source: "zarr3://http://h/m-ft2/", shader: "s", blend: "additive" });
  });

  it("keeps the new layer's own settings when it is there already, and still drops the old one", () => {
    const ft2 = { type: "image", name: "m-ft2", source: "zarr3://http://h/m-ft2/", shader: "s", opacity: 0.2 };
    const next = replaceLayers({ layers: [ft1, ft2] }, { "m-ft2": { type: "image", source: ft2.source, shader: "s" } }, { "m-ft2": "m-ft1" });
    expect(layersOf(next)).toEqual([ft2]);
  });

  it("appends the new layer when the one it replaces is not there", () => {
    const next = replaceLayers({ layers: [base] }, { "m-ft2": { type: "image", source: "zarr3://x" } }, { "m-ft2": "m-ft1" });
    expect(layersOf(next).map((l) => l.name)).toEqual(["m", "m-ft2"]);
  });
});

describe("sourceUrls", () => {
  it("lists every URL of a string, a {url} and a list of them", () => {
    expect(sourceUrls("zarr://a")).toEqual(["zarr://a"]);
    expect(sourceUrls({ url: "zarr://a", transform: {} })).toEqual(["zarr://a"]);
    expect(sourceUrls([{ url: "zarr://a" }, "zarr://b"])).toEqual(["zarr://a", "zarr://b"]);
    expect(sourceUrls(undefined)).toEqual([]);
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

describe("pointerNm (ai_annotate._zyx_nm: the mouse state's voxel coordinates in nm)", () => {
  it("converts as positionNm does, from a Float32Array as Neuroglancer's mouse state holds it", () => {
    expect(pointerNm(new Float32Array([10.5, 20.5, 30.5]), state.dimensions)).toEqual([84, 164, 244]);
    expect(pointerNm([10.5, 20.5, 30.5], state.dimensions)).toEqual(positionNm(state));
  });
  it("puts the axes in z y x order by name, whatever order the viewer has them in", () => {
    expect(pointerNm([100, 200, 300], { x: [4e-9, "m"], y: [4e-9, "m"], z: [8e-9, "m"] })).toEqual([2400, 800, 400]);
  });
  it("is null before the pointer has been over the data (an empty or NaN position), or without dimensions", () => {
    expect(pointerNm(new Float32Array(0), state.dimensions)).toBeNull();  // Neuroglancer's initial position
    expect(pointerNm([NaN, NaN, NaN], state.dimensions)).toBeNull();
    expect(pointerNm([1, NaN, 3], state.dimensions)).toBeNull();
    expect(pointerNm([1, 2], state.dimensions)).toBeNull();
    expect(pointerNm(null, state.dimensions)).toBeNull();
    expect(pointerNm([1, 2, 3], undefined)).toBeNull();
    expect(pointerNm([1, 2, 3], { z: [8e-9, "m"], y: [8e-9, "m"] })).toBeNull();  // fewer than three axes (the Python refuses too)
  });
});

describe("depthAxisFromLayout (geometry.depth_axis_from_layout: docs/ai_annotate.md's table under Choosing the plane)", () => {
  it("maps each single-panel layout to the data axis it looks along, with z y x display dimensions", () => {
    expect(depthAxisFromLayout("yz")).toBe(0);  // the XY plane, at the hovered z
    expect(depthAxisFromLayout("xz")).toBe(1);  // the XZ plane, at the hovered y
    expect(depthAxisFromLayout("xy")).toBe(2);  // the YZ plane, at the hovered x
  });
  it("gives the first panel's (xy, the YZ plane) for 4panel, 3d and anything else", () => {
    for (const layout of ["4panel", "3d", "4panel-alt", "nonsense", "", null, undefined, 7, {}]) {
      expect(depthAxisFromLayout(layout), JSON.stringify(layout)).toBe(2);
    }
  });
  it("drops the -3d suffix, reads an object's type, and ignores case", () => {
    expect(depthAxisFromLayout("yz-3d")).toBe(0);
    expect(depthAxisFromLayout("xz-3d")).toBe(1);
    expect(depthAxisFromLayout("xy-3d")).toBe(2);
    expect(depthAxisFromLayout({ type: "yz", orthographicProjection: true })).toBe(0);
    expect(depthAxisFromLayout({ type: "xz-3d" })).toBe(1);
    expect(depthAxisFromLayout("YZ")).toBe(0);
  });
  it("lets the display dimensions' names decide which data axis each slot is", () => {
    // Shown x, y, z: the xy panel looks along slot 2, which is z.
    expect(depthAxisFromLayout("xy", ["x", "y", "z"])).toBe(0);
    expect(depthAxisFromLayout("xz", ["x", "y", "z"])).toBe(1);
    expect(depthAxisFromLayout("yz", ["x", "y", "z"])).toBe(2);
    // Shown y, x, z: xz looks along slot 1, which is x.
    expect(depthAxisFromLayout("xz", ["y", "x", "z"])).toBe(2);
    expect(depthAxisFromLayout("xz", ["z", "y", "x"])).toBe(1);  // the default, spelled out
  });
  it("is 0 when the slot's dimension is not z, y or x, or there are fewer than three", () => {
    expect(depthAxisFromLayout("xy", ["z", "y", "c"])).toBe(0);
    expect(depthAxisFromLayout("xy", ["z", "y"])).toBe(0);
    expect(depthAxisFromLayout("yz", ["t", "y", "x"])).toBe(0);
    expect(depthAxisFromLayout("xz", [])).toBe(1);  // no names given: the default z y x
  });
});

describe("viewDepthAxisOf (geometry.depth_axis_for_view)", () => {
  it("reads the state's layout over the first three dimensions", () => {
    expect(viewDepthAxisOf({ ...state, layout: "yz" })).toBe(0);
    expect(viewDepthAxisOf({ ...state, layout: "xz" })).toBe(1);
    expect(viewDepthAxisOf(state)).toBe(2);  // layout "xy"
    expect(viewDepthAxisOf({ ...state, layout: { type: "yz-3d" } })).toBe(0);
    expect(viewDepthAxisOf({ ...state, layout: "4panel" })).toBe(2);
  });
  it("prefers the display dimensions the user set", () => {
    expect(viewDepthAxisOf({ ...state, layout: "xy", displayDimensions: ["x", "y", "z"] })).toBe(0);
    expect(viewDepthAxisOf({ ...state, layout: "xy", displayDimensions: [] })).toBe(2);  // none set: the dimensions' order
  });
  it("displayDimensionsOf: those the state sets, else the first three dimensions, else none", () => {
    expect(displayDimensionsOf({ ...state, displayDimensions: ["x", "y", "z"] })).toEqual(["x", "y", "z"]);
    expect(displayDimensionsOf(state)).toEqual(Object.keys(state.dimensions!).slice(0, 3));
    expect(displayDimensionsOf(null)).toEqual([]);
  });
  it("takes the first three of more dimensions, in the viewer's order", () => {
    const withChannel: NgState = { dimensions: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"], "c^": [1, ""] }, layout: "xy" };
    expect(viewDepthAxisOf(withChannel)).toBe(0);
  });
  it("is the default viewer's (the xy panel over z y x) without a state", () => {
    expect(viewDepthAxisOf(null)).toBe(2);
    expect(viewDepthAxisOf({})).toBe(2);
    expect(viewDepthAxisOf({ layout: "yz" })).toBe(0);
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
