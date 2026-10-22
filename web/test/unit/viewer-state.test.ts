// The pure functions over a Neuroglancer state's JSON (src/viewer/state.ts):
// replacing layers as the Python dashboard's replace_layer does, keeping the
// user's settings on them (carriedLayer), the position in nm, the data axis
// the layout looks along (cellmap_flow/ai_annotate/geometry.py's
// depth_axis_for_view), the dataset a state shows.
import { describe, expect, it } from "vitest";

import { carriedLayer, datasetUrlFromState, layersOf, type NgState, positionNm, replaceLayers, viewDepthAxisOf } from "../../src/viewer/state";

const raw = { type: "image", name: "data", source: "zarr://https://x/dataset.zarr/recon-1/em/fibsem-uint8", shader: "void main(){}" };
const model = { type: "image", name: "model", source: "zarr3://http://h/m/", shader: "old", visible: false };
const state: NgState = {
  dimensions: { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] },
  position: [10.5, 20.5, 30.5],
  layers: [raw, model],
  layout: "xy",
};

describe("replaceLayers (replace_layer)", () => {
  it("replaces a layer in place and keeps it hidden where it was hidden", () => {
    const next = replaceLayers(state, { model: { type: "image", source: "zarr3://http://h/m2/", shader: "new" } });
    expect(next.layers).toEqual([raw, { type: "image", name: "model", source: "zarr3://http://h/m2/", shader: "new", visible: false }]);
    // The rest of the state, and the input, are untouched.
    expect(next.position).toEqual([10.5, 20.5, 30.5]);
    expect(next.layout).toBe("xy");
    expect(state.layers).toEqual([raw, model]);
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
});

describe("positionNm", () => {
  it("matches axes by name, so x y z dimensions still give z y x", () => {
    const xyz: NgState = { dimensions: { x: [4e-9, "m"], y: [4e-9, "m"], z: [8e-9, "m"] }, position: [100, 200, 300] };
    expect(positionNm(xyz)).toEqual([2400, 800, 400]);
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
  it("takes the first three of more dimensions, in the viewer's order", () => {
    const withChannel: NgState = { dimensions: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"], "c^": [1, ""] }, layout: "xy" };
    expect(viewDepthAxisOf(withChannel)).toBe(0);
  });
});

describe("datasetUrlFromState", () => {
  it("gives the raw layer's URL without its format", () => {
    expect(datasetUrlFromState(state)).toBe("https://x/dataset.zarr/recon-1/em/fibsem-uint8");
    expect(datasetUrlFromState({ layers: [{ name: "data", type: "image", source: [{ url: "n5://s3://bucket/x.n5/em" }] }] })).toBe("s3://bucket/x.n5/em");
    expect(datasetUrlFromState({ layers: [{ name: "data", type: "image", source: "precomputed://gs://bucket/vol" }] })).toBe("gs://bucket/vol");
  });
});
