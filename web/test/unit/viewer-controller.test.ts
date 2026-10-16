// The controller (src/viewer/index.ts) before it has an iframe: what the
// backend may call while the page is still booting. Everything after the
// iframe exists touches the DOM and is checked by typecheck and in a
// browser, except the first mount, which the tests below do against a
// stand-in document whose iframe only records the URL it is given: enough
// to read the state the viewer would open on.
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionState, SessionStore } from "../../src/contracts";
import type { DatasetInfo } from "../../src/contracts";
import { createViewer, EXAMPLE_DATASET_URL, liveEquivalences, NG_PAGE, withSelectedLayer } from "../../src/viewer/index";
import { decodeState, TAB_STATE_KEY } from "../../src/viewer/share";
import type { NgState } from "../../src/viewer/state";

function fakeSession(state: Partial<SessionState> = {}): SessionStore {
  const current: SessionState = {
    datasetUrl: null, chain: { input_norm: [], postprocess: [] }, models: [], resample: false, prefs: {}, paintVolume: null, ...state,
  };
  return {
    get state() { return current; },
    async update(change) { Object.assign(current, change); },
    onChange() { return () => {}; },
  };
}

/** A document with nothing on it and an iframe that only records its URL:
 * what mountFrame touches. The frame's window stays unreadable (null), so
 * the controller reads the state back from the frame's src, as it does
 * while Neuroglancer is still loading. */
function stubDocument() {
  const frame = { id: "", title: "", style: { cssText: "" }, src: "", contentWindow: null, addEventListener() {}, removeEventListener() {} };
  vi.stubGlobal("document", { getElementById: () => null, querySelector: () => null, createElement: () => frame, body: { appendChild() {} } });
  vi.stubGlobal("sessionStorage", { getItem: () => null, setItem() {}, removeItem() {} });
  return { frame, shown: () => decodeState(frame.src.replace(/^[^#]*/, ""))! };
}

const paint = { type: "segmentation", source: { url: "s3+http://localhost/paint/v/annotation", subsources: { default: { writingEnabled: true }, bounds: {} } }, tab: "Draw" };

describe("createViewer before the iframe exists", () => {
  afterEach(() => vi.unstubAllGlobals());
  it("answers without a viewer", () => {
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    expect(viewer.position()).toBeNull();
    expect(viewer.viewerUrl()).toBeNull();
    expect(viewer.layerShader("data")).toBeNull();
  });
  it("takes layers to show once there is an iframe, and forgets a removed one, without touching the page", () => {
    const lines: string[] = [];
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    viewer.setLayers({ mito: { type: "image", source: "zarr3://a" }, er: { type: "image", source: "zarr3://b" } });
    viewer.removeLayer("er");
    expect(lines).toEqual([]);
  });
  it("prepares a dataset for the reload: the state kept by the tab, no iframe, the viewer's URL answered", () => {
    // Only the tab's storage is touched (connect.js reloads the page a
    // second later); without a document here, mounting an iframe would throw.
    const items = new Map<string, string>();
    vi.stubGlobal("sessionStorage", { getItem: (k: string) => items.get(k) ?? null, setItem: (k: string, v: string) => void items.set(k, v), removeItem: (k: string) => void items.delete(k) });
    const lines: string[] = [];
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    const raw = { type: "image", source: "zarr://https://example.org/em.zarr" };
    const url = viewer.prepareDataset("https://example.org/em.zarr", raw);
    const hash = items.get(TAB_STATE_KEY)!;
    expect(hash).toMatch(/^#!/);
    expect(url).toBe(NG_PAGE + hash);
    expect(decodeState(hash)?.layers).toEqual([{ name: "data", ...raw }]);
    expect(viewer.viewerUrl()).toBeNull();  // nothing shown yet
    expect(lines).toEqual(["viewer prepared for https://example.org/em.zarr: shown when the page reloads"]);
  });
  it("reaches a loaded segmentation layer's equivalence sets, and nothing else", () => {
    const sets = { clear() {}, restoreState() {} };
    const layers: Record<string, { layer: unknown }> = {
      mito: { layer: { displayState: { segmentationGroupState: { value: { segmentEquivalences: sets } } } } },
      raw: { layer: { displayState: {} } },  // an image layer
      loading: { layer: null },  // not loaded yet
    };
    const viewer = { layerManager: { getLayerByName: (name: string) => layers[name] as { layer: never } | undefined } };
    expect(liveEquivalences(viewer, "mito")).toBe(sets);
    expect(liveEquivalences(viewer, "raw")).toBeNull();
    expect(liveEquivalences(viewer, "loading")).toBeNull();
    expect(liveEquivalences(viewer, "gone")).toBeNull();
    expect(liveEquivalences({}, "mito")).toBeNull();  // an older Neuroglancer without layerManager on the viewer
  });
  it("names the Neuroglancer page beside this one and a real public dataset as the example", () => {
    expect(NG_PAGE).toBe("./ng/index.html");
    expect(EXAMPLE_DATASET_URL).toMatch(/^https:\/\/.*\.zarr\/recon-1\/em\/fibsem-uint8$/);
  });
});

describe("withSelectedLayer (s.selected_layer.layer = name; visible = True; layer.tab)", () => {
  const state: NgState = { layers: [{ name: "data", type: "image" }, { name: "annotation_v", type: "segmentation" }], layout: "xy" };

  it("selects the layer with its panel open, the rest of the state untouched", () => {
    expect(withSelectedLayer(state, "annotation_v")).toEqual({ ...state, selectedLayer: { layer: "annotation_v", visible: true } });
    expect(state.selectedLayer).toBeUndefined();
  });

  it("puts the layer's panel on the tab when one is given", () => {
    const next = withSelectedLayer(state, "annotation_v", "Draw")!;
    expect(next.layers).toEqual([{ name: "data", type: "image" }, { name: "annotation_v", type: "segmentation", tab: "Draw" }]);
    expect(next.selectedLayer).toEqual({ layer: "annotation_v", visible: true });
  });

  it("keeps the panel's other settings, and reads the oldest links' object form of layers", () => {
    const sized: NgState = { layers: { data: { type: "image" } }, selectedLayer: { layer: "other", visible: false, size: 400, side: "left" } };
    expect(withSelectedLayer(sized, "data", "Rendering")).toEqual({
      layers: [{ name: "data", type: "image", tab: "Rendering" }],
      selectedLayer: { layer: "data", visible: true, size: 400, side: "left" },
    });
  });

  it("is null for a layer the state has not got, which Neuroglancer would not select either", () => {
    expect(withSelectedLayer(state, "gone")).toBeNull();
    expect(withSelectedLayer({}, "data")).toBeNull();
  });
});

describe("selectLayer", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("touches nothing before there is an iframe", () => {
    const lines: string[] = [];
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    viewer.selectLayer("annotation_v", "Draw");
    expect(viewer.viewerUrl()).toBeNull();
    expect(lines).toEqual([]);
  });

  it("carries a selection made before the iframe into the first state, with the queued layers", async () => {
    const { shown } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    viewer.setLayers({ annotation_v: paint });
    viewer.selectLayer("annotation_v", "Draw");
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    const state = shown();
    expect(state.layers).toEqual([
      { name: "data", type: "image", source: "zarr://https://example.org/em.zarr" },
      { name: "annotation_v", ...paint },
    ]);
    expect(state.selectedLayer).toEqual({ layer: "annotation_v", visible: true });
  });

  it("selects a layer the viewer has not got yet when setLayers brings it, and not before", async () => {
    const { frame, shown } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    const opened = frame.src;
    viewer.selectLayer("annotation_v", "Draw");
    expect(frame.src).toBe(opened);  // nothing to select yet: the viewer is left alone
    viewer.setLayers({ annotation_v: paint });
    const state = shown();
    expect(state.selectedLayer).toEqual({ layer: "annotation_v", visible: true });
    expect(state.layers).toEqual([
      { name: "data", type: "image", source: "zarr://https://example.org/em.zarr" },
      { name: "annotation_v", ...paint },
    ]);
    // Once selected, a later setLayers does not select again (the user may have moved on).
    viewer.setLayers({ mito: { type: "image", source: "zarr3://m" } });
    expect(shown().selectedLayer).toEqual({ layer: "annotation_v", visible: true });  // Neuroglancer's own state keeps it; nothing here changed it
  });

  it("selects a layer the viewer has at once, its panel on the tab", async () => {
    const { shown } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    viewer.setLayers({ annotation_v: { ...paint, tab: undefined } });
    viewer.selectLayer("annotation_v", "Draw");
    const state = shown();
    expect(state.selectedLayer).toEqual({ layer: "annotation_v", visible: true });
    expect((state.layers as { name: string; tab?: string }[]).find((l) => l.name === "annotation_v")?.tab).toBe("Draw");
  });

  it("forgets a pending selection of a layer that is removed", async () => {
    const { shown } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    viewer.selectLayer("annotation_v", "Draw");
    viewer.removeLayer("annotation_v");
    viewer.setLayers({ annotation_v: paint });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    expect(shown().selectedLayer).toBeUndefined();
  });
});

/** jrc_mus-liver's raw as the engine reads it: 8 nm at its finest, voxel 0's centre at 4 nm. */
const LIVER: DatasetInfo = {
  url: "https://example.org/liver.zarr",
  dtype: "uint8",
  channels: 1,
  axes: [{ name: "z", unit: "nanometer" }, { name: "y", unit: "nanometer" }, { name: "x", unit: "nanometer" }],
  levels: [
    { shape: [8932, 12728, 12747], voxel: [8, 8, 8], origin: [0, 0, 0] },
    { shape: [4466, 6364, 6373], voxel: [16, 16, 16], origin: [4, 4, 4] },
  ],
};

describe("datasetOpened", () => {
  afterEach(() => vi.unstubAllGlobals());

  /** A page with no `#!` state of its own (restore() reads the address bar first). */
  function bareAddress() {
    vi.stubGlobal("location", { hash: "", href: "http://localhost/" });
    vi.stubGlobal("history", { replaceState() {} });
  }

  it("mounts a viewer restored from the session alone once the volume is read, with its dimensions", async () => {
    const { frame, shown } = stubDocument();
    bareAddress();
    const lines: string[] = [];
    const viewer = createViewer({ session: fakeSession({ datasetUrl: LIVER.url }), log: (l) => lines.push(l) });
    await viewer.restore();
    expect(frame.src).toBe("");  // waiting: nothing mounted without the geometry
    viewer.setLayers({ model: { type: "image", source: "zarr3://http://h/model/" } });  // queued meanwhile
    viewer.datasetOpened(LIVER.url, LIVER);
    const state = shown();
    expect(state.dimensions).toEqual({ z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] });
    expect(state.position).toEqual([4466, 6364, 6373.5]);
    expect((state.layers as { name: string }[]).map((l) => l.name)).toEqual(["data", "model"]);
    expect(lines).toContain(`viewer restored on ${LIVER.url}`);
    const src = frame.src;
    viewer.datasetOpened(LIVER.url, LIVER);
    expect(frame.src).toBe(src);  // once
  });

  it("mounts it without the geometry when the volume cannot be opened, for Neuroglancer to try", async () => {
    const { shown } = stubDocument();
    bareAddress();
    const viewer = createViewer({ session: fakeSession({ datasetUrl: LIVER.url }), log: () => {} });
    await viewer.restore();
    viewer.datasetOpened(LIVER.url, null);
    expect(shown().dimensions).toBeUndefined();
    expect((shown().layers as { name: string }[]).map((l) => l.name)).toEqual(["data"]);
  });

  it("leaves a viewer that did not wait alone: one opened with its geometry, or another dataset's", async () => {
    const { frame } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset(LIVER.url, { type: "image", source: `zarr://${LIVER.url}` }, LIVER);
    const src = frame.src;
    viewer.datasetOpened(LIVER.url, LIVER);
    expect(frame.src).toBe(src);
    const other = stubDocument();
    bareAddress();
    const waiting = createViewer({ session: fakeSession({ datasetUrl: LIVER.url }), log: () => {} });
    await waiting.restore();
    waiting.datasetOpened("https://example.org/other.zarr", LIVER);
    expect(other.frame.src).toBe("");
  });
});
