// The controller (src/viewer/index.ts) before it has an iframe: what the
// backend may call while the page is still booting. Everything after the
// iframe exists touches the DOM and is checked by typecheck and in a
// browser, except the first mount, which the tests below do against a
// stand-in document whose iframe only records the URL it is given: enough
// to read the state the viewer would open on; and the reach into a live
// viewer for the AI-assisted annotation (the pointer, the layout's depth
// axis, a key binding, a paint layer re-read), done against a stand-in
// `viewer` object on the iframe's window with the few Neuroglancer members
// the controller touches (mouseState, inputEventBindings.global,
// bindAction, state).
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionState, SessionStore } from "../../src/contracts";
import type { DatasetInfo } from "../../src/contracts";
import { createViewer, EXAMPLE_DATASET_URL, liveEquivalences, NG_PAGE, withSelectedLayer } from "../../src/viewer/index";
import { decodeState, TAB_STATE_KEY } from "../../src/viewer/share";
import type { NgLayer, NgState } from "../../src/viewer/state";

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

/** A stand-in for Neuroglancer's `window.viewer`, with the members the
 * controller reaches: the state as JSON (restoreState replaces it, as a
 * test can see), the mouse state, the global key map and bindAction, which
 * record what was bound; `press(action)` runs the action's listeners as a
 * key press would. */
function stubNgViewer(state: NgState) {
  let current = state;
  const keys = new Map<string, string>();
  const listeners = new Map<string, ((event: Event) => void)[]>();
  const viewer = {
    state: { toJSON: () => current, restoreState: (next: NgState) => { current = next; } },
    mouseState: { position: new Float32Array(0) as ArrayLike<number>, active: false },
    inputEventBindings: { global: { set: (key: string, action: string) => void keys.set(key, action), delete: (key: string) => void keys.delete(key) } },
    bindAction: (action: string, handler: (event: Event) => void) => void listeners.set(action, [...(listeners.get(action) ?? []), handler]),
    bindActionCalls: 0,
  };
  const bindAction = viewer.bindAction;
  viewer.bindAction = (action, handler) => { viewer.bindActionCalls++; bindAction(action, handler); };
  return { viewer, keys, listeners, press: (action: string) => (listeners.get(action) ?? []).forEach((fn) => fn(new Event("action"))) };
}

/** stubDocument with a live viewer in the iframe's window (its `viewer`
 * and a location whose hash is empty, as Neuroglancer's is until it writes
 * its state): `replaceViewer` is a new page of Neuroglancer (a new dataset
 * opened), `load()` fires the iframe's load event. */
function liveDocument(state: NgState) {
  const listeners: Record<string, (() => void)[]> = {};
  let stub = stubNgViewer(state);
  // The window also takes the mirror's hashchange listener (share.ts).
  const win = { viewer: stub.viewer, location: { hash: "" }, addEventListener() {}, removeEventListener() {} };
  const frame = {
    id: "", title: "", style: { cssText: "" }, src: "", contentWindow: win,
    addEventListener: (type: string, fn: () => void) => void (listeners[type] = [...(listeners[type] ?? []), fn]),
    removeEventListener: (type: string, fn: () => void) => void (listeners[type] = (listeners[type] ?? []).filter((f) => f !== fn)),
    // A clone is a new iframe, a fresh page with no viewer yet; replaceWith records it.
    replacedBy: null as Record<string, unknown> | null,
    cloneNode() {
      return { ...frame, src: "", contentWindow: { location: { hash: "" }, addEventListener() {}, removeEventListener() {} }, removeAttribute(name: string) { if (name === "src") this.src = ""; } };
    },
    replaceWith(fresh: Record<string, unknown>) { frame.replacedBy = fresh; },
  };
  vi.stubGlobal("document", { getElementById: () => null, querySelector: () => null, createElement: () => frame, body: { appendChild() {} } });
  vi.stubGlobal("sessionStorage", { getItem: () => null, setItem() {}, removeItem() {} });
  return {
    frame,
    get stub() { return stub; },
    load: () => (listeners.load ?? []).forEach((fn) => fn()),
    replaceViewer(next: NgState) {
      stub = stubNgViewer(next);
      win.viewer = stub.viewer;
      return stub;
    },
  };
}

const live: NgState = {
  dimensions: { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] },
  position: [10.5, 20.5, 30.5],
  layers: [{ name: "data", type: "image", source: "zarr://https://example.org/em.zarr" }, { name: "annotation_v", ...paint, paintValue: "3", brushSize: 7 }],
  layout: "yz",
};

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
    expect(state.position).toEqual([6373.5, 6364, 4466]);
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

describe("the reach into a live viewer for AI-assisted annotation", () => {
  afterEach(() => vi.unstubAllGlobals());

  async function mounted(state: NgState = live) {
    const doc = liveDocument(state);
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    return { doc, viewer };
  }

  it("pointerPosition: the mouse state's voxel coordinates in nm, z y x, while the pointer is over a panel", async () => {
    const { doc, viewer } = await mounted();
    expect(viewer.pointerPosition()).toBeNull();  // not over the viewer yet (active false, position empty)
    doc.stub.viewer.mouseState.position = new Float32Array([10.5, 20.5, 30.5]);
    expect(viewer.pointerPosition()).toBeNull();  // a position, but the pointer has left (active false)
    doc.stub.viewer.mouseState.active = true;
    expect(viewer.pointerPosition()).toEqual([84, 164, 244]);
    doc.stub.viewer.mouseState.position = new Float32Array([NaN, NaN, NaN]);
    expect(viewer.pointerPosition()).toBeNull();
  });

  it("pointerPosition is null without a viewer, or with one that has no mouse state", async () => {
    expect(createViewer({ session: fakeSession(), log: () => {} }).pointerPosition()).toBeNull();
    const { doc, viewer } = await mounted();
    delete (doc.stub.viewer as { mouseState?: unknown }).mouseState;
    expect(viewer.pointerPosition()).toBeNull();
  });

  it("viewDepthAxis: the live state's layout over its dimensions; the default viewer's without one", async () => {
    const { doc, viewer } = await mounted();
    expect(viewer.viewDepthAxis()).toBe(0);  // "yz": the XY plane
    doc.stub.viewer.state.restoreState({ ...live, layout: "xz" });
    expect(viewer.viewDepthAxis()).toBe(1);
    doc.stub.viewer.state.restoreState({ ...live, layout: "4panel" });
    expect(viewer.viewDepthAxis()).toBe(2);
    doc.stub.viewer.state.restoreState({ ...live, layout: "xy", displayDimensions: ["x", "y", "z"] });
    expect(viewer.viewDepthAxis()).toBe(0);
    expect(createViewer({ session: fakeSession(), log: () => {} }).viewDepthAxis()).toBe(2);
  });

  it("holds Neuroglancer's coordinate space through a layer change, so its x, y, z order stays", async () => {
    // Without the hold, restoreState's dropping every layer emptied the
    // space, and the new layers made it z, y, x again with the position's
    // numbers kept: the brush then painted across the slice shown.
    const { doc, viewer } = await mounted();
    let holds = 0;
    const heldDuring: boolean[] = [];
    (doc.stub.viewer as unknown as Record<string, unknown>).layerSpecification = { coordinateSpaceCombiner: { retain: () => (holds++, () => void holds--) } };
    const restore = doc.stub.viewer.state.restoreState;
    doc.stub.viewer.state.restoreState = (next: NgState) => { heldDuring.push(holds > 0); restore(next); };
    viewer.setLayers({ boxes: { type: "annotation", source: "local://annotations" } });
    expect(heldDuring).toEqual([true]);
    expect(holds).toBe(0);
  });

  it("viewDepthAxis reads the state in the iframe's hash while Neuroglancer is still loading", async () => {
    const { shown } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    expect(shown().layout).toBe("xy");  // DEFAULT_LAYOUT
    expect(viewer.viewDepthAxis()).toBe(2);
  });

  it("bindKey binds the key to the action in the global map and the handler to the action, once per viewer", async () => {
    const { doc, viewer } = await mounted();
    const presses: string[] = [];
    viewer.bindKey("shift+keyg", "ai-annotate", () => presses.push("g"));
    expect(doc.stub.keys.get("shift+keyg")).toBe("ai-annotate");
    expect(doc.stub.viewer.bindActionCalls).toBe(1);
    // The viewer is noticed again and again (the load, the mirror's poll, every read): bound once all the same.
    doc.load();
    doc.load();
    viewer.position();
    viewer.pointerPosition();
    expect(doc.stub.viewer.bindActionCalls).toBe(1);
    doc.stub.press("ai-annotate");
    expect(presses).toEqual(["g"]);
  });

  it("bindKey puts the bindings on a new viewer when it appears, exactly once each", async () => {
    const { doc, viewer } = await mounted();
    const presses: string[] = [];
    viewer.bindKey("shift+keyg", "ai-annotate", () => presses.push("g"));
    viewer.bindKey("keyu", "ai-undo", () => presses.push("u"));
    expect(doc.stub.viewer.bindActionCalls).toBe(2);
    // A new dataset: a new page of Neuroglancer, a new viewer object with no bindings.
    const next = doc.replaceViewer({ ...live, layout: "xz" });
    expect(next.keys.size).toBe(0);
    doc.load();
    expect(next.keys.get("shift+keyg")).toBe("ai-annotate");
    expect(next.keys.get("keyu")).toBe("ai-undo");
    expect(next.viewer.bindActionCalls).toBe(2);
    doc.load();
    viewer.viewDepthAxis();
    expect(next.viewer.bindActionCalls).toBe(2);
    next.press("ai-annotate");
    next.press("ai-undo");
    expect(presses).toEqual(["g", "u"]);
  });

  it("bindKey before there is a viewer binds when one appears; the unbind takes the key out and silences the handler", async () => {
    const doc = liveDocument(live);
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    const presses: string[] = [];
    const unbind = viewer.bindKey("shift+keyg", "ai-annotate", () => presses.push("g"));
    expect(doc.stub.keys.size).toBe(0);  // no iframe yet
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    doc.load();
    expect(doc.stub.keys.get("shift+keyg")).toBe("ai-annotate");
    doc.stub.press("ai-annotate");
    unbind();
    expect(doc.stub.keys.has("shift+keyg")).toBe(false);
    doc.stub.press("ai-annotate");  // Neuroglancer's listener is still there; it does nothing now
    expect(presses).toEqual(["g"]);
    unbind();  // twice is fine
    // A viewer that appears later does not get the unbound binding.
    const next = doc.replaceViewer(live);
    doc.load();
    expect(next.keys.size).toBe(0);
    expect(next.viewer.bindActionCalls).toBe(0);
  });

  it("bindKey leaves a key another binding still wants in the map when one of them is unbound", async () => {
    const { doc, viewer } = await mounted();
    const unbindA = viewer.bindKey("shift+keyg", "a", () => {});
    viewer.bindKey("shift+keyg", "b", () => {});
    unbindA();
    expect(doc.stub.keys.get("shift+keyg")).toBe("b");
  });

  it("bindKey says once per viewer, and binds nothing, when the viewer has no bindings API", async () => {
    const lines: string[] = [];
    const doc = liveDocument(live);
    delete (doc.stub.viewer as { bindAction?: unknown }).bindAction;
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    viewer.bindKey("shift+keyg", "ai-annotate", () => {});
    viewer.bindKey("keyu", "ai-undo", () => {});
    doc.load();
    expect(doc.stub.keys.size).toBe(0);
    expect(lines.filter((line) => line.includes("bound"))).toEqual(["this viewer has no key bindings API: shift+keyg is not bound"]);
  });

  it("refreshLayer respells the paint layer's source URL through the state, keeping the layer's settings and the rest of the state", async () => {
    const { doc, viewer } = await mounted();
    expect(viewer.refreshLayer("annotation_v")).toBe(true);
    const state = doc.stub.viewer.state.toJSON();
    const layers = state.layers as Record<string, unknown>[];
    expect(layers.map((l) => l.name)).toEqual(["data", "annotation_v"]);
    expect(layers[1]).toEqual({
      name: "annotation_v", ...paint, paintValue: "3", brushSize: 7,
      source: { url: "s3+http://localhost/paint/v~1/annotation", subsources: { default: { writingEnabled: true }, bounds: {} } },
    });
    expect(layers[0]).toEqual(live.layers![0 as never]);
    expect(state.position).toEqual([10.5, 20.5, 30.5]);
    expect(state.layout).toBe("yz");
    // Again: the suffix counts up.
    expect(viewer.refreshLayer("annotation_v")).toBe(true);
    const again = (doc.stub.viewer.state.toJSON().layers as Record<string, unknown>[])[1].source as { url: string };
    expect(again.url).toBe("s3+http://localhost/paint/v~2/annotation");
  });

  it("refreshLayer opens the viewer again on the re-read layer when Neuroglancer throws taking the state, and says so", async () => {
    const lines: string[] = [];
    const doc = liveDocument(live);
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    doc.load();
    doc.stub.viewer.state.restoreState = () => {
      throw new Error(`Error restoring property "layout": Cannot read properties of undefined (reading 'generation')`);
    };
    expect(viewer.refreshLayer("annotation_v")).toBe(true);
    const fresh = doc.frame.replacedBy as { src: string } | null;
    expect(fresh).not.toBeNull();
    const reopened = decodeState(fresh!.src.replace(/^[^#]*/, ""));
    expect((reopened?.layers as Record<string, unknown>[])[1].source).toEqual({
      url: "s3+http://localhost/paint/v~1/annotation", subsources: { default: { writingEnabled: true }, bounds: {} },
    });
    expect(reopened?.layout).toBe("yz");
    expect(lines.some((l) => l.includes("could not take the re-read annotation_v layer") && l.includes("opening it again"))).toBe(true);
  });

  it("marks the mouse off the viewer while a state goes in, so Neuroglancer's hover tracking skips the half-made layers", async () => {
    const doc = liveDocument(live);
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    doc.load();
    // The pointer over a panel, as when a finetune's new model comes in while the user paints.
    const mouse = doc.stub.viewer.mouseState as { active: boolean; setActive?(active: boolean): void };
    mouse.active = true;
    const told: boolean[] = [];
    mouse.setActive = (active) => { told.push(active); mouse.active = active; };
    let activeDuringRestore: boolean | null = null;
    const restore = doc.stub.viewer.state.restoreState;
    doc.stub.viewer.state.restoreState = (next: NgState) => { activeDuringRestore = mouse.active; restore(next); };
    viewer.setLayers({ "mito-ft2": { type: "image", source: "zarr3://http://h/mito-ft2/" } });
    expect(activeDuringRestore).toBe(false);
    expect(told).toEqual([false]);
    expect((doc.stub.viewer.state.toJSON().layers as { name: string }[]).map((l) => l.name)).toContain("mito-ft2");
  });

  it("opens the viewer again on the new layers when Neuroglancer throws taking them, instead of throwing to the caller", async () => {
    const lines: string[] = [];
    const doc = liveDocument(live);
    const viewer = createViewer({ session: fakeSession(), log: (l) => lines.push(l) });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    doc.load();
    doc.stub.viewer.state.restoreState = () => {
      throw new TypeError("Cannot set properties of undefined (setting 'localPositionValid')");
    };
    expect(() => viewer.setLayers({ "mito-ft2": { type: "image", source: "zarr3://http://h/mito-ft2/" } })).not.toThrow();
    const fresh = doc.frame.replacedBy as { src: string } | null;
    expect(fresh).not.toBeNull();
    const reopened = decodeState(fresh!.src.replace(/^[^#]*/, ""));
    expect((reopened?.layers as { name: string }[]).map((l) => l.name)).toEqual(["data", "annotation_v", "mito-ft2"]);
    expect(lines.some((l) => l.includes("could not take the layer mito-ft2") && l.includes("opening it again"))).toBe(true);
    // Taking a layer out goes the same way.
    expect(() => viewer.removeLayer("annotation_v")).not.toThrow();
  });

  it("refreshLayer keeps a hidden paint layer hidden", async () => {
    const hidden: NgState = { ...live, layers: [{ name: "annotation_v", ...paint, visible: false }] };
    const { doc, viewer } = await mounted(hidden);
    expect(viewer.refreshLayer("annotation_v")).toBe(true);
    expect((doc.stub.viewer.state.toJSON().layers as Record<string, unknown>[])[0].visible).toBe(false);
  });

  it("refreshLayer is false for a layer that is not there, one that is not a paint layer, and without a viewer", async () => {
    const { doc, viewer } = await mounted();
    const before = doc.stub.viewer.state.toJSON();
    expect(viewer.refreshLayer("gone")).toBe(false);
    expect(viewer.refreshLayer("data")).toBe(false);  // the raw layer: not the paint route's
    expect(doc.stub.viewer.state.toJSON()).toBe(before);  // nothing restored
    expect(createViewer({ session: fakeSession(), log: () => {} }).refreshLayer("annotation_v")).toBe(false);
  });

  it("refreshLayer works through the iframe's hash while Neuroglancer is still loading", async () => {
    const { shown } = stubDocument();
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    viewer.setLayers({ annotation_v: paint });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    expect(viewer.refreshLayer("annotation_v")).toBe(true);
    const layer = (shown().layers as Record<string, unknown>[]).find((l) => l.name === "annotation_v")!;
    expect((layer.source as { url: string }).url).toBe("s3+http://localhost/paint/v~1/annotation");
  });
});

describe("layer: a layer as the viewer has it, for Run on a region's box", () => {
  afterEach(() => vi.unstubAllGlobals());
  const box = { type: "axis_aligned_bounding_box", id: "1", pointA: [0, 0, 0], pointB: [80, 160, 240] };
  const region = { type: "annotation", source: [{ url: "local://annotations" }], annotations: [] as unknown[] };

  it("is the layer in the live viewer's state, with what the user drew in it since the page set it", async () => {
    const doc = liveDocument(live);
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    // Drawn in Neuroglancer: its own state has the box, no state the page set does.
    doc.stub.viewer.state.restoreState({ ...live, layers: [...(live.layers as NgLayer[]), { name: "run_region", ...region, annotations: [box] }] });
    expect(viewer.layer("run_region")).toEqual({ name: "run_region", ...region, annotations: [box] });
    expect(viewer.layer("gone")).toBeNull();
  });

  it("is the layer queued before there is an iframe, and the one in the iframe's hash while Neuroglancer loads", async () => {
    const early = createViewer({ session: fakeSession(), log: () => {} });
    early.setLayers({ run_region: region });
    expect(early.layer("run_region")).toEqual({ ...region, name: "run_region" });
    expect(early.layer("data")).toBeNull();
    stubDocument();
    const loading = createViewer({ session: fakeSession(), log: () => {} });
    loading.setLayers({ run_region: { ...region, annotations: [box] } });
    await loading.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    expect(loading.layer("run_region")).toEqual({ name: "run_region", ...region, annotations: [box] });
  });
});
