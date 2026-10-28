// The controller (src/viewer/index.ts) before it has an iframe: what the
// backend may call while the page is still booting. Everything after the
// iframe exists touches the DOM and is checked by typecheck and in a
// browser, except the first mount, which the tests below do against a
// stand-in document whose iframe only records the URL it is given: enough
// to read the state the viewer would open on; and a few reaches into a live
// viewer (a layer change, a paint layer re-read, a box the user drew), done
// against a stand-in `viewer` object on the iframe's window with the
// Neuroglancer members the controller touches.
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionState, SessionStore } from "../../src/contracts";
import type { DatasetInfo } from "../../src/contracts";
import { defaultState } from "../../src/state";
import { createViewer, NG_PAGE } from "../../src/viewer/index";
import { decodeState, TAB_STATE_KEY } from "../../src/viewer/share";
import type { NgLayer, NgState } from "../../src/viewer/state";

function fakeSession(state: Partial<SessionState> = {}): SessionStore {
  const current: SessionState = { ...defaultState(), chain: { input_norm: [], postprocess: [] }, ...state };
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
 * test can see) and the mouse state. */
function stubNgViewer(state: NgState) {
  let current = state;
  const viewer = {
    state: { toJSON: () => current, restoreState: (next: NgState) => { current = next; } },
    mouseState: { position: new Float32Array(0) as ArrayLike<number>, active: false },
  };
  return { viewer };
}

/** stubDocument with a live viewer in the iframe's window (its `viewer`
 * and a location whose hash is empty, as Neuroglancer's is until it writes
 * its state): `load()` fires the iframe's load event. */
function liveDocument(state: NgState) {
  const listeners: Record<string, (() => void)[]> = {};
  const stub = stubNgViewer(state);
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
  return { frame, stub, load: () => (listeners.load ?? []).forEach((fn) => fn()) };
}

const live: NgState = {
  dimensions: { z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] },
  position: [10.5, 20.5, 30.5],
  layers: [{ name: "data", type: "image", source: "zarr://https://example.org/em.zarr" }, { name: "annotation_v", ...paint, paintValue: "3", brushSize: 7 }],
  layout: "yz",
};

describe("createViewer before the iframe exists", () => {
  afterEach(() => vi.unstubAllGlobals());
  it("prepares a dataset for the reload: the state kept by the tab, no iframe, the viewer's URL answered", () => {
    // Only the tab's storage is touched (connect.js reloads the page a
    // second later); without a document here, mounting an iframe would throw.
    const items = new Map<string, string>();
    vi.stubGlobal("sessionStorage", { getItem: (k: string) => items.get(k) ?? null, setItem: (k: string, v: string) => void items.set(k, v), removeItem: (k: string) => void items.delete(k) });
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    const raw = { type: "image", source: "zarr://https://example.org/em.zarr" };
    const url = viewer.prepareDataset("https://example.org/em.zarr", raw);
    const hash = items.get(TAB_STATE_KEY)!;
    expect(hash).toMatch(/^#!/);
    expect(url).toBe(NG_PAGE + hash);
    expect(decodeState(hash)?.layers).toEqual([{ name: "data", ...raw }]);
    expect(viewer.viewerUrl()).toBeNull();  // nothing shown yet
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
    const viewer = createViewer({ session: fakeSession({ datasetUrl: LIVER.url }), log: () => {} });
    await viewer.restore();
    expect(frame.src).toBe("");  // waiting: nothing mounted without the geometry
    viewer.setLayers({ model: { type: "image", source: "zarr3://http://h/model/" } });  // queued meanwhile
    viewer.datasetOpened(LIVER.url, LIVER);
    const state = shown();
    expect(state.dimensions).toEqual({ z: [8e-9, "m"], y: [8e-9, "m"], x: [8e-9, "m"] });
    expect(state.position).toEqual([6373.5, 6364, 4466]);
    expect((state.layers as { name: string }[]).map((l) => l.name)).toEqual(["data", "model"]);
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
});

describe("a live viewer", () => {
  afterEach(() => vi.unstubAllGlobals());

  async function mounted(state: NgState = live) {
    const doc = liveDocument(state);
    const viewer = createViewer({ session: fakeSession(), log: () => {} });
    await viewer.openDataset("https://example.org/em.zarr", { type: "image", source: "zarr://https://example.org/em.zarr" });
    return { doc, viewer };
  }

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

  it("opens the viewer again on the new layers when Neuroglancer throws taking them, instead of throwing to the caller", async () => {
    const { doc, viewer } = await mounted();
    doc.load();
    doc.stub.viewer.state.restoreState = () => {
      throw new TypeError("Cannot set properties of undefined (setting 'localPositionValid')");
    };
    expect(() => viewer.setLayers({ "mito-ft2": { type: "image", source: "zarr3://http://h/mito-ft2/" } })).not.toThrow();
    const fresh = doc.frame.replacedBy as { src: string } | null;
    expect(fresh).not.toBeNull();
    const reopened = decodeState(fresh!.src.replace(/^[^#]*/, ""));
    expect((reopened?.layers as { name: string }[]).map((l) => l.name)).toEqual(["data", "annotation_v", "mito-ft2"]);
    // Taking a layer out goes the same way.
    expect(() => viewer.removeLayer("annotation_v")).not.toThrow();
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

  it("gives Run on a region the layer as the live viewer has it, with what the user drew in it since the page set it", async () => {
    const box = { type: "axis_aligned_bounding_box", id: "1", pointA: [0, 0, 0], pointB: [80, 160, 240] };
    const region = { type: "annotation", source: [{ url: "local://annotations" }], annotations: [] as unknown[] };
    const { doc, viewer } = await mounted();
    // Drawn in Neuroglancer: its own state has the box, no state the page set does.
    doc.stub.viewer.state.restoreState({ ...live, layers: [...(live.layers as NgLayer[]), { name: "run_region", ...region, annotations: [box] }] });
    expect(viewer.layer("run_region")).toEqual({ name: "run_region", ...region, annotations: [box] });
    expect(viewer.layer("gone")).toBeNull();
  });
});
