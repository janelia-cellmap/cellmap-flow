// The controller (src/viewer/index.ts) before it has an iframe: what the
// backend may call while the page is still booting. Everything after the
// iframe exists touches the DOM and is checked by typecheck and in a browser.
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionState, SessionStore } from "../../src/contracts";
import { createViewer, EXAMPLE_DATASET_URL, liveEquivalences, NG_PAGE } from "../../src/viewer/index";
import { decodeState, TAB_STATE_KEY } from "../../src/viewer/share";

function fakeSession(state: Partial<SessionState> = {}): SessionStore {
  const current: SessionState = {
    datasetUrl: null, chain: { input_norm: [], postprocess: [] }, models: [], resample: false, prefs: {}, ...state,
  };
  return {
    get state() { return current; },
    async update(change) { Object.assign(current, change); },
    onChange() { return () => {}; },
  };
}

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
