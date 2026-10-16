// The viewer controller: the Neuroglancer iframe and the state it shows.
//
// In server mode the dashboard's JavaScript never touches Neuroglancer:
// Python builds every layer through the neuroglancer package and the page
// embeds the Python viewer in <iframe id="my_iframe">. In the browser there
// is no Python, so this controller owns the iframe (a same-origin
// Neuroglancer build at ./ng/, so its window is reachable) and the state in
// it: the layers (layers.ts, the port of viewer/layers.py), the position,
// and the page's `#!` hash that mirrors the viewer's (share.ts).
//
// Two ways to change the viewer. When Neuroglancer's page has made its
// viewer (it puts it on its window as `viewer`), a state is applied through
// `viewer.state.restoreState(json)`, built from `viewer.state.toJSON()`:
// no reload, the camera stays. Until then (the iframe still loading, say)
// the state goes into the iframe's URL hash, which Neuroglancer reads when
// it starts, or on hashchange if it is already up: the hash is the queue.
// Either way the tab keeps a copy of the viewer's hash (share.ts), for the
// reload connect.js does after Connect and any other reload of the tab.
//
// Three things reach into the live viewer beyond its state, for the
// AI-assisted annotation (the Finetune tab's Shift+G; the Python counterpart
// is dashboard/routes/finetune/ai_annotate.py, which reaches the Python
// viewer the same three ways): where the pointer is (`viewer.mouseState`),
// a key bound to an action (`viewer.inputEventBindings.global` and
// `viewer.bindAction`), and a paint layer re-read after labels were written
// behind its back (through the state, its source URL respelled as
// overlay.refresh_annotation_layer respells the server's). A new dataset is
// a new page of Neuroglancer and so a new `viewer` object, which forgets
// its bindings: they are kept here and put on each viewer as it appears.
import type { DatasetInfo, LayerSpec, SessionStore, ViewerController } from "../contracts";
import { initialState, RAW_LAYER_NAME, rawLayer, respelledPaintSource, stateWithoutInfo } from "./layers";
import { withEquivalences } from "./equivalences";
import { decodeState, encodeState, isStateHash, mirrorFrameHash, rememberedState, rememberState, withHash } from "./share";
import { datasetUrlFromState, layerShaderOf, layersOf, type NgState, pointerNm, positionNm, replaceLayers, viewDepthAxisOf, withoutLayer } from "./state";

/** Neuroglancer's page, built beside this one (vite.config.ts serves ./ng/). */
export const NG_PAGE = "./ng/index.html";

/** The ids the dashboard's template (templates/index.html) gives the viewer's
 * iframe and, without a viewer, the connect panel and its parts. The
 * dashboard's own JavaScript looks them up (connect.js, the finetune tab's
 * reloadViewer), so they are kept. */
const FRAME_ID = "my_iframe";
const PANEL_ID = "noNeuroglancerPanel";
const COLUMN_ID = "iframe-column";
const PATH_INPUT_ID = "datasetPathInput";

/** A public OME-Zarr volume with CORS, as the connect field's example: the
 * raw EM of an OpenOrganelle dataset. */
export const EXAMPLE_DATASET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8";

/** The connect panel's text in the browser, in place of the server's
 * "No Neuroglancer URL configured. Enter a dataset path below". */
const CONNECT_DESCRIPTION = "No dataset open. Enter the URL of a public OME-Zarr volume (one served with CORS, "
  + "such as an OpenOrganelle dataset), or pick a folder on this machine, to open it in the viewer. The models run in this browser.";

/** What Neuroglancer's default page puts on its window: its state, its
 * layers (for a segmentation layer's equivalence sets, changed in place),
 * where the pointer is, and its key bindings. The names are Neuroglancer's
 * public ones (viewer.ts), checked against the fork's bundle: `mouseState`
 * is its MouseSelectionState (`position`, a Float32Array of voxel
 * coordinates in the viewer's coordinate space, empty until the pointer
 * has been over a panel; `active` while it is over one), `inputEventBindings
 * .global` the EventActionMap that keys pressed anywhere over the viewer
 * go through (`set(key, action)`, `delete(key)`; the Python's
 * `input_event_bindings.viewer`), and `bindAction(action, handler)` listens
 * for the action on the viewer's element (an `action:<name>` event). The
 * last three are optional here so that an older build without them fails
 * soft: no binding, not an exception. */
interface NgViewer {
  state: { toJSON(): NgState; restoreState(state: NgState): void };
  layerManager?: { getLayerByName(name: string): { layer: NgUserLayer | null } | undefined };
  mouseState?: { position?: ArrayLike<number>; active?: boolean };
  inputEventBindings?: { global?: { set(key: string, action: string): void; delete?(key: string): void } };
  bindAction?(action: string, handler: (event: Event) => void): void;
}

/** A key binding the page asked for (bindKey): put on every viewer as it
 * appears, `appliedTo` the viewers that have it so that none gets it twice
 * (Neuroglancer would run the handler once per bindAction). `active` is
 * cleared by the unbind: a viewer's own listener cannot be removed through
 * bindAction (it registers a disposer, and returns nothing), so a listener
 * left on a viewer checks it and does nothing. */
interface KeyBinding {
  key: string;
  action: string;
  handler: () => void;
  active: boolean;
  appliedTo: WeakSet<object>;
}

/** A loaded layer, as far as this file reaches into it: a segmentation
 * layer's disjoint sets of equivalent ids (SharedDisjointUint64Sets). */
interface NgUserLayer {
  displayState?: { segmentationGroupState?: { value?: { segmentEquivalences?: NgEquivalences } } };
}
export interface NgEquivalences {
  clear(): void;
  /** Sets of decimal uint64 strings, as a layer's `equivalences` JSON. */
  restoreState(sets: string[][]): void;
}

/** The live viewer's equivalence sets for layer `name`: null when the layer
 * is not there, not loaded yet, or not a segmentation layer. */
export function liveEquivalences(viewer: Pick<NgViewer, "layerManager">, name: string): NgEquivalences | null {
  const layer = viewer.layerManager?.getLayerByName(name)?.layer;
  const sets = layer?.displayState?.segmentationGroupState?.value?.segmentEquivalences;
  return sets && typeof sets.restoreState === "function" && typeof sets.clear === "function" ? sets : null;
}

/** `state` with layer `name` selected and its panel open (Neuroglancer's
 * `selectedLayer: {layer, visible}`, what the Python API writes for
 * `s.selected_layer.layer = name; s.selected_layer.visible = True`; the
 * panel's other settings, its side and size, kept) and, with `tab`, its
 * panel on that tab (the layer's own `tab` key, as `layer.tab = "Draw"`
 * sets it). Null when `state` has no layer of that name: Neuroglancer
 * clears a selection naming a layer it has not got, so the selection waits
 * until the layer is set. */
export function withSelectedLayer(state: NgState, name: string, tab?: string): NgState | null {
  const layers = layersOf(state);
  if (!layers.some((layer) => layer.name === name)) return null;
  const panel = state.selectedLayer;
  const kept = panel && typeof panel === "object" && !Array.isArray(panel) ? (panel as Record<string, unknown>) : {};
  const next: NgState = { ...state, selectedLayer: { ...kept, layer: name, visible: true } };
  if (tab !== undefined) next.layers = layers.map((layer) => (layer.name === name ? { ...layer, tab } : layer));
  return next;
}

/** The page boots with data-viewer="pending" on <html> when the session has
 * a dataset (index.html's inline script), which keeps the connect panel
 * invisible (browser.css) until the viewer replaces it; cleared when the
 * viewer is mounted, or when restore() finds nothing to show. */
function viewerSettled(): void {
  if (typeof document !== "undefined" && document.documentElement?.dataset) delete document.documentElement.dataset.viewer;
}

/** The controller, with the one addition to the shared contract: openDataset
 * takes the dataset's geometry when the caller has it, for the viewer's
 * dimensions and centre. */
export interface BrowserViewer extends ViewerController {
  openDataset(url: string, raw: LayerSpec, info?: DatasetInfo): Promise<string>;
}

/** The dataset the page's own `#!` hash shows (a link's raw layer), or
 * null: for the boot to point the session at it before anything opens the
 * session's dataset, so a link to another volume wins over what this tab had
 * open. restore() does the same; this is for doing it earlier. */
export function datasetUrlFromPage(): string | null {
  return datasetUrlFromState(decodeState(location.hash));
}

export function createViewer(opts: { session: SessionStore; log: (line: string) => void }): BrowserViewer {
  const { session, log } = opts;
  let frame: HTMLIFrameElement | null = null;
  /** The state last put in the iframe's hash, until the viewer itself can be read. */
  let pending: NgState | null = null;
  /** Layers set before there is an iframe to show them in (the backend serves
   * the session's models again before the viewer is restored, so that their
   * views exist when the viewer asks): put in the first state mounted. */
  let queued: Record<string, LayerSpec> = {};
  /** The layer to select, its panel on `tab`, once the viewer has a layer
   * of that name: selectLayer called before there is an iframe, or before
   * setLayers has put the layer in (the backend selects the paint layer it
   * sets in the same breath, as the server's add_annotation_layer does both
   * in one transaction). */
  let selection: { name: string; tab?: string } | null = null;
  let stopMirror: (() => void) | null = null;
  /** The dataset restore() left the viewer waiting for (the session names
   * it, the tab has no view of it): mounted by datasetOpened, with the
   * geometry the engine read, rather than at once without it. */
  let awaiting: string | null = null;
  /** The key bindings asked for and not unbound, in the order asked. */
  const bindings: KeyBinding[] = [];
  /** The viewers told they have no bindings API, so it is said once each. */
  const unbindable = new WeakSet<object>();

  /** Neuroglancer's viewer in the iframe, once its page has made it. This
   * is where a viewer is noticed, so the key bindings it has not got yet
   * are put on it here: on the iframe's load (Neuroglancer's scripts are
   * deferred, so `viewer` is on its window by then), on the mirror's poll
   * of its hash (which it writes once it is up), and whenever the page
   * reads the viewer. */
  function liveViewer(): NgViewer | null {
    const win = frame?.contentWindow as (Window & { viewer?: NgViewer }) | null | undefined;
    const viewer = win?.viewer;
    if (!viewer?.state || typeof viewer.state.toJSON !== "function") return null;
    for (const binding of bindings) applyBinding(viewer, binding);
    return viewer;
  }

  /** Put `binding` on `viewer` unless it is there already: the key to the
   * action in the global map (keys pressed over any panel), the handler on
   * the action (ai_annotate._ensure_binding: `viewer.actions.add(ACTION,
   * handler)` and `input_event_bindings.viewer[KEY] = ACTION`). */
  function applyBinding(viewer: NgViewer, binding: KeyBinding): void {
    if (!binding.active || binding.appliedTo.has(viewer)) return;
    const map = viewer.inputEventBindings?.global;
    if (!map || typeof map.set !== "function" || typeof viewer.bindAction !== "function") {
      if (!unbindable.has(viewer)) {
        unbindable.add(viewer);
        log(`this viewer has no key bindings API: ${binding.key} is not bound`);
      }
      return;
    }
    binding.appliedTo.add(viewer);
    map.set(binding.key, binding.action);
    viewer.bindAction(binding.action, () => {
      if (binding.active) binding.handler();
    });
  }

  /** The state as the viewer has it now: the live viewer's, else the one in
   * the iframe's hash (its own window's when readable, else its src), else
   * the one last set. Null without an iframe. */
  function currentState(): NgState | null {
    const live = liveViewer();
    if (live) return live.state.toJSON();
    if (!frame) return null;
    let hash = "";
    try {
      hash = frame.contentWindow?.location.hash ?? "";
    } catch {
      // Not readable yet.
    }
    if (!isStateHash(hash)) hash = frame.src.replace(/^[^#]*/, "");
    return decodeState(hash) ?? pending;
  }

  /** Put `state` in the viewer: through the live viewer when there is one,
   * else through the iframe's hash (a reset to that state when Neuroglancer
   * reads it). Either way the page's hash follows. */
  function applyState(state: NgState): void {
    const live = liveViewer();
    if (live) {
      live.state.restoreState(state);
      // Neuroglancer rewrites its own hash from the new state a moment
      // later; the mirror picks it up on its next poll.
      return;
    }
    const hash = encodeState(state);
    pending = state;
    mountFrame(hash);
    rememberState(hash);
  }

  /** The iframe showing Neuroglancer at `hash`: created in place of the
   * connect panel (or in the viewer column, or the body, whichever the page
   * has) the first time, with the id and title the template gives it; after
   * that its URL is set to the new hash. */
  function mountFrame(hash: string): HTMLIFrameElement {
    if (!frame) {
      frame = document.getElementById(FRAME_ID) as HTMLIFrameElement | null;
    }
    if (!frame) {
      frame = document.createElement("iframe");
      frame.id = FRAME_ID;
      frame.title = "Neuroglancer Data Viewer";
      frame.style.cssText = "width:100%;height:100%;border:none;";
      const panel = document.getElementById(PANEL_ID);
      if (panel) panel.replaceWith(frame);
      else (document.getElementById(COLUMN_ID) ?? document.body).appendChild(frame);
    }
    if (!stopMirror) {
      // The mirror's poll reads the hash the viewer writes once it is up,
      // and the load is when its deferred scripts have run: either way a
      // new viewer is there to put the key bindings on.
      const noticeViewer = () => void liveViewer();
      const stop = mirrorFrameHash(frame, (hash) => {
        rememberState(hash);
        noticeViewer();
      });
      const mounted = frame;
      mounted.addEventListener("load", noticeViewer);
      stopMirror = () => {
        stop();
        mounted.removeEventListener("load", noticeViewer);
      };
    }
    frame.src = NG_PAGE + hash;
    viewerSettled();
    return frame;
  }

  /** A fresh page of Neuroglancer at `hash`, in a new iframe in the old
   * one's place (its id, title and style): for a state the live viewer threw
   * on part way through taking, which a hash change would only hand the same
   * viewer again. Without an iframe yet, mountFrame makes the first. */
  function reopenFrame(hash: string): void {
    const old = frame;
    stopMirror?.();
    stopMirror = null;
    if (old) {
      const fresh = old.cloneNode(false) as HTMLIFrameElement;
      fresh.removeAttribute("src");
      old.replaceWith(fresh);
      frame = fresh;
    }
    mountFrame(hash);
  }

  /** The connect panel, worded for the browser: a URL, not a cluster path. */
  function adjustConnectPanel(): void {
    const input = document.getElementById(PATH_INPUT_ID) as HTMLInputElement | null;
    if (input) input.placeholder = EXAMPLE_DATASET_URL;
    const description = document.querySelector(`#${PANEL_ID} .no-data-desc`);
    if (description) description.textContent = CONNECT_DESCRIPTION;
  }

  /** The viewer's own URL: its window's, which carries the state Neuroglancer
   * last wrote, when readable; else what the iframe was given. */
  function viewerUrl(): string | null {
    if (!frame) return null;
    try {
      const href = frame.contentWindow?.location.href;
      if (href && href !== "about:blank") return href;
    } catch {
      // Not readable: the src is the best there is.
    }
    return frame.src || null;
  }

  /** `state` with the layers set before there was an iframe, which are then
   * no longer queued, and the pending selection when it names one of the
   * state's layers. */
  function withQueued(state: NgState): NgState {
    if (!Object.keys(queued).length) return withSelection(state);
    const merged = replaceLayers(state, queued);
    queued = {};
    return withSelection(merged);
  }

  /** `state` with the pending selection applied, when it names a layer of
   * `state`; the selection is then no longer pending. */
  function withSelection(state: NgState): NgState {
    if (!selection) return state;
    const selected = withSelectedLayer(state, selection.name, selection.tab);
    if (!selected) return state;
    selection = null;
    return selected;
  }

  /** Show `state` as a new viewer: through the hash, which has Neuroglancer
   * reset to it (a new dataset is a new state, not a merge into the old
   * one), and kept as the tab's. */
  function openState(state: NgState): void {
    const shown = withQueued(state);
    const hash = encodeState(shown);
    pending = shown;
    mountFrame(hash);
    rememberState(hash);
  }

  /** The state to put back: the page's own `#!` hash when it has one (a
   * link someone opened; it wins, and the address bar is cleared of it so a
   * later tab completed from this URL starts fresh), else the tab's copy of
   * the viewer's hash (this page reloaded: by connect.js after /api/set-data,
   * or by the user). Null when there is neither. The session follows the
   * state's dataset, so the dashboard and the viewer agree on what is open. */
  async function stateFromPage(): Promise<NgState | null> {
    let hash = location.hash;
    if (isStateHash(hash)) {
      history.replaceState(null, "", withHash(location.href, ""));
    } else {
      hash = rememberedState() ?? "";
    }
    if (!hash) return null;
    const state = decodeState(hash);
    if (!state) {
      log("a viewer state that could not be read was found; ignoring it");
      return null;
    }
    const url = datasetUrlFromState(state);
    if (url && url !== session.state.datasetUrl) {
      await session.update({ datasetUrl: url });
      log(`dataset from the link: ${url}`);
    }
    return state;
  }

  return {
    async restore() {
      const linked = await stateFromPage();
      if (linked) {
        openState(linked);
        return;
      }
      const url = session.state.datasetUrl;
      if (url) {
        // The session knows the dataset but not its geometry, which the
        // engine reads as the session is restored (backend restoreSession,
        // a second or so): the viewer waits for it (datasetOpened), the
        // connect panel staying hidden meanwhile, and layers set before
        // then are queued. Mounted at once without it, Neuroglancer took
        // the coordinate space from whichever layer's metadata came first,
        // 8 nm (the raw) or 16 nm (the model's output), and adding the
        // 16 nm paint layer switched an 8 nm one to 16 nm with the
        // position's voxel numbers kept, which put the view past the
        // data's far corner; setting the dimensions once the viewer was up
        // did not hold either. Connect's state carries them from the start
        // (prepareDataset), and the Python viewer sets the raw's before any
        // layer (bootstrap.raw_dimensions), for the same reason.
        awaiting = url;
        return;
      }
      adjustConnectPanel();
      viewerSettled();
    },

    datasetOpened(url, info) {
      if (awaiting === null || awaiting !== url) return;
      awaiting = null;
      if (info) {
        openState(initialState(info, { [RAW_LAYER_NAME]: rawLayer(url, info) }));
        log(`viewer restored on ${url}`);
      } else {
        // The engine could not open it: Neuroglancer tries on its own, and
        // says why it cannot in its layer panel.
        openState(stateWithoutInfo(url));
        log(`viewer restored on ${url}, without its geometry`);
      }
    },

    async openDataset(url, raw, info) {
      awaiting = null;
      const layers = { [RAW_LAYER_NAME]: raw };
      openState(info ? initialState(info, layers) : stateWithoutInfo(url, raw));
      log(`viewer opened on ${url}`);
      return viewerUrl() ?? NG_PAGE + encodeState(pending ?? {});
    },

    prepareDataset(url, raw, info) {
      // Into the tab's state only, which the reload connect.js does a second
      // after /api/set-data answers restores from (restore() above): an iframe
      // mounted now would show the volume, go with the page, and show it
      // again. An open viewer's hash is no longer followed, or its next poll
      // would put the old state back before the reload.
      stopMirror?.();
      stopMirror = null;
      const state = info ? initialState(info, { [RAW_LAYER_NAME]: raw }) : stateWithoutInfo(url, raw);
      const hash = encodeState(state);
      pending = state;
      rememberState(hash);
      log(`viewer prepared for ${url}: shown when the page reloads`);
      return NG_PAGE + hash;
    },

    setLayers(layers, replacing = {}) {
      const state = currentState();
      if (!state) {
        // No iframe yet: shown when restore() or openDataset() makes one.
        // A layer replaced meanwhile is only queued, with nothing set on it.
        for (const old of Object.values(replacing)) delete queued[old];
        Object.assign(queued, layers);
        return;
      }
      applyState(withSelection(replaceLayers(state, layers, replacing)));
    },

    removeLayer(name) {
      delete queued[name];
      if (selection?.name === name) selection = null;
      const state = currentState();
      if (!state) return;
      applyState(withoutLayer(state, name));
    },

    setEquivalences(name, sets) {
      // Every report carries the whole set of joined ids so far, so one
      // that cannot be shown yet is only a delay: the next brings it too.
      const live = liveViewer();
      if (live) {
        // The loaded layer's own sets, replaced in place. Not through the
        // state: restoring the whole state re-creates the layers, and with
        // the mouse over the viewer Neuroglancer's hover tracking reads the
        // new segmentation layer before its selection state exists
        // ("Cannot set properties of undefined (setting
        // 'localPositionValid')"), which blanks the viewer. The sets are in
        // the viewer's toJSON(), so the page's hash follows all the same.
        const equivalences = liveEquivalences(live, name);
        if (equivalences) {
          equivalences.clear();
          equivalences.restoreState(sets);
          return;
        }
        // Not loaded yet (or not a segmentation layer): through the state.
      }
      const state = currentState();
      if (!state) {
        // No iframe yet: a layer waiting for the first state carries them
        // in (the merging step is part of the layer's own chain).
        if (queued[name]) queued[name] = { ...queued[name], equivalences: sets.map((set) => [...set]) };
        return;
      }
      const next = withEquivalences(state, name, sets);
      if (!next) return;  // no segmentation layer of that name: removed, or not drawn yet
      applyState(next);
    },

    position() {
      return positionNm(currentState());
    },

    viewerUrl,

    layerShader(name) {
      return layerShaderOf(currentState(), name);
    },
    selectLayer(name, tab) {
      selection = tab === undefined ? { name } : { name, tab };
      const state = currentState();
      if (!state) return;  // no iframe yet: in the first state mounted, with the queued layers
      const selected = withSelection(state);
      if (selected !== state) applyState(selected);  // else not a layer yet: selected when setLayers brings it
    },

    pointerPosition() {
      // The mouse state's position is in the viewer's coordinate space,
      // the state's `dimensions`, as the view's own position is (the
      // Python's action_state.mouse_voxel_coordinates, converted by
      // ai_annotate._zyx_nm). Not over a panel (`active` false, which
      // Neuroglancer sets as the pointer leaves) is null: the Shift+G
      // handler then says to hover over the data first.
      const live = liveViewer();
      if (!live?.mouseState?.active) return null;
      return pointerNm(live.mouseState.position, live.state.toJSON().dimensions);
    },

    viewDepthAxis() {
      return viewDepthAxisOf(currentState());
    },

    bindKey(key, action, handler) {
      const binding: KeyBinding = { key, action, handler, active: true, appliedTo: new WeakSet() };
      bindings.push(binding);
      liveViewer();  // on the viewer now, when there is one
      return () => {
        if (!binding.active) return;
        binding.active = false;
        bindings.splice(bindings.indexOf(binding), 1);
        // The key is taken out of the live viewer's map, unless another
        // binding still wants it; the action listener stays, inert.
        const live = liveViewer();
        const map = live?.inputEventBindings?.global;
        if (live && binding.appliedTo.has(live) && typeof map?.delete === "function" && !bindings.some((other) => other.key === key && other.appliedTo.has(live))) {
          map.delete(key);
        }
      };
    },

    refreshLayer(name) {
      // Through the state, not by rebuilding the layer: the layer's own
      // settings (its paint value, brush, tool, colours) are in its JSON
      // and go with it; only the source's spelling moves, and the view
      // and the other layers are left alone (overlay.refresh_annotation_layer).
      const state = currentState();
      if (!state) return false;
      const layer = layersOf(state).find((candidate) => candidate.name === name);
      if (!layer) return false;
      const source = respelledPaintSource(layer.source);
      if (source === null) return false;  // not a paint layer: nothing of ours to re-read
      const next = replaceLayers(state, { [name]: { ...layer, source } });
      try {
        applyState(next);
      } catch (e) {
        // Neuroglancer can throw part way through taking a state, leaving
        // the viewer half restored (seen after a Seed from Prediction:
        // "Error restoring property "layout": Cannot read properties of
        // undefined (reading 'generation')"). The labels are written by
        // then, so the viewer is opened again on the new state, rather than
        // the write reported failed or the iframe reloaded on the state it
        // was first given, which lacks every layer added since.
        log(`the viewer could not take the re-read ${name} layer (${(e as Error).message ?? e}); opening it again on that state`);
        const hash = encodeState(next);
        pending = next;
        reopenFrame(hash);
        rememberState(hash);
      }
      return true;
    },
  };
}
