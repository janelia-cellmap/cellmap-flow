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
import type { DatasetInfo, LayerSpec, SessionStore, ViewerController } from "../contracts";
import { initialState, RAW_LAYER_NAME, stateWithoutInfo } from "./layers";
import { withEquivalences } from "./equivalences";
import { decodeState, encodeState, isStateHash, mirrorFrameHash, mirrorToPage } from "./share";
import { datasetUrlFromState, layerShaderOf, type NgState, positionNm, replaceLayers, withoutLayer } from "./state";

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
  + "such as an OpenOrganelle dataset) to open it in the viewer. The models run in this browser.";

/** What Neuroglancer's default page puts on its window. */
interface NgViewer {
  state: { toJSON(): NgState; restoreState(state: NgState): void };
}

/** The controller, with the one addition to the shared contract: openDataset
 * takes the dataset's geometry when the caller has it, for the viewer's
 * dimensions and centre. */
export interface BrowserViewer extends ViewerController {
  openDataset(url: string, raw: LayerSpec, info?: DatasetInfo): Promise<string>;
}

/** The dataset the page's own `#!` hash shows (a shared link's raw layer),
 * or null: for the boot to point the session at it before anything opens
 * the session's dataset, so a link to another volume wins over what this
 * browser had open. restore() does the same; this is for doing it earlier. */
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
  let stopMirror: (() => void) | null = null;

  /** Neuroglancer's viewer in the iframe, once its page has made it. */
  function liveViewer(): NgViewer | null {
    const win = frame?.contentWindow as (Window & { viewer?: NgViewer }) | null | undefined;
    const viewer = win?.viewer;
    return viewer?.state && typeof viewer.state.toJSON === "function" ? viewer : null;
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
    mirrorToPage(hash);
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
    stopMirror ??= mirrorFrameHash(frame);
    frame.src = NG_PAGE + hash;
    return frame;
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
   * no longer queued. */
  function withQueued(state: NgState): NgState {
    if (!Object.keys(queued).length) return state;
    const merged = replaceLayers(state, queued);
    queued = {};
    return merged;
  }

  /** Show `state` as a new viewer: through the hash, which has Neuroglancer
   * reset to it (a new dataset is a new state, not a merge into the old
   * one), with the page's hash following. */
  function openState(state: NgState): void {
    const shown = withQueued(state);
    const hash = encodeState(shown);
    pending = shown;
    mountFrame(hash);
    mirrorToPage(hash);
  }

  /** The state the page's `#!` hash holds (a shared link, or this page
   * reloaded by connect.js after /api/set-data), or null. The session
   * follows the state's dataset, so the dashboard and the viewer agree on
   * what is open. */
  async function stateFromPage(): Promise<NgState | null> {
    const hash = location.hash;
    if (!isStateHash(hash)) return null;
    const state = decodeState(hash);
    if (!state) {
      log("the page's URL holds a viewer state that could not be read; ignoring it");
      return null;
    }
    const url = datasetUrlFromState(state);
    if (url && url !== session.state.datasetUrl) {
      await session.update({ datasetUrl: url });
      log(`dataset from the page's URL: ${url}`);
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
        // The session knows the dataset but not its geometry (that comes
        // from the inference module's open): Neuroglancer reads it itself.
        // The backend puts the proper raw layer and the models' layers in
        // through setLayers when it serves them again.
        openState(stateWithoutInfo(url));
        log(`viewer restored on ${url}`);
        return;
      }
      adjustConnectPanel();
    },

    async openDataset(url, raw, info) {
      const layers = { [RAW_LAYER_NAME]: raw };
      openState(info ? initialState(info, layers) : stateWithoutInfo(url, raw));
      log(`viewer opened on ${url}`);
      return viewerUrl() ?? NG_PAGE + encodeState(pending ?? {});
    },

    setLayers(layers) {
      const state = currentState();
      if (!state) {
        // No iframe yet: shown when restore() or openDataset() makes one.
        Object.assign(queued, layers);
        return;
      }
      applyState(replaceLayers(state, layers));
    },

    removeLayer(name) {
      delete queued[name];
      const state = currentState();
      if (!state) return;
      applyState(withoutLayer(state, name));
    },

    setEquivalences(name, sets) {
      // Every report carries the whole set of joined ids so far, so one
      // that cannot be shown yet is only a delay: the next brings it too.
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
  };
}
