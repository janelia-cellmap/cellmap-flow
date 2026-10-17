// Pure functions over a Neuroglancer state's JSON (what
// `viewer.state.toJSON()` gives and `viewer.state.restoreState()` takes, and
// what the page's `#!` hash encodes): replacing layers the way the Python
// dashboard does across a redraw, reading the position (and the pointer's)
// in nm, the data axis the layout looks along, and the user's shader of a
// layer. No DOM here, so all of it is unit-tested.
import { AXIS_NAMES, LAYOUT_DEPTH_SLOT, slotAxis } from "../ai-annotate/geometry";
import type { LayerSpec } from "../contracts";

/** A layer in the state: at least its name; `visible: false` when hidden
 * (Neuroglancer leaves the key out when it is shown). */
export interface NgLayer {
  name: string;
  type?: string;
  source?: unknown;
  visible?: boolean;
  shader?: string;
  shaderControls?: Record<string, unknown>;
  [key: string]: unknown;
}

/** The viewer state's fields this module reads; everything else passes
 * through untouched. */
export interface NgState {
  dimensions?: Record<string, [number, string]>;
  position?: number[];
  layers?: NgLayer[] | Record<string, Omit<NgLayer, "name">>;
  /** Neuroglancer's layout: a name ("xy", "4panel", "xy-3d") or an object
   * with a `type` (when its 3D view has settings of its own). */
  layout?: unknown;
  /** The names of the dimensions the panels show, when the user picked
   * them; otherwise Neuroglancer shows the first three of `dimensions`. */
  displayDimensions?: string[];
  [key: string]: unknown;
}

/** nm per unit of a dimension, as Neuroglancer names units (raw.py's
 * _NM_PER_UNIT and review_routes' nm_per_unit). */
const NM_PER_UNIT: Record<string, number> = { m: 1e9, mm: 1e6, um: 1e3, "µm": 1e3, nm: 1 };

/** The state's layers as an array. Neuroglancer's current JSON is an array
 * of layers with `name`; its oldest links held an object keyed by name,
 * which Neuroglancer still reads, so both are accepted. */
export function layersOf(state: NgState | null | undefined): NgLayer[] {
  const layers = state?.layers;
  if (!layers) return [];
  if (Array.isArray(layers)) return layers;
  return Object.entries(layers).map(([name, layer]) => ({ ...layer, name }));
}

/** A layer's properties that belong to the data its source serves rather
 * than to how the user shows it: a layer drawn again keeps them from the one
 * it replaces only while the source is the same (carriedLayer). The
 * segments shown, the merger's equivalences and the segment query and
 * colours name segments of that data; the channel shown (local dimensions,
 * position, channel dimensions) counts that data's channels; the shader and
 * its controls are over its range (the app sets its own for a new chain);
 * annotations and a legacy top-level transform are the source's own. The
 * same list as cellmap_flow/viewer/layers.py's SOURCE_BOUND_LAYER_KEYS
 * (tests/web/test_layer_carry.py holds them equal). */
export const SOURCE_BOUND_LAYER_KEYS: readonly string[] = [
  "annotations",
  "channelDimensions",
  "equivalences",
  "localDimensions",
  "localPosition",
  "segmentColors",
  "segmentQuery",
  "segments",
  "shader",
  "shaderControls",
  "transform",
];
const SOURCE_BOUND = new Set(SOURCE_BOUND_LAYER_KEYS);

/** Every URL a layer's source names, in order: a string, a {url, ...}, or
 * a list of either (sourceUrl gives the first). */
export function sourceUrls(source: unknown): string[] {
  if (Array.isArray(source)) return source.flatMap(sourceUrls);
  const url = sourceUrl(source);
  return url === null ? [] : [url];
}

/** `source` with the URLs it names (sourceUrls') replaced by `fn`, in the
 * form it is in: a string, a {url, ...} with its other keys kept, or a list
 * of either. `fn` gives a URL's replacement, or null to keep it; an entry
 * whose URL is kept, or that names none, stays as it is. null when `fn`
 * keeps every URL. */
export function mapSourceUrls(source: unknown, fn: (url: string) => string | null): unknown | null {
  if (Array.isArray(source)) {
    let replaced = false;
    const entries = source.map((entry) => {
      const next = mapSourceUrls(entry, fn);
      if (next === null) return entry;
      replaced = true;
      return next;
    });
    return replaced ? entries : null;
  }
  const url = sourceUrl(source);
  const next = url === null ? null : fn(url);
  if (next === null) return null;
  return typeof source === "string" ? next : { ...(source as Record<string, unknown>), url: next };
}

/** `spec` as the layer `name` that takes the place of `old`, keeping what the
 * user set on `old` (routes/pipeline.py's replace_layer, through
 * cellmap_flow/viewer/layers.py). A redraw (Submit, a model added, a mark
 * drawn) builds each layer anew, and Neuroglancer shows the new one as
 * built: the user's blending (additive), opacity, resolution, volume
 * rendering, segment colouring or tab were back at their defaults, and a
 * layer they had switched off came back on. So the new layer is `spec`
 * with every property of `old` that `spec` does not set: the app's own
 * properties (the source, a new chain's shader, the annotations drawn) win,
 * the user's other settings stay. When the source is another one, the
 * properties tied to its data (SOURCE_BOUND_LAYER_KEYS) are not kept. A
 * layer of another type (an image layer become a segmentation when the
 * chain now ends in labels) has no settings this one would read, and keeps
 * only being hidden. */
export function carriedLayer(old: NgLayer | undefined, spec: LayerSpec, name: string): NgLayer {
  const next: NgLayer = { ...spec, name };
  if (!old) return next;
  if (old.type !== spec.type) {
    if (old.visible === false) next.visible = false;
    return next;
  }
  const sameSource = JSON.stringify(sourceUrls(old.source)) === JSON.stringify(sourceUrls(spec.source));
  for (const [key, value] of Object.entries(old)) {
    if (next[key] !== undefined || key === "name") continue;
    if (!sameSource && SOURCE_BOUND.has(key)) continue;
    next[key] = value;
  }
  return next;
}

/** `state` with each of `layers` in place of the layer of the same name,
 * keeping what the user set on it (carriedLayer), or appended when there was
 * none. `replacing` names, for a layer of `layers` that is new, the layer it
 * takes over from (a finetuned model's next iteration, `x-ft2` for `x-ft1`):
 * that one's place in the list and its settings go to the new layer, and it
 * is gone. */
export function replaceLayers(state: NgState, layers: Record<string, LayerSpec>, replacing: Record<string, string> = {}): NgState {
  const current = layersOf(state).map((layer) => ({ ...layer }));
  for (const [name, spec] of Object.entries(layers)) {
    let index = current.findIndex((layer) => layer.name === name);
    const previous = replacing[name];
    const at = previous === undefined || previous === name ? -1 : current.findIndex((layer) => layer.name === previous);
    if (at !== -1) {
      if (index === -1) {
        index = at;  // in the old one's place, with its settings
      } else {
        // Both there already: the new one keeps its own, the old one goes.
        current.splice(at, 1);
        index = current.findIndex((layer) => layer.name === name);
      }
    }
    const replacement = carriedLayer(index >= 0 ? current[index] : undefined, spec, name);
    if (index >= 0) current[index] = replacement;
    else current.push(replacement);
  }
  return { ...state, layers: current };
}

/** `state` without the layer `name`. */
export function withoutLayer(state: NgState, name: string): NgState {
  return { ...state, layers: layersOf(state).filter((layer) => layer.name !== name) };
}

/** The viewer's position in nm, z, y, x, or null when the state has no
 * position or dimensions yet (the browser has not reported them). The
 * conversion is pointerNm's: see there. */
export function positionNm(state: NgState | null | undefined): number[] | null {
  return pointerNm(state?.position, state?.dimensions);
}

/** Viewer voxel coordinates `coords` as nm, z, y, x, over the viewer's
 * `dimensions`; null when either is missing, they disagree on the rank,
 * fewer than three axes are there, a coordinate is not a number (the mouse
 * state's position before the pointer has been over the data), or an axis
 * is not a length.
 *
 * Neuroglancer's positions (the view's, the mouse's) are in voxels of the
 * viewer's coordinate space, whose scales it reports in SI units (8 nm
 * reads back as 8e-9 m), so each coordinate is multiplied by its axis's
 * scale in nm. Axes are matched by name (z, y, x), falling back to the
 * first three dimensions in order, as review_routes._navigate_viewer,
 * finetune/common.viewer_position_and_scales and ai_annotate._zyx_nm read
 * them. */
export function pointerNm(coords: ArrayLike<number> | null | undefined, dimensions: Record<string, [number, string]> | null | undefined): number[] | null {
  if (!dimensions || !coords) return null;
  const names = Object.keys(dimensions);
  if (names.length < 3 || coords.length !== names.length) return null;
  const nm: number[] = [];
  zyxIndices(names).forEach((i) => {
    const [scale, unit] = dimensions[names[i]];
    const perUnit = NM_PER_UNIT[unit];
    if (perUnit === undefined) return; // not a length: the position is not in nm
    const coord = coords[i];
    if (typeof coord !== "number" || Number.isNaN(coord)) return;
    // Rounded to a millionth of a nanometre, which drops the floating-point
    // noise of the unit conversion (10.5 voxels * 8e-9 m * 1e9 is
    // 84.00000000000001) and nothing a viewer position means.
    nm.push(Math.round(coord * scale * perUnit * 1e6) / 1e6);
  });
  return nm.length === 3 ? nm : null;
}

/** pointerNm backwards: a point in nm (z, y, x) as coordinates of
 * `dimensions`, matched to z, y, x by name as pointerNm matches them; a
 * dimension of `dimensions` that is not one of them is 0. null when the
 * space has fewer than three dimensions, or one of the three is not a
 * length. */
export function coordsFromNm(nm: readonly number[], dimensions: Record<string, [number, string]>): number[] | null {
  const names = Object.keys(dimensions);
  if (names.length < 3 || nm.length !== 3) return null;
  const coords = names.map(() => 0);
  for (const [axis, i] of zyxIndices(names).entries()) {
    const [scale, unit] = dimensions[names[i]];
    const perUnit = NM_PER_UNIT[unit];
    if (perUnit === undefined) return null;
    // Rounded as pointerNm rounds: to a millionth of the coordinate.
    coords[i] = Math.round((nm[axis] / (scale * perUnit)) * 1e6) / 1e6;
  }
  return coords;
}

/** Where z, y and x are among a coordinate space's dimension `names`, in
 * that order: each found by its name, or, when it is not among them, at
 * its own place in z, y, x (the first three dimensions in order), as
 * pointerNm and coordsFromNm match them. */
function zyxIndices(names: readonly string[]): number[] {
  return AXIS_NAMES.map((axis, fallback) => (names.includes(axis) ? names.indexOf(axis) : fallback));
}

// ---------------------------------------------------------------------------
// The plane the viewer shows: cellmap_flow/ai_annotate/geometry.py's
// depth_axis_from_layout and depth_axis_for_view, for the state's JSON. The
// axis names and the panels' slots are ai-annotate/geometry.ts's, the port
// of the rest of that module.
// ---------------------------------------------------------------------------

/** The panel name of a Neuroglancer layout (geometry._layout_type): the
 * string itself or an object's `type`, lower-cased, a "-3d" suffix (the
 * layouts with a 3D view beside the panel) dropped; null for anything
 * else. */
function layoutType(layout: unknown): string | null {
  const type = layout && typeof layout === "object" ? (layout as { type?: unknown }).type : layout;
  if (typeof type !== "string") return null;
  const name = type.toLowerCase();
  return name.endsWith("-3d") ? name.slice(0, -"-3d".length) : name;
}

/** The data axis (0 z, 1 y, 2 x) the panel named by `layout` looks along
 * (geometry.depth_axis_from_layout).
 *
 * `displayDimensions` are the names of the viewer's display dimensions in
 * order (z, y, x when not given, as a viewer that sets none over z, y, x
 * shows them); see ai-annotate/geometry.ts LAYOUT_DEPTH_SLOT for why they
 * matter. Only a single-plane layout says
 * which plane the user is looking at: Neuroglancer keeps one layout for the
 * whole viewer, so in "4panel" or "3d" the panel the pointer is in is not
 * known. Those (and anything unrecognised) get the "xy" panel's axis, the
 * panel Neuroglancer puts first. A slot whose dimension is not z, y or x
 * (a channel axis shown as a display dimension, fewer than three) is axis
 * 0, as the Python answers. A rotated cross-section is not followed. */
export function depthAxisFromLayout(layout: unknown, displayDimensions?: readonly string[] | null): 0 | 1 | 2 {
  const names: readonly string[] = displayDimensions?.length ? displayDimensions : AXIS_NAMES;
  const slot = LAYOUT_DEPTH_SLOT[layoutType(layout) ?? ""] ?? LAYOUT_DEPTH_SLOT.xy;
  const axis = slotAxis(names, slot);
  return axis === 1 || axis === 2 ? axis : 0;
}

/** A viewer state's display dimensions (geometry.display_dimensions_for_view):
 * those it sets, else the first three of its coordinate space, as
 * Neuroglancer takes them; empty when it has neither. */
export function displayDimensionsOf(state: NgState | null | undefined): string[] {
  const set = state?.displayDimensions;
  return Array.isArray(set) && set.length ? set.filter((name): name is string => typeof name === "string") : Object.keys(state?.dimensions ?? {}).slice(0, 3);
}

/** depthAxisFromLayout for a viewer state's JSON (geometry.depth_axis_for_view),
 * over its display dimensions (displayDimensionsOf). Without a state (no
 * viewer yet) the default viewer's: the "xy" panel over z, y, x. */
export function viewDepthAxisOf(state: NgState | null | undefined): 0 | 1 | 2 {
  const display = displayDimensionsOf(state);
  return depthAxisFromLayout(state?.layout, display.length ? display : null);
}

/** The layer `name`'s shader and shader controls as the user left them in
 * `state`, to put back on the layer when it is rebuilt (routes/pipeline.py's
 * _save_shaders_from_viewer): null when the state has no such layer, else
 * whichever of the two the layer has. */
export function layerShaderOf(state: NgState | null | undefined, name: string): { shader?: string; shaderControls?: Record<string, unknown> } | null {
  const layer = layersOf(state).find((l) => l.name === name);
  if (!layer) return null;
  const kept: { shader?: string; shaderControls?: Record<string, unknown> } = {};
  if (typeof layer.shader === "string" && layer.shader && layer.shader !== "None") kept.shader = layer.shader;
  if (layer.shaderControls && typeof layer.shaderControls === "object" && Object.keys(layer.shaderControls).length) {
    kept.shaderControls = layer.shaderControls;
  }
  return kept;
}

/** The URL of a layer's source, whichever of Neuroglancer's forms it is in:
 * a string, {url, ...}, or an array of either (the Python API's); null when
 * there is none. */
export function sourceUrl(source: unknown): string | null {
  if (typeof source === "string") return source;
  if (Array.isArray(source)) return source.length ? sourceUrl(source[0]) : null;
  if (source && typeof source === "object" && typeof (source as { url?: unknown }).url === "string") {
    return (source as { url: string }).url;
  }
  return null;
}

/** The dataset a state shows: the raw layer's source URL without the data
 * format in front of it (`zarr://https://...` gives `https://...`), as the
 * session's datasetUrl holds it. The raw layer is the one named "data"
 * (RAW_LAYER_NAME), else the first image layer. Null without one. */
export function datasetUrlFromState(state: NgState | null | undefined): string | null {
  const layers = layersOf(state);
  const raw = layers.find((layer) => layer.name === "data") ?? layers.find((layer) => layer.type === "image");
  const url = raw ? sourceUrl(raw.source) : null;
  if (!url) return null;
  return url.replace(/^(zarr|zarr2|zarr3|n5|precomputed):\/\//, "");
}
