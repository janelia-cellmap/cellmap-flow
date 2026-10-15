// Pure functions over a Neuroglancer state's JSON (what
// `viewer.state.toJSON()` gives and `viewer.state.restoreState()` takes, and
// what the page's `#!` hash encodes): replacing layers the way the Python
// dashboard does across a redraw, reading the position in nm, and the
// user's shader of a layer. No DOM here, so all of it is unit-tested.
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

/** `state` with each of `layers` in place of the layer of the same name, or
 * appended when there was none (routes/pipeline.py's _replace_layer). A
 * redraw replaces each layer with a new one, which Neuroglancer shows, so a
 * layer the user had switched off would come back on with every Submit:
 * the new layer stays hidden where the old one was. */
export function replaceLayers(state: NgState, layers: Record<string, LayerSpec>): NgState {
  const current = layersOf(state).map((layer) => ({ ...layer }));
  for (const [name, spec] of Object.entries(layers)) {
    const index = current.findIndex((layer) => layer.name === name);
    const hidden = index >= 0 && current[index].visible === false;
    const replacement: NgLayer = { ...spec, name };
    if (hidden) replacement.visible = false;
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
 * position or dimensions yet (the browser has not reported them).
 *
 * Neuroglancer's position is in voxels of the viewer's coordinate space,
 * whose scales it reports in SI units (8 nm reads back as 8e-9 m), so each
 * coordinate is multiplied by its axis's scale in nm. Axes are matched by
 * name (z, y, x), falling back to the first three dimensions in order, as
 * review_routes._navigate_viewer and finetune/common.viewer_position_and_scales
 * read it. */
export function positionNm(state: NgState | null | undefined): number[] | null {
  const dimensions = state?.dimensions;
  const position = state?.position;
  if (!dimensions || !position) return null;
  const names = Object.keys(dimensions);
  if (names.length < 3 || position.length !== names.length) return null;
  const nm: number[] = [];
  ["z", "y", "x"].forEach((axis, fallback) => {
    const i = names.includes(axis) ? names.indexOf(axis) : fallback;
    const [scale, unit] = dimensions[names[i]];
    const perUnit = NM_PER_UNIT[unit];
    if (perUnit === undefined) return; // not a length: the position is not in nm
    // Rounded to a millionth of a nanometre, which drops the floating-point
    // noise of the unit conversion (10.5 voxels * 8e-9 m * 1e9 is
    // 84.00000000000001) and nothing a viewer position means.
    nm.push(Math.round(position[i] * scale * perUnit * 1e6) / 1e6);
  });
  return nm.length === 3 ? nm : null;
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
