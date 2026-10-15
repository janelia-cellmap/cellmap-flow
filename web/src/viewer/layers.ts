// The layers the viewer shows, as Neuroglancer state JSON: a port of
// cellmap_flow/viewer/layers.py (prediction_source, prediction_shader_for,
// prediction_layer, display_channel, raw_layer) and of the viewer the
// Python dashboard starts (viewer/bootstrap.py's new_viewer and
// raw_dimensions), for the browser, where no Python builds them.
//
// A layer here is the JSON Neuroglancer's state holds, which is also what
// the Python API's `ImageLayer(...).to_json()` gives, with two differences
// to keep in mind when comparing: the Python API wraps a layer's `source` in
// a one-element list (`"source": [{"url": ...}]`), while Neuroglancer takes
// the string or object itself as well, and the Python API spells
// `local_position` as a keyword, emitted as `localPosition`. Neuroglancer
// expects camelCase in its state JSON (`shaderControls`, `localPosition`,
// `crossSectionScale`), and that is what is written here.
import type { Chain, DatasetInfo, LayerSpec, ModelInfo, ServedModel, Step } from "../contracts";
import { isSegmentation, outputChannels, outputDtype } from "../infer/post";
import { outputDisplayRange, PREDICTION_COLORS, predictionShader, rawShader } from "./display";
import type { NgState } from "./state";

export { PREDICTION_COLORS };

/** The raw layer's name, as bootstrap.new_viewer names it (raw_name="data")
 * and routes/pipeline.py looks it up. */
export const RAW_LAYER_NAME = "data";

/** Neuroglancer's limit on the GPU memory it keeps chunks in, twice its
 * default of 1 GB, as bootstrap.apply_viewer_defaults sets it. */
export const GPU_MEMORY_LIMIT = 2_000_000_000;

/** The layout a new viewer opens with. The Python viewer leaves
 * Neuroglancer's default, 4panel (three cross sections and a 3D view). In
 * the browser every prediction chunk is a forward pass of the model on this
 * machine's GPU, and three cross sections ask for about three times the
 * chunks one does, so the viewer opens on one; the layout control in
 * Neuroglancer's top bar switches it. */
export const DEFAULT_LAYOUT = "xy";

/** A layer's shader and shader controls as the user left them, put back on
 * the layer when it is rebuilt (routes/pipeline.py's session.shaders). */
export interface KeptShader {
  shader?: string;
  shaderControls?: Record<string, unknown>;
}

/** A layer's source when it needs a transform: Neuroglancer's
 * {url, transform} form. */
export interface LayerSource {
  url: string;
  transform: {
    outputDimensions: Record<string, [number, string]>;
    inputDimensions: Record<string, [number, string]>;
  };
}

/** Neuroglancer's dimensions: each axis's scale and unit, as the Python
 * CoordinateSpace emits them (8 nm as [8e-9, "m"]). */
export type NgDimensions = Record<string, [number, string]>;

/** A shader the user left: anything but an unset one. The Python API
 * reports a layer with no shader as the string "None", which routes/
 * pipeline.py guards against; the guard is kept here so a session that
 * went through the Python dashboard restores the same way. */
function keptShader(keep?: KeptShader): string | undefined {
  return keep?.shader && keep.shader !== "None" ? keep.shader : undefined;
}

function keptControls(keep?: KeptShader): Record<string, unknown> | undefined {
  return keep?.shaderControls && Object.keys(keep.shaderControls).length ? keep.shaderControls : undefined;
}

/** `nm` in metres, as Neuroglancer's dimensions and transforms want lengths.
 * Divided by 1e9 rather than multiplied by 1e-9: 1e9 is an exact double and
 * 1e-9 is not, so this rounds once and 4.5 nm comes out as 4.5e-9, the way
 * the Python CoordinateSpace(units="nm") prints it, not 4.500000000000001e-9. */
export function nmToMetres(nm: number): number {
  return nm / 1e9;
}

// ---------------------------------------------------------------------------
// The raw layer
// ---------------------------------------------------------------------------

/** The data formats Neuroglancer names in front of a location. */
const SOURCE_FORMAT = /^(zarr|zarr2|zarr3|n5|precomputed):\/\//;

/** The source Neuroglancer reads the raw data at `url` from itself
 * (raw.py's neuroglancer_source): the format in front of the location,
 * `zarr://` for a .zarr container (v2 or v3, an OME-Zarr multiscale group
 * or one array; Neuroglancer tells them apart and reads a group's levels
 * itself), `n5://` for a .n5 one, `precomputed://` for a gs:// path that
 * names neither (Neuroglancer's cloud volumes are precomputed). A url that
 * already names its format is kept. */
export function rawSource(url: string): string {
  if (SOURCE_FORMAT.test(url)) return url;
  const container = /\.(zarr|n5)(?=\/|$)/g;
  let format: string | null = null;
  for (let match = container.exec(url); match; match = container.exec(url)) format = match[1];
  if (format === null && url.startsWith("gs://")) return `precomputed://${url}`;
  return `${format ?? "zarr"}://${url}`;
}

/** The raw data's layer (layers.py's raw_layer over raw.py's get_raw_layer
 * with wrap_raw=False): an image layer reading `url` itself, with the raw
 * shader over the dtype's range, or with the user's `keep` shader and
 * controls in place of it. */
export function rawLayer(url: string, info: DatasetInfo, keep?: KeptShader): LayerSpec {
  const layer: LayerSpec = {
    type: "image",
    name: RAW_LAYER_NAME,
    source: rawSource(url),
    shader: keptShader(keep) ?? rawShader(info.dtype),
  };
  const controls = keptControls(keep);
  if (controls) layer.shaderControls = controls;
  return layer;
}

// ---------------------------------------------------------------------------
// A prediction layer
// ---------------------------------------------------------------------------

/** The voxel size (nm, z, y, x) to draw a model's output at, or null to draw
 * it at the size the engine declares (layers.py's prediction_voxel_override).
 *
 * The engine reads the raw at the pyramid level closest to the model's input
 * voxel size and treats that level as if it were at the model's size, so when
 * the two differ its output really lies at a proportionally different size
 * than declared: a model trained at 16 nm on raw at 6, 12, 24 nm reads 12 nm
 * data, and its output is at 12 nm, not 16. The engine says so in
 * `effective_output_voxel_size`; the layer's transform then declares the
 * array at that size, so it overlays the raw. (Python also guesses the
 * override for a server too old to say, from the raw's levels; the browser's
 * engine is never that old.) */
export function predictionVoxelOverride(info: ModelInfo): number[] | null {
  const declared = info.output_voxel_size;
  const effective = info.effective_output_voxel_size;
  if (!declared?.length || !effective?.length) return null;
  if (effective.length === declared.length && effective.every((size, i) => size === declared[i])) return null;
  return [...effective];
}

/** The source of a model's layer (layers.py's prediction_source): `url` as
 * the engine serves it (zarr3://...), plain when the output is drawn at its
 * declared size; with `overrideScales` (z, y, x nm, from
 * predictionVoxelOverride) the {url, transform} form whose transform
 * declares the array at those scales. Neuroglancer wants the same rank on
 * both sides of a transform, so a served array with a channel axis
 * (`hasChannel`, the engine's last axis) keeps it as a unit-less `c^`.
 * Neuroglancer reads a served channel axis as `c'`, a slider showing one
 * channel at a time; a transform renames the source dimensions it names and
 * passes the rest through. */
export function predictionSource(url: string, overrideScales: readonly number[] | null, hasChannel: boolean): string | LayerSource {
  if (!overrideScales) return url;
  const dimensions: NgDimensions = {};
  ["z", "y", "x"].forEach((axis, i) => {
    dimensions[axis] = [nmToMetres(overrideScales[i]), "m"];
  });
  if (hasChannel) dimensions["c^"] = [1, ""];
  return { url, transform: { outputDimensions: dimensions, inputDimensions: { ...dimensions } } };
}

/** The channel a model's layer opens on (layers.py's display_channel): the
 * one the engine names (`display_channel`, a Cellpose model's cell
 * probability rather than its first flow), while the chain keeps the model's
 * channels; null otherwise (the first). */
export function displayChannel(info: ModelInfo, postprocess: Step[]): number | null {
  const channel = info.display_channel;
  const channels = info.output_channels;
  if (channel === null || channel === undefined || !channels || info.has_channel === false) return null;
  if (outputChannels(postprocess, channels) !== channels) return null;
  return Math.trunc(channel);
}

/** A served model's layer (layers.py's prediction_layer): a segmentation
 * layer when the chain ends in labels, else an image layer in `color`
 * (PREDICTION_COLORS[index % length], the model's place in the submit
 * order) over the range the chain produces, with the user's `keep` shader
 * and controls instead when they left some. The source URL is the engine's
 * (`served.url`, zarr3://...), transformed when the output must be drawn at
 * another voxel size than declared. With a display channel, the layer opens
 * on it: Neuroglancer's channel axis is a local dimension whose position is
 * the channel shown, at the channel's centre as Neuroglancer places them. */
export function predictionLayer(served: ServedModel, chain: Chain, color: string, keep?: KeptShader): LayerSpec {
  const { info } = served;
  const source = predictionSource(served.url, predictionVoxelOverride(info), info.has_channel ?? true);
  if (isSegmentation(chain.postprocess) === true) {
    return { type: "segmentation", name: served.entry.name, source };
  }
  let shader = keptShader(keep);
  if (shader === undefined) {
    // The type the chain serves when a step declares one (uint8 after a
    // DefaultPostprocessor), which the shader's bounds must be values of.
    const range = outputDisplayRange(chain.postprocess, info.output_class);
    shader = predictionShader(color, range, outputDtype(chain.postprocess));
  }
  const layer: LayerSpec = { type: "image", name: served.entry.name, source, shader };
  const controls = keptControls(keep);
  if (controls) layer.shaderControls = controls;
  const channel = displayChannel(info, chain.postprocess);
  if (channel !== null) layer.localPosition = [channel + 0.5];
  return layer;
}

// ---------------------------------------------------------------------------
// The viewer's initial state
// ---------------------------------------------------------------------------

/** Length units a dataset's axes may be in, to metres. DatasetInfo's voxel
 * sizes are in nm whatever the axis says; the unit only tells a spatial axis
 * from a channel or time axis. */
const LENGTH_UNITS = new Set(["nm", "nanometer", "nanometre", "um", "µm", "micrometer", "micrometre", "mm", "millimeter", "millimetre", "m", "meter", "metre", ""]);

/** The spatial axes of `info`, each with its index into the levels' shape,
 * voxel and origin arrays. The arrays may cover every axis or only the last
 * ones (chunkmirage keeps channels apart from the spatial shape), so an axis
 * is paired with them from the end. A channel axis ("c") is never spatial;
 * an axis with no unit is spatial when it is x, y or z. */
function spatialAxes(info: DatasetInfo): { name: string; index: number }[] {
  const count = info.levels[0]?.voxel.length ?? info.axes.length;
  const first = Math.max(0, info.axes.length - count);
  const axes: { name: string; index: number }[] = [];
  info.axes.forEach((axis, i) => {
    if (i < first) return;
    const unit = axis.unit ?? "";
    const spatial = axis.name !== "c" && LENGTH_UNITS.has(unit) && (unit !== "" || ["x", "y", "z"].includes(axis.name));
    if (spatial) axes.push({ name: axis.name, index: i - first });
  });
  return axes;
}

/** The viewer's dimensions for the raw data (bootstrap.raw_dimensions): its
 * axes at the finest level's voxel size, in metres, as the Python
 * CoordinateSpace with units "nm" emits them. Set before any layer is
 * added, so that the raw decides the viewer's coordinate space rather than
 * whichever layer Neuroglancer takes it from, such as a model's output at
 * another voxel size. */
export function dimensionsOf(info: DatasetInfo): NgDimensions {
  const finest = info.levels[0];
  const dimensions: NgDimensions = {};
  for (const { name, index } of spatialAxes(info)) dimensions[name] = [nmToMetres(finest.voxel[index]), "m"];
  return dimensions;
}

/** The volume's centre as a viewer position: in voxels of dimensionsOf, the
 * finest level's, measured from the coordinate origin (the level's origin,
 * nm, is the corner of its voxel 0, so it is part of the position). */
export function centrePosition(info: DatasetInfo): number[] {
  const finest = info.levels[0];
  return spatialAxes(info).map(({ index }) => (finest.origin?.[index] ?? 0) / finest.voxel[index] + finest.shape[index] / 2);
}

/** `layers` as Neuroglancer's state holds them: an array, each with its
 * `name` (the record's key; the Python API's dict of layers is its own
 * convenience), in the record's order. */
export function layerList(layers: Record<string, LayerSpec>): NgState["layers"] {
  return Object.entries(layers).map(([name, spec]) => ({ ...spec, name }));
}

/** A new viewer's state on the raw data `info` describes (bootstrap.new_viewer):
 * the raw's dimensions, the position at the volume's centre, one voxel per
 * screen pixel (crossSectionScale 1, Neuroglancer's default, which the Python
 * viewer leaves), `layers` in their order, the raw first, and the layout. */
export function initialState(info: DatasetInfo, layers: Record<string, LayerSpec>, options: { layout?: string } = {}): NgState {
  return {
    gpuMemoryLimit: GPU_MEMORY_LIMIT,
    dimensions: dimensionsOf(info),
    position: centrePosition(info),
    crossSectionScale: 1,
    layers: layerList(layers),
    layout: options.layout ?? DEFAULT_LAYOUT,
  };
}

/** A new viewer's state when the raw's geometry is not known: only the raw
 * layer, Neuroglancer taking the dimensions and the centre from the data it
 * reads. Used to restore the viewer from the session alone (a dataset URL,
 * no DatasetInfo) when the page's URL carries no state. */
export function stateWithoutInfo(url: string, raw?: LayerSpec): NgState {
  const layer: LayerSpec = raw ?? { type: "image", source: rawSource(url) };
  return {
    gpuMemoryLimit: GPU_MEMORY_LIMIT,
    layers: layerList({ [RAW_LAYER_NAME]: layer }),
    layout: DEFAULT_LAYOUT,
  };
}
