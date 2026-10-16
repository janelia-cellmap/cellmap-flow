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
import { PAINT_ROUTE } from "../contracts";
import type { Chain, DatasetInfo, LayerSpec, ModelInfo, PaintVolume, ServedModel, Step } from "../contracts";
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

// ---------------------------------------------------------------------------
// The paint layer: the annotation volume as the fork's writable segmentation
// layer, as dashboard/routes/finetune/overlay.py's add_annotation_layer
// builds it over MinIO (and selects it: viewer/index.ts selectLayer).
// ---------------------------------------------------------------------------

/** What a paint layer's name starts with: `annotation_<volume id>`, the
 * name the server's add-to-viewer route gives it (`annotation_{crop_id}`,
 * the crop id being the volume's id, as sessions.js sends it) and the one
 * the tab's crops.js expects: Load Crops looks for that layer before adding
 * its own. */
export const PAINT_LAYER_PREFIX = "annotation_";

/** The paint layer's name for the volume `volumeId`. */
export function paintLayerName(volumeId: string): string {
  return PAINT_LAYER_PREFIX + volumeId;
}

/** The keys the layer's draw tools are bound to (overlay.py's
 * ANNOTATION_TOOL_BINDINGS): the fork's brush and flood fill, from the
 * voxel_annotation module of google/neuroglancer#858, a single capital
 * letter each as Neuroglancer's tool bindings want. */
export const PAINT_TOOL_BINDINGS: Readonly<Record<string, string>> = { A: "vox-brush", F: "vox-flood-fill" };

/** The tab the layer's panel opens on: the fork's Draw tab, where the brush
 * and flood fill are (voxel_annotation/index.ts adds it to a layer with a
 * writable subsource). */
export const PAINT_TAB = "Draw";

/** The URL the viewer paints `volumeId` through: the page's paint route
 * (PAINT_ROUTE, which the service worker answers from the paint store) as
 * an S3-compatible endpoint, in the fork's `s3+http(s)://<host>/<path>`
 * scheme, whose provider reads and writes `http(s)://<host>/<path>/<key>`
 * unsigned, as the server's `s3+<MinIO url>/annotation` does against MinIO.
 * `pageBase` is the page's directory with its trailing slash
 * (`https://janelia-cellmap.github.io/cellmap-flow/app/`,
 * `http://localhost:5174/`), the service worker's scope; by default this
 * page's own. */
export function paintLayerUrl(volumeId: string, pageBase: string = typeof location === "undefined" ? "http://localhost/" : new URL(".", location.href).href): string {
  const base = pageBase.endsWith("/") ? pageBase : `${pageBase}/`;
  return `s3+${base}${PAINT_ROUTE}${volumeId}/annotation`;
}

/** The paint route's volume segment in a layer's source URL, with the
 * optional refresh suffix: `.../paint/<volume>[~<n>]/`. The volume id has
 * no "~" or "/" (`vol-<8 hex>-<YYYYmmdd-HHMMSS>`), so the suffix is
 * unambiguous. The same shape paint-route.ts strips before the store.
 * (PAINT_ROUTE is `paint/`, letters and a slash: nothing to escape.) */
const PAINT_VOLUME_SEGMENT = new RegExp(`(/${PAINT_ROUTE})([^/~]+)(?:~(\\d+))?(?=/)`);

/** `url` with the paint route's volume segment respelled, so that
 * Neuroglancer takes it for a new source and reads its chunks afresh: the
 * server's refresh_annotation_layer does the same by respelling the MinIO
 * port (overlay._respelled_port, `:5000` to `:05000`), since the viewer
 * caches a source by its URL and labels written behind its back (an
 * accepted AI annotation, an undo) show only once the chunks are read
 * again. The volume segment takes a `~<n>` suffix the service worker
 * strips (paint-route.ts), so every spelling reaches the same volume; `n`
 * counts up (`<volume>` to `<volume>~1`, `~1` to `~2`, ...) rather than
 * toggling, so no refresh returns to a spelling the viewer may still hold
 * chunks for. Null when `url` has no paint route in it. */
export function respelledPaintUrl(url: string): string | null {
  const match = PAINT_VOLUME_SEGMENT.exec(url);
  if (!match) return null;
  const [whole, route, volume, suffix] = match;
  const next = suffix === undefined ? 1 : Number(suffix) + 1;
  return url.slice(0, match.index) + `${route}${volume}~${next}` + url.slice(match.index + whole.length);
}

/** A layer's `source` with its paint-route URL respelled by
 * respelledPaintUrl, whichever of Neuroglancer's forms the source is in (a
 * string, {url, ...} with the rest kept, since `subsources` is where the
 * fork's `writingEnabled` lives, or an array of either, the first paint
 * source respelled and the others kept); null when no paint-route URL is
 * in it. */
export function respelledPaintSource(source: unknown): unknown | null {
  if (typeof source === "string") return respelledPaintUrl(source);
  if (Array.isArray(source)) {
    for (let i = 0; i < source.length; i++) {
      const respelled = respelledPaintSource(source[i]);
      if (respelled !== null) return source.map((entry, j) => (j === i ? respelled : entry));
    }
    return null;
  }
  if (source && typeof source === "object" && typeof (source as { url?: unknown }).url === "string") {
    const url = respelledPaintUrl((source as { url: string }).url);
    return url === null ? null : { ...(source as Record<string, unknown>), url };
  }
  return null;
}

/** The writable segmentation layer on `url` for `volume`, as
 * add_annotation_layer builds it: the volume's `annotation` group as its
 * source, its default subsource writable (the fork paints into a subsource
 * whose `writingEnabled` is set, and shows the Draw tab for it) and the
 * bounds subsource on, the brush and flood fill bound to their keys, the
 * panel on the Draw tab. The fork's own settings (`brushSize` 3,
 * `brushShape` disk, `floodFillMaxVoxels` 10000, `paintValue` "1", which
 * is background, the first label to paint) are left at their defaults, as
 * the server leaves them. */
export function paintLayer(url: string, volume: PaintVolume): LayerSpec {
  return {
    type: "segmentation",
    name: paintLayerName(volume.id),
    source: { url, subsources: { default: { writingEnabled: true }, bounds: {} } },
    toolBindings: { ...PAINT_TOOL_BINDINGS },
    tab: PAINT_TAB,
  };
}

// ---------------------------------------------------------------------------
// The good regions: the boxes Mark as Good made, as the server's
// refresh_good_regions_layer draws them (dashboard/routes/finetune/
// good_regions.py), a local annotation layer of axis-aligned boxes in nm.
// ---------------------------------------------------------------------------

/** The good regions' layer name (good_regions.GOOD_REGIONS_LAYER). */
export const GOOD_REGIONS_LAYER = "good_regions";

/** A region as the layer draws it: its box in nm, (z, y, x), and its label. */
export interface RegionBox {
  label: string;
  offset_nm: number[];
  shape_nm: number[];
}

/** The layer of `regions`: neuroglancer.LocalAnnotationLayer's JSON over a
 * coordinate space of `axes` (the viewer's dimension names, the raw's) at
 * 1 nm, one axis_aligned_bounding_box per region from its offset to its
 * offset plus its shape, its id its place in the list and its description
 * its label, as the Python builds it. A point's coordinates follow the
 * names: z, y and x take the box's z, y and x wherever they are among the
 * axes (an axis by another name takes the box's coordinate in its place,
 * as pointerNm reads the viewer's). A null entry (one that is not a region)
 * is skipped and keeps its place, so the ids stay the regions' indices. */
export function goodRegionsLayer(regions: readonly (RegionBox | null)[], axes: readonly string[] = ["z", "y", "x"]): LayerSpec {
  const zyx = ["z", "y", "x"];
  const names = axes.length === 3 ? axes : zyx;
  const order = names.map((name, k) => (zyx.includes(name) ? zyx.indexOf(name) : k));
  const annotations: Record<string, unknown>[] = [];
  regions.forEach((region, index) => {
    if (!region) return;
    const hi = region.offset_nm.map((o, a) => o + region.shape_nm[a]);
    annotations.push({
      type: "axis_aligned_bounding_box",
      pointA: order.map((a) => region.offset_nm[a]),
      pointB: order.map((a) => hi[a]),
      id: String(index),
      description: region.label,
    });
  });
  const outputDimensions = Object.fromEntries(names.map((name) => [name, [nmToMetres(1), "m"]]));
  return { type: "annotation", source: [{ url: "local://annotations", transform: { outputDimensions } }], annotations };
}
