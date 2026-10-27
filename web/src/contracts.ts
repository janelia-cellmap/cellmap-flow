// What the parts of the browser mode agree on: the types each module exports
// to the others, and the few constants they share. Each module (backend,
// infer, viewer, models, state) implements its side of these; main.ts wires
// them together. Change a type here only together with every module that
// uses it.
//
// The dashboard's own JavaScript (cellmap_flow/dashboard/static/js) knows
// nothing of this file: it speaks to the backend over fetch("/api/...") and
// EventSource exactly as it speaks to the Flask routes, and that HTTP contract
// (request and response JSON per route) is defined by the Python routes. The
// backend here must answer the same way.

// ---------------------------------------------------------------------------
// Chains (cellmap_flow/pipeline_spec.py)
// ---------------------------------------------------------------------------

/** A step is flat: the op's class name and its constructor arguments. The
 * dashboard's forms send every value as a string and the constructors parse
 * them; the browser keeps values as given, like PipelineSpec does. */
export type StepValue = string | number | boolean | null;
export interface Step {
  name: string;
  [param: string]: StepValue | undefined;
}
/** The two chains, in the order they run. JSON key order matters for the URL
 * blob: input_norm first, then postprocess, then extras. */
export interface Chain {
  input_norm: Step[];
  postprocess: Step[];
}

/** The marker around the args blob in a layer's URL (serving/protocol.py). */
export const ARGS_KEY = "__CFLOW_ARGS__";

// ---------------------------------------------------------------------------
// The /api/* contract between the dashboard's JavaScript and the backend
// ---------------------------------------------------------------------------

export interface ApiRequest {
  method: string; // GET, POST, PUT, DELETE
  /** The path without the page's base, starting "/api/". */
  path: string;
  query: URLSearchParams;
  /** The parsed JSON body, or null when there is none or it is not JSON. */
  body: unknown;
  /** Header names lower-cased. Last-Event-ID is the one that matters. */
  headers: Record<string, string>;
}

export type ApiResponse =
  | { status: number; json: unknown }
  | { status: number; body: string | ArrayBuffer; type: string; headers?: Record<string, string> }
  /** A server-sent event stream, or any other response written over time. */
  | { status: number; stream: ReadableStream<Uint8Array>; type: string };

/** A route's handler. `params` are the named parts of the route's pattern
 * (e.g. the job id in /api/finetune/job/:id/status). */
export type RouteHandler = (req: ApiRequest, params: Record<string, string>, ctx: AppContext) => ApiResponse | Promise<ApiResponse>;

export interface Backend {
  handle(req: ApiRequest): Promise<ApiResponse>;
}

/** How a route that only a server can serve answers in the browser: the same
 * shape the dashboard's JavaScript already handles for a failed request, with
 * a reason it can show. */
export interface Unavailable {
  success: false;
  available: false;
  error: string;
}

/** The bodies the Python routes parse with pydantic (dashboard/requests.py),
 * as generated into src/generated/cellmap-flow.ts by `npm run gen`, for the
 * handlers here that answer the same routes; the route table (Route, ROUTES)
 * and the constants both modes share come from the same file. */
export type {
  AddModel,
  AIAnnotateDecision,
  AIAnnotateResend,
  AIAnnotateRun,
  AIAnnotateSettings,
  CreateVolume,
  FinetuneRestart,
  FinetuneSubmit,
  PipelineUpdate,
  RemoveModel,
  ResolveModel,
  Route,
  ServerConfigUpdate,
  SetData,
  SubmitModels,
} from "./generated/cellmap-flow";

// ---------------------------------------------------------------------------
// Session state (what dashboard/state.py keeps between requests)
// ---------------------------------------------------------------------------

export interface SessionState {
  /** The raw volume's URL (http(s) to an OME-Zarr group or array, s3://, gs://), or null. */
  datasetUrl: string | null;
  /** The chains as last submitted. */
  chain: Chain;
  /** The models being served, in the order they were submitted (layer colours follow it). */
  models: ModelEntry[];
  resample: boolean;
  /** /api/finetune/user-prefs and the like. */
  prefs: Record<string, unknown>;
  /** The annotation volume the Finetune tab paints into and trains on: its
   * id in the paint store (a PaintVolume), or null. Tab-scoped like the
   * models; the store's listing is the durable one. */
  paintVolume: string | null;
  /** The channel each model's layer shows, by model name, as POST
   * /api/finetune/channel sets it (the training form's Channel, and Use this
   * model's pick): the channel's index into the model's output channels. A
   * layer built for the model later opens on it, and so does one for a
   * model finetuned from it (serving.ts shownChannel). The server's
   * `shown_channels` on its dashboard session. */
  shownChannels: Record<string, number>;
}

/** The session, kept in the tab's sessionStorage so it survives the page reload that
 * connect.js does after /api/set-data. */
export interface SessionStore {
  readonly state: SessionState;
  /** Change the state and persist it. */
  update(change: Partial<SessionState>): Promise<void>;
  onChange(listener: (state: SessionState) => void): () => void;
}

// ---------------------------------------------------------------------------
// Models (Hugging Face cellmap/onnx-web-models, one manifest.json per model)
// ---------------------------------------------------------------------------

export type VariantName = "fp32" | "fp16" | "int8w" | "int8w_fp16";

export interface ModelVariant {
  variant: VariantName;
  input_size: number[]; // z, y, x voxels
  output_size: number[];
  file: string; // file name within the model's folder
  file_mb: number;
  // fidelity against the fp32 PyTorch model, from the manifest
  max_abs?: number;
  mean_abs?: number;
  flip_frac?: number;
  ort_cpu_s?: number;
  /** A model finetuned in this browser: the Hugging Face file its bytes
   * were made from (FINETUNED_REPO entries only, since 2026-10-16). Its
   * graph is that file's, byte for byte but for the weights trained, so
   * the base's finetune block (ModelManifest.finetune) describes it by
   * that file's name, and its finetune files are in that folder. */
  source?: { repo: string; folder: string; file: string };
}

export interface ModelManifest {
  model: string;
  task: string;
  architecture: string;
  checkpoint: string;
  params: number;
  channels: string[];
  input_voxel_size_nm: number[];
  output_voxel_size_nm: number[];
  /** What the output is: "logits" (sigmoid or threshold at 0) for Marwan's exports. */
  output: string;
  /** The normalizers the model expects, as cellmap-flow steps. */
  input_norm: Step[];
  io: { input: string; output: string; [k: string]: unknown };
  variants: ModelVariant[];
  /** Where the export came from, as Marwan's manifests say since
   * 2026-10-16: the PyTorch model's Hugging Face repo, the 3D model's a 2D
   * student was distilled from, and how it was exported. */
  source_repo?: string;
  distilled_from_repo?: string;
  torch_version?: string;
  exporter?: string;
  opset?: number;
  /** What the browser can train of the model, as the export declares it.
   * Absent from a manifest written before 2026-10-16, whose head the
   * browser finds in the graph itself (finetune/modes.ts). */
  finetune?: FinetuneSpec;
}

/** A manifest's `finetune` block: one entry per way to train the model,
 * named as the Finetune tab's modes are (finetune/modes.ts). A mode the
 * browser does not know is left out when the manifest is read. */
export interface FinetuneSpec {
  head?: HeadFinetuneSpec;
  full?: FullFinetuneSpec;
}

/** The model's last layer, a 1×1 convolution, trained on frozen features. */
export interface HeadFinetuneSpec {
  /** The ONNX initializers of its weight and bias, and the tensor it reads. */
  weight: string;
  bias: string | null;
  features: string;
  onnx_node?: string;
  in_channels?: number;
  out_channels?: number;
  /** The files whose head can be written: fp32 and fp16. An int8w file's
   * weights feed a DequantizeLinear and are not plain initializers. */
  files: string[];
  initializers?: string;
}

/** The whole network, trained in the browser by `library`'s runtime. */
export interface FullFinetuneSpec {
  /** "tfjs": TensorFlow.js, from `files` (the graph, Marwan's runtime). */
  library: string;
  files: string[];
  /** "identity": the graph's params are named as the ONNX initializers (the
   * 2D models). Else the path of weight_map.json (the 3D models, since
   * 2026-10-17): the graph's params are PyTorch's state_dict tensors, and
   * this says how each initializer is cut from one. */
  weight_map: string;
  /** With a weight_map file, the path of torch_params.json: how each
   * PyTorch param is rebuilt from the initializers. */
  torch_params?: string;
  /** The folder's initializers.json: each file's initializers' offsets. */
  initializers: string;
  /** The files the trained weights can be written back into. */
  write_back: string[];
  /** Params that never train (BatchNorm's running statistics). */
  frozen: string[];
  /** What a training step cost where Marwan measured it: per tile side,
   * one step of one tile, and how. */
  spike?: { setup?: string; date?: string; full: Record<string, { step_ms: number; peak_mb: number }> };
}

/** A model as the dashboard serves it: one manifest, one chosen variant. */
export interface ModelEntry {
  /** Stable id, e.g. "cellmap/onnx-web-models/setup_54". */
  id: string;
  /** The layer and job name the dashboard shows, a valid URL path segment. */
  name: string;
  repo: string; // "cellmap/onnx-web-models"
  folder: string; // "setup_54"
  manifest: ModelManifest;
  variant: ModelVariant;
  /** Where the .onnx bytes are fetched from (Hugging Face resolve URL). */
  onnxUrl: string;
}

/** What /__control__/model_info says about a served model (models/geometry.py,
 * chunkmirage_server.py), as the dashboard's layer code reads it. */
export interface ModelInfo {
  output_channels: number;
  channels: string[];
  write_shape: number[]; // output block in nm (z, y, x), as the Python server sends it
  read_shape: number[]; // input block in nm
  output_voxel_size: number[];
  input_voxel_size: number[];
  effective_output_voxel_size?: number[];
  has_channel: boolean;
  /** Channels first, as chunkmirage serves them. */
  output_axes: string[];
  available: boolean;
  output_class?: string | null;
  output_min?: number | null;
  output_max?: number | null;
  display_channel?: number | null;
  engine: "browser";
}

export interface GpuLimits {
  maxBufferSize: number | null;
  maxStorageBufferBindingSize: number | null;
  shaderF16: boolean;
}

export interface Catalog {
  /** Every model the browser can run, one entry per manifest with its default variant. */
  list(): Promise<ModelEntry[]>;
  /** The same, fetched afresh from the Hub (the dashboard's Refresh button). */
  refresh(): Promise<ModelEntry[]>;
  /** An entry from a reference the dashboard sends: an id, a repo/folder, or a name. */
  resolve(ref: string): Promise<ModelEntry | null>;
  /** The manifest as /api/huggingface-models shows a repo's metadata.json. */
  metadataFor(entry: ModelEntry): Record<string, unknown>;
  /** The variant to run on this machine. */
  chooseVariant(manifest: ModelManifest, limits: GpuLimits): ModelVariant;
  /** The .onnx bytes, from IndexedDB when already fetched (Hugging Face rate-limits anonymous downloads). */
  modelBytes(entry: ModelEntry, onProgress?: (loaded: number, total: number | null) => void): Promise<ArrayBuffer>;
  /** Keep a model finetuned in this browser: its ONNX bytes under
   * `record.entry.onnxUrl` (the byte cache) and the record itself
   * (IndexedDB), so that list() and resolve() answer its entry on later
   * visits too and modelBytes() serves it without a download. */
  addFinetuned(record: FinetunedRecord, bytes: ArrayBuffer): Promise<void>;
  /** The finetuned models kept, newest first. */
  finetuned(): Promise<FinetunedRecord[]>;
  /** Forget a finetuned model and its bytes. */
  removeFinetuned(id: string): Promise<void>;
}

/** The "repo" of the entries finetuned in this browser (ModelEntry.repo and
 * the first segment of their ids), where Hugging Face models have theirs. */
export const FINETUNED_REPO = "finetuned";

/** The metadata_props key a finetuned model's downloaded .onnx carries its
 * entry's manifest under, as JSON (GET /api/finetune/model/:name/download),
 * so that the file alone says its voxel sizes, blocks and channels: the
 * server's onnx model type reads it from there (cellmap_flow/models/configs/
 * onnx.py MANIFEST_KEY). */
export const MANIFEST_METADATA_KEY = "cellmap_flow.manifest";

/** A model finetuned in this browser: the entry it is served as (the base's
 * manifest under the new name, one variant whose file is `<folder>.onnx`,
 * `onnxUrl` the key its bytes are kept under), and where it came from. */
export interface FinetunedRecord {
  /** The entry's id, `finetuned/<folder>`. */
  id: string;
  /** The layer and job name, `<base name>-ft<n>`; also the folder. */
  name: string;
  base: { id: string; name: string; file: string };
  /** The annotation volume it was trained on (PaintVolume.id). */
  volume: string;
  created_at: string;
  /** The training parameters as the Finetune tab sent them. */
  params: Record<string, unknown>;
  epochs: number;
  final_loss: number | null;
  entry: ModelEntry;
  /** Set when the model came from a shared link rather than a run in this
   * browser (link.ts rebuildFinetuned): the name the link gave it, which is
   * `name` unless this browser had a model of that name already. Such a
   * record's `base` is the Hugging Face model it was rebuilt on, and it has
   * no volume here (""), no parameters and no epochs: the training was in
   * the browser the link was made in. */
  from_link?: { name: string };
}

// ---------------------------------------------------------------------------
// Viewer (what viewer/layers.py and the Python viewer do today)
// ---------------------------------------------------------------------------

/** A Neuroglancer layer as JSON, the shape neuroglancer.ImageLayer(...).to_json() gives. */
export type LayerSpec = Record<string, unknown>;

export interface ViewerController {
  /** Put the viewer on the page if the session has a dataset, else leave the
   * connect panel (its field takes a URL in the browser). */
  restore(): Promise<void>;
  /** Open the raw volume at `url` in the viewer; returns the viewer's own URL.
   * With `info`, the viewer's dimensions and position come from the volume
   * (the finest voxel size, the centre), as the CLI viewer sets them;
   * without, Neuroglancer derives them from the layer. */
  openDataset(url: string, raw: LayerSpec, info?: DatasetInfo): Promise<string>;
  /** Make `url` the dataset the viewer shows after the page reloads, without
   * showing it now: the state goes into the page's URL, where restore()
   * reads it. For /api/set-data, which connect.js follows with a reload (the
   * server's route builds its viewer without showing it either); returns the
   * viewer's URL for that state. */
  prepareDataset(url: string, raw: LayerSpec, info?: DatasetInfo): string;
  /** Replace the named layers, each keeping what the user set on the one it
   * replaces (viewer/state.ts's carriedLayer: blending, opacity, being
   * hidden...). `replacing` names, for a new layer, the layer it takes over
   * from, whose place and settings it gets (a finetuned model's next
   * iteration); that one is removed. */
  setLayers(layers: Record<string, LayerSpec>, replacing?: Record<string, string>): void;
  removeLayer(name: string): void;
  /** The viewer's position, in nm (z, y, x), or null without a viewer. */
  position(): number[] | null;
  viewerUrl(): string | null;
  /** Show `sets` as layer `name`'s equivalences, as the dashboard's
   * /update/equivalences route does: sets of decimal uint64 strings, each
   * set the ids that are one object. */
  setEquivalences(name: string, sets: string[][]): void;
  /** The raw layer's shader/contrast as the user left it, to keep across redraws. */
  layerShader(name: string): { shader?: string; shaderControls?: Record<string, unknown> } | null;
  /** Select layer `name`, its panel open on `tab` (the fork's "Draw" tab for
   * a paint layer, where the brush is), as the server's
   * add_annotation_layer does. A name the viewer has no layer of yet is
   * selected when the layer is set. */
  selectLayer(name: string, tab?: string): void;
  /** The session's dataset as the engine opened it, for a viewer restore()
   * left waiting: when the session names a dataset but the tab has no view
   * of it (a link, or a reload that lost the view), restore() mounts
   * nothing, and this mounts the viewer with the raw's geometry (its
   * dimensions explicit, the view at its centre), as Connect's state has
   * them. Mounted without the geometry, Neuroglancer takes the coordinate
   * space from whichever layer's metadata arrives first (8 nm raw, 16 nm
   * model), and a layer at another voxel size added later can switch it and
   * move the view; dimensions set after the viewer is up did not hold.
   * `info` null (the volume could not be opened) mounts the viewer without
   * it, for Neuroglancer to try. Does nothing when restore() did not wait. */
  datasetOpened(url: string, info: DatasetInfo | null): void;
  /** Where the pointer is over the data, in nm (z, y, x), as Neuroglancer's
   * mouse state has it (the `mouse_voxel_coordinates` the Python's Shift+G
   * action gets): null when it is not over the viewer, or there is none. */
  pointerPosition(): number[] | null;
  /** The data axis the viewer's single-panel layout looks along (0 z: the
   * XY plane, 1 y: XZ, 2 x: YZ), the first panel's (XY) in a multi-panel
   * layout, as geometry.depth_axis_for_view reads the Python viewer's. */
  viewDepthAxis(): 0 | 1 | 2;
  /** The viewer's display dimensions (geometry.display_dimensions_for_view):
   * those its state sets, else the first three of its coordinate space;
   * empty when it has neither. They say how a panel draws its plane. */
  viewDisplayDimensions(): string[];
  /** Bind `key` (Neuroglancer's spelling, "shift+keyg") to `action` in the
   * viewer, `handler` run when it is pressed, now and in every viewer
   * mounted after (a new dataset is a new page of Neuroglancer); returns
   * the unbind. */
  bindKey(key: string, action: string, handler: () => void): () => void;
  /** Make the viewer read layer `name`'s chunks again (labels written behind
   * its back, as a server's refresh_annotation_layer does), keeping the
   * layer's own settings; whether a layer of that name was found. */
  refreshLayer(name: string): boolean;
  /** Layer `name` as the viewer has it now, its JSON in the viewer's state
   * (what the user did in it included: the boxes drawn in an annotation
   * layer), or null when the viewer has no such layer. */
  layer(name: string): LayerSpec | null;
}

// ---------------------------------------------------------------------------
// Inference (the core: chunks computed on demand for the viewer)
// ---------------------------------------------------------------------------

export interface DatasetInfo {
  url: string;
  dtype: string;
  channels: number;
  axes: { name: string; unit: string }[];
  /** Finest first; voxel sizes in nm. */
  levels: { shape: number[]; voxel: number[]; origin: number[] }[];
}

export interface ServedModel {
  entry: ModelEntry;
  /** The engine view's name: `${entry.name}${ARGS_KEY}${blob}${ARGS_KEY}`. */
  view: string;
  /** The layer's source URL for Neuroglancer (zarr3://...). */
  url: string;
  info: ModelInfo;
  /** Stop serving it and free its session if no other served model shares it. */
  dispose(): Promise<void>;
}

export type Provider = "webgpu" | "wasm";

export interface InferenceStats {
  provider: Provider | null;
  device: string | null;
  blocks: number;
  inflight: number;
  lastMs: number | null;
  avgMs: number | null;
  status: string;
}

export interface Inference {
  /** Open the raw volume as the engine's "raw" view; the first call starts the engine. */
  openDataset(url: string): Promise<DatasetInfo>;
  dataset(): DatasetInfo | null;
  /** Serve `entry` through `chain`; `blob` is the chain's URL blob (shared/args.ts), which names the view.
   * `modelBytes` gives the .onnx bytes (Catalog.modelBytes); it is called once per entry, when its
   * session is first built. */
  serve(entry: ModelEntry, chain: Chain, blob: string, modelBytes: () => Promise<ArrayBuffer>): Promise<ServedModel>;
  served(): ServedModel[];
  gpuLimits(): Promise<GpuLimits>;
  readonly stats: InferenceStats;
  onStats(listener: (stats: InferenceStats) => void): () => void;
  /** Ids a served layer's merging step has joined so far, as sets of
   * decimal uint64 strings, each set the ids that are one object, whenever
   * they grow: the viewer shows them as the segmentation layer's
   * equivalences. `view` is the served view's name. */
  onEquivalences(listener: (view: string, sets: string[][]) => void): () => void;
  /** Drop every session and the WebGPU device (recovers a hung run); views are re-created on the next chunk. */
  reset(): Promise<void>;
  /** Voxels [lo, hi) of level `level` of the open dataset (DatasetInfo.levels,
   * finest first), within the level, as the engine's reader reads them for a
   * chunk: the bytes of the level's dtype in C order (z, y, x). The caller
   * clips the box to the level and pads what lies outside (chunkmirage's
   * `padded`, as the engine does for a stage's input). */
  readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer>;
  /** The served model `view`'s own output (ServedModel.view): what its model
   * stage computes, the chain's normalizers and none of its postprocessors,
   * over voxels [lo, hi) of the level it is served at (its output grid,
   * z, y, x), as [C, z, y, x] float32 in C order (`shape` says it). What the
   * server's Seed from Prediction reads of a model server, with no
   * postprocessing (view_labels.read_prediction): a threshold is set on the
   * raw output. Chunks already computed for the viewer come from the
   * engine's caches; the rest are computed. The box must lie within the
   * level. */
  readOutput(view: string, lo: number[], hi: number[]): Promise<{ data: Float32Array; shape: number[] }>;
  /** Compute the served model `view` (ServedModel.view: the layer's view,
   * the chain's postprocessors and all) over voxels [lo, hi) of the level it
   * is served at, and write it into `store` as an OME-Zarr of exactly that
   * box: chunkmirage's Engine.materialize, which walks the model's chunks
   * the box reaches (through the engine's caches and demand queue, as the
   * viewer's requests go), stitches the box's own chunks from them and
   * writes the zarr: one level of hi − lo voxels, the level's scale, a
   * translation that puts its voxel 0 on the level's voxel lo, empty chunks
   * skipped. For Run on a region (src/run-region). The box must lie within
   * the level. */
  materialize(view: string, store: RegionStore, options: RegionWrite): Promise<RegionWritten>;
  /** What materialize would hold over that box, before anything is
   * computed (chunkmirage's Engine.materializePlan): the most bytes of the
   * saved array's chunks held at once while they are stitched from the
   * model's, and the axes the box's corner is off the model's chunk edges
   * along (z 0, y 1, x 2), the ones putting it on an edge shrinks that. */
  materializePlan?(view: string, box: { lo: number[]; hi: number[] }): Promise<RegionPlan>;
  /** A function of cellmap_flow.portable, the Python both modes run, by
   * module path (`"cellmap_flow.portable.calls:rank_tally"`), run in the
   * engine's Pyodide worker as chunkmirage's `Engine.call` runs it:
   * `fn(args, arrays)` with the JSON arguments and the arrays' bytes (which
   * the call takes over), answering JSON, one array (an ArrayBuffer) or
   * `{value, arrays}`. Python loads when the dataset opens, numpy with it,
   * so the first call rarely waits for it. */
  python<T>(fn: string, args: Record<string, unknown>, arrays?: ArrayBuffer[]): Promise<T>;
}

/** Where materialize writes a zarr's files, by their path within it
 * (`.zattrs`, `0/.zarray`, `0/0.1.2`): chunkmirage's WritableStore, which
 * its directoryStore (a folder on this machine) and memoryStore are. */
export interface RegionStore {
  put(path: string, bytes: Uint8Array): Promise<void>;
}

/** How far a materialize has got: the saved array's chunks done of its
 * `total`, those `written` (the rest were empty), and the bytes written so
 * far; and the model's chunks `computed` of those the box `needed`, which a
 * box off the chunks' edges has more of than it saves (its chunks are
 * stitched from them). */
export interface RegionProgress {
  done: number;
  total: number;
  written: number;
  bytes: number;
  computed?: number;
  needed?: number;
}

/** What materialize is asked to write: the box [lo, hi) in voxels of the
 * served level (z, y, x), attributes merged into the zarr group's (the
 * provenance), a signal that stops it and a progress callback. */
export interface RegionWrite {
  lo: number[];
  hi: number[];
  /** The chunks' compression: gzip (the default, smaller; zarr-python,
   * Neuroglancer and napari read it) or none (null), for a reader without
   * gzip, such as Fiji's. */
  compression?: "gzip" | null;
  attributes?: Record<string, unknown>;
  signal?: AbortSignal;
  onProgress?: (progress: RegionProgress) => void;
}

/** What a materialize would hold (Inference.materializePlan). */
export interface RegionPlan {
  peakBytes: number;
  split: number[];
}

/** What a materialize did: the box's chunks, those written and skipped
 * (empty), the bytes written, and whether the signal stopped it first. */
export interface RegionWritten {
  chunks: number;
  written: number;
  skipped: number;
  bytes: number;
  aborted: boolean;
}

// ---------------------------------------------------------------------------
// The whole
// ---------------------------------------------------------------------------

export interface AppContext {
  session: SessionStore;
  catalog: Catalog;
  infer: Inference;
  viewer: ViewerController;
  /** The paint: the annotation volumes' files, shared with the service worker (IndexedDB). */
  paint: PaintStore;
  /** In-page events the backend turns into server-sent events: "log" (a
   * CustomEvent whose detail is a line) and "status". */
  events: EventTarget;
  /** Append a line to the page's log (shown in the dashboard's status area and the console). */
  log(line: string): void;
}

// ---------------------------------------------------------------------------
// Finetuning in the browser (src/finetune, backend/routes/finetune.ts): the
// paint store the service worker serves and the volumes in it
// ---------------------------------------------------------------------------

/** Where the paint is served: `<page directory>paint/<volume>/<path>`, an
 * S3-style endpoint (GET, HEAD, PUT, DELETE of whole objects) the service
 * worker (src/sw.ts) answers from the paint store, which the Neuroglancer
 * fork's writable zarr source writes to as it writes to MinIO in server
 * mode. Directly under the page's directory, like `local/`. */
export const PAINT_ROUTE = "paint/";

/** An annotation volume's files, by volume id and path within the volume
 * (`.zattrs`, `annotation/s0/.zarray`, `annotation/s0/3.4.5`). The page
 * writes a new volume's metadata and reads the paint for training; the
 * service worker reads and writes on the viewer's behalf. IndexedDB in the
 * browser (both can open it), memory in tests. */
export interface PaintStore {
  get(volume: string, path: string): Promise<Uint8Array | undefined>;
  put(volume: string, path: string, bytes: Uint8Array): Promise<void>;
  delete(volume: string, path: string): Promise<void>;
  /** The paths stored under `volume`, those under `prefix` (a directory
   * within it, with its trailing slash) when given, sorted. */
  list(volume: string, prefix?: string): Promise<string[]>;
  /** The ids of the volumes with any file stored, sorted. */
  volumes(): Promise<string[]>;
  /** Every file of `volume`. */
  removeVolume(volume: string): Promise<void>;
}

/** An annotation volume: the server's record of one
 * (finetune/session/volume.py: VolumeGeometry.record and the root attrs
 * create_volume_zarr writes), which is what the volume's root `.zattrs`
 * holds here too, so a volume downloaded from the browser reads as one made
 * on the server (read_volume). Labels: 0 unpainted, 1 background, 2 and up
 * objects. One chunk is one model output block, so each is one training
 * sample. `dataset_offset_nm` is voxel 0's centre, the OME translation. */
export interface PaintVolume {
  /** `vol-<8 hex>-<YYYYmmdd-HHMMSS>` (new_volume_id). */
  id: string;
  type: "annotation_volume";
  /** The model the volume is on the grid of; null for one painted before
   * any model (planPaintVolume), on the raw's grid, which is not trained on. */
  model_name: string | null;
  /** The ModelEntry's id (the browser's own field; the server has none); null as model_name. */
  model_id: string | null;
  /** The raw volume's URL (the server's dataset_path). */
  dataset_path: string;
  /** The model's output block in voxels (z, y, x): the chunk. */
  chunk_size: number[];
  /** The model's input block in voxels. */
  input_size: number[];
  /** nm: the grid the volume lies on, the prediction's own. */
  output_voxel_size: number[];
  input_voxel_size: number[];
  /** The model's own voxel sizes, for provenance, where the ones above are the raw level's. */
  claimed_output_voxel_size?: number[];
  claimed_input_voxel_size?: number[];
  dataset_offset_nm: number[];
  /** The volume's extent, padded to whole chunks. */
  dataset_shape_voxels: number[];
  /** uint8, or uint16 for an instance model's ids (annotation_dtype_for). */
  dtype: "uint8" | "uint16";
  /** The chains when the volume was made: the trainer normalizes through input_norm. */
  input_norm: Step[];
  postprocess: Step[];
  created_at: string;
  resample?: boolean;
}
