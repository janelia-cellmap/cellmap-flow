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
}

/** The session, kept in IndexedDB so it survives the page reload that
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
  /** Replace the named layers, keeping a layer's visibility if it was hidden. */
  setLayers(layers: Record<string, LayerSpec>): void;
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
}

// ---------------------------------------------------------------------------
// The whole
// ---------------------------------------------------------------------------

export interface AppContext {
  session: SessionStore;
  catalog: Catalog;
  infer: Inference;
  viewer: ViewerController;
  /** In-page events the backend turns into server-sent events: "log" (a
   * CustomEvent whose detail is a line) and "status". */
  events: EventTarget;
  /** Append a line to the page's log (shown in the dashboard's status area and the console). */
  log(line: string): void;
}
