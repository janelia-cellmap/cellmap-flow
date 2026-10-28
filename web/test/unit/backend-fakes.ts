// Fakes of the AppContext's members for the backend tests: an Inference
// that records what it serves, a Catalog of two models, a ViewerController
// that records its layers, and the in-memory session. Each implements its
// contract (src/contracts.ts) and nothing more.
import { ARGS_KEY } from "../../src/contracts";
import type {
  AppContext,
  Catalog,
  Chain,
  DatasetInfo,
  FinetunedRecord,
  GpuLimits,
  Inference,
  InferenceStats,
  LayerSpec,
  ModelEntry,
  ModelInfo,
  ModelManifest,
  OutputActivation,
  RegionStore,
  RegionWrite,
  RegionWritten,
  ServedModel,
  SessionState,
  ViewerController,
} from "../../src/contracts";
import { entryFor } from "../../src/models";
import { networkActivation, outputClass } from "../../src/shared/model-info";
import { memorySession } from "../../src/state";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { portablePython } from "./pyodide";

export function manifest(name: string, channels: string[], voxel: number): ModelManifest {
  return {
    model: name,
    task: `${channels[0]}_task`,
    architecture: "ResUNet3DValid",
    checkpoint: "model_checkpoint_1",
    params: 1000,
    channels,
    input_voxel_size_nm: [voxel, voxel, voxel],
    output_voxel_size_nm: [voxel, voxel, voxel],
    output: "raw logits (apply sigmoid / threshold at 0)",
    input_norm: [
      { name: "MinMaxNormalizer", min_value: 0, max_value: 255 },
      { name: "LambdaNormalizer", expression: "2*x-1" },
    ],
    io: { input: "input", output: "output" },
    variants: [
      { variant: "fp32", input_size: [148, 148, 148], output_size: [60, 60, 60], file: `${name}_fp32.onnx`, file_mb: 42.9 },
      { variant: "int8w", input_size: [116, 116, 116], output_size: [28, 28, 28], file: `${name}_int8w.onnx`, file_mb: 10.9 },
    ],
  };
}

/** The catalog's entry for a model: its default variant (the first), named
 * as the catalog names it (models/manifest.ts entryName: task_model_voxel). */
export function entry(folder: string, task = "mito_affinities", channels: string[] = ["mito_aff_z", "mito_aff_y", "mito_aff_x"], voxel = 16): ModelEntry {
  const m = { ...manifest(folder, channels, voxel), task };
  return entryFor(folder, m, m.variants[0]);
}

/** The names the fakes' two catalog models serve under. */
export const NAME_54 = "mito_aff_setup54_16nm";
export const NAME_56 = "nuc_setup56_128nm";

export const DATASET: DatasetInfo = {
  url: "https://example.org/data.zarr",
  dtype: "uint8",
  channels: 1,
  axes: [
    { name: "z", unit: "nanometer" },
    { name: "y", unit: "nanometer" },
    { name: "x", unit: "nanometer" },
  ],
  levels: [{ shape: [1000, 1000, 1000], voxel: [8, 8, 8], origin: [0, 0, 0] }],
};

export class FakeInference implements Inference {
  opened: string[] = [];
  private current: DatasetInfo | null = null;
  private views: ServedModel[] = [];
  /** Every serve() call: the entry's name and variant, and the blob. */
  serves: { name: string; variant: string; blob: string; chain: Chain }[] = [];
  limits: GpuLimits = { maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false };
  disposed: string[] = [];
  /** How many times each entry's bytes were asked for. */
  bytesAsked = new Map<string, number>();
  stats: InferenceStats = { provider: null, device: null, blocks: 0, inflight: 0, lastMs: null, avgMs: null, status: "idle" };
  failOpen: string | null = null;

  constructor(dataset: DatasetInfo | null = null) {
    this.current = dataset;
  }

  async openDataset(url: string): Promise<DatasetInfo> {
    if (this.failOpen) throw new Error(this.failOpen);
    this.opened.push(url);
    this.current = { ...DATASET, url };
    return this.current;
  }

  dataset(): DatasetInfo | null {
    return this.current;
  }

  async serve(entry: ModelEntry, chain: Chain, blob: string, modelBytes: () => Promise<ArrayBuffer>): Promise<ServedModel> {
    this.serves.push({ name: entry.name, variant: entry.variant.variant, blob, chain });
    this.bytesAsked.set(entry.name, (this.bytesAsked.get(entry.name) ?? 0) + 1);
    await modelBytes();
    const info: ModelInfo = {
      output_channels: entry.manifest.channels.length,
      channels: entry.manifest.channels,
      write_shape: entry.variant.output_size,
      read_shape: entry.variant.input_size,
      output_voxel_size: entry.manifest.output_voxel_size_nm,
      input_voxel_size: entry.manifest.input_voxel_size_nm,
      has_channel: true,
      output_axes: ["c", "z", "y", "x"],
      available: true,
      // As the real one reads it: off the network, else the manifest (shared/model-info.ts).
      ...outputClass(entry.manifest, await this.outputActivation(entry, modelBytes)),
      engine: "browser",
    };
    const view = `${entry.name}${ARGS_KEY}${blob}${ARGS_KEY}`;
    const served: ServedModel = {
      entry,
      view,
      url: `zarr3://http://localhost/virtual/page/${view}/`,
      info,
      dispose: async () => {
        this.views = this.views.filter((s) => s !== served);
        this.disposed.push(view);
      },
    };
    this.views.push(served);
    return served;
  }

  served(): ServedModel[] {
    return [...this.views];
  }

  /** What the network in the catalog's bytes ends in, as the engine reads
   * it (shared/model-info.ts networkActivation, in node's Pyodide); null
   * when the bytes are not an ONNX model (the fake catalog's 8 bytes) or
   * there is no Pyodide. Not kept: the fake reads it every time. */
  async outputActivation(entry: ModelEntry, modelBytes: () => Promise<ArrayBuffer>): Promise<OutputActivation | null> {
    try {
      return await networkActivation(await modelBytes(), entry.manifest.io?.output, this);
    } catch {
      return null;
    }
  }

  async gpuLimits(): Promise<GpuLimits> {
    return this.limits;
  }

  onEquivalences(_listener: (view: string, sets: string[][]) => void): () => void {
    return () => {};
  }

  onStats(): () => void {
    return () => {};
  }

  async reset(): Promise<void> {}

  /** Every readRaw call's box, and zeros of the box's size for it (uint8),
   * or, with `rawPattern` set, a pattern a test can recognise. */
  reads: { level: number; lo: number[]; hi: number[] }[] = [];

  /** What readRaw returns: zeros, or a gradient along x (the voxel's x
   * index modulo 256, so the left of every plane is dark), which gives the
   * fake AI provider dark pixels to mask. */
  rawPattern: "zeros" | "gradient" = "zeros";

  async readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
    if (!this.current) throw new Error("open a dataset before reading it");
    this.reads.push({ level, lo, hi });
    const shape = hi.map((h, a) => h - lo[a]);
    const n = shape.reduce((acc, s) => acc * s, 1);
    const bytes = new Uint8Array(n);
    if (this.rawPattern === "gradient") {
      const [, , nx] = shape;
      for (let i = 0; i < n; i++) bytes[i] = (lo[2] + (i % nx)) & 255;
    }
    return bytes.buffer;
  }

  /** Every readOutput call: the view and the box on its level. */
  outputReads: { view: string; lo: number[]; hi: number[] }[] = [];

  /** The model's output at voxel (z, y, x) of channel `c` of a served
   * view's level, what readOutput answers; a test sets it. By default a
   * logit of -4 everywhere: background, well clear of the boundary. */
  outputAt: (view: string, c: number, z: number, y: number, x: number) => number = () => -4;

  async readOutput(view: string, lo: number[], hi: number[]): Promise<{ data: Float32Array; shape: number[] }> {
    const served = this.views.find((s) => s.view === view);
    if (!served) throw new Error(`${view} is not served`);
    this.outputReads.push({ view, lo, hi });
    const shape = [served.info.output_channels, ...hi.map((h, a) => h - lo[a])];
    const data = new Float32Array(shape.reduce((acc, s) => acc * s, 1));
    let i = 0;
    for (let c = 0; c < shape[0]; c++) {
      for (let z = lo[0]; z < hi[0]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[2]; x < hi[2]; x++) data[i++] = this.outputAt(view, c, z, y, x);
    }
    return { data, shape };
  }

  /** The portable Python, in node's Pyodide (pyodide.ts); a test that calls it skips when there is none. */
  async python<T>(fn: string, args: Record<string, unknown>, arrays: ArrayBuffer[] = []): Promise<T> {
    const call = await portablePython();
    if (!call) throw new Error("Pyodide is not available in this test run");
    return call<T>(fn, args, arrays);
  }

  /** Every materialize call: the view, the store and the options. */
  materialized: { view: string; store: RegionStore; options: RegionWrite }[] = [];

  /** Called with each chunk's grid index before the fake writes it: a test
   * aborts there, or throws to fail the run. */
  beforeChunk: (view: string, index: number[]) => void | Promise<void> = () => {};

  /** chunkmirage's Engine.materialize as far as a run on a region sees it:
   * the box rounded out to the served model's output blocks (its chunks),
   * each chunk a file `0/z.y.x` of one byte, the group's `.zattrs` the
   * attributes, progress after each chunk, and a stop between chunks once
   * the signal is aborted, which is answered (not thrown) with `aborted`. */
  async materialize(view: string, store: RegionStore, options: RegionWrite): Promise<RegionWritten> {
    const served = this.views.find((s) => s.view === view);
    if (!served) throw new Error(`${view} is not served`);
    this.materialized.push({ view, store, options });
    const chunk = served.entry.variant.output_size;
    const first = options.lo.map((l, a) => Math.floor(l / chunk[a]));
    const last = options.hi.map((h, a) => Math.ceil(h / chunk[a]));
    const total = last.reduce((n, l, a) => n * (l - first[a]), 1);
    const attrs = new TextEncoder().encode(JSON.stringify({ ...options.attributes }));
    await store.put(".zattrs", attrs);
    let done = 0, bytes = attrs.length;
    for (let z = first[0]; z < last[0]; z++) for (let y = first[1]; y < last[1]; y++) for (let x = first[2]; x < last[2]; x++) {
      if (options.signal?.aborted) return { chunks: total, written: done, skipped: 0, bytes, aborted: true };
      await this.beforeChunk(view, [z, y, x]);
      if (options.signal?.aborted) return { chunks: total, written: done, skipped: 0, bytes, aborted: true };
      await store.put(`0/${z}.${y}.${x}`, new Uint8Array([1]));
      done++;
      bytes++;
      options.onProgress?.({ done, total, written: done, bytes });
    }
    return { chunks: total, written: done, skipped: 0, bytes, aborted: false };
  }
}

export class FakeCatalog implements Catalog {
  constructor(public entries: ModelEntry[] = [entry("setup_54"), entry("setup_56", "nuc", ["nuc"], 128)]) {}
  failList: string | null = null;
  refreshed = 0;

  async list(): Promise<ModelEntry[]> {
    if (this.failList) throw new Error(this.failList);
    return this.entries;
  }

  async refresh(): Promise<ModelEntry[]> {
    this.refreshed++;
    return this.list();
  }

  /** The id, the folder or the name, as the real catalog resolves them. */
  async resolve(ref: string): Promise<ModelEntry | null> {
    return this.entries.find((e) => e.id === ref || e.folder === ref || e.name === ref) ?? null;
  }

  metadataFor(e: ModelEntry): Record<string, unknown> {
    return { model_name: e.name, description: e.manifest.task, channels_names: e.manifest.channels };
  }

  /** The small variant when the GPU's buffer limit is known and small, else the default. */
  chooseVariant(m: ModelManifest, limits: GpuLimits) {
    return limits.maxBufferSize !== null && limits.maxBufferSize < 2 ** 30 ? m.variants[m.variants.length - 1] : m.variants[0];
  }

  async modelBytes(entry: ModelEntry): Promise<ArrayBuffer> {
    return this.finetunedBytes.get(entry.onnxUrl) ?? new ArrayBuffer(8);
  }

  /** The finetuned models kept, newest first, and their bytes by onnxUrl. */
  finetunedRecords: FinetunedRecord[] = [];
  finetunedBytes = new Map<string, ArrayBuffer>();

  async addFinetuned(record: FinetunedRecord, bytes: ArrayBuffer): Promise<void> {
    this.finetunedRecords = [record, ...this.finetunedRecords.filter((r) => r.id !== record.id)];
    this.finetunedBytes.set(record.entry.onnxUrl, bytes);
    if (!this.entries.some((e) => e.id === record.entry.id)) this.entries = [...this.entries, record.entry];
  }

  async finetuned(): Promise<FinetunedRecord[]> {
    return this.finetunedRecords;
  }

  async removeFinetuned(id: string): Promise<void> {
    const record = this.finetunedRecords.find((r) => r.id === id);
    this.finetunedRecords = this.finetunedRecords.filter((r) => r.id !== id);
    if (record) this.finetunedBytes.delete(record.entry.onnxUrl);
    this.entries = this.entries.filter((e) => e.id !== id);
  }
}

export class FakeViewer implements ViewerController {
  layers: Record<string, LayerSpec> = {};
  removed: string[] = [];
  shaders: Record<string, { shader?: string; shaderControls?: Record<string, unknown> }> = {};
  openedWith: { url: string; raw: LayerSpec; info?: DatasetInfo }[] = [];
  setLayersCalls = 0;

  async restore(): Promise<void> {}

  async openDataset(url: string, raw: LayerSpec, info?: DatasetInfo): Promise<string> {
    this.openedWith.push({ url, raw, info });
    this.layers = { data: raw };
    return "http://localhost/ng/#!state";
  }

  /** What prepareDataset was given: the dataset the reload would show. */
  prepared: { url: string; raw: LayerSpec; info?: DatasetInfo }[] = [];

  prepareDataset(url: string, raw: LayerSpec, info?: DatasetInfo): string {
    this.prepared.push({ url, raw, info });
    return "http://localhost/ng/#!state";
  }

  /** What each setLayers call's `replacing` named, in order. */
  replacings: Record<string, string>[] = [];

  setLayers(layers: Record<string, LayerSpec>, replacing: Record<string, string> = {}): void {
    this.setLayersCalls++;
    this.replacings.push(replacing);
    for (const old of Object.values(replacing)) delete this.layers[old];
    Object.assign(this.layers, layers);
  }

  removeLayer(name: string): void {
    delete this.layers[name];
    this.removed.push(name);
  }

  /** The viewer's position in nm (z, y, x), or none; a test sets it (the AI-annotate run's default point). */
  positionNm: number[] | null = null;

  position(): number[] | null {
    return this.positionNm;
  }

  viewerUrl(): string | null {
    return "http://localhost/ng/#!state";
  }

  setEquivalences(_name: string, _sets: string[][]): void {}

  layerShader(name: string) {
    return this.shaders[name] ?? null;
  }

  /** Every selectLayer call. */
  selected: { name: string; tab?: string }[] = [];

  selectLayer(name: string, tab?: string): void {
    this.selected.push(tab === undefined ? { name } : { name, tab });
  }

  /** Every datasetOpened call: the URL, and the geometry or null. */
  datasetsOpened: { url: string; info: DatasetInfo | null }[] = [];

  datasetOpened(url: string, info: DatasetInfo | null): void {
    this.datasetsOpened.push({ url, info });
  }

  /** Where the pointer is, in nm (z, y, x); a test sets it. */
  pointer: number[] | null = null;

  pointerPosition(): number[] | null {
    return this.pointer;
  }

  /** The depth axis of the layout a test pretends the viewer shows. */
  depthAxis: 0 | 1 | 2 = 0;

  viewDepthAxis(): 0 | 1 | 2 {
    return this.depthAxis;
  }

  /** The display dimensions a test pretends the viewer shows. */
  displayDimensions: string[] = ["x", "y", "z"];

  viewDisplayDimensions(): string[] {
    return this.displayDimensions;
  }

  /** Every binding made and not yet unbound: a test presses a key by calling its handler. */
  bindings: { key: string; action: string; handler: () => void }[] = [];

  bindKey(key: string, action: string, handler: () => void): () => void {
    const binding = { key, action, handler };
    this.bindings.push(binding);
    return () => {
      this.bindings = this.bindings.filter((b) => b !== binding);
    };
  }

  /** Every refreshLayer call; a layer the fake has is found. */
  refreshed: string[] = [];

  refreshLayer(name: string): boolean {
    this.refreshed.push(name);
    return name in this.layers;
  }

  /** The layer as set, with its name; a test puts what the user drew in it. */
  layer(name: string): LayerSpec | null {
    return name in this.layers ? { ...this.layers[name], name } : null;
  }
}

export interface FakeContext extends AppContext {
  infer: FakeInference;
  catalog: FakeCatalog;
  viewer: FakeViewer;
  paint: MemoryPaintStore;
  logged: string[];
}

/** A context whose dataset is open (unless `dataset: null`) and whose session starts as `state`. */
export function fakeContext(options: { dataset?: DatasetInfo | null; state?: Partial<SessionState> } = {}): FakeContext {
  const logged: string[] = [];
  const ctx: FakeContext = {
    session: memorySession(options.state),
    catalog: new FakeCatalog(),
    infer: new FakeInference(options.dataset === undefined ? DATASET : options.dataset),
    viewer: new FakeViewer(),
    paint: new MemoryPaintStore(),
    events: new EventTarget(),
    log: (line) => void logged.push(line),
    logged,
  };
  return ctx;
}

/** An ApiRequest the way relay.ts builds one from the dashboard's fetch. */
export function request(method: string, path: string, body?: unknown, headers: Record<string, string> = {}) {
  const url = new URL(path, "http://localhost/");
  return { method, path: url.pathname, query: url.searchParams, body: body ?? null, headers };
}
