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
  GpuLimits,
  Inference,
  InferenceStats,
  LayerSpec,
  ModelEntry,
  ModelInfo,
  ModelManifest,
  ServedModel,
  SessionState,
  ViewerController,
} from "../../src/contracts";
import { entryFor } from "../../src/models";
import { memorySession } from "../../src/state";

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

  /** What displayRange() answers (and puts on the open dataset's info). */
  sampledRange: [number, number] | null = null;

  async displayRange(): Promise<[number, number] | null> {
    if (this.current && this.sampledRange) this.current.displayRange = this.sampledRange;
    return this.current ? this.sampledRange : null;
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

  async modelBytes(): Promise<ArrayBuffer> {
    return new ArrayBuffer(8);
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

  setLayers(layers: Record<string, LayerSpec>): void {
    this.setLayersCalls++;
    Object.assign(this.layers, layers);
  }

  removeLayer(name: string): void {
    delete this.layers[name];
    this.removed.push(name);
  }

  position(): number[] | null {
    return null;
  }

  viewerUrl(): string | null {
    return "http://localhost/ng/#!state";
  }

  setEquivalences(_name: string, _sets: string[][]): void {}

  layerShader(name: string) {
    return this.shaders[name] ?? null;
  }
}

export interface FakeContext extends AppContext {
  infer: FakeInference;
  catalog: FakeCatalog;
  viewer: FakeViewer;
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
