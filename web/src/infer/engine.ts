// The Inference of contracts.ts on chunkmirage's browser Engine: the raw
// volume opened as the engine view "raw", and one served view per model and
// chain whose produce() is produce.ts on an ORT session (session.ts). The
// engine brings the demand queue (finest level first, chunks the viewer gave
// up on dropped unrun), the 256 MB chunk cache, the zarr v3 metadata and the
// service worker relay; this file only has to describe each view's grid and
// compute its chunks.
import { Engine } from "../../chunkmirage/web/src/engine";
import type { ViewInfo } from "../../chunkmirage/web/src/types";
import type { Chain, DatasetInfo, GpuLimits, Inference, InferenceStats, ModelEntry, ServedModel, SessionStore } from "../contracts";
import { viewName } from "../shared/args";
import { modelInfo } from "../shared/model-info";
import { chooseRawLevel, datasetInfoFrom, effectiveOutputVoxelSize, geometryOf, outputLevel, outputShape } from "./geometry";
import { validateNormalizers } from "./norms";
import { buildPostprocessors, isSegmentation, outputChannels, outputDtype, validatePostprocessors } from "./post";
import { type ModelOutput, type ProduceContext, produceChunk } from "./produce";
import { Sessions } from "./session";

/** The engine view the raw volume is opened as, so Engine.read can read it. */
const RAW_VIEW = "raw";
/** Its chunk shape is metadata for a view we never serve (the viewer reads the
 * raw volume from its own URL); the Engine wants one all the same. */
const RAW_CHUNK = [64, 64, 64];

export class InferenceEngine implements Inference {
  readonly stats: InferenceStats = { provider: null, device: null, blocks: 0, inflight: 0, lastMs: null, avgMs: null, status: "idle" };
  private listeners = new Set<(stats: InferenceStats) => void>();
  private engine: Engine | null = null;
  private info: DatasetInfo | null = null;
  private servedList: ServedModel[] = [];
  private sessions: Sessions;
  private announcedFirst = false;
  private totalMs = 0;
  private lastFailure: string | null = null;

  /** `session` is the dashboard's session store: the engine keeps nothing in
   * it (what is served is re-served by the backend after a reload, through
   * serve()); it is here so the inference can read its settings later. */
  constructor(private opts: { session: SessionStore; log: (line: string) => void }) {
    this.sessions = new Sessions(opts.log, (line) => this.setStatus(line));
  }

  // ------------------------------------------------------------- the dataset

  async openDataset(url: string): Promise<DatasetInfo> {
    if (this.engine && this.info && this.info.url === url) return this.info;
    if (this.engine) {
      // Another URL. The Engine opens a source in start(), once, or in
      // add()/edit(), which also plan the view's ops in its Pyodide workers
      // and so wait for Python to load (tens of seconds on a first visit). A
      // fresh Engine opens the new source with its reader at once. The old
      // one keeps listening for its own page id and answers nothing of ours;
      // its workers live until the page unloads (there is no stop()), which
      // is rare: the dashboard reloads the page after a dataset change
      // anyway. The served models were on the old dataset's grid, so they
      // are dropped; the backend serves them again. (Upstream wish: a public
      // Engine.open(views), or a start() option without Pyodide.)
      this.opts.log(`the dataset changed to ${url}: starting a new engine for it`);
      for (const served of this.servedList.splice(0)) await served.dispose();
      this.engine = null;
      this.info = null;
    }
    // start() registers the service worker (the one relay.ts registered, by
    // the same name), opens the source with its reader worker, and starts the
    // Pyodide workers in the background. Our views never wait for Python:
    // answer() awaits `ready` only for a view without infos, and serve()
    // sets ours. (Upstream wish: a start() option that skips Pyodide.)
    const engine = new Engine(() => this.engineChanged());
    await engine.start({ [RAW_VIEW]: { source: url, chunk: RAW_CHUNK } }, (line) => this.setStatus(line));
    const source = engine.sources[RAW_VIEW];
    if (!source) throw new Error(`the engine opened ${url} but reports no source for it`);
    this.engine = engine;
    this.info = datasetInfoFrom(url, source);
    const finest = this.info.levels[0];
    this.opts.log(`opened ${url}: ${source.dtype}, ${source.levels.length} level(s), finest ${finest.shape.join("×")} voxels at ${finest.voxel.join("×")} nm`);
    this.setStatus("dataset open");
    return this.info;
  }

  dataset(): DatasetInfo | null { return this.info; }

  // ------------------------------------------------------------- the models

  async serve(entry: ModelEntry, chain: Chain, blob: string, modelBytes: () => Promise<ArrayBuffer>): Promise<ServedModel> {
    const engine = this.engine, dataset = this.info;
    if (!engine || !dataset) throw new Error("open a dataset before serving a model");
    // A chain step the browser cannot run is refused here, before the layer
    // exists, rather than failing on every chunk.
    validateNormalizers(chain.input_norm);
    validatePostprocessors(chain.postprocess);

    const geometry = geometryOf(entry);
    const rawLevel = chooseRawLevel(dataset.levels, geometry.input_voxel_size);
    const raw = dataset.levels[rawLevel];
    const level = outputLevel(raw, geometry);
    const info = modelInfo(entry);
    const effective = effectiveOutputVoxelSize(raw, geometry);
    if (effective) {
      // The level read is not at the model's voxel size: it is read as if it
      // were, and the viewer draws the output at its real size (layers.py).
      info.effective_output_voxel_size = effective;
      this.opts.log(`${entry.name}: no raw level at ${geometry.input_voxel_size.join("×")} nm; reading level ${rawLevel} (${raw.voxel.join("×")} nm) as if it were, so the output really lies at ${effective.join("×")} nm`);
    }
    const channels = outputChannels(chain.postprocess, info.output_channels);
    const dtype = outputDtype(chain.postprocess, "float32");
    const block = outputShape(geometry);
    const view = viewName(entry.name, blob);

    // One level at the output voxel size, as the Python server serves it. A
    // channel axis is declared through `added`: the engine puts it first in
    // the array and the chunk (see encode.ts for the lines), so the chunk's
    // bytes are the model's own [C, z, y, x]. `halo`, `lead` and `channels`
    // are read only for views the Pyodide workers compute.
    const viewInfo: ViewInfo = {
      dtype, out: dtype, channels: 1, lead: 0, halo: [0, 0, 0],
      kind: isSegmentation(chain.postprocess) ? "label" : "image",
      axes: dataset.axes.map((a) => ({ name: a.name, unit: a.unit })),
      levels: [level],
      ...(info.has_channel ? { added: [channels] } : {}),
    };
    const key = entry.onnxUrl;
    // The steps once for this layer, as the server builds its PostprocessOp
    // once per layer: the merger's equivalences live across chunks and reach
    // the viewer through the listeners.
    const postprocessors = buildPostprocessors(chain.postprocess, {
      onEquivalences: (sets) => { for (const listener of this.equivalenceListeners) listener(view, sets); },
    });
    const ctx: ProduceContext = {
      geometry, rawShape: raw.shape, rawDtype: dataset.dtype, chain, postprocessors, hasChannel: info.has_channel,
      read: (lo, hi) => engine.read(RAW_VIEW, rawLevel, lo, hi),
      run: (input, dims) => this.run(key, modelBytes, entry.name, input, dims),
    };
    engine.serve(view, viewInfo, block, (lvl, index) => this.produce(ctx, view, lvl, index));
    this.sessions.retain(key);

    const served: ServedModel = {
      entry, view, url: `zarr3://${engine.url(view)}`, info,
      dispose: async () => {
        const i = this.servedList.indexOf(served);
        if (i >= 0) this.servedList.splice(i, 1);
        // The engine has no unserve(): the view stays registered, but with
        // its layer gone from the viewer nothing asks for it any more.
        await this.sessions.release(key);
      },
    };
    this.servedList.push(served);
    this.opts.log(`serving ${entry.name} (${entry.variant.variant}): ${block.join("×")} voxels per block at ${level.voxel.join("×")} nm, ${channels} channel(s), ${dtype}, ${level.shape.join("×")} voxels in all`);
    return served;
  }

  served(): ServedModel[] { return [...this.servedList]; }

  private readonly equivalenceListeners = new Set<(view: string, sets: string[][]) => void>();

  onEquivalences(listener: (view: string, sets: string[][]) => void): () => void {
    this.equivalenceListeners.add(listener);
    return () => { this.equivalenceListeners.delete(listener); };
  }

  /** The engine's produce() for a model view: one output chunk. */
  private async produce(ctx: ProduceContext, view: string, level: number, index: number[]): Promise<ArrayBuffer> {
    if (level !== 0) throw new Error(`${view}: level ${level} is not served (a model view has one level)`);
    this.stats.inflight++;
    this.notify();
    try {
      return await produceChunk(ctx, index);
    } finally {
      this.stats.inflight--;
      this.notify();
    }
  }

  /** The forward pass for produce(): the model's session, built on the first
   * call from the catalog's bytes, run one pass at a time. */
  private async run(key: string, modelBytes: () => Promise<ArrayBuffer>, name: string, input: Float32Array, dims: number[]): Promise<ModelOutput> {
    const s = await this.sessions.get(key, modelBytes, name);
    if (!this.announcedFirst) {
      this.announcedFirst = true;
      this.opts.log("first inference: compiling GPU kernels, the first block takes a few seconds");
    }
    const { out, ms } = await this.sessions.run(s, input, dims);
    this.stats.blocks++;
    this.stats.lastMs = ms;
    this.totalMs += ms;
    this.stats.avgMs = this.totalMs / this.stats.blocks;
    this.stats.provider = this.sessions.provider;
    if (this.stats.blocks === 1 || this.stats.blocks % 10 === 0) {
      this.setStatus(`running · ${this.stats.blocks} block(s) done · last ${Math.round(ms)} ms`);
    }
    return out;
  }

  // ------------------------------------------------------------- the device

  async gpuLimits(): Promise<GpuLimits> {
    const limits = await this.sessions.gpuLimits();
    if (this.stats.device === null) {
      this.stats.device = await this.sessions.deviceName();
      this.notify();
    }
    return limits;
  }

  async reset(): Promise<void> {
    await this.sessions.reset();
    this.announcedFirst = false;
    this.setStatus("reset: sessions released; the next block builds them again");
    this.opts.log("inference reset");
  }

  // ------------------------------------------------------------- the stats

  onStats(listener: (stats: InferenceStats) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  private setStatus(line: string) {
    this.stats.status = line;
    this.notify();
  }

  private notify() {
    for (const listener of this.listeners) listener(this.stats);
  }

  /** The Engine's onChange: its counts moved, and maybe a chunk failed. */
  private engineChanged() {
    const failure = this.engine?.failure;
    if (failure && failure.message !== this.lastFailure) {
      this.lastFailure = failure.message;
      this.opts.log(`a chunk of ${failure.view.split("__CFLOW_ARGS__")[0]} failed: ${failure.message}`);
    }
    this.notify();
  }
}
