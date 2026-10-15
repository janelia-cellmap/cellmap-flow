// The Inference of contracts.ts on chunkmirage's browser Engine: the raw
// volume opened as the engine view "raw", and one staged view per model and
// chain (Engine.serveStages), its stages the model and the postprocessors
// (stages.ts) on an ONNX runner (session.ts, or chunkmirage's onnx.ts). The
// engine brings the reader, the input box of each chunk with its padding,
// the demand queue (finest level first, chunks the viewer gave up on dropped
// unrun), the per-stage cache and the 256 MB chunk cache, the zarr v3
// metadata and the service worker relay; this file describes each view's
// grid, builds its stages and keeps the stats.
import { TYPED as ENGINE_TYPED } from "../../chunkmirage/web/src/asread";
import { Engine } from "../../chunkmirage/web/src/engine";
import type { ViewInfo } from "../../chunkmirage/web/src/types";
import type { Chain, DatasetInfo, GpuLimits, Inference, InferenceStats, ModelEntry, ServedModel, SessionStore } from "../contracts";
import { viewName } from "../shared/args";
import { modelInfo } from "../shared/model-info";
import { chooseRawLevel, datasetInfoFrom, effectiveOutputVoxelSize, geometryOf, outputLevel, outputShape } from "./geometry";
import { validateNormalizers } from "./norms";
import { buildPostprocessors, isSegmentation, outputChannels, outputDtype, validatePostprocessors } from "./post";
import { type ModelRunner, type RunnerModel, Sessions } from "./session";
import { type ModelOutput, modelStage, modelStageKey, postprocessStage, postprocessStageKey } from "./stages";

/** The engine view the raw volume is opened as: what the model stages read. */
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
  private runner: ModelRunner;
  private announcedFirst = false;
  private totalMs = 0;
  private lastFailure: string | null = null;

  /** `session` is the dashboard's session store: the engine keeps nothing in
   * it (what is served is re-served by the backend after a reload, through
   * serve()); it is here so the inference can read its settings later.
   * `runner` runs the models; the browser mode's own sessions by default
   * (chunkmirage's `new Onnx({ log })` fits the same interface). */
  constructor(private opts: { session: SessionStore; log: (line: string) => void; runner?: ModelRunner }) {
    this.runner = opts.runner ?? new Sessions(opts.log, (line) => this.setStatus(line));
  }

  // ------------------------------------------------------------- the dataset

  async openDataset(url: string): Promise<DatasetInfo> {
    if (this.engine && this.info && this.info.url === url) return this.info;
    if (this.engine) {
      // Another URL: the served models were on the old dataset's grid, so
      // they go (the backend serves them again), and so does the engine, its
      // reader and its caches with it. A fresh one opens the new source and
      // serves under a new page id, so a viewer that cached chunks by URL
      // fetches afresh.
      this.opts.log(`the dataset changed to ${url}: starting a new engine for it`);
      for (const served of this.servedList.splice(0)) await served.dispose();
      this.engine.stop();
      this.engine = null;
      this.info = null;
    }
    // start() registers the service worker (the one relay.ts registered, by
    // the same name) and opens the source with its reader worker. The raw
    // view has no ops and our views are computed here, so no Pyodide starts.
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
    if (dataset.channels > 1) throw new Error(`${dataset.url} has ${dataset.channels} channels; the browser mode reads single-channel volumes (it does not select a channel yet)`);

    const geometry = geometryOf(entry);
    const rawLevel = chooseRawLevel(dataset.levels, geometry.input_voxel_size);
    const raw = dataset.levels[rawLevel];  // in nm, for the geometry
    const sourceRaw = engine.sources[RAW_VIEW].levels[rawLevel];  // as the engine declares it, for the view
    const info = modelInfo(entry);
    const effective = effectiveOutputVoxelSize(raw, geometry);
    if (effective) {
      // The level read is not at the model's voxel size: it is read as if it
      // were, and the output lies at its real size (outputLevel declares it
      // there; the viewer's transform from this field changes nothing then).
      info.effective_output_voxel_size = effective;
      this.opts.log(`${entry.name}: no raw level at ${geometry.input_voxel_size.join("×")} nm; reading level ${rawLevel} (${raw.voxel.join("×")} nm) as if it were, so the output really lies at ${effective.join("×")} nm`);
    }
    const channels = outputChannels(chain.postprocess, info.output_channels);
    const dtype = outputDtype(chain.postprocess, "float32");
    if (!(dtype in ENGINE_TYPED)) {
      // chunkmirage's stages know the typed arrays of asread.ts's TYPED, which
      // stop at 32-bit integers: a stage declaring uint64 (the segmentation
      // steps' ids) fails its `instanceof` check there, and a chunk at the
      // volume's edge would be laid out with the wrong element size. Until
      // chunkmirage adds BigUint64Array to its Numbers and TYPED, such a
      // chain is refused here, before the layer exists.
      throw new Error(`${entry.name}: the chain's output is ${dtype}, which chunkmirage's browser stages cannot serve yet (their typed arrays stop at 32-bit integers); the segmentation steps wait on that`);
    }
    const block = outputShape(geometry);
    const level = outputLevel(sourceRaw, geometry);
    const view = viewName(entry.name, blob);

    // One level at the output voxel size, as the Python server serves it, in
    // the source's units and axes (the engine reads the raw level at the
    // voxel size the source declares and grows the chunk by the ratio of the
    // two grids, so both must be in the same units). A channel axis is
    // declared through `added`: the engine puts it first in the array and the
    // chunk, so a chunk's bytes are the model's own [C, z, y, x]. `out` and
    // `added` are the last stage's (serveStages sets them); `halo`, `lead`
    // and `channels` are read only for views the Pyodide workers compute.
    const viewInfo: ViewInfo = {
      dtype, out: dtype, channels: 1, lead: 0, halo: [0, 0, 0],
      kind: isSegmentation(chain.postprocess) ? "label" : "image",
      axes: engine.sources[RAW_VIEW].axes.map((a) => ({ name: a.name, unit: a.unit })),
      levels: [level],
      added: info.has_channel ? [channels] : [],
    };
    const model: RunnerModel = { key: entry.onnxUrl, bytes: modelBytes, input: entry.manifest.io?.input, output: entry.manifest.io?.output };
    const modelKey = modelStageKey(entry, chain);
    // The steps once for this layer, as the server builds its PostprocessOp
    // once per layer: the merger's equivalences live across chunks and reach
    // the viewer through the listeners.
    const postprocessors = buildPostprocessors(chain.postprocess, {
      onEquivalences: (sets) => { for (const listener of this.equivalenceListeners) listener(view, sets); },
    });
    const stages = [
      modelStage({
        key: modelKey, geometry, rawShape: [...sourceRaw.shape], inputVoxelSize: [...sourceRaw.voxel], normalizers: chain.input_norm,
        channels: info.output_channels, hasChannel: info.has_channel,
        run: (data, dims) => this.run(model, data, dims),
      }),
      postprocessStage({
        key: postprocessStageKey(modelKey, chain), geometry, steps: chain.postprocess, postprocessors,
        modelChannels: info.output_channels, hasChannel: info.has_channel,
      }),
    ];
    // Zero past the volume, as the Python server's Pipeline(padding="zero").
    engine.serveStages(view, viewInfo, block, RAW_VIEW, stages, "zero");
    this.runner.retain(model.key);

    const served: ServedModel = {
      entry, view, url: `zarr3://${engine.url(view)}`, info,
      dispose: async () => {
        const i = this.servedList.indexOf(served);
        if (i >= 0) this.servedList.splice(i, 1);
        // The view's chunks go (viewers cache by URL, and an edit is a new
        // name); its stages' outputs stay in the engine's staged cache for a
        // view served again with a later stage changed.
        engine.unserve(view);
        await this.runner.release(model.key);
      },
    };
    this.servedList.push(served);
    this.opts.log(`serving ${entry.name} (${entry.variant.variant}): ${block.join("×")} voxels per block at ${level.voxel.join("×")} ${viewInfo.axes[0]?.unit || "nm"}, ${channels} channel(s), ${dtype}, ${level.shape.join("×")} voxels in all`);
    return served;
  }

  served(): ServedModel[] { return [...this.servedList]; }

  private readonly equivalenceListeners = new Set<(view: string, sets: string[][]) => void>();

  onEquivalences(listener: (view: string, sets: string[][]) => void): () => void {
    this.equivalenceListeners.add(listener);
    return () => { this.equivalenceListeners.delete(listener); };
  }

  /** The forward pass for the model stage: the runner's, with the stats
   * kept here (blocks, the pass's time, the provider). */
  private async run(model: RunnerModel, data: Float32Array, dims: number[]): Promise<ModelOutput> {
    if (!this.announcedFirst) {
      this.announcedFirst = true;
      this.opts.log("first inference: compiling GPU kernels, the first block takes a few seconds");
    }
    this.stats.inflight++;
    this.notify();
    try {
      const out = await this.runner.run(model, data, dims);
      this.stats.blocks++;
      this.stats.lastMs = out.ms;
      this.totalMs += out.ms;
      this.stats.avgMs = this.totalMs / this.stats.blocks;
      this.stats.provider = this.runner.provider;
      if (this.stats.blocks === 1 || this.stats.blocks % 10 === 0) {
        this.setStatus(`running · ${this.stats.blocks} block(s) done · last ${Math.round(out.ms)} ms`);
      }
      return out;
    } finally {
      this.stats.inflight--;
      this.notify();
    }
  }

  // ------------------------------------------------------------- the device

  async gpuLimits(): Promise<GpuLimits> {
    const limits = await this.runner.limits();
    if (this.stats.device === null) {
      this.stats.device = await this.runner.device();
      this.notify();
    }
    return limits;
  }

  async reset(): Promise<void> {
    await this.runner.reset();
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
