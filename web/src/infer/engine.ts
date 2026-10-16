// The Inference of contracts.ts on chunkmirage's browser Engine: the raw
// volume opened as the engine view "raw", and one staged view per model and
// chain (Engine.serveStages), its stages the model and the postprocessors
// (stages.ts), the model run by chunkmirage's Onnx (onnxruntime-web on
// WebGPU, one pass at a time per session, WASM when the browser has no GPU).
// The engine brings the reader, the input box of each chunk with its padding,
// the demand queue (finest level first, chunks the viewer gave up on dropped
// unrun), the per-stage cache and the 256 MB chunk cache, the zarr v3
// metadata and the service worker relay; this file describes each view's
// grid, builds its stages and keeps the stats.
//
// A model's own output (readOutput, for the Finetune tab's Seed from
// Prediction) is a second staged view beside the layer's, the model stage
// alone, served under `<view>~output` once first asked for. It is the same
// stage object, so its outputs are the ones kept under the stage's key: a
// chunk the viewer had computed is read from the engine's staged cache, not
// run again. Its chunks are read as any client reads a view, over its URL
// through the service worker (the engine answers a page's requests by the
// route it registers there and has no other door to a chunk); they are
// whole chunks of uncompressed little-endian float32, channels first, as the
// view's zarr.json declares them.
import { Engine } from "../../chunkmirage/web/src/engine";
import { Onnx, type OnnxModel } from "../../chunkmirage/web/src/onnx";
import type { Stage } from "../../chunkmirage/web/src/stages";
import type { ViewInfo } from "../../chunkmirage/web/src/types";
import type { Chain, DatasetInfo, GpuLimits, Inference, InferenceStats, ModelEntry, ServedModel, SessionStore } from "../contracts";
import { viewName } from "../shared/args";
import { modelInfo } from "../shared/model-info";
import { chooseRawLevel, datasetInfoFrom, effectiveOutputVoxelSize, geometryOf, outputLevel, outputShape } from "./geometry";
import { validateNormalizers } from "./norms";
import { boxFromChunks } from "./output";
import { buildPostprocessors, isSegmentation, outputChannels, outputDtype, validatePostprocessors } from "./post";
import { type ModelOutput, modelStage, modelStageKey, postprocessStage, postprocessStageKey } from "./stages";

/** What runs the models: chunkmirage's Onnx, or anything of its shape (a
 * test's fake): sessions built on first use and shared by the views using
 * them, one forward pass at a time per session. */
export type ModelRunner = Pick<Onnx, "run" | "retain" | "release" | "reset" | "limits" | "device" | "provider">;

/** The engine view the raw volume is opened as: what the model stages read. */
const RAW_VIEW = "raw";
/** Its chunk shape is metadata for a view we never serve (the viewer reads the
 * raw volume from its own URL); the Engine wants one all the same. */
const RAW_CHUNK = [64, 64, 64];

/** The suffix of the view a served model's own output is read from (readOutput). */
const OUTPUT_SUFFIX = "~output";

/** A served model's own output, as the view readOutput reads: its name,
 * its ViewInfo and chunk (the layer's, in float32 with the model's
 * channels), the model stage, the level's shape, and whether the engine
 * serves it yet. */
interface OutputView {
  name: string;
  info: ViewInfo;
  chunk: number[];
  stage: Stage;
  channels: number;
  hasChannel: boolean;
  levelShape: number[];
  registered: boolean;
}

export class InferenceEngine implements Inference {
  readonly stats: InferenceStats = { provider: null, device: null, blocks: 0, inflight: 0, lastMs: null, avgMs: null, status: "idle" };
  private listeners = new Set<(stats: InferenceStats) => void>();
  private engine: Engine | null = null;
  private info: DatasetInfo | null = null;
  private servedList: ServedModel[] = [];
  /** The output view of each served model, by the served view's name. */
  private outputs = new Map<string, OutputView>();
  private runner: ModelRunner;
  private announcedFirst = false;
  private totalMs = 0;
  private lastFailure: string | null = null;

  /** `session` is the dashboard's session store: the engine keeps nothing in
   * it (what is served is re-served by the backend after a reload, through
   * serve()); it is here so the inference can read its settings later.
   * `runner` runs the models: chunkmirage's Onnx by default, what it says
   * about a session (built on WebGPU, or on WASM and why) in the log and on
   * the dashboard's status line. ORT's WebAssembly runtime comes from ./ort/
   * beside the page, where chunkmirage's Vite plugin puts it (vite.config.ts). */
  constructor(private opts: { session: SessionStore; log: (line: string) => void; runner?: ModelRunner }) {
    this.runner = opts.runner ?? new Onnx({ log: (line) => { opts.log(line); this.setStatus(line); } });
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
    // The model under a key that is its identity (one session per model
    // file, shared by the layers using it), its bytes from the catalog (so
    // its IndexedDB cache is used) when the first chunk needs them.
    const name = `${entry.id}/${entry.variant.file}`;
    const model: OnnxModel = {
      key: name,
      bytes: async () => {
        this.setStatus(`loading ${name}…`);
        const bytes = await modelBytes();
        this.setStatus(`${name}: building the session…`);
        return bytes;
      },
      input: entry.manifest.io?.input, output: entry.manifest.io?.output,
    };
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
    // The model's own output, for readOutput: served only once asked for.
    const output: OutputView = {
      name: `${view}${OUTPUT_SUFFIX}`,
      info: { ...viewInfo, dtype: "float32", out: "float32", kind: "image", added: info.has_channel ? [info.output_channels] : [] },
      chunk: block, stage: stages[0], channels: info.output_channels, hasChannel: info.has_channel, levelShape: [...level.shape], registered: false,
    };
    this.outputs.set(view, output);

    const served: ServedModel = {
      entry, view, url: `zarr3://${engine.url(view)}`, info,
      dispose: async () => {
        const i = this.servedList.indexOf(served);
        if (i >= 0) this.servedList.splice(i, 1);
        // The view's chunks go (viewers cache by URL, and an edit is a new
        // name); its stages' outputs stay in the engine's staged cache for a
        // view served again with a later stage changed.
        engine.unserve(view);
        if (this.outputs.get(view) === output) {
          if (output.registered) engine.unserve(output.name);
          this.outputs.delete(view);
        }
        await this.runner.release(model.key);
      },
    };
    this.servedList.push(served);
    this.opts.log(`serving ${entry.name} (${entry.variant.variant}): ${block.join("×")} voxels per block at ${level.voxel.join("×")} ${viewInfo.axes[0]?.unit || "nm"}, ${channels} channel(s), ${dtype}, ${level.shape.join("×")} voxels in all`);
    return served;
  }

  served(): ServedModel[] { return [...this.servedList]; }

  /** Voxels [lo, hi) of raw level `level`, as the engine's reader reads them
   * for a stage's input (Engine.read on the raw view): what the finetune's
   * samples are cut from. The box must lie within the level. */
  readRaw(level: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
    if (!this.engine || !this.info) return Promise.reject(new Error("open a dataset before reading it"));
    if (level < 0 || level >= this.info.levels.length) return Promise.reject(new Error(`the dataset has no level ${level}`));
    return this.engine.read(RAW_VIEW, level, lo, hi);
  }

  /** The served model `view`'s own output over voxels [lo, hi) of its level
   * (contracts.ts), from its output view (see the top of this file): the
   * chunks the box touches fetched at once, each placed in the box
   * (output.ts). The engine's demand queue runs the ones not yet computed. */
  async readOutput(view: string, lo: number[], hi: number[]): Promise<{ data: Float32Array; shape: number[] }> {
    const engine = this.engine, output = this.outputs.get(view);
    if (!engine || !output) throw new Error(`${view.split("__CFLOW_ARGS__")[0]} is not served`);
    if (lo.length !== 3 || hi.length !== 3 || lo.some((l, a) => !Number.isInteger(l) || l < 0 || hi[a] > output.levelShape[a] || hi[a] <= l)) {
      throw new Error(`the box ${lo.join(",")}..${hi.join(",")} is not within the model's ${output.levelShape.join("×")} output voxels`);
    }
    if (!output.registered) {
      engine.serveStages(output.name, output.info, output.chunk, RAW_VIEW, [output.stage], "zero");
      output.registered = true;
    }
    const base = engine.url(output.name);
    return boxFromChunks(lo, hi, output.chunk, output.channels, async (index) => {
      // Level 0, the default chunk key encoding: c/<channel chunk>/z/y/x.
      const response = await fetch(`${base}0/c/${output.hasChannel ? "0/" : ""}${index.join("/")}`);
      if (!response.ok) throw new Error(`chunk ${index.join(",")} of the model's output: HTTP ${response.status} ${(await response.text()).slice(0, 200)}`);
      return new Float32Array(await response.arrayBuffer());
    });
  }

  private readonly equivalenceListeners = new Set<(view: string, sets: string[][]) => void>();

  onEquivalences(listener: (view: string, sets: string[][]) => void): () => void {
    this.equivalenceListeners.add(listener);
    return () => { this.equivalenceListeners.delete(listener); };
  }

  /** The forward pass for the model stage: the runner's, with the stats
   * kept here (blocks, the pass's time, the provider). */
  private async run(model: OnnxModel, data: Float32Array, dims: number[]): Promise<ModelOutput> {
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
