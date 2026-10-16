// Run on a region: the run itself. Each chosen model's view is written over
// its box (region.ts planModel) by Inference.materialize, which is
// chunkmirage's Engine.materialize: the engine walks the box's chunks
// through the stages, caches and demand queue the viewer's requests use, and
// writes the OME-Zarr into the store it is handed. This file is the order
// (one model after the other, so the GPU's turns go to one model's chunks at
// a time and a cancelled run leaves whole models behind it), the progress
// across the run, the stop, the provenance put in each zarr's group
// attributes, and what the log says was written where. The targets (a
// folder on this machine, or memory for a zip) are output.ts's, the DOM
// panel.ts's; one run at a time is the panel's rule.
import type { DatasetInfo, Inference, RegionStore, RegionWritten } from "../contracts";
import { formatBytes, formatDuration, type ModelPlan, type RegionNm } from "./region";

/** The key of cellmap-flow's own attributes in each zarr's group attributes
 * (`.zattrs`), beside the OME-Zarr `multiscales` the engine writes. */
export const PROVENANCE_KEY = "cellmap_flow";

/** What a zarr written by a run says about itself, under PROVENANCE_KEY:
 * the model (its name, catalog id, repository, folder, variant and file),
 * the chain it ran through, the dataset, the region asked for in nm and the
 * box written on the level in voxels, the level's voxel size, when, and the
 * page that ran it (the app has no version of its own; its address names
 * the deployment). The keys are snake_case, as an annotation volume's
 * record (PaintVolume) and the server's attributes are. */
export function provenance(plan: ModelPlan, dataset: Pick<DatasetInfo, "url">, region: RegionNm, now: Date, appUrl: string): Record<string, unknown> {
  const { entry } = plan.served;
  return {
    [PROVENANCE_KEY]: {
      model_name: entry.name,
      model_id: entry.id,
      repo: entry.repo,
      folder: entry.folder,
      variant: entry.variant.variant,
      model_file: entry.variant.file,
      input_norm: plan.chain.input_norm,
      postprocess: plan.chain.postprocess,
      dataset_path: dataset.url,
      region_offset_nm: [...region.offset],
      region_shape_nm: [...region.size],
      box_voxels: { lo: [...plan.lo], hi: [...plan.hi] },
      voxel_size_nm: [...plan.level.voxel],
      created_at: now.toISOString(),
      app_url: appUrl,
      engine: "browser",
    },
  };
}

/** Where a run writes: one store per model, and how the log names it. */
export interface RunTarget {
  /** Where model `name`'s zarr goes, in words ("Predictions/mito.zarr"). */
  where(name: string): string;
  /** The store model `name`'s zarr is written into. */
  store(name: string): RegionStore;
}

/** How far a run has got: the model being written (`model` of `models`,
 * from 1), its chunks done of its total, those written and the bytes, and
 * across the run the chunks done of all models', the time since the start
 * and a rough time left (null until a chunk is done). */
export interface RunProgress {
  model: number;
  models: number;
  name: string;
  done: number;
  total: number;
  written: number;
  bytes: number;
  runDone: number;
  runTotal: number;
  runBytes: number;
  elapsedMs: number;
  remainingMs: number | null;
}

/** One model's zarr: its plan, where it went, and what materialize did. */
export interface ModelResult {
  plan: ModelPlan;
  where: string;
  written: RegionWritten;
}

/** What a run did: the models written (the last one partly, when it was
 * cancelled or failed), whether it was cancelled, why it failed (null when
 * it did not), and how long it took. */
export interface RunOutcome {
  results: ModelResult[];
  aborted: boolean;
  error: string | null;
  elapsedMs: number;
}

export interface RunOptions {
  infer: Pick<Inference, "materialize">;
  plans: ModelPlan[];
  target: RunTarget;
  /** Each model's group attributes (provenance()). */
  attributes: (plan: ModelPlan) => Record<string, unknown>;
  /** The chunks' compression (RegionWrite.compression); gzip when absent. */
  compression?: "gzip" | null;
  /** Cancel: checked by the engine between chunks, and here between models. */
  signal: AbortSignal;
  onProgress?: (progress: RunProgress) => void;
  log: (line: string) => void;
  /** Milliseconds, for the times; performance.now() by default. */
  clock?: () => number;
}

/** "12 of 18 chunks (12 written, 3.1 MB)", one model's count. */
function counted(written: RegionWritten): string {
  const empty = written.skipped ? `, ${written.skipped.toLocaleString("en-US")} empty and not written` : "";
  return `${written.written.toLocaleString("en-US")} of ${written.chunks.toLocaleString("en-US")} chunks written (${formatBytes(written.bytes)}${empty})`;
}

/** Write each of `plans` in turn into its store, as above. A model that
 * fails stops the run (the next would most likely fail the same way, and
 * a message per model would bury the first); the zarrs written before it
 * stay, and so do its own chunks written so far. */
export async function runRegion(opts: RunOptions): Promise<RunOutcome> {
  const clock = opts.clock ?? (() => performance.now());
  const start = clock();
  const runTotal = opts.plans.reduce((n, p) => n + p.chunks, 0);
  const results: ModelResult[] = [];
  let before = 0, beforeBytes = 0;
  for (let i = 0; i < opts.plans.length; i++) {
    const plan = opts.plans[i];
    const where = opts.target.where(plan.name);
    if (opts.signal.aborted) break;
    opts.log(`${plan.name}: writing ${plan.chunks.toLocaleString("en-US")} chunk(s) of voxels ${plan.lo.join(",")}..${plan.hi.join(",")} to ${where}`);
    let last: RegionWritten = { chunks: plan.chunks, written: 0, skipped: 0, bytes: 0, aborted: false };
    try {
      const written = await opts.infer.materialize(plan.served.view, opts.target.store(plan.name), {
        lo: plan.lo, hi: plan.hi, compression: opts.compression === undefined ? "gzip" : opts.compression, attributes: opts.attributes(plan), signal: opts.signal,
        onProgress: (p) => {
          last = { chunks: p.total, written: p.written, skipped: p.done - p.written, bytes: p.bytes, aborted: false };
          const runDone = before + p.done;
          const elapsedMs = clock() - start;
          opts.onProgress?.({
            model: i + 1, models: opts.plans.length, name: plan.name, done: p.done, total: p.total, written: p.written, bytes: p.bytes,
            runDone, runTotal, runBytes: beforeBytes + p.bytes, elapsedMs,
            remainingMs: runDone > 0 ? (elapsedMs / runDone) * Math.max(0, runTotal - runDone) : null,
          });
        },
      });
      results.push({ plan, where, written });
      if (written.aborted || opts.signal.aborted) {
        opts.log(`${plan.name}: cancelled, ${counted(written)} to ${where}`);
        return { results, aborted: true, error: null, elapsedMs: clock() - start };
      }
      opts.log(`${plan.name}: done, ${counted(written)} to ${where}`);
      before += plan.chunks;
      beforeBytes += written.bytes;
    } catch (e) {
      // A cancel may come back as the engine's AbortError rather than an
      // answer with `aborted`: either way the run was cancelled, and what
      // was written of the model is what the last progress said.
      results.push({ plan, where, written: { ...last, aborted: opts.signal.aborted } });
      if (opts.signal.aborted) {
        opts.log(`${plan.name}: cancelled, ${counted(last)} to ${where}`);
        return { results, aborted: true, error: null, elapsedMs: clock() - start };
      }
      const error = `${plan.name}: ${e instanceof Error ? e.message : String(e)}`;
      opts.log(`${plan.name}: failed after ${counted(last)} to ${where}: ${e instanceof Error ? e.message : String(e)}`);
      return { results, aborted: false, error, elapsedMs: clock() - start };
    }
  }
  const elapsedMs = clock() - start;
  if (!opts.signal.aborted) opts.log(`run done in ${formatDuration(elapsedMs)}: ${results.length} model(s)`);
  return { results, aborted: opts.signal.aborted, error: null, elapsedMs };
}

/** One line of progress for the panel: the model, its chunks, the run's
 * share, the megabytes, the time so far and the time left. */
export function describeProgress(p: RunProgress): string {
  const model = p.models > 1 ? `${p.name} (${p.model} of ${p.models})` : p.name;
  const left = p.remainingMs === null ? "" : `, about ${formatDuration(p.remainingMs)} left`;
  return `${model}: ${p.done.toLocaleString("en-US")} of ${p.total.toLocaleString("en-US")} chunks, ${p.written.toLocaleString("en-US")} written, ${formatBytes(p.runBytes)} in all; ${formatDuration(p.elapsedMs)} so far${left}`;
}
