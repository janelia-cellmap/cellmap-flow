// The finetune jobs of the page: what cellmap_flow/finetune/job_manager
// (manager.py's FinetuneJobManager, state.py's FinetuneJob and JobStatus) is
// on the server, for runs that happen in this tab. A job is one submission
// of the Finetune tab's Start Finetuning: it runs the model's head training,
// or its whole network's (src/finetune/run.ts and full/run-full.ts, through
// the `run` seam, as the options' mode says) on the paint of its volume,
// keeps the result as a finetuned model in the catalog, and, with
// Auto-serve, serves it beside the model it came from; then it waits,
// as the server's trainer does, for a Restart, which runs again from the
// trained head (or network) with the form's new parameters under the same
// job id. A job keeps its mode: switching needs a new job, as switching
// between LoRA and a full finetune does on the server.
//
// The routes (routes/finetune.ts) read a job as the Python routes read
// theirs: `toDict()` is state.py's to_dict, `statusRecord()` adds the
// `loss`, `loss_epoch` and `progress_percent` of manager.get_job_status,
// and the log is kept as lines with the byte offset of the log after each,
// which is what the log stream's event ids are (sse.ts). The epoch counter
// and the loss are parsed from the run's lines exactly as monitor.py's
// _parse_training_progress parses the trainer's, so a run that logs
// "Epoch 3/20 - Loss: 0.0113" is followed by the dashboard's card and plot
// as a cluster job is.
//
// Statuses (state.py's JobStatus): PENDING at submit, RUNNING while the run
// goes, then COMPLETED (kept, not served), WAITING_FOR_RESTART (kept and
// served; restartable), FAILED or CANCELLED. The last three but one are
// final (TERMINAL_STATUSES): nothing moves a job again once it has one.
import type { AppContext, ModelEntry, PaintVolume } from "../contracts";
import { trainingEntry } from "../finetune/full/run-full";
import { loadGoodRegions } from "../finetune/good-regions";
import { MODE_NOUN, MODE_WORDS } from "../finetune/modes";
import type { Finetuned, FinetuneOptions, RunFinetune, RunReporter } from "../finetune/types";
import { PYTHON, pyRegExp } from "../generated/python-data";
import { finetunedRecord, nextFinetunedName } from "../models/finetuned";
import { stamp } from "../shared/format";
import { keepInTab } from "../state";
import { message } from "./router";
import { serveModels } from "./serving";

export type JobStatus = "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED" | "WAITING_FOR_RESTART";

/** state.py's TERMINAL_STATUSES: once a job has one of these, neither a run nor a request moves it. */
export const TERMINAL_STATUSES: ReadonlySet<JobStatus> = new Set(PYTHON.finetune.terminal_statuses as JobStatus[]);

/** Where a job's output goes, where the server's jobs have a run directory:
 * the finetuned model is a catalog entry in this browser's IndexedDB. */
export const OUTPUT_DIR = "this browser (IndexedDB)";

/** A job as state.py's FinetuneJob.to_dict gives it. The fields of a
 * cluster job that the browser has no equivalent of are null, in place, so
 * the dashboard's JavaScript reads the same keys. */
export interface JobRecord {
  job_id: string;
  lsf_job_id: null;
  model_name: string;
  output_dir: string;
  params: Record<string, unknown>;
  status: JobStatus;
  created_at: string;
  log_file: null;
  finetuned_model_name: string | null;
  model_yaml_path: null;
  current_epoch: number;
  total_epochs: number;
  latest_loss: number | null;
  latest_loss_epoch: number | null;
  inference_server_url: null;
  inference_server_ready: boolean;
  /** The annotation volume's id, where the server's is its corrections directory. */
  corrections_path: string;
}

/** What manager.get_job_status answers (and list_jobs, per job): to_dict
 * with latest_loss and latest_loss_epoch renamed loss and loss_epoch, which
 * job-card.js reads, and progress_percent. */
export interface JobStatusRecord extends Omit<JobRecord, "latest_loss" | "latest_loss_epoch"> {
  loss: number | null;
  loss_epoch: number | null;
  progress_percent: number;
}

/** What a job runs on: the context, the model (the served entry, whose
 * variant is the one the layer shows), the volume, the training options
 * as run.ts takes them and the parameters as the tab sent them (the
 * record's `params`), and whether to serve the result. */
export interface SubmitArgs {
  ctx: AppContext;
  entry: ModelEntry;
  volume: PaintVolume;
  options: FinetuneOptions;
  params: Record<string, unknown>;
  autoServe: boolean;
}

export interface JobsDeps {
  run: RunFinetune;
  /** The clock, for job ids and created_at (tests fix it). */
  now?: () => Date;
}

/** Whether `job` can take a restart (state.py's can_restart): one waiting
 * for one, or a running one whose model is served. */
export function canRestart(job: FinetuneJob): boolean {
  return job.status === "WAITING_FOR_RESTART" || (job.status === "RUNNING" && job.inferenceServerReady);
}

// The trainer's lines the monitor reads the progress from (markers.py's
// EPOCH_START_RE and EPOCH_SUMMARY_RE, exported): the start of an epoch
// moves the counter on as soon as it begins; the summary at its end carries
// the loss, which is that epoch's.
const EPOCH_START = pyRegExp(PYTHON.finetune.epoch_start);
const EPOCH_SUMMARY = pyRegExp(PYTHON.finetune.epoch_summary);

/** The marker job-card.js resets the card on, printed as the first line of a
 * restarted run (markers.py's RESTARTING_TRAINING). */
export const RESTARTING_MARKER = PYTHON.finetune.restarting_marker;

export class FinetuneJob {
  status: JobStatus = "PENDING";
  currentEpoch = 0;
  totalEpochs: number;
  latestLoss: number | null = null;
  latestLossEpoch: number | null = null;
  inferenceServerReady = false;
  finetunedModelName: string | null = null;
  /** The log's lines, and the UTF-8 byte count of the log through each line
   * and its newline: the event id a line goes out with (sse.ts logEvents's
   * definition), and what GET …/logs answers as `offset`. */
  readonly lines: string[] = [];
  readonly offsets: number[] = [];
  /** The finetuned entry the session serves for this job, which the next run's replaces. */
  servedEntryId: string | null = null;
  cancelRequested = false;
  /** The finished epoch after which Stop Early ends the run, or null when none was asked for. */
  stopAfterEpoch: number | null = null;
  private waiters: (() => void)[] = [];
  private current: Promise<void> = Promise.resolve();

  constructor(
    readonly id: string,
    readonly createdAt: string,
    /** What the job runs on; a restart replaces the options and params. */
    public args: SubmitArgs,
  ) {
    this.totalEpochs = args.options.epochs;
  }

  get modelName(): string {
    return this.args.entry.name;
  }

  get params(): Record<string, unknown> {
    return this.args.params;
  }

  isFinal(): boolean {
    return TERMINAL_STATUSES.has(this.status);
  }

  toDict(): JobRecord {
    return {
      job_id: this.id,
      lsf_job_id: null,
      model_name: this.modelName,
      output_dir: OUTPUT_DIR,
      params: this.params,
      status: this.status,
      created_at: this.createdAt,
      log_file: null,
      finetuned_model_name: this.finetunedModelName,
      model_yaml_path: null,
      current_epoch: this.currentEpoch,
      total_epochs: this.totalEpochs,
      latest_loss: this.latestLoss,
      latest_loss_epoch: this.latestLossEpoch,
      inference_server_url: null,
      inference_server_ready: this.inferenceServerReady,
      corrections_path: this.args.volume.id,
    };
  }

  statusRecord(): JobStatusRecord {
    const { latest_loss, latest_loss_epoch, ...rest } = this.toDict();
    return {
      ...rest,
      loss: latest_loss,
      loss_epoch: latest_loss_epoch,
      progress_percent: this.totalEpochs > 0 ? (this.currentEpoch / this.totalEpochs) * 100 : 0,
    };
  }

  /** The whole log, each line with its newline. */
  logText(): string {
    return this.lines.map((line) => `${line}\n`).join("");
  }

  /** The byte count of the log so far: where a live stream opened after GET …/logs starts. */
  logOffset(): number {
    return this.offsets.length ? this.offsets[this.offsets.length - 1] : 0;
  }

  /** Append `text` to the log (several lines when it holds newlines), read
   * the progress off it, and wake whoever waits on the job. */
  log(text: string): void {
    const encoder = new TextEncoder();
    for (const line of text.split("\n")) {
      this.lines.push(line);
      this.offsets.push(this.logOffset() + encoder.encode(line).byteLength + 1);
      this.parse(line);
    }
    this.notify();
  }

  private parse(line: string): void {
    const start = EPOCH_START.exec(line);
    if (start) {
      this.currentEpoch = Number(start[1]);
      this.totalEpochs = Number(start[2]);
    }
    const summary = EPOCH_SUMMARY.exec(line);
    if (summary) {
      const epoch = Number(summary[1]);
      this.currentEpoch = Math.max(this.currentEpoch, epoch);
      this.totalEpochs = Number(summary[2]);
      const loss = Number(summary[3]);
      if (Number.isFinite(loss)) {
        this.latestLoss = loss;
        this.latestLossEpoch = epoch;
      }
    }
  }

  setStatus(status: JobStatus): void {
    this.status = status;
    this.notify();
  }

  /** state.py's start_iteration: a new run's progress from zero and its
   * model not served again until it completes; the new run's epoch count. */
  startIteration(totalEpochs: number): void {
    this.currentEpoch = 0;
    this.latestLoss = null;
    this.latestLossEpoch = null;
    this.totalEpochs = totalEpochs;
    this.inferenceServerReady = false;
    this.stopAfterEpoch = null;
    this.setStatus("RUNNING");
  }

  /** How the run speaks to this job: its lines into the log; cancelled once
   * Cancel was asked for, or Stop Early and the epoch it was to end after
   * has finished (its summary line seen), so that the run stops between
   * steps with that epoch's head. */
  reporter(): RunReporter {
    return {
      log: (line) => this.log(line),
      cancelled: () => this.cancelRequested || (this.stopAfterEpoch !== null && (this.latestLossEpoch ?? 0) >= this.stopAfterEpoch),
    };
  }

  /** Resolves on the next change: a line logged, or the status moved. */
  changed(): Promise<void> {
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  private notify(): void {
    const waiting = this.waiters;
    this.waiters = [];
    for (const wake of waiting) wake();
  }

  /** Make `run` the job's current run (submit, restart). */
  start(run: Promise<void>): void {
    this.current = run;
  }

  /** The end of the current run (for whoever must wait for it, such as a test). */
  settled(): Promise<void> {
    return this.current;
  }
}

/** A refused request, with the HTTP status the Python route gives it. */
export interface Refusal {
  ok: false;
  status: number;
  error: string;
}

export class FinetuneJobs {
  private readonly jobs = new Map<string, FinetuneJob>();
  private readonly now: () => Date;

  constructor(private readonly deps: JobsDeps) {
    this.now = deps.now ?? (() => new Date());
  }

  /** Every job, oldest first (the order the Python list_jobs has them in,
   * which job-monitor.js sorts by created_at itself). */
  list(): FinetuneJob[] {
    return [...this.jobs.values()];
  }

  get(id: string): FinetuneJob | undefined {
    return this.jobs.get(id);
  }

  /** Start a job: it runs at once (there is no queue in the browser), and
   * this returns as soon as it has begun, with the job to follow. */
  submit(args: SubmitArgs): FinetuneJob {
    const now = this.now();
    const job = new FinetuneJob(`ft-${this.jobs.size + 1}-${timestamp(now)}`, now.toISOString(), args);
    this.jobs.set(job.id, job);
    job.log(`Job ${job.id}: ${args.entry.name}${args.options.mode === "full" ? "'s whole network" : ""} on the paint of ${args.volume.id}, ${args.options.epochs} epoch(s) in this browser${args.autoServe ? ", served when done" : ""}`);
    job.start(this.run(job));
    return job;
  }

  /** manager.cancel_job: the job is CANCELLED now; its run stops at its next
   * step (RunReporter.cancelled) and whatever it was making is not kept. A
   * job waiting for a restart has nothing running: its model stays kept and
   * served, since those are the catalog's and the session's, not the job's. */
  cancel(id: string): { ok: true } | Refusal {
    const job = this.jobs.get(id);
    if (!job) return { ok: false, status: 404, error: `Job ${id} not found` };
    if (job.isFinal()) return { ok: false, status: 400, error: `Job ${id} already finished with status ${job.status}` };
    job.cancelRequested = true;
    job.log(
      job.inferenceServerReady
        ? `Cancelled by the user. ${job.finetunedModelName} stays kept and served (the Models tab removes it).`
        : "Cancelled by the user: the run stops at its next step, and nothing of it is kept.",
    );
    job.setStatus("CANCELLED");
    return { ok: true };
  }

  /** Stop Early (the route's stop_signal.json on the server): the run ends
   * after the epoch now running, and its head is kept (and served, with
   * Auto-serve) as a completed run's would be. Only while a run trains. */
  stopEarly(id: string): { ok: true; afterEpoch: number } | Refusal {
    const job = this.jobs.get(id);
    if (!job) return { ok: false, status: 404, error: `Job ${id} not found` };
    if (job.status !== "RUNNING" || job.inferenceServerReady) return { ok: false, status: 400, error: `Job ${id} is not training (its status is ${job.status})` };
    if (job.stopAfterEpoch === null) {
      job.stopAfterEpoch = (job.latestLossEpoch ?? 0) + 1;
      job.log(`Stop requested: training ends after epoch ${job.stopAfterEpoch}, and ${trained(job)} by then is kept${job.args.autoServe ? " and served" : ""}.`);
    }
    return { ok: true, afterEpoch: job.stopAfterEpoch };
  }

  /** manager.restart_finetuning_job: run again under the same job id, from
   * the model's own weights on the paint as it is now, as the server's
   * restart starts from the model the job started from, with the form's
   * `options` and `params` (over the job's): a run's result never depends
   * on the runs before it. Only a job that canRestart takes one (a 409 as the
   * Python route answers, in its words). */
  restart(id: string, options: FinetuneOptions, params: Record<string, unknown>): { ok: true; job: FinetuneJob } | Refusal {
    const job = this.jobs.get(id);
    if (!job) return { ok: false, status: 404, error: `Job ${id} not found` };
    if (!canRestart(job)) {
      return {
        ok: false,
        status: 409,
        error: `Job ${id} is in state ${job.status} - can only restart a job that is waiting for a restart (its training iteration has finished or diverged)`,
      };
    }
    const mode = job.args.options.mode;
    if (options.mode !== mode) {
      return {
        ok: false,
        status: 409,
        error: `Job ${id} trains ${MODE_WORDS[mode]}; switching to ${MODE_WORDS[options.mode]} needs a new job, not a restart (Start Finetuning, with the Training field as it is now)`,
      };
    }
    // A 3D model's network lies in the file of the block it trained on
    // (run-full.ts trainingEntry), which another block's file cannot start from.
    const before = trainingEntry(job.args.entry, job.args.options), after = trainingEntry(job.args.entry, options);
    if (mode === "full" && after.variant.file !== before.variant.file) {
      return {
        ok: false,
        status: 409,
        error: `Job ${id} trains ${job.args.entry.name}'s whole network on blocks of ${before.variant.input_size[0]}³, in its ${before.variant.input_size[0]}³ file; a Block Size of ${after.variant.input_size[0]}³ needs a new job, not a restart (Start Finetuning, with the form as it is now)`,
      };
    }
    job.args = { ...job.args, options, params: { ...job.params, ...params } };
    // The marker first (job-card.js resets the card and the loss plot on
    // it), then the progress from zero, so the first status poll after it
    // already shows the new run.
    job.log(`${RESTARTING_MARKER}: a new run from ${job.modelName}'s own ${MODE_NOUN[mode]} on the paint as it is now, ${options.epochs} epoch(s) with the form's parameters`);
    job.startIteration(options.epochs);
    job.start(this.run(job));
    return { ok: true, job };
  }

  /** One run of `job`, start to finish: the base's bytes and the volume's
   * good regions as they are now (a restart takes the regions marked since,
   * as the server's trainer reads good_regions.json when it starts), the
   * run itself, then the result kept and served (keep). A cancel while it runs leaves
   * the job CANCELLED and the result unkept; any failure is logged into
   * the job and makes it FAILED, never thrown (a job's end is read from its
   * status, as a cluster job's is). */
  private async run(job: FinetuneJob): Promise<void> {
    const { ctx, entry: served, volume, options } = job.args;
    // The file it trains from and into: a 3D model's of the run's block,
    // which may not be the one served (run-full.ts trainingEntry).
    const entry = trainingEntry(served, options);
    job.setStatus("RUNNING");
    let result: Finetuned;
    try {
      const dataset = ctx.infer.dataset();
      if (!dataset) throw new Error("No dataset is open: open the volume the paint was made on first");
      if (entry !== served) {
        job.log(`Training ${entry.variant.file}, the ${entry.variant.input_size[0]}³ file, where ${served.name} is served from its ${served.variant.input_size[0]}³ one: the finetune is served at the block it trains on`);
      }
      const baseBytes = await ctx.catalog.modelBytes(entry);
      const goodRegions = await loadGoodRegions(ctx.paint, volume.id);
      result = await this.deps.run({
        entry,
        baseBytes,
        volume,
        chain: ctx.session.state.chain,
        dataset,
        infer: ctx.infer,
        paint: ctx.paint,
        options,
        report: job.reporter(),
        ...(goodRegions.length ? { goodRegions } : {}),
      });
    } catch (e) {
      if (job.isFinal()) {
        job.log(`The run stopped: ${message(e)}`);
        return;
      }
      job.log(`Training failed: ${message(e)}`);
      job.setStatus("FAILED");
      return;
    }
    if (job.isFinal()) {
      job.log("The run stopped; nothing of it is kept.");
      return;
    }
    if (job.stopAfterEpoch !== null) job.log(`Stopped early after epoch ${result.epochs}: ${trained(job)} so far is kept.`);
    try {
      await this.keep(job, result, entry);
    } catch (e) {
      if (job.isFinal()) return;
      job.log(`Could not keep or serve the finetuned model: ${message(e)}`);
      job.setStatus("FAILED");
    }
  }

  /** The run's result as a finetuned model: a catalog entry named after the
   * base (nextFinetunedName) with the result's bytes, recorded with where it
   * came from (`trained`, the file the run trained, so the finetune is served
   * at its block); then, with Auto-serve, served in the session beside its base
   * (in place of this job's previous model, on a restart, which is then
   * removed from the catalog: a job keeps its latest model only), and the
   * job left WAITING_FOR_RESTART, its model ready; without, COMPLETED. */
  private async keep(job: FinetuneJob, result: Finetuned, trained: ModelEntry): Promise<void> {
    const { ctx, entry, volume, autoServe } = job.args;
    // The names of the list's other models are passed over too: a model of
    // an added repo published as `<base>-ft<n>` would otherwise lose its
    // name to this one (models/index.ts withUniqueNames) while served.
    const listed = await ctx.catalog.list().catch(() => [] as ModelEntry[]);
    const name = nextFinetunedName(entry, await ctx.catalog.finetuned(), listed.map((e) => e.name));
    const record = finetunedRecord(trained, name, result.bytes.byteLength, { volume: volume.id, created_at: this.now().toISOString(), params: job.params, epochs: result.epochs, final_loss: result.finalLoss });
    const fentry = record.entry;
    await ctx.catalog.addFinetuned(record, result.bytes);
    // Made here, so listed here, served or not (src/kept).
    await keepInTab(ctx.session, { finetuned: [record.id] });
    const loss = result.finalLoss === null ? "" : `, final loss ${result.finalLoss.toFixed(6)}`;
    const on = result.blocks !== undefined ? `in ${result.blocks} block(s)` : result.tiles !== undefined ? `in ${result.tiles} tile(s)` : `of ${result.chunks} chunk(s)`;
    const kept = `Finetuned model ${name} kept in this browser (${fentry.variant.file_mb} MB): ${result.epochs} epoch(s) on ${result.pixels} painted voxels ${on}${loss}.`;
    job.log(`${kept} The Models tab lists it; Download gives the .onnx.`);
    ctx.log(kept);
    job.finetunedModelName = name;
    if (job.isFinal()) return;
    if (!autoServe) {
      job.setStatus("COMPLETED");
      return;
    }
    // The session's models with the new one added: in the place of the
    // model this job served before (a restart's, so its layer keeps its
    // colour), else right after the base, else at the end. The base stays
    // served, so the two can be compared side by side and a new job can
    // start from it again.
    const current = ctx.session.state.models;
    const previous = job.servedEntryId;
    const models = current.filter((m) => m.id !== previous);
    let at = previous === null ? -1 : current.findIndex((m) => m.id === previous);
    if (at === -1) {
      const base = models.findIndex((m) => m.id === entry.id);
      at = base === -1 ? models.length : base + 1;
    }
    models.splice(at, 0, fentry);
    // The previous model's layer is handed to the new one, so what the user
    // set on it (additive blending, say) stays on the layer they look at.
    const previousName = previous === null ? undefined : current.find((m) => m.id === previous)?.name;
    await serveModels(ctx, models, ctx.session.state.chain, previousName ? { replacing: { [name]: previousName } } : {});
    await ctx.session.update({ models });
    if (job.isFinal()) return;
    job.servedEntryId = fentry.id;
    job.inferenceServerReady = true;
    // A restart's model is the job's model from now on: the iteration it
    // replaces is forgotten, its record and bytes (3.9 to 43 MB), so the
    // Models tab lists one model per job, the latest, as the server serves
    // only a job's latest iteration (keeping the earlier ones' files on
    // disk, the run's record). It went out of serving and out of the viewer
    // above, its layer handed to this one. Only this job's own model is
    // ever removed (servedEntryId is set nowhere else), never its base.
    if (previous !== null && previous !== fentry.id) {
      const was = previousName ?? previous;
      try {
        await ctx.catalog.removeFinetuned(previous);
        job.log(`${name} replaces ${was}, which is removed from this browser.`);
      } catch (error) {
        job.log(`${was} could not be removed from this browser (${message(error)}); it stays on the Models tab, beside ${name}.`);
      }
    }
    job.log(`Serving ${name} beside ${entry.name}. WAITING_FOR_RESTART: paint where it is still wrong and Restart to train again, from ${job.modelName}'s own ${MODE_NOUN[job.args.options.mode]}, on all of it.`);
    job.setStatus("WAITING_FOR_RESTART");
  }
}

/** What a job's run trains, in a log line's words: "the head trained", "the network trained". */
function trained(job: FinetuneJob): string {
  return `the ${MODE_NOUN[job.args.options.mode]} trained`;
}

/** `YYYYmmdd-HHMMSS`, as the volume ids and the server's run directories are stamped. */
function timestamp(date: Date): string {
  return stamp(date);
}
