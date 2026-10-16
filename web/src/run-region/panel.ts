// Run on a region: the Run tab (templates/_run_tab.html, the browser mode's
// alone) and its flows. The tab runs the models served on the Models tab
// over a region of the open volume, or all of it, and saves each model's
// predictions as an OME-Zarr in a folder on this machine; then the zarrs
// open in the viewer, read back from the folder, beside the live layers.
//
// It is wired here, in the page, rather than by a dashboard script talking
// to /api/ routes: the folder is picked with the File System Access API,
// whose picker and permission prompt need the user's click in this window,
// and the chunks are written by the engine itself (Inference.materialize,
// chunkmirage's Engine.materialize) into a directory handle, which no
// request through the service worker can carry. The template is the
// browser's alone, so no dashboard script needs to know of it (as the
// header's Share button is src/share-link.ts's).
//
// What the user does: tick the models (all, by default); draw a box in the
// viewer (Draw a box adds the run_region annotation layer, its tool the
// bounding box, drawn with two Ctrl+clicks; the latest box drawn fills the
// fields), or choose Whole
// volume; edit the fields if need be; Choose folder… once; Run. The summary
// says each model's box in voxels and chunks at its level, and warns of a
// large run. The run goes on while the user browses; Cancel stops it. The
// region's geometry is region.ts's, the run run.ts's, the folder and the
// zip output.ts's.
import { directoryStore, memoryStore } from "../../chunkmirage/web/src/materialize";
import { pageBase } from "../backend/serving";
import type { AppContext, LayerSpec, ServedModel } from "../contracts";
import { localKeyFor, localUrl, supportsLocalFolders } from "../local-folder";
import { carriesFiles, DROP_CLASS, type HandleItem } from "../local-folder/panel";
import { getHandle, putHandle } from "../state/local-store";
import { shaderColor } from "../viewer/display";
import { dimensionsOf, PREDICTION_COLORS } from "../viewer/layers";
import {
  type FolderStore, type MemoryStore, OUTPUT_FOLDER_KEY, OUTPUT_PICKER_OPTIONS, type OutputFolder, prepareFolder, savedLayer, savedLayerName,
  ZIP_LIMIT_BYTES, zarrName, zipFileName, zipRun,
} from "./output";
import {
  describePlan, formatBytes, formatDuration, isPlan, largeRunWarning, latestBox, type ModelPlan, planModel, type RegionNm, regionError,
  roundedOut, RUN_REGION_LAYER, runRegionLayer, sameRegion, snappedRegion, wholeVolumeNm, withBox,
} from "./region";
import { describeProgress, provenance, type RunOutcome, type RunProgress, runRegion, type RunTarget } from "./run";

/** The tab's elements, by the ids templates/_run_tab.html and _dashboard.html
 * give them (tests/web/test_render.py checks each is on the browser's page
 * and on no other). */
export const IDS = {
  tab: "run-tab",
  pane: "run",
  models: "runModels",
  modeBox: "runRegionBox",
  modeWhole: "runRegionWhole",
  drawBox: "runDrawBoxBtn",
  offsetZ: "runOffsetZ",
  offsetY: "runOffsetY",
  offsetX: "runOffsetX",
  sizeZ: "runSizeZ",
  sizeY: "runSizeY",
  sizeX: "runSizeX",
  summary: "runRegionSummary",
  warning: "runRegionWarning",
  peakWarning: "runPeakWarning",
  chooseFolder: "runChooseFolderBtn",
  folderDrop: "runFolderDrop",
  folderName: "runFolderName",
  folderNote: "runFolderNote",
  compression: "runCompression",
  start: "runStartBtn",
  cancel: "runCancelBtn",
  status: "runStatus",
  progress: "runProgress",
  progressBar: "runProgressBar",
  progressText: "runProgressText",
  results: "runResults",
  log: "runLog",
} as const;

/** The viewer's iframe (viewer/index.ts mounts it under this id), focused
 * by Draw a box so that the next clicks go to Neuroglancer. */
const FRAME_ID = "my_iframe";

/** How often the tab, while shown, looks for a new box in the viewer and a
 * change in the models served. */
const POLL_MS = 1000;

/** What the tab says where the browser cannot write to a folder. */
export const NO_FOLDER_NOTE = `Saving to a folder needs Chrome or Edge (the File System Access API). In this browser a run of up to ${formatBytes(ZIP_LIMIT_BYTES)} of predictions is kept in memory and saved as a zip of the .zarr folders.`;

/** How long typing in the region's fields pauses before the box in the
 * viewer is redrawn to them (showTypedBox): each redraw sets the viewer's
 * state, which is not worth doing at every key. */
export const TYPED_BOX_DELAY_MS = 400;

/** Above this, the Run tab says how much of a model's stitched chunks a
 * run holds at once, and offers to snap the box onto a chunk edge: a tab
 * has a few GB at most. */
export const PEAK_WARN_BYTES = 512e6;

/** The picker as Chromium has it (not in lib.dom, so `window` is read as this). */
interface PickerWindow {
  showDirectoryPicker?(options?: { id?: string; mode?: "read" | "readwrite"; startIn?: "desktop" | "documents" | "downloads" }): Promise<OutputFolder>;
}

/** The IndexedDB store of picked handles (state/local-store.ts), which the
 * service worker reads the local/ route's folders from. The casts are the
 * gap in lib.dom that OutputFolder fills. */
const indexedDbFolders: FolderStore = {
  get: (key) => getHandle(key) as unknown as Promise<OutputFolder | undefined>,
  put: (key, folder) => putHandle(key, folder as unknown as FileSystemDirectoryHandle),
};

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** The Run tab, wired: does nothing on a page without it (the Flask
 * dashboard's). Call it once the page's DOM is there; it needs nothing of
 * the dashboard's own script. */
export function installRunRegionUi(ctx: AppContext, opts: { folders?: FolderStore; supported?: boolean } = {}): void {
  const byId = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T | null;
  const pane = byId(IDS.pane);
  if (!pane) return;
  const el = {
    models: byId(IDS.models)!, modeBox: byId<HTMLInputElement>(IDS.modeBox)!, modeWhole: byId<HTMLInputElement>(IDS.modeWhole)!,
    drawBox: byId<HTMLButtonElement>(IDS.drawBox)!,
    offset: [IDS.offsetZ, IDS.offsetY, IDS.offsetX].map((id) => byId<HTMLInputElement>(id)!),
    size: [IDS.sizeZ, IDS.sizeY, IDS.sizeX].map((id) => byId<HTMLInputElement>(id)!),
    summary: byId(IDS.summary)!, warning: byId(IDS.warning)!, peakWarning: byId(IDS.peakWarning)!,
    chooseFolder: byId<HTMLButtonElement>(IDS.chooseFolder)!, folderDrop: byId(IDS.folderDrop)!, folderName: byId(IDS.folderName)!,
    folderNote: byId(IDS.folderNote)!,
    start: byId<HTMLButtonElement>(IDS.start)!, cancel: byId<HTMLButtonElement>(IDS.cancel)!, status: byId(IDS.status)!,
    progress: byId(IDS.progress)!, progressBar: byId(IDS.progressBar)!, progressText: byId(IDS.progressText)!,
    results: byId(IDS.results)!, log: byId<HTMLTextAreaElement>(IDS.log)!,
  };
  const folders = opts.folders ?? indexedDbFolders;
  const supported = opts.supported ?? supportsLocalFolders();

  /** The folder the predictions go to, once chosen (or remembered). */
  let folder: OutputFolder | null = null;
  /** The run going on, if any: one at a time. */
  let running: AbortController | null = null;
  /** The served views the model list shows, to rebuild it only when they
   * change (null before it is first built). */
  let listed: string | null = null;
  /** The models the user unticked, by name: a model served later is ticked. */
  const unticked = new Set<string>();
  /** The latest box seen in the viewer, as JSON, to take a new one when it is drawn. */
  let seenBox = "";
  /** The last zip offered, freed when the next run starts. */
  let zipHref: string | null = null;

  const log = (line: string) => {
    el.log.value += `${line}\n`;
    el.log.scrollTop = el.log.scrollHeight;
    ctx.log(`run on a region: ${line}`);
  };
  const status = (text: string, tone: "info" | "error" | "success" = "info") => {
    el.status.textContent = text;
    el.status.className = `small mb-2 ${tone === "error" ? "text-danger" : tone === "success" ? "text-success" : "text-muted"}`;
  };

  // ---------------------------------------------------------------- models

  /** The model list, rebuilt when the served views changed; whether it was. */
  function listModels(): boolean {
    const served = ctx.infer.served();
    const signature = served.map((s) => s.view).join("\n");
    if (signature === listed) return false;
    listed = signature;
    if (!served.length) {
      el.models.replaceChildren(Object.assign(document.createElement("div"), {
        className: "small text-muted", textContent: "No models are served: tick some on the Models tab and click Submit Models.",
      }));
      return true;
    }
    el.models.replaceChildren(...served.map((s) => {
      const row = document.createElement("div");
      row.className = "form-check";
      const label = document.createElement("label");
      label.className = "form-check-label small";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.className = "form-check-input";
      box.checked = !unticked.has(s.entry.name);
      box.addEventListener("change", () => {
        if (box.checked) unticked.delete(s.entry.name);
        else unticked.add(s.entry.name);
        summarize();
      });
      label.append(box, ` ${s.entry.name} `, Object.assign(document.createElement("span"), {
        className: "text-muted", textContent: `(${s.entry.variant.variant}, ${s.entry.variant.output_size.join("×")} voxels a chunk)`,
      }));
      row.append(label);
      return row;
    }));
    return true;
  }

  const ticked = (): ServedModel[] => ctx.infer.served().filter((s) => !unticked.has(s.entry.name));

  // ---------------------------------------------------------------- the region

  const fields = () => [...el.offset, ...el.size];

  /** The fields as a region, or null while they are all blank. A blank or
   * unreadable one among filled ones is NaN, which regionError names. */
  function readRegion(): RegionNm | null {
    if (fields().every((f) => f.value.trim() === "")) return null;
    const value = (f: HTMLInputElement) => (f.value.trim() === "" ? NaN : Number(f.value));
    return { offset: el.offset.map(value), size: el.size.map(value) };
  }

  function fillRegion(region: RegionNm | null): void {
    const r = region && roundedOut(region);
    el.offset.forEach((f, a) => (f.value = r ? String(r.offset[a]) : ""));
    el.size.forEach((f, a) => (f.value = r ? String(r.size[a]) : ""));
  }

  /** The plans for the ticked models over the fields' region, or what is
   * missing for there to be any. */
  function plan(): { plans: ModelPlan[]; refused: string[] } | { missing: string } {
    const dataset = ctx.infer.dataset();
    if (!dataset) return { missing: "Open a volume first." };
    const region = readRegion();
    if (!region) return { missing: el.modeBox.checked ? "Draw a box in the viewer (Draw a box, above), or choose Whole volume." : "Fill in the region." };
    const bad = regionError(region);
    if (bad) return { missing: bad };
    const models = ticked();
    if (!models.length) return { missing: ctx.infer.served().length ? "Tick a model above." : "No models are served: tick some on the Models tab and click Submit Models." };
    const plans: ModelPlan[] = [], refused: string[] = [];
    for (const p of models.map((s) => planModel(s, dataset, region))) {
      if (isPlan(p)) plans.push(p);
      else refused.push(p.error);
    }
    return { plans, refused };
  }

  /** The summary under the fields: each model's box and chunks, and the warning for a large run. */
  function summarize(): void {
    const planned = plan();
    if ("missing" in planned) {
      el.summary.textContent = planned.missing;
      el.warning.hidden = true;
      return;
    }
    el.summary.replaceChildren(...[...planned.plans.map(describePlan), ...planned.refused].map((line) => Object.assign(document.createElement("div"), { textContent: line })));
    const chunks = planned.plans.reduce((n, p) => n + p.chunks, 0);
    const bytes = planned.plans.reduce((n, p) => n + p.bytes, 0);
    const warnings = [largeRunWarning(chunks, ctx.infer.stats.avgMs)];
    if (!supported && bytes > ZIP_LIMIT_BYTES) warnings.push(zipTooLarge(bytes));
    const text = warnings.filter((w): w is string => w !== null).join(" ");
    el.warning.textContent = text;
    el.warning.hidden = !text;
    void checkPeak(planned.plans);
  }

  /** The latest checkPeak, so an answer for fields since changed is dropped. */
  let peakAsked = 0;

  /** Say so when saving a model's box would hold more than PEAK_WARN_BYTES
   * of its stitched chunks at once (a box whose corner is off the model's
   * chunk edges along several axes), with a button that moves the corner
   * onto the chunk edge along the axis that shrinks it most. Asked of the
   * engine (Inference.materializePlan), which computes nothing for it. */
  async function checkPeak(plans: ModelPlan[]): Promise<void> {
    const ask = ++peakAsked;
    el.peakWarning.hidden = true;
    const planOf = ctx.infer.materializePlan?.bind(ctx.infer);
    if (!planOf || !el.modeBox.checked) return;
    try {
      for (const p of plans) {
        const peak = await planOf(p.served.view, { lo: p.lo, hi: p.hi });
        if (ask !== peakAsked) return;
        if (peak.peakBytes <= PEAK_WARN_BYTES) continue;
        let best: { axis: number; bytes: number } | null = null;
        for (const axis of peak.split) {
          const lo = [...p.lo];
          lo[axis] = Math.floor(lo[axis] / p.chunk[axis]) * p.chunk[axis];
          const snapped = await planOf(p.served.view, { lo, hi: p.hi });
          if (!best || snapped.peakBytes < best.bytes) best = { axis, bytes: snapped.peakBytes };
        }
        if (ask !== peakAsked) return;
        const line = Object.assign(document.createElement("span"), {
          textContent: `Saving ${p.name} holds up to ${formatBytes(peak.peakBytes)} of its chunks in this browser at once, as the box's corner is inside the model's chunks.`
            + (best ? ` Starting the box on a chunk edge along ${"zyx"[best.axis]} holds ${formatBytes(best.bytes)}. ` : " "),
        });
        el.peakWarning.replaceChildren(line);
        if (best) {
          const axis = best.axis;
          const snap = Object.assign(document.createElement("button"), {
            type: "button", className: "btn btn-sm btn-outline-warning py-0", textContent: `Snap ${"zyx"[axis]}`,
            title: "Moves the box's corner back to the start of the model chunk it is in along this axis; its far side stays.",
          });
          snap.addEventListener("click", () => {
            const region = readRegion();
            if (!region) return;
            fillRegion(snappedRegion(region, p, axis));
            summarize();
            showTypedBox();
          });
          el.peakWarning.append(snap);
        }
        el.peakWarning.hidden = false;
        return;
      }
    } catch (e) {
      // Only a hint: a plan that cannot be made says nothing, and Run says why.
      log(`the memory check could not be made: ${message(e)}`);
    }
  }

  function zipTooLarge(bytes: number): string {
    return `Up to ${formatBytes(bytes)} of predictions: more than the ${formatBytes(ZIP_LIMIT_BYTES)} a zip from this browser holds. Choose a smaller region, or use Chrome or Edge to write to a folder.`;
  }

  /** The latest box in the run_region layer, into the fields when it is a new one. */
  function takeNewBox(): void {
    const box = latestBox(ctx.viewer.layer(RUN_REGION_LAYER));
    const key = JSON.stringify(box);
    if (key === seenBox) return;
    seenBox = key;
    if (!box) return;
    // The box drawn from the fields (showTypedBox) reads back a little off,
    // its corners kept as 32-bit floats: it is not a new one.
    const typed = readRegion();
    if (typed && sameRegion(box, typed)) return;
    fillRegion(box);
    summarize();
  }

  /** The fields' region drawn as the run_region layer's box, so the box in
   * the viewer follows what is typed, as the fields follow a box drawn
   * (takeNewBox). Only in box mode, once every field is a number and the
   * size positive; the layer is added when it is not there yet. */
  function showTypedBox(): void {
    if (typedBoxTimer !== null) clearTimeout(typedBoxTimer);
    typedBoxTimer = null;
    if (!el.modeBox.checked) return;
    const region = readRegion();
    if (!region || ![...region.offset, ...region.size].every(Number.isFinite) || region.size.some((s) => s <= 0)) return;
    const dataset = ctx.infer.dataset();
    if (!dataset) return;
    const layer = ctx.viewer.layer(RUN_REGION_LAYER) ?? runRegionLayer(Object.keys(dimensionsOf(dataset)));
    const shown = latestBox(layer);
    if (shown && sameRegion(shown, region)) return;
    const next = withBox(layer, region);
    seenBox = JSON.stringify(latestBox(next));
    ctx.viewer.setLayers({ [RUN_REGION_LAYER]: next });
  }

  /** showTypedBox once typing pauses: each redraw sets the viewer's state. */
  let typedBoxTimer: ReturnType<typeof setTimeout> | null = null;
  function showTypedBoxSoon(): void {
    if (typedBoxTimer !== null) clearTimeout(typedBoxTimer);
    typedBoxTimer = setTimeout(showTypedBox, TYPED_BOX_DELAY_MS);
  }

  function chooseMode(): void {
    const dataset = ctx.infer.dataset();
    if (el.modeWhole.checked) {
      fillRegion(dataset ? wholeVolumeNm(dataset) : null);
    } else {
      // The box drawn, or nothing: fields left on the whole volume would be
      // run as if they were the box.
      const box = latestBox(ctx.viewer.layer(RUN_REGION_LAYER));
      seenBox = JSON.stringify(box);
      fillRegion(box);
    }
    summarize();
  }

  function drawBox(): void {
    const dataset = ctx.infer.dataset();
    if (!dataset) return status("Open a volume first.", "error");
    // Added only when it is not there: setting it again would replace the
    // layer, and the boxes drawn in it with it.
    if (!ctx.viewer.layer(RUN_REGION_LAYER)) ctx.viewer.setLayers({ [RUN_REGION_LAYER]: runRegionLayer(Object.keys(dimensionsOf(dataset))) });
    ctx.viewer.selectLayer(RUN_REGION_LAYER);
    if (!el.modeBox.checked) {
      el.modeBox.checked = true;
      chooseMode();
    }
    byId(FRAME_ID)?.focus();
    status("The run_region layer is in the viewer: hold Ctrl and click two opposite corners of the box in the viewer, scrolling between the clicks to give it depth.");
  }

  // ---------------------------------------------------------------- the folder

  function showFolder(): void {
    if (!supported) {
      el.chooseFolder.style.display = "none";
      el.folderName.textContent = "No folder: this browser cannot write to one";
      el.folderNote.textContent = NO_FOLDER_NOTE;
      return;
    }
    el.folderName.textContent = folder ? folder.name : "No folder chosen";
    el.folderNote.textContent = folder
      ? `Each model goes to ${folder.name}/<model>.zarr. The browser asks to allow writing to the folder when you click Run, when it has not yet (after a reload, say).`
      : "Choose where the predictions go, or drop a folder here: each model is written to <folder>/<model>.zarr. Chrome won't let a page write straight into Downloads or your home folder (\"it contains system files\"): make a new folder inside it (the picker has a New Folder button) and choose that.";
  }

  /** The picker, in the user's click; the folder chosen, or null when the dialog was cancelled. */
  async function pickFolder(): Promise<OutputFolder | null> {
    try {
      const picked = await (window as unknown as Required<PickerWindow>).showDirectoryPicker(OUTPUT_PICKER_OPTIONS);
      await useFolder(picked);
      return picked;
    } catch (e) {
      // AbortError: the dialog was cancelled, which needs no message.
      if ((e as { name?: string })?.name !== "AbortError") status(`Could not pick a folder: ${message(e)}`, "error");
      return null;
    }
  }

  /** `picked` as the output folder, shown and kept for the next visit (a
   * store that refuses, in a private window, costs only that). */
  async function useFolder(picked: OutputFolder): Promise<void> {
    folder = picked;
    showFolder();
    log(`output folder: ${picked.name}`);
    await folders.put(OUTPUT_FOLDER_KEY, picked).catch((e: unknown) => log(`the folder is not remembered for the next visit: ${message(e)}`));
  }

  /** A folder dropped on the Output folder row, taken as the picker's would
   * be; the browser asks to allow writing to it when Run is clicked (start).
   * Asked for in the drop event: the drag's items are gone once it ends. */
  function acceptDroppedFolder(): void {
    const zone = el.folderDrop;
    zone.addEventListener("dragover", (event) => {
      if (!carriesFiles(event.dataTransfer)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "link";
      zone.classList.add(DROP_CLASS);
    });
    zone.addEventListener("dragleave", (event) => {
      // Leaving for one of the row's own children is not leaving.
      if (!zone.contains(event.relatedTarget as Node | null)) zone.classList.remove(DROP_CLASS);
    });
    zone.addEventListener("drop", (event) => {
      zone.classList.remove(DROP_CLASS);
      if (!carriesFiles(event.dataTransfer)) return;
      event.preventDefault();
      const items = Array.from(event.dataTransfer.items).filter((item) => item.kind === "file") as HandleItem[];
      if (items.length !== 1) return status("Drop one folder: the one the predictions go to.", "error");
      const item = items[0];
      if (!item.getAsFileSystemHandle) return status("This browser gives no handle for a dropped folder: use Choose folder… in Chrome or Edge.", "error");
      item.getAsFileSystemHandle().then(
        (handle) => {
          if (!handle || handle.kind !== "directory") return status(`${handle ? handle.name : "What was dropped"} is not a folder: drop the folder the predictions go to.`, "error");
          status(`Output folder: ${handle.name}. The browser asks to allow writing to it when you click Run.`);
          return useFolder(handle as unknown as OutputFolder);
        },
        (e: unknown) => status(`Could not take the dropped folder: ${message(e)}`, "error"),
      );
    });
  }

  // ---------------------------------------------------------------- the run

  function setBusy(busy: boolean): void {
    el.start.disabled = busy;
    el.cancel.disabled = !busy;
    el.chooseFolder.disabled = busy;
    if (busy) el.progress.hidden = false;
  }

  function showProgress(p: RunProgress): void {
    const percent = p.runTotal ? Math.min(100, (100 * p.runDone) / p.runTotal) : 0;
    el.progressBar.style.width = `${percent.toFixed(1)}%`;
    el.progress.setAttribute("aria-valuenow", String(Math.round(percent)));
    el.progressText.textContent = describeProgress(p);
  }

  async function start(): Promise<void> {
    if (running) return;
    const planned = plan();
    if ("missing" in planned) return status(planned.missing, "error");
    if (planned.refused.length) return status(planned.refused.join(" "), "error");
    const { plans } = planned;
    const dataset = ctx.infer.dataset()!, region = readRegion()!;

    let target: RunTarget;
    let used: OutputFolder | null = null;
    let memory: Map<string, MemoryStore> | null = null;
    if (supported) {
      const chosen = folder ?? (await pickFolder());
      if (!chosen) return;
      // Permission asked in this click when Chrome has dropped it (a
      // reload; a folder just picked in "readwrite" mode has it already),
      // then the zarrs in the way, if the user agrees.
      const ready = await prepareFolder(chosen, plans.map((p) => p.name), (question) => window.confirm(question));
      if (!ready.ready) return status(ready.reason, ready.refused ? "info" : "error");
      if (ready.removed.length) log(`removed ${ready.removed.map((e) => `${chosen.name}/${e}`).join(", ")} before writing again`);
      used = chosen;
      target = {
        where: (name) => `${chosen.name}/${zarrName(name)}`,
        store: (name) => directoryStore(chosen as unknown as FileSystemDirectoryHandle, zarrName(name)),
      };
    } else {
      const bytes = plans.reduce((n, p) => n + p.bytes, 0);
      if (bytes > ZIP_LIMIT_BYTES) return status(zipTooLarge(bytes), "error");
      const stores = new Map<string, MemoryStore>();
      memory = stores;
      target = {
        where: (name) => `${zarrName(name)} (in memory, for the zip)`,
        store: (name) => {
          const store = memoryStore();
          stores.set(name, store);
          return store;
        },
      };
    }

    const controller = new AbortController();
    running = controller;
    setBusy(true);
    el.results.replaceChildren();
    if (zipHref) URL.revokeObjectURL(zipHref);
    zipHref = null;
    const now = new Date();
    const chunks = plans.reduce((n, p) => n + p.chunks, 0);
    status(`Running ${plans.map((p) => p.name).join(", ")}: ${chunks.toLocaleString("en-US")} chunk(s). You can go on browsing.`);
    log(`run of ${dataset.url}, offset ${region.offset.join(", ")} nm, size ${region.size.join(", ")} nm`);
    let outcome: RunOutcome;
    try {
      outcome = await runRegion({
        infer: ctx.infer, plans, target, signal: controller.signal,
        // The tab's Compression: gzip, or none for a reader without it (Fiji).
        compression: (byId<HTMLSelectElement>(IDS.compression)?.value ?? "gzip") === "none" ? null : "gzip",
        attributes: (p) => provenance(p, dataset, region, now, pageBase()),
        onProgress: showProgress, log,
      });
    } catch (e) {
      outcome = { results: [], aborted: false, error: message(e), elapsedMs: 0 };
    } finally {
      running = null;
      setBusy(false);
    }
    if (outcome.error) status(`The run stopped: ${outcome.error}`, "error");
    else if (outcome.aborted) status("Cancelled. What was written stays.");
    else status(`Done in ${formatDuration(outcome.elapsedMs)}.`, "success");
    if (used) offerViewer(outcome, used);
    if (memory) offerZip(outcome, memory, now);
  }

  /** After a folder run: a button that opens the zarrs written as layers. */
  function offerViewer(outcome: RunOutcome, written: OutputFolder): void {
    const shown = outcome.results.filter((r) => r.written.written > 0);
    if (!shown.length) return;
    const button = Object.assign(document.createElement("button"), {
      type: "button", className: "btn btn-sm btn-outline-secondary", textContent: "Show the saved predictions in the viewer",
      title: "Adds each model's saved zarr as a layer, read back from the folder, beside the model's live layer.",
    });
    button.addEventListener("click", () => void openInViewer(shown.map((r) => r.plan), written));
    el.results.replaceChildren(button);
  }

  /** The zarrs of `plans` in `written` as layers: the folder's handle stored
   * under a fresh local/ key (the service worker reads it from there; a new
   * key per opening, output.ts says why), each zarr's layer named after its
   * model, coloured and shaded as the model's live layer is. */
  async function openInViewer(plans: ModelPlan[], written: OutputFolder): Promise<void> {
    const key = localKeyFor(written.name);
    try {
      await folders.put(key, written);
    } catch (e) {
      return status(`Could not open ${written.name} for the viewer: ${message(e)}`, "error");
    }
    const url = localUrl(key);
    const served = ctx.infer.served();
    const layers: Record<string, LayerSpec> = {};
    plans.forEach((p, i) => {
      const keep = ctx.viewer.layerShader(p.name) ?? undefined;
      const place = served.findIndex((s) => s.entry.name === p.name);
      const color = shaderColor(keep?.shader) ?? PREDICTION_COLORS[(place >= 0 ? place : i) % PREDICTION_COLORS.length];
      layers[savedLayerName(p.name)] = savedLayer(p.served, p.chain, url, color, keep);
    });
    ctx.viewer.setLayers(layers);
    log(`in the viewer: ${Object.keys(layers).join(", ")}, read from ${written.name} at ${url}`);
  }

  /** After a run kept in memory: the zip of its zarrs, as a link to save. */
  function offerZip(outcome: RunOutcome, stores: Map<string, MemoryStore>, now: Date): void {
    if (!outcome.results.some((r) => r.written.written > 0)) return;
    let zip: Uint8Array;
    try {
      zip = zipRun(stores, now);
    } catch (e) {
      return status(`Could not make the zip: ${message(e)}`, "error");
    }
    zipHref = URL.createObjectURL(new Blob([zip as Uint8Array<ArrayBuffer>], { type: "application/zip" }));
    const name = zipFileName(now);
    const link = Object.assign(document.createElement("a"), {
      className: "btn btn-sm btn-outline-secondary", href: zipHref, download: name,
      textContent: `Save the predictions (${name}, ${formatBytes(zip.length)})`,
    });
    el.results.replaceChildren(link);
    log(`${name}: ${formatBytes(zip.length)}, ready to save`);
  }

  // ---------------------------------------------------------------- wiring

  function tick(): void {
    if (!pane!.classList.contains("active")) return;
    if (listModels()) summarize();
    if (el.modeBox.checked) takeNewBox();
  }

  el.modeBox.addEventListener("change", chooseMode);
  el.modeWhole.addEventListener("change", chooseMode);
  for (const f of fields()) {
    f.addEventListener("input", () => {
      summarize();
      showTypedBoxSoon();
    });
    f.addEventListener("change", showTypedBox);
  }
  if (supported) acceptDroppedFolder();
  el.drawBox.addEventListener("click", drawBox);
  el.chooseFolder.addEventListener("click", () => void pickFolder());
  el.start.addEventListener("click", () => void start());
  el.cancel.addEventListener("click", () => {
    if (!running) return;
    running.abort();
    // materialize gives up the chunks not yet written and resolves once
    // the writes under way have ended (chunkmirage's writeChunks).
    status("Cancelling: the chunks being written finish, the rest are given up.");
  });
  // A reload or a new volume (connect.js reloads the page) would end the
  // run: the browser asks first.
  window.addEventListener("beforeunload", (event) => {
    if (running) event.preventDefault();
  });
  byId(IDS.tab)?.addEventListener("shown.bs.tab", tick);
  setInterval(tick, POLL_MS);

  showFolder();
  listModels();
  summarize();
  if (supported) {
    folders.get(OUTPUT_FOLDER_KEY).then(
      (remembered) => {
        if (remembered && !folder) {
          folder = remembered;
          showFolder();
        }
      },
      () => {},  // no stored folder to offer: Choose folder… asks
    );
  }
}
