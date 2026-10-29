// A link to the page that opens a volume and runs models on it: the query
// ?dataset=<url>&model=<ref>&model=<ref> (a model's ref is a name such as
// setup_53, or the repository path cellmap/onnx-web-models/setup_53, or any
// Hugging Face repo's `<owner>/<repo>/<folder>`, which the catalog looks up
// there: models/index.ts resolve), as the README's example link has it.
// main.ts reads it once at boot, after the service worker's first-visit
// reload (which keeps the URL): the dataset and the models go into the tab's
// session, where Connect and Submit would have put them, so the session's
// restore opens the volume and serves the models as after any reload (a
// link with models and no dataset, as a published model's card has it,
// serves them once a volume is opened); and the query is taken off the
// address bar, so a later tab completed from this URL starts fresh, as a
// `#!` state link's is (viewer/index.ts). A model the catalog does not know
// is logged and left out, and the page says which and why (linkNotes, in
// the notice main.ts shows): the catalog's reason, such as a model
// published from a server with no copy for the browser, and the smaller
// copy to open instead where its repo has one; the rest run.
//
// A model finetuned in a browser is kept in that browser alone (IndexedDB,
// models/finetuned.ts), so its name would mean nothing anywhere else; the
// link carries what rebuilds it instead (finetunedRef). Browser finetuning
// trains only the model's last 1×1 convolution, its head (src/finetune), so
// the model is its base's ONNX file with other numbers in that one layer, a
// few hundred of them: they travel in the link, and the page that opens it
// downloads the base, writes them in (onnx-head.ts withHead) and keeps the
// result as a model finetuned in this browser, under the link's name unless
// this browser has a model of that name already, which is never replaced.
// The paint it was trained on does not travel: it is in the paint store of
// the browser that painted it. A link's `#!` view is opened without the
// layers this browser cannot show (linkedView), the page saying which and
// why (linkNotes), rather than with layers whose sources never answer.
//
// A model run at another voxel size than the one its ref gives it (the
// Models tab's Run at (nm): ModelEntry.runVoxelSize) has that size after its
// ref (runAtRef: `setup_54@32nm`), and runs at it where the link opens; so
// does a finetuned model trained at another voxel size than its base's own,
// which is rebuilt as a model of that size.
import { message } from "./backend/router";
import { ARGS_KEY, type Catalog, type Chain, type ModelEntry, type PaintStore } from "./contracts";
import { headOf, withHead } from "./finetune/onnx-head";
import type { Head } from "./finetune/types";
import { PYTHON } from "./generated/python-data";
import { parseVoxelSize, withRunVoxelSize } from "./infer/geometry";
import { finetunedRecord, isFinetuned, nextFinetunedName, publishedRef } from "./models/finetuned";
import { NoBrowserCopyError } from "./models/hf";
import { hubName, voxelLabel, withVariant } from "./models/manifest";
import { base64UrlToBytes, bytesToBase64Url, decodeArgs } from "./shared/args";
import type { Hint } from "./shared/hint";
import { pastedPath } from "./shared/pasted";
import { paintVolumeOf } from "./viewer/layers";
import { decodeState, encodeState } from "./viewer/share";
import { layersOf, type NgLayer, type NgState, sourceUrl } from "./viewer/state";

// The query's parameters, as a published model's card writes them too
// (cellmap_flow/portable/publish.py, exported by build/export_data.py).
export const DATASET_PARAM = PYTHON.links.dataset_param;
export const MODEL_PARAM = PYTHON.links.model_param;

export interface LinkRequest {
  /** The volume to open, or null when the link names none. */
  datasetUrl: string | null;
  /** The models to run, as given, each once. */
  models: string[];
}

/** What `search` (location.search) asks for: repeated `model` parameters
 * and comma-separated lists both work, and a dataset pasted from
 * Neuroglancer with its format (shared/pasted.ts). */
export function parseLink(search: string): LinkRequest {
  const params = new URLSearchParams(search);
  const datasetUrl = pastedPath(params.get(DATASET_PARAM) ?? "") || null;
  const models = params.getAll(MODEL_PARAM).flatMap((m) => m.split(",")).map((m) => m.trim()).filter(Boolean);
  return { datasetUrl, models: [...new Set(models)] };
}

/** `href` without the link's parameters; anything else in the URL stays. */
export function withoutLink(href: string): string {
  const url = new URL(href);
  url.searchParams.delete(DATASET_PARAM);
  url.searchParams.delete(MODEL_PARAM);
  return url.href;
}

/** A served view's name in a layer's source URL: the model's name, then the
 * chain's blob between ARGS_KEY markers (shared/args.ts). */
const MARKED_VIEW = new RegExp(`${ARGS_KEY}([^/]+?)${ARGS_KEY}`);

/** The chain the model layers of a `#!` state were drawn through: a served
 * view is named after its model with the chain's blob between ARGS_KEY
 * markers (shared/args.ts), and a layer's source URL names the view. The
 * first layer's that reads; null when the state has no such layer or the
 * blob cannot be read, and a shared link then runs this browser's own
 * chain. */
export function chainFromHash(hash: string): Chain | null {
  for (const layer of layersOf(decodeState(hash))) {
    const blob = sourceUrl(layer.source)?.match(MARKED_VIEW)?.[1];
    if (!blob) continue;
    try {
      return decodeArgs(blob).chain;
    } catch {
      // Not a blob this build reads: the next layer may carry one it does.
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The voxel size a model of the link runs at
// ---------------------------------------------------------------------------

/** What follows a ref whose model runs at another voxel size: `@<z>x<y>x<x>nm`,
 * one number for a cube (voxelLabel). An "@" before a revision (hubRef's
 * `@main`) is never numbers and "nm", and no other part of a ref holds an
 * "@", so the two are told apart. */
const RUN_AT = /@(\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)*)nm$/;

/** `ref` for a model run at `size` (nm, z, y, x): `setup_54@32nm`,
 * "setup_54 at 32 nm"; `ref` itself for null, its own. */
export function runAtRef(ref: string, size: readonly number[] | null): string {
  return size ? `${ref}@${voxelLabel([...size])}nm` : ref;
}

/** The ref a link's model ref is without its voxel size (runAtRef), and that
 * size (null when it has none); an Error for a size that is not one or
 * three positive numbers. */
export function splitRunAt(linked: string): { ref: string; runAt: number[] | null } {
  const match = RUN_AT.exec(linked);
  if (!match) return { ref: linked, runAt: null };
  return { ref: linked.slice(0, match.index), runAt: parseVoxelSize(match[1].split("x"), `the voxel size ${match[1]}nm`) };
}

// ---------------------------------------------------------------------------
// A finetuned model in the link: its name, its base and its head's numbers
// ---------------------------------------------------------------------------

/** What separates the parts of a finetuned model's ref: unreserved in a URL
 * (RFC 3986), so never escaped, and in no model name, folder, file name or
 * URL-safe base64, so a ref without one is a plain ref, as every link made
 * before finetuned refs is. */
export const FINETUNED_SEPARATOR = "~";

/** A finetuned model as a link carries it: its name; the Hugging Face model
 * it was finetuned from (`base`, its ref as a plain ref names a model: a
 * folder of the built-in catalog, or `<owner>/<repo>/<folder>` of another
 * repo, a model published from a browser among them) and the file of that
 * model whose head was trained (`file`, the variant: its precision and
 * block); and the trained head, `weight` [outChannels × inChannels] as
 * Head.weight has it, and `bias`, one per output channel. */
export interface LinkedFinetuned {
  name: string;
  base: string;
  file: string;
  weight: Float32Array;
  bias: Float32Array;
}

/** What each part of a ref may hold: a name is a layer name and a URL path
 * segment; a base a folder or a repository path; a file an .onnx file's
 * name; the shape `<out>x<in>`; the numbers URL-safe base64. */
const NAME = /^[\w.-]+$/;
const BASE = /^[\w.-]+(?:\/[\w.-]+)*$/;
const FILE = /^[\w.-]+\.onnx$/;
const SHAPE = /^([1-9]\d*)x([1-9]\d*)$/;
const NUMBERS = /^[A-Za-z0-9_-]+$/;

/** `model`'s ref in a link: `<name>~<base>~<file>~<out>x<in>~<numbers>`,
 * the numbers the weight matrix then the biases as float32 little-endian
 * (the byte order ONNX keeps them in) in URL-safe base64 without padding:
 * exact, where decimals would round, and short (a 3 × 32 head is 99 numbers,
 * 528 characters). The shape is in the ref so that a ref cut short or
 * edited by hand is told from a good one without the base, and a base whose
 * head is another shape is refused by name. An fp16 model's head is carried
 * as float32 too: every float16 value is one, so it goes back exactly. */
export function finetunedRef(model: LinkedFinetuned): string {
  const out = model.bias.length;
  const inChannels = out ? model.weight.length / out : 0;
  if (!out || !Number.isInteger(inChannels) || inChannels < 1) throw new Error(`a head of ${model.weight.length} weights and ${out} biases is not [out, in] and [out]`);
  for (const [what, value, pattern] of [["name", model.name, NAME], ["base", model.base, BASE], ["file", model.file, FILE]] as const) {
    if (!pattern.test(value)) throw new Error(`the ${what} ${JSON.stringify(value)} cannot go in a link`);
  }
  const numbers = new Uint8Array(4 * (model.weight.length + model.bias.length));
  const view = new DataView(numbers.buffer);
  model.weight.forEach((w, i) => view.setFloat32(4 * i, w, true));
  model.bias.forEach((b, i) => view.setFloat32(4 * (model.weight.length + i), b, true));
  return [model.name, model.base, model.file, `${out}x${inChannels}`, bytesToBase64Url(numbers)].join(FINETUNED_SEPARATOR);
}

/** Whether `ref` is a finetuned model's ref rather than a plain one. */
export function isFinetunedRef(ref: string): boolean {
  return ref.includes(FINETUNED_SEPARATOR);
}

/** The model a finetuned ref names (finetunedRef's format), or an Error
 * saying what is wrong with one that is not (cut short, or edited). */
export function parseFinetunedRef(ref: string): LinkedFinetuned {
  const parts = ref.split(FINETUNED_SEPARATOR);
  if (parts.length !== 5) throw new Error(`a finetuned model's ref has five parts separated by ${FINETUNED_SEPARATOR}, and this has ${parts.length}`);
  const [name, base, file, shape, encoded] = parts;
  if (!NAME.test(name)) throw new Error(`${JSON.stringify(name)} is not a model name`);
  if (!BASE.test(base)) throw new Error(`${JSON.stringify(base)} is not a model to finetune from`);
  if (!FILE.test(file)) throw new Error(`${JSON.stringify(file)} is not an .onnx file`);
  const dims = SHAPE.exec(shape);
  if (!dims) throw new Error(`${JSON.stringify(shape)} is not a head's shape (<out>x<in>)`);
  const out = Number(dims[1]), inChannels = Number(dims[2]);
  if (!NUMBERS.test(encoded) || encoded.length % 4 === 1) throw new Error("the head's numbers are not URL-safe base64");
  const numbers = base64UrlToBytes(encoded);
  const expected = 4 * out * (inChannels + 1);
  if (numbers.byteLength !== expected) throw new Error(`the head's numbers are ${numbers.byteLength} bytes, where a ${out}×${inChannels} head and its biases take ${expected}`);
  const view = new DataView(numbers.buffer, numbers.byteOffset, numbers.byteLength);
  const weight = Float32Array.from({ length: out * inChannels }, (_, i) => view.getFloat32(4 * i, true));
  const bias = Float32Array.from({ length: out }, (_, i) => view.getFloat32(4 * (out * inChannels + i), true));
  return { name, base, file, weight, bias };
}

/** Whether a head has exactly the link's numbers, bit for bit (so -0 is not
 * 0, and a NaN is equal to itself): the test for a model already kept here. */
export function sameHead(head: Pick<Head, "weight" | "bias">, model: Pick<LinkedFinetuned, "weight" | "bias">): boolean {
  return sameBits(head.weight, model.weight) && sameBits(head.bias, model.bias);
}

function sameBits(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return x.every((bits, i) => bits === y[i]);
}

// ---------------------------------------------------------------------------
// The link's models in this browser
// ---------------------------------------------------------------------------

/** What the link's models need of the catalog: looking a ref up, and for a
 * finetuned ref the finetuned models kept, their bytes and the base's, and
 * keeping the rebuilt one. */
export type LinkCatalog = Pick<Catalog, "resolve" | "modelBytes" | "finetuned" | "addFinetuned">;

/** The models a link names, as this browser runs them. */
export interface LinkedModels {
  /** The catalog's entries, in the link's order, each once. */
  entries: ModelEntry[];
  /** The link's names of the models served here under another name, to
   * that name: this browser had a model of the link's name, with other
   * weights (a finetuned one kept here; or a Hugging Face model whose name
   * this list gives another, models/index.ts withUniqueNames). The linked
   * view's layers are renamed with them. */
  renamed: Record<string, string>;
  /** What the link names that this browser does not run, and why, in words
   * that follow "is left out: ". */
  leftOut: { name: string; why: string }[];
}

/** The catalog's entries for `refs`, in their order and each once: a plain
 * ref looked up, a finetuned one rebuilt (rebuildFinetuned), each run at
 * the voxel size after it (splitRunAt), or at its own. A ref the catalog
 * does not know, cannot look up, cannot read or cannot rebuild is logged
 * and left out, with why; the rest run. */
export async function resolveLinkedModels(
  catalog: LinkCatalog,
  refs: string[],
  log: (line: string) => void,
  options: { now?: () => Date } = {},
): Promise<LinkedModels> {
  const now = options.now ?? (() => new Date());
  const linked: LinkedModels = { entries: [], renamed: {}, leftOut: [] };
  const leave = (name: string, why: string) => {
    log(`The link's model ${name} is left out: ${why}`);
    linked.leftOut.push({ name, why });
  };
  // The link's own names for its finetuned models: one renamed here takes
  // none of them, so that each of the others keeps its own where it can.
  const reserved = refs.filter(isFinetunedRef).map((ref) => ref.split(FINETUNED_SEPARATOR)[0]);
  for (const linkedRef of refs) {
    let ref: string, runAt: number[] | null;
    try {
      ({ ref, runAt } = splitRunAt(linkedRef));
    } catch (e) {
      leave(linkedRef, `its voxel size cannot be read (${message(e)})`);
      continue;
    }
    let entry: ModelEntry | null;
    if (isFinetunedRef(ref)) {
      let model: LinkedFinetuned;
      try {
        model = parseFinetunedRef(ref);
      } catch (e) {
        leave(ref.split(FINETUNED_SEPARATOR)[0] || ref, `its part of the link cannot be read (${message(e)})`);
        continue;
      }
      try {
        entry = await rebuildFinetuned(catalog, model, { reserved, claimed: linked.entries, runAt, log, now });
      } catch (e) {
        leave(model.name, `it could not be rebuilt from ${model.base} (${message(e)})`);
        continue;
      }
      if (entry.name !== model.name) linked.renamed[model.name] = entry.name;
    } else {
      try {
        entry = await catalog.resolve(ref);
      } catch (e) {
        // A repo that has no browser copy of the model says why itself, and what to open instead.
        leave(ref, e instanceof NoBrowserCopyError ? e.message : `it could not be looked up (${message(e)})`);
        continue;
      }
      if (!entry) {
        leave(ref, "it is not a model this browser can run");
        continue;
      }
      // Listed here under another name than its own, which the sender's
      // view names its layer by.
      if (!isFinetuned(entry) && hubName(entry.manifest) !== entry.name) linked.renamed[hubName(entry.manifest)] = entry.name;
    }
    if (!linked.entries.some((x) => x.id === entry.id)) linked.entries.push(withRunVoxelSize(entry, runAt));
  }
  return linked;
}

/** The entry a link's finetuned model is served as here, and the model kept
 * if it was not already.
 *
 * Already kept: a model finetuned in this browser with exactly the link's
 * head (sameHead), named as the link names it or kept from an earlier
 * opening of a link that did (FinetunedRecord.from_link), is served as it
 * is; opening one link twice keeps one model. Else the base is looked up
 * (a Hugging Face model; a finetuned one would be this browser's own, not
 * the sender's, unless the link names it by where it was published from
 * here: those are the sender's bytes), its bytes are fetched in the link's
 * file when its manifest still has that file (else in the catalog's
 * default, said in the log), the link's numbers are written over its head,
 * whose shape must be theirs, and the result is kept as a finetuned model
 * whose record says it came from a link. It takes the link's name unless a
 * model kept here (or a Hugging Face model) has it already: a model of this
 * browser is never replaced, and the new one is the next of the base's
 * series instead (nextFinetunedName), passing over `reserved`, the link's
 * names for its other finetuned models. `claimed`, the entries resolved for
 * the link so far, are not served twice. With `runAt`, the voxel size after
 * the link's ref, the model was trained at that size, and is kept as a
 * model of that size (models/finetuned.ts finetunedEntry, on the base run
 * there). Throws what went wrong; nothing is kept then. */
export async function rebuildFinetuned(
  catalog: LinkCatalog,
  model: LinkedFinetuned,
  opts: { reserved: readonly string[]; claimed: readonly ModelEntry[]; runAt?: number[] | null; log: (line: string) => void; now: () => Date },
): Promise<ModelEntry> {
  const { log } = opts;
  const records = await catalog.finetuned();
  for (const record of records) {
    if (record.name !== model.name && record.from_link?.name !== model.name) continue;
    if (opts.claimed.some((e) => e.id === record.id)) continue;
    let head: Head;
    try {
      head = headOf(await catalog.modelBytes(record.entry));
    } catch {
      // Its bytes are gone or unreadable: it is not the link's model to
      // serve, and its name stays taken all the same.
      continue;
    }
    if (sameHead(head, model)) {
      log(`The link's ${model.name}: ${record.name}, kept in this browser, has its weights already`);
      return record.entry;
    }
  }
  const base = await catalog.resolve(model.base);
  if (!base) throw new Error(`${model.base} is not a model this browser can run`);
  const published = records.find((r) => r.id === base.id)?.published;
  if (isFinetuned(base) && (!published || publishedRef(published) !== model.base)) {
    throw new Error(`${model.base} is a model finetuned in this browser, not one from Hugging Face`);
  }
  const variant = base.manifest.variants.find((v) => v.file === model.file);
  if (!variant) log(`The link's ${model.name} was finetuned on ${base.name}'s ${model.file}, which its manifest no longer lists; rebuilding it on ${base.variant.file}`);
  const source = variant ? withVariant(base, variant) : base;
  const bytes = await catalog.modelBytes(source);
  const head = headOf(bytes);
  const out = model.bias.length, inChannels = model.weight.length / out;
  if (head.outChannels !== out || head.inChannels !== inChannels) {
    throw new Error(`the link's last layer is ${out}×${inChannels} and ${source.name}'s is ${head.outChannels}×${head.inChannels}`);
  }
  const data = withHead(bytes, head, model.weight, model.bias);
  // Taken by a model kept here, or by any entry the name would resolve to
  // (a Hugging Face model's folder or name, in a link edited by hand).
  const holder = await catalog.resolve(model.name).catch(() => null);
  const taken = holder !== null || records.some((r) => r.name === model.name);
  const name = taken ? nextFinetunedName(source, records, opts.reserved) : model.name;
  const record = finetunedRecord(withRunVoxelSize(source, opts.runAt ?? null), name, data.byteLength, { volume: "", created_at: opts.now().toISOString(), params: {}, epochs: 0, final_loss: null, from_link: { name: model.name } });
  await catalog.addFinetuned(record, data);
  log(taken
    ? `The link's ${model.name} is kept in this browser as ${name}, rebuilt from ${source.name}: this browser has a model named ${model.name} already, with other weights, and keeps it`
    : `The link's ${model.name} is kept in this browser, rebuilt from ${source.name}`);
  return record.entry;
}

// ---------------------------------------------------------------------------
// The link's view in this browser
// ---------------------------------------------------------------------------

/** A layer of a linked `#!` view that this browser does not show: "paint",
 * its paint volume (`volume`) is not in this browser's paint store, being
 * in the store of the browser that painted it; "model", its model is not
 * served here (left out of the link's models, or not named by the link). */
export interface DroppedLayer {
  name: string;
  why: "paint" | "model";
  volume?: string;
}

/** The paint volumes `state`'s paint layers paint into, each once. */
export function paintVolumesIn(state: NgState | null | undefined): string[] {
  const volumes = layersOf(state).map((layer) => paintVolumeOf(layer.source)).filter((v): v is string => v !== null);
  return [...new Set(volumes)];
}

/** `state`, a linked view, as this browser can show it, and the layers left
 * out. A paint layer whose volume is not among `volumes` (those this
 * browser's paint store holds) is left out. A model layer (one whose source
 * names a served view) first takes its new name from `renamed`, and is left
 * out when no model is served under it (`served`, the names of the session's
 * models) or another layer has it already; a renamed one is renamed in its
 * source's view and in the selection too, so the model served under the new
 * name takes the layer's place as it would its own (backend/serving.ts
 * showLayers sets each served model's layer by name, keeping a hidden
 * layer hidden and the shader the user set). Every other layer stays as it
 * is. `state` itself when nothing changes. */
export function linkedView(
  state: NgState,
  here: { served: ReadonlySet<string>; renamed: Readonly<Record<string, string>>; volumes: ReadonlySet<string> },
): { state: NgState; dropped: DroppedLayer[] } {
  const dropped: DroppedLayer[] = [];
  const layers: NgLayer[] = [];
  const names = new Set<string>();
  let changed = false;
  for (const layer of layersOf(state)) {
    const volume = paintVolumeOf(layer.source);
    if (volume !== null && !here.volumes.has(volume)) {
      dropped.push({ name: layer.name, why: "paint", volume });
      changed = true;
      continue;
    }
    if (volume === null && MARKED_VIEW.test(sourceUrl(layer.source) ?? "")) {
      const name = Object.hasOwn(here.renamed, layer.name) ? here.renamed[layer.name] : layer.name;
      if (!here.served.has(name) || names.has(name)) {
        dropped.push({ name: layer.name, why: "model" });
        changed = true;
        continue;
      }
      if (name !== layer.name) {
        layers.push({ ...layer, name, source: renamedView(layer.source, layer.name, name) });
        names.add(name);
        changed = true;
        continue;
      }
    }
    layers.push(layer);
    names.add(layer.name);
  }
  if (!changed) return { state, dropped };
  const next: NgState = { ...state, layers };
  const selected = state.selectedLayer as { layer?: unknown } | undefined;
  if (selected && typeof selected === "object" && typeof selected.layer === "string" && Object.hasOwn(here.renamed, selected.layer)) {
    next.selectedLayer = { ...selected, layer: here.renamed[selected.layer] };
  }
  return { state: next, dropped };
}

/** A model layer's source with the view it names renamed from `from` to
 * `to` (`.../<from>__CFLOW_ARGS__<blob>__CFLOW_ARGS__/`), the chain's blob
 * kept, whichever of Neuroglancer's forms the source is in. */
function renamedView(source: unknown, from: string, to: string): unknown {
  const respell = (url: string) => url.split(`/${from}${ARGS_KEY}`).join(`/${to}${ARGS_KEY}`);
  if (typeof source === "string") return respell(source);
  if (Array.isArray(source)) return source.map((entry) => renamedView(entry, from, to));
  if (source && typeof source === "object" && typeof (source as { url?: unknown }).url === "string") {
    return { ...(source as Record<string, unknown>), url: respell((source as { url: string }).url) };
  }
  return source;
}

/** linkedView over a `#!` hash (the page's, a link's), with `paint`, this
 * browser's paint store, asked which of the view's paint volumes it holds:
 * one does when any of its files is there (the Finetune tab's delete route
 * asks the same), as when the link's sender opens it in another tab. The
 * hash as this browser can show it (`hash` itself when nothing changed, and
 * for a hash that holds no state), and the layers left out, each logged
 * with why, as resolveLinkedModels logs the models it leaves out. */
export async function linkedHash(
  hash: string,
  here: { served: ReadonlySet<string>; renamed: Readonly<Record<string, string>>; paint: Pick<PaintStore, "list">; log: (line: string) => void },
): Promise<{ hash: string; dropped: DroppedLayer[] }> {
  const state = decodeState(hash);
  if (!state) return { hash, dropped: [] };
  const volumes = new Set<string>();
  for (const volume of paintVolumesIn(state)) {
    const files = await here.paint.list(volume).catch(() => [] as string[]);
    if (files.length) volumes.add(volume);
  }
  const view = linkedView(state, { served: here.served, renamed: here.renamed, volumes });
  for (const layer of view.dropped) here.log(layerNote(layer));
  return { hash: view.state === state ? hash : encodeState(view.state), dropped: view.dropped };
}

/** What the page says when a link did not open whole: a line for each
 * model left out, each model renamed and each layer left out of the view,
 * the why under an "i"; none when it opened as it was made. */
export function linkNotes(models: Pick<LinkedModels, "renamed" | "leftOut">, dropped: readonly DroppedLayer[]): Hint[] {
  const notes: Hint[] = models.leftOut.map(({ name, why }) => ({ short: `The link's model ${name} is left out.`, more: `Why: ${why}.` }));
  for (const [from, to] of Object.entries(models.renamed)) {
    notes.push({ short: `The link's ${from} is served here as ${to}.`, more: `This browser has a model named ${from} already, with other weights, and keeps it.` });
  }
  for (const layer of dropped) {
    notes.push(layer.why === "paint" ? { short: `Layer ${layer.name} is left out of the view.`, more: `Why: ${paintKept(layer)}.` } : { short: layerNote(layer) });
  }
  return notes;
}

/** Why a layer of a link's view is left out, as a line for the log. */
function layerNote(layer: DroppedLayer): string {
  return layer.why === "paint"
    ? `Layer ${layer.name} is left out of the view: ${paintKept(layer)}.`
    : `Layer ${layer.name} is left out of the view: its model is not served here.`;
}

function paintKept(layer: DroppedLayer): string {
  return `its paint (${layer.volume}) is kept in the browser the link was made in, and this browser has no copy of it`;
}
