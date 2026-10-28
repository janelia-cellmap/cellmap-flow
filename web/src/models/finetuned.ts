// Models finetuned in this browser: their records (IndexedDB) and the
// entries they are served as, beside the Hugging Face catalog's
// (index.ts). See PLAN_browser_finetune.md, "The finetuned model as a
// catalog entry".
//
// A finetuned model is an ordinary ONNX file (the base's graph with its last
// layer's weights replaced; src/finetune) that exists only in this browser,
// so it is kept as two halves in the models database (byte-cache.ts): its
// bytes in the "bytes" store, under the key its entry's onnxUrl names, and
// its FinetunedRecord in the "finetuned" store, which says what it is (the
// entry, the base, the volume, the training). The record is what the
// catalog lists on the next visit; the bytes are what it serves.
//
// FinetunedRegistry is the record store as the catalog uses it;
// IndexedDbFinetunedRegistry is the browser's implementation and
// MemoryFinetunedRegistry stands in without IndexedDB (tests, or a browser
// with storage blocked: a model then lasts as long as the page).
import { FINETUNED_REPO } from "../contracts";
import type { FinetunedRecord, ModelEntry, ModelManifest, ModelVariant } from "../contracts";

import { sourceOf } from "../finetune/modes";
import { MODEL_DB_NAME, ModelDatabase } from "./byte-cache";
import { entryFor } from "./manifest";

export interface FinetunedRegistry {
  /** Every record kept, newest first. */
  list(): Promise<FinetunedRecord[]>;
  /** Keep `record`, replacing one with the same id. */
  put(record: FinetunedRecord): Promise<void>;
  /** Forget the record with `id`; nothing happens when there is none. */
  remove(id: string): Promise<void>;
}

/** Whether `entry` is a model finetuned in this browser. */
export function isFinetuned(entry: ModelEntry): boolean {
  return entry.repo === FINETUNED_REPO;
}

/** The entry a model finetuned from `base` is served as, named `name`
 * (nextFinetunedName): repo FINETUNED_REPO, id `finetuned/<name>`, folder
 * and name both `name`, the base's manifest under the new name with one
 * variant, the base's block sizes and precision in a file `<name>.onnx`,
 * and the Hugging Face file it was made from (ModelVariant.source).
 * `fileMb` is the ONNX file's size in MB (bytes / 1e6, as the manifests
 * give it); without it the base's is kept, which is right to the rounding
 * (the head is a few KB of the file).
 *
 * The variant keeps none of the base's fidelity numbers (max_abs and the
 * rest measure the base export against its PyTorch model), and the manifest
 * says in `checkpoint` where the weights came from, since the base's
 * checkpoint name no longer describes them; a published base's `published`
 * block is not this model's and is left out. */
export function finetunedEntry(base: ModelEntry, name: string, fileMb?: number): ModelEntry {
  const variant: ModelVariant = {
    variant: base.variant.variant,
    input_size: base.variant.input_size.slice(),
    output_size: base.variant.output_size.slice(),
    file: `${name}.onnx`,
    file_mb: fileMb === undefined ? base.variant.file_mb : Math.round(fileMb * 100) / 100,
    // The Hugging Face file at the root of its lineage, whose layout the
    // bytes keep: what the manifest's finetune block names (modes.ts).
    source: sourceOf(base),
  };
  const { published: _published, ...inherited } = base.manifest;
  const manifest: ModelManifest = {
    ...inherited,
    model: name,
    checkpoint: `finetuned in the browser from ${base.manifest.checkpoint || base.name}`,
    variants: [variant],
  };
  return entryFor(name, manifest, variant, FINETUNED_REPO);
}

/** The record of a model finetuned from `base` as `name`, whose ONNX file is `byteLength` bytes. */
export function finetunedRecord(base: ModelEntry, name: string, byteLength: number, history: Pick<FinetunedRecord, "volume" | "created_at" | "params" | "epochs" | "final_loss" | "from_link">): FinetunedRecord {
  const entry = finetunedEntry(base, name, byteLength / 1e6);
  return { id: entry.id, name, base: { id: base.id, name: base.name, file: base.variant.file }, ...history, entry };
}

/** The name for the next model finetuned from `base`: `<base name>-ft<n>`,
 * n one past the highest among `existing` for that base (a record whose
 * base is this entry, or whose name is `<base name>-ft<k>`); the first is
 * -ft1. A base that is itself finetuned (`x-ft1`) continues its series
 * (`x-ft2`, not `x-ft1-ft1`): the record keeps the exact lineage in `base`,
 * and the layer list reads as one series of attempts. `taken` are names
 * counted as used besides the records' (a shared link's names for its other
 * finetuned models, which they keep where they can: link.ts). */
export function nextFinetunedName(base: ModelEntry, existing: FinetunedRecord[], taken: readonly string[] = []): string {
  const series = base.name.replace(/-ft\d+$/, "");
  const inSeries = new RegExp(`^${escapeRegExp(series)}-ft(\\d+)$`);
  let highest = 0;
  for (const record of existing) {
    const match = inSeries.exec(record.name) ?? (record.base.id === base.id ? /-ft(\d+)$/.exec(record.name) : null);
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  for (const name of taken) {
    const match = inSeries.exec(name);
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  return `${series}-ft${highest + 1}`;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Newest first by created_at (ISO 8601), the name deciding a tie, so two
 * listings of the same records come in the same order. */
export function newestFirst(a: FinetunedRecord, b: FinetunedRecord): number {
  return time(b) - time(a) || b.created_at.localeCompare(a.created_at) || b.name.localeCompare(a.name);
}

function time(record: FinetunedRecord): number {
  const t = Date.parse(record.created_at);
  return Number.isNaN(t) ? 0 : t;
}

/** A registry that lasts as long as the page. */
export class MemoryFinetunedRegistry implements FinetunedRegistry {
  private readonly records = new Map<string, FinetunedRecord>();

  async list(): Promise<FinetunedRecord[]> {
    return [...this.records.values()].sort(newestFirst);
  }

  async put(record: FinetunedRecord): Promise<void> {
    this.records.set(record.id, record);
  }

  async remove(id: string): Promise<void> {
    this.records.delete(id);
  }
}

/** The IndexedDB registry: the "finetuned" store of the models database,
 * records keyed by id. A row that is not a record (a future shape, a
 * corrupted write) is left out of the listing rather than failing it. */
export class IndexedDbFinetunedRegistry implements FinetunedRegistry {
  private readonly db: ModelDatabase;

  constructor(factory: IDBFactory, name: string = MODEL_DB_NAME) {
    this.db = new ModelDatabase(factory, name);
  }

  async list(): Promise<FinetunedRecord[]> {
    const rows = await this.db.run("finetuned", "readonly", (store) => store.getAll() as IDBRequest<unknown[]>);
    return rows.filter(isRecord).sort(newestFirst);
  }

  async put(record: FinetunedRecord): Promise<void> {
    await this.db.run("finetuned", "readwrite", (store) => store.put(record, record.id));
  }

  async remove(id: string): Promise<void> {
    await this.db.run("finetuned", "readwrite", (store) => store.delete(id));
  }
}

// The fields the catalog reads of a row before trusting it as a record.
function isRecord(row: unknown): row is FinetunedRecord {
  if (!row || typeof row !== "object") return false;
  const r = row as Record<string, unknown>;
  const entry = r.entry as Record<string, unknown> | undefined;
  return (
    typeof r.id === "string" &&
    typeof r.name === "string" &&
    typeof r.created_at === "string" &&
    !!entry &&
    typeof entry === "object" &&
    typeof entry.id === "string" &&
    typeof entry.onnxUrl === "string" &&
    entry.repo === FINETUNED_REPO
  );
}

/** The registry for this runtime: IndexedDB when there is one, else memory.
 * No log line of its own: the byte cache's opener (openByteCache) notes a
 * missing models database once, for both halves. */
export function openFinetunedRegistry(factory: IDBFactory | undefined): FinetunedRegistry {
  return factory ? new IndexedDbFinetunedRegistry(factory) : new MemoryFinetunedRegistry();
}
