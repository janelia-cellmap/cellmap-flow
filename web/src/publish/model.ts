// A model finetuned in this browser as a publish of it says it: the
// Publication that cellmap_flow/portable/publish.py, the Python both modes
// run, writes the model card, the manifest's "published" block and the
// app's links from (the publish route calls it through
// portable/publish_calls.py). What is here is what only this browser's
// records say: how the model came to be here, what it was finetuned from
// and how.
//
// <folder> is the model's name (its record's, `<base name>-ft<n>`), and the
// manifest is the record's entry's, its one variant the bytes the browser
// serves for it, which the route uploads as `<folder>/<file>.onnx`.
import type { FinetunedRecord, ModelManifest } from "../contracts";
import { PYTHON } from "../generated/python-data";
import { lineage } from "../models/finetuned";

/** A publish as cellmap_flow/portable/publish.py takes it (its `pub`, whose
 * docstring says each field): where it goes and when, the manifest whose
 * variants are the files uploaded, how the weights came to be here, the
 * training as far as it is known (null where not), and the dataset's URL
 * to name, or null to name none. */
export interface Publication {
  repo: string;
  folder: string;
  published_at: string;
  manifest: ModelManifest;
  origin: "trained" | "link" | "file";
  where: "browser" | "server";
  finetuned_from: string | null;
  finetune_mode: string | null;
  epochs: number | null;
  final_loss: number | null;
  trained_at: string | null;
  app_version: string;
  dataset: string | null;
}

/** The publish of `record` to `folder` in `repo` (`owner/name`) at
 * `publishedAt`, naming `dataset` (or none), with what it was finetuned
 * from traced through `records` (finetunedFrom). A model trained here has
 * its annotation volume, and its training is its record's; one that came
 * in a shared link or a file was trained elsewhere, and its record knows
 * nothing of that training. */
export function publicationOf(
  record: FinetunedRecord,
  records: readonly FinetunedRecord[],
  at: { repo: string; folder: string; publishedAt: string; dataset: string | null },
): Publication {
  const trained = Boolean(record.volume);
  return {
    repo: at.repo,
    folder: at.folder,
    published_at: at.publishedAt,
    manifest: record.entry.manifest,
    origin: trained ? "trained" : record.from_link ? "link" : "file",
    where: "browser",
    finetuned_from: finetunedFrom(record, records) ?? null,
    finetune_mode: finetuneModeOf(record) ?? null,
    epochs: trained ? record.epochs : null,
    final_loss: trained ? record.final_loss : null,
    trained_at: trained ? record.created_at : null,
    app_version: PYTHON.version,
    dataset: at.dataset,
  };
}

/** The ref of the model `record` was finetuned from: its base's id (a
 * Hugging Face model's id is its ref), or, for a base finetuned in this
 * browser too, that model's ref where it was published, and failing that
 * the ref of the model it came from, as far back as `records` go. A model
 * that was neither published nor is kept any more leaves its id. A model
 * opened from a file (its base has no id) was finetuned from what its own
 * manifest's "published" block says, if anything. */
export function finetunedFrom(record: FinetunedRecord, records: readonly FinetunedRecord[]): string | undefined {
  const { ancestry, base, published } = lineage(record, records);
  return published ?? (base.id || ancestry[ancestry.length - 1].entry.manifest.published?.finetuned_from);
}

/** How `record` was trained, where it says: its whole network ("full") or
 * its last layer ("head") in this browser, or as a published manifest it
 * came with says; undefined for a model whose training was elsewhere and
 * is not said. */
function finetuneModeOf(record: FinetunedRecord): string | undefined {
  const mode = record.params.finetune_mode;
  return mode === "full" || mode === "head" ? mode : record.entry.manifest.published?.finetune_mode;
}
