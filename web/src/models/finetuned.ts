// Models finetuned in this browser: their records (IndexedDB) and the
// entries they are served as, beside the Hugging Face catalog's
// (index.ts). See PLAN_browser_finetune.md, "The finetuned model as a
// catalog entry".
import type { FinetunedRecord, ModelEntry } from "../contracts";

/** The entry a model finetuned from `base` is served as, named `name`. */
export function finetunedEntry(_base: ModelEntry, _name: string): ModelEntry {
  throw new Error("finetunedEntry is not implemented yet");
}

/** The name for the next model finetuned from `base`: `<base name>-ft<n>`, n past those `existing` already has. */
export function nextFinetunedName(_base: ModelEntry, _existing: FinetunedRecord[]): string {
  throw new Error("nextFinetunedName is not implemented yet");
}
