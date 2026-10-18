// The ways the browser can finetune a served model, from what its manifest
// declares (ModelManifest.finetune), as the server's finetune_modes() says
// which ways a served PyTorch model takes: the Finetune tab offers exactly
// these (GET /api/finetune/models), and submit refuses any other.
//
//   head  the last 1×1 convolution, trained on the features of the painted
//         voxels (run.ts): offered when the served file is one of the
//         head's `files` (fp32 and fp16; an int8w file's weights feed a
//         DequantizeLinear and cannot be written);
//   full  the whole network, trained with TensorFlow.js from the folder's
//         training graph (full/run-full.ts): offered when the library is
//         "tfjs", the served file is one of `write_back`, and the model
//         reads one slice. Since 2026-10-17 the 3D models' manifests declare
//         it too, with NDHWC graphs that full/ does not build yet, so a 3D
//         model keeps the head until it does.
//
// A manifest with no finetune block (written before 2026-10-16, or by
// another exporter) keeps the head as it always had it: the browser finds
// the last convolution in the graph itself (onnx-head.ts headOf), and a
// graph that has none fails the run with the reason.
import type { ModelEntry } from "../contracts";
import { geometryOf, inputShape } from "../infer/geometry";

export type FinetuneMode = "head" | "full";

/** The Hugging Face file `entry`'s bytes are, or were made from: a model
 * finetuned in this browser names its base's (ModelVariant.source), whose
 * layout its bytes keep and whose folder holds its finetune files. */
export function sourceOf(entry: ModelEntry): { repo: string; folder: string; file: string } {
  return entry.variant.source ?? { repo: entry.repo, folder: entry.folder, file: entry.variant.file };
}

/** The ways `entry` can be finetuned in the browser, the head first. */
export function finetuneModes(entry: ModelEntry): FinetuneMode[] {
  const spec = entry.manifest.finetune;
  if (!spec) return ["head"];
  const { file } = sourceOf(entry);
  const modes: FinetuneMode[] = [];
  if (spec.head?.files.includes(file)) modes.push("head");
  if (spec.full?.library === "tfjs" && spec.full.write_back.includes(file) && readsOneSlice(entry)) modes.push("full");
  return modes;
}

/** Whether `entry` reads one slice, a 2D model, whose whole network full/ can build. */
function readsOneSlice(entry: ModelEntry): boolean {
  try {
    return inputShape(geometryOf(entry))[0] === 1;
  } catch {
    return false;
  }
}

/** What each mode trains, in the tab's words. */
export const MODE_WORDS: Record<FinetuneMode, string> = { head: "the last layer", full: "the whole network" };
export const MODE_NOUN: Record<FinetuneMode, string> = { head: "head", full: "network" };

/** Why `entry` cannot be trained as `mode`, in a sentence, or null when it can. */
export function refuseMode(entry: ModelEntry, mode: FinetuneMode): string | null {
  const modes = finetuneModes(entry);
  if (modes.includes(mode)) return null;
  const { file } = sourceOf(entry);
  if (!modes.length) {
    return `${entry.name} cannot be finetuned in the browser: its manifest declares no training for ${file}${/int8w/.test(file) ? " (an int8w file's weights are quantized and cannot be written; serve the fp32 or fp16 file)" : ""}.`;
  }
  const offered = modes.map((m) => MODE_WORDS[m]).join(" or ");
  const why = mode !== "full" ? "" : entry.manifest.finetune?.full && !readsOneSlice(entry)
    ? " (the browser trains a 3D model's whole network not yet, only a 2D model's)"
    : ` (its manifest has no whole-network training for ${file})`;
  return `${entry.name} trains ${offered} in the browser, not ${MODE_WORDS[mode]}${why}.`;
}
