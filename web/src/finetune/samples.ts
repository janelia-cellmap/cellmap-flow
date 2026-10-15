// The training samples of a finetune: each painted chunk of the volume with
// the model's input block around it, read from the raw level and normalized
// as the model stage does (infer/stages.ts modelInput). The port of
// cellmap_flow/finetune/data/reader.py for the browser. See
// PLAN_browser_finetune.md, "Samples".
import type { FinetuneRun, Sample } from "./types";

/** The samples of `run`'s volume, one per painted chunk, in chunk order. */
// eslint-disable-next-line require-yield
export async function* collectSamples(_run: Pick<FinetuneRun, "volume" | "entry" | "chain" | "dataset" | "infer" | "paint">): AsyncGenerator<Sample> {
  throw new Error("collectSamples is not implemented yet");
}
