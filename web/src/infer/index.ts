// Inference: chunkmirage's Engine with one serve() view per model, whose
// produce() runs onnxruntime-web. See engine.ts for the pieces.
import type { Inference, SessionStore } from "../contracts";
import { InferenceEngine } from "./engine";

/** The page's inference. Nothing heavy happens here: the Engine starts with
 * the first openDataset(), and a model's ORT session with its first chunk. */
export async function createInference(opts: { session: SessionStore; log: (line: string) => void }): Promise<Inference> {
  return new InferenceEngine(opts);
}
