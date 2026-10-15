// Inference: chunkmirage's Engine with one serve() view per model, whose
// produce() runs onnxruntime-web. STUB: replaced by the infer module.
import type { Inference, SessionStore } from "../contracts";

export async function createInference(_opts: { session: SessionStore; log: (line: string) => void }): Promise<Inference> {
  throw new Error("infer/index.ts is not implemented yet");
}
