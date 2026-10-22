// Seed from Prediction's segmentation in the browser is the Python both modes
// run: cellmap_flow/portable/label_it.py (the request's options, the
// channels a seed reads, the probability and the components method), called
// through label_it_calls.py in the engine's Pyodide worker
// (Inference.python). backend/routes/view-labels.ts makes those calls; this
// file holds the one step the Python cannot take here, and the probability
// the rank reads.
//
// That step is the mutex watershed. mwatershed is Rust, with no Pyodide
// wheel, so for an affinity model the Python answers the watershed's input
// (label_it.affinities: the affinity channels as probabilities) and
// infer/affinity.ts's mutexWatershed, parity-tested against the server's
// segment.mutex_watershed (test/parity/label-it.test.ts), joins them here.
import type { Inference } from "../contracts";
import { mutexWatershed } from "../infer/affinity";

/** The Python's entry points for Label it, by module path. */
export const LABEL_IT = "cellmap_flow.portable.label_it_calls";

/** A block of values, channels first: `shape` is [C, z, y, x]. */
export interface Block {
  data: Float32Array;
  shape: number[];
}

/** A typed array's bytes to hand to the worker: its own buffer when it spans
 * all of it (transferred, so the caller's copy is gone after), else a copy. */
export function bufferOf(a: Float32Array): ArrayBuffer {
  const whole = a.byteOffset === 0 && a.byteLength === a.buffer.byteLength && a.buffer instanceof ArrayBuffer;
  return whole ? (a.buffer as ArrayBuffer) : (a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength) as ArrayBuffer);
}

/** The objects of the mutex watershed (segment.mutex_watershed's, as the
 * server's seed runs it): `affinities`, label_it.affinities' answer in [0, 1],
 * joined where the affinity is over `threshold` (the bias), fragments
 * below it background, objects of fewer than `minSize` voxels too; ids
 * 1..n over the block's z, y, x. */
export function mutexObjects(affinities: Block, offsets: number[][], threshold: number, minSize: number): Uint32Array {
  const labels = mutexWatershed(affinities, offsets, threshold, minSize);
  const out = new Uint32Array(labels.length);
  for (let i = 0; i < labels.length; i++) out[i] = Number(labels[i]);
  return out;
}

/** A model's output block as a probability (label_it.probability over every
 * channel), by the served model's output class (ModelInfo.output_class;
 * null: read off the values): 0.5 at the model's decision boundary,
 * whatever its activation. `block` is handed to the worker. */
export async function probability(infer: Pick<Inference, "python">, block: Block, outputClass: string | null | undefined): Promise<Block> {
  const bytes = await infer.python<ArrayBuffer>(`${LABEL_IT}:probability`, { shape: block.shape, output_class: outputClass ?? null }, [bufferOf(block.data)]);
  return { data: new Float32Array(bytes), shape: block.shape };
}
