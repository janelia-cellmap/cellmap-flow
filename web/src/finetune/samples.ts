// The training samples of a finetune: each painted chunk of the volume with
// the model's input block around it, read from the raw level and normalized
// as the model stage does (infer/stages.ts modelInput). The port of
// cellmap_flow/finetune/data/reader.py (PatchReader) for the browser. See
// PLAN_browser_finetune.md, "Samples".
//
// A sample must be, byte for byte, the block the model stage feeds the model
// when it serves a chunk of the layer, or the head would train on inputs it
// never sees at inference. So the steps here are the engine's for a stage's
// input (web/chunkmirage/web/src/engine.ts: stagedChunk, input), with the
// same functions where there are functions to share:
//
//   - the chunk's box on the served output grid, clipped to the level
//     (a chunk the level ends in is smaller than the block);
//   - the input box on the raw level: the chunk's box scaled by the ratio
//     of the grids and grown by the model's halo (chunkmirage's inputBox);
//   - the raw read where the box lies inside the level (Inference.readRaw,
//     the engine's reader) and zero past it (chunkmirage's padded with the
//     "zero" padding the engine serves the models with, as the Python
//     server's Pipeline(padding="zero"));
//   - that block converted to float32, the voxels inside the level
//     normalized through the volume's input_norm and the padding left at
//     zero, placed in a block of the model's input shape (modelInput).
//
// The normalizers are the volume's (PaintVolume.input_norm, the chain when
// the volume was made), as the server's PatchReader normalizes through the
// session's own input_norm_config: the paint was drawn over the prediction
// that chain produced.
import { inputBox, padded, type StageAt } from "../../chunkmirage/web/src/stages";
import { chooseRawLevel, geometryOf, halo as haloOf, inputShape, outputLevel, outputShape, ratio } from "../infer/geometry";
import { modelInput } from "../infer/stages";
import type { FinetuneRun, Sample } from "./types";
import { paintedChunkIndices, readPaintedChunk } from "./volume";

/** The samples of `run`'s volume, one per painted chunk, in chunk order
 * (the store's sorted listing). A chunk whose index lies past the served
 * level is skipped, as the engine never serves one; a chunk deleted between
 * the listing and its read is skipped too. The volume's chunk must be the
 * model's output block, or the samples would not be the blocks the model
 * stage computes. */
export async function* collectSamples(run: Pick<FinetuneRun, "volume" | "entry" | "chain" | "dataset" | "infer" | "paint">): AsyncGenerator<Sample> {
  const { volume, dataset, entry } = run;
  // The grid the model is served on, as engine.ts's serve() finds it: the
  // raw level the model reads, and the output level over it.
  const g = geometryOf(entry);
  const rawLevel = chooseRawLevel(dataset.levels, g.input_voxel_size);
  const raw = dataset.levels[rawLevel];
  const level = outputLevel(raw, g);
  const chunk = outputShape(g);
  if (!sameShape(volume.chunk_size, chunk)) {
    throw new Error(`the volume's chunks are ${volume.chunk_size.join("×")} voxels, but ${entry.name} (${entry.variant.variant}) writes ${chunk.join("×")}: the volume was made for another block size`);
  }
  const halo = haloOf(g), r = ratio(g), inShape = inputShape(g);
  const spec = { normalizers: volume.input_norm, rawShape: raw.shape };

  for (const index of await paintedChunkIndices(run.paint, volume.id)) {
    if (index.length !== chunk.length) continue;  // not a chunk of a (z, y, x) array
    // The chunk's box on the output level, clipped to it as the engine clips
    // a chunk (Engine.stagedChunk); nothing to compute past the level.
    const lo = index.map((i, a) => i * chunk[a]), hi = lo.map((o, a) => Math.min(o + chunk[a], level.shape[a]));
    if (hi.some((h, a) => h <= lo[a])) continue;
    const painted = await readPaintedChunk(run.paint, volume, index);
    if (!painted) continue;
    // The input box on the raw level and what lies inside it, read; the rest
    // zero (Engine.input). Nothing is read for a box wholly outside.
    const { lo: inLo, hi: inHi } = inputBox(lo, hi, r, halo);
    const readLo = inLo.map((o) => Math.max(o, 0)), readHi = inHi.map((h, a) => Math.min(h, raw.shape[a]));
    const data = readHi.every((h, a) => h > readLo[a]) ? await run.infer.readRaw(rawLevel, readLo, readHi) : null;
    const block = padded(data, dataset.dtype, dataset.channels, readLo, readHi, inLo, inHi, "zero");
    // The stage's view of the chunk: only its box matters to modelInput (the
    // view and level name the engine's cache keys, which no sample has).
    const at: StageAt = { view: "", level: 0, index, lo, hi, voxel: level.voxel };
    const input = modelInput(spec, block, at, halo, r, inShape);
    yield { chunk: painted, input, inputDims: [1, 1, ...inShape] };
  }
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
