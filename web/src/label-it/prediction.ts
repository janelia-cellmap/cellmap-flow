// The prediction under a box of the annotation volume: the port of
// cellmap_flow/dashboard/routes/finetune/view_labels.py's read_prediction
// for the browser, where the served model's output is the page's own.
//
// The server reads its model server's array with the dashboard's input
// normalization and no postprocessing: the trainer's target is set on the
// raw output, and that is where the model's decision boundary is. Here the
// same output is Inference.readOutput's, the model stage's (the chain's
// normalizers, none of its postprocessors), from the engine's caches when
// the viewer has had those chunks computed, so a seed copies the prediction
// the user is looking at without running the model again.
//
// Where the two grids meet: each annotation voxel takes the served voxel its
// centre lies in, so a seed from a model served at another voxel size (any
// served model can seed, not only the one the volume is for) lies under that
// model's layer. Both grids are placed by their OME translation, voxel 0's
// centre (the volume's dataset_offset_nm, the served level's origin), so a
// grid's corner is half a voxel before it. The box shrinks to what the
// served level covers; a box it does not reach at all is refused.
import type { DatasetInfo, Inference, ModelEntry, PaintVolume } from "../contracts";
import { volumeCornerNm } from "../ai-annotate/fill";
import { chooseRawLevel, geometryOf, type Level, outputLevel } from "../infer/geometry";
import type { Block } from "./segment";

/** The level a model's output is served at over `dataset`, in nm: the output
 * grid infer/engine.ts serves the model's view on, from the raw level
 * chooseRawLevel picks for its input voxel size (finetune/volume.ts plans a
 * volume on the same grid, which is how a volume's chunk is a chunk of its
 * model's layer). */
export function servedLevel(dataset: DatasetInfo, entry: ModelEntry): Level {
  const g = geometryOf(entry);
  return outputLevel(dataset.levels[chooseRawLevel(dataset.levels, g.input_voxel_size)], g);
}

/** Which served voxels the annotation voxels [lo, hi) read: `indices[a]`,
 * per axis, the served voxel each annotation voxel's centre lies in, for the
 * voxels the level covers, which are [lo, hi) of the answer (the box shrunk
 * to them). null when the level covers no voxel of the box along some axis
 * (read_prediction's "The view is outside what the model's server
 * predicts."). */
export function servedIndices(
  volume: Pick<PaintVolume, "output_voxel_size" | "dataset_offset_nm">,
  lo: readonly number[],
  hi: readonly number[],
  level: Level,
): { lo: number[]; hi: number[]; indices: number[][] } | null {
  const voxel = volume.output_voxel_size;
  const corner = volumeCornerNm(volume.dataset_offset_nm, voxel);
  // The server's corner of the served level (ome_corner of its translation)
  // rescaled from the voxel size declared to the one Neuroglancer draws it
  // at, `corner / declared * drawn`; the browser draws a layer at the size
  // it declares, and the same arithmetic keeps the floor below on the same
  // side of a voxel boundary as the Python's.
  const servedCorner = level.origin.map((o, a) => ((o - level.voxel[a] / 2) / level.voxel[a]) * level.voxel[a]);
  const newLo = [...lo], newHi = [...hi], indices: number[][] = [];
  for (let a = 0; a < 3; a++) {
    const inside: number[] = [];
    let first = -1, last = -1;
    for (let i = lo[a]; i < hi[a]; i++) {
      const centre = corner[a] + (i + 0.5) * voxel[a];
      const index = Math.floor((centre - servedCorner[a]) / level.voxel[a]);
      if (index < 0 || index >= level.shape[a]) continue;
      if (first < 0) first = i;
      last = i;
      inside.push(index);
    }
    if (!inside.length) return null;
    // The voxels inside are one run: the centres rise along the axis.
    newLo[a] = first;
    newHi[a] = last + 1;
    indices.push(inside);
  }
  return { lo: newLo, hi: newHi, indices };
}

/** The model's raw output over annotation voxels [lo, hi), as
 * read_prediction answers it: `prediction` [C, z, y, x] float32 over the
 * box shrunk to what the served level covers, and that box. `view` is the
 * served model's (ServedModel.view) and `level` its served level
 * (servedLevel). null when the level covers nothing of the box. */
export async function readPrediction(
  infer: Pick<Inference, "readOutput">,
  view: string,
  level: Level,
  volume: Pick<PaintVolume, "output_voxel_size" | "dataset_offset_nm">,
  lo: readonly number[],
  hi: readonly number[],
): Promise<{ prediction: Block; lo: number[]; hi: number[] } | null> {
  const placed = servedIndices(volume, lo, hi, level);
  if (!placed) return null;
  const { indices } = placed;
  // The served box the indices span, read once, then each annotation voxel
  // its served voxel (np.ix_ in the Python); with the two grids equal, the
  // whole region in order.
  const readLo = indices.map((ix) => ix[0]);
  const readHi = indices.map((ix) => ix[ix.length - 1] + 1);
  const region = await infer.readOutput(view, readLo, readHi);
  const [C, rz, ry, rx] = region.shape;
  const span = readHi.map((h, a) => h - readLo[a]);
  if (region.shape.length !== 4 || rz !== span[0] || ry !== span[1] || rx !== span[2]) {
    throw new Error(`asked for ${span.join("×")} voxels of the model's output, got a block of ${region.shape.join("×")}`);
  }
  const [iz, iy, ix] = indices.map((idx, a) => idx.map((v) => v - readLo[a]));
  const shape = [C, iz.length, iy.length, ix.length];
  const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  let k = 0;
  for (let c = 0; c < C; c++) {
    for (const z of iz) {
      for (const y of iy) {
        const row = ((c * rz + z) * ry + y) * rx;
        for (const x of ix) data[k++] = region.data[row + x];
      }
    }
  }
  return { prediction: { data, shape }, lo: placed.lo, hi: placed.hi };
}
