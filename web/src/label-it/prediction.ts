// The prediction under a box of the annotation volume, in the browser: where
// it is read and the reading, around the Python both modes run
// (cellmap_flow/portable/label_it.py's placement, through label_it_calls.py
// in the engine's Pyodide worker), which says which served voxel lies under
// each annotation voxel. The server reads its model server's array the same
// way (view_labels.read_prediction).
//
// The served model's output is the page's own here: Inference.readOutput's,
// the model stage's (the chain's normalizers, none of its postprocessors, as
// the server reads its model server's array with no postprocessing: the
// trainer's target is set on the raw output, and that is where the model's
// decision boundary is), from the engine's caches when the viewer has had
// those chunks computed, so a seed copies the prediction the user is looking
// at without running the model again.
import type { DatasetInfo, Inference, ModelEntry, PaintVolume } from "../contracts";
import { chooseRawLevel, geometryOf, type Level, outputLevel } from "../infer/geometry";
import { type Block, LABEL_IT } from "./segment";

/** The level a model's output is served at over `dataset`, in nm: the output
 * grid infer/engine.ts serves the model's view on, from the raw level
 * chooseRawLevel picks for its input voxel size (finetune/volume.ts plans a
 * volume on the same grid, which is how a volume's chunk is a chunk of its
 * model's layer). */
export function servedLevel(dataset: DatasetInfo, entry: ModelEntry): Level {
  const g = geometryOf(entry);
  return outputLevel(dataset.levels[chooseRawLevel(dataset.levels, g.input_voxel_size)], g);
}

/** Which served voxels the annotation voxels of a box read
 * (label_it.placement): `indices[a]`, per axis, the served voxel each
 * annotation voxel's centre lies in, for the voxels the level covers, which
 * are [lo, hi) (the box shrunk to them); and [read_lo, read_hi), the box of
 * the level those indices span, the one to read. */
export interface Placement {
  lo: number[];
  hi: number[];
  indices: number[][];
  read_lo: number[];
  read_hi: number[];
}

/** The box and the two grids as label_it_calls' placement takes them: the
 * annotation voxels [lo, hi) of `volume`, on the served `level` (its origin
 * is its OME translation, voxel 0's centre, as the volume's
 * dataset_offset_nm is). */
export function placementArgs(volume: Pick<PaintVolume, "output_voxel_size" | "dataset_offset_nm">, lo: readonly number[], hi: readonly number[], level: Level) {
  return {
    lo: [...lo],
    hi: [...hi],
    volume: { output_voxel_size: volume.output_voxel_size, dataset_offset_nm: volume.dataset_offset_nm },
    level: { origin: level.origin, voxel: level.voxel, shape: level.shape },
  };
}

/** Where the annotation voxels [lo, hi) of `volume` lie on `level`
 * (placementArgs), by the Python; null when the level covers no voxel of the
 * box along some axis. */
export function place(
  infer: Pick<Inference, "python">,
  volume: Pick<PaintVolume, "output_voxel_size" | "dataset_offset_nm">,
  lo: readonly number[],
  hi: readonly number[],
  level: Level,
): Promise<Placement | null> {
  return infer.python<Placement | null>(`${LABEL_IT}:place`, placementArgs(volume, lo, hi, level));
}

/** The served model's raw output over the box `placement` says to read:
 * [C, z, y, x] float32, for the Python to put on the annotation grid
 * (label_it.gather, in label_it_calls.seed). `view` is the served model's
 * (ServedModel.view). */
export async function readRegion(infer: Pick<Inference, "readOutput">, view: string, placement: Placement): Promise<Block> {
  const region = await infer.readOutput(view, placement.read_lo, placement.read_hi);
  const span = placement.read_hi.map((h, a) => h - placement.read_lo[a]);
  const [, rz, ry, rx] = region.shape;
  if (region.shape.length !== 4 || rz !== span[0] || ry !== span[1] || rx !== span[2]) {
    throw new Error(`asked for ${span.join("×")} voxels of the model's output, got a block of ${region.shape.join("×")}`);
  }
  return region;
}
