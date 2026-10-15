// Pure functions over a Neuroglancer state's JSON for a segmentation
// layer's `equivalences`: the ids a merging step has joined across chunks
// (SimpleBlockwiseMerger, src/infer/merger.ts), shown as one object. The
// browser side of routes/pipeline.py's update_equivalences, which sets
// `layer.equivalences` on the layer whose source ends with the dataset;
// here the layer is found by name (a served view is named after its
// model, and so is its layer). No DOM: the controller (index.ts) reads the
// state, puts it through here and restores it.
import { type NgLayer, type NgState, layersOf } from "./state";

/** `state` with `sets` as the equivalences of the segmentation layer
 * named `name` (each set the ids that are one object, the shape the layer's
 * JSON takes and the Python's to_json() gives), replacing what it had; null
 * when the state has no segmentation layer of that name (gone, or not drawn
 * yet), so the caller can leave the viewer alone. The ids go in as decimal
 * strings: Neuroglancer reads a uint64 from a number or a string, and a
 * string is exact. The other layers and the rest of the state pass through. */
export function withEquivalences(state: NgState, name: string, sets: readonly (readonly string[])[]): NgState | null {
  const layers = layersOf(state);
  const index = layers.findIndex((layer) => layer.name === name && layer.type === "segmentation");
  if (index < 0) return null;
  const updated: NgLayer[] = layers.map((layer) => ({ ...layer }));
  updated[index] = { ...layers[index], equivalences: sets.map((set) => set.map(String)) };
  return { ...state, layers: updated };
}

/** The equivalences the layer `name` shows in `state`, as Neuroglancer
 * writes them (lists of ids, numbers or strings), or null when the layer
 * has none or is not there. */
export function equivalencesOf(state: NgState | null | undefined, name: string): (string | number)[][] | null {
  const layer = layersOf(state).find((l) => l.name === name);
  const equivalences = layer?.equivalences;
  if (!Array.isArray(equivalences)) return null;
  return equivalences.filter((group): group is (string | number)[] => Array.isArray(group));
}
