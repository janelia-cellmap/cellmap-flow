// A folder's whole-network finetune files, fetched from Hugging Face as the
// catalog fetches the manifests (models/hf.ts rawUrl, fetchJson): the
// training graph (finetune/train_graph.json, checked by graph.ts) and the
// initializers' offsets in each .onnx file (finetune/initializers.json),
// once per folder and page. Both are a few tens of KB (initializers.json of
// a 3D folder 190 KB: twelve files). Marwan's runtime, the folder's other
// file, is vendored (cellmap-tfjs-train.ts), not fetched.
//
// A 3D folder's graph names its params as the PyTorch state_dict, not as the
// ONNX initializers, since the export splits a Conv3d into a convolution per
// depth tap: two more files say how to rebuild each param from the
// initializers (torch_params.json, its from_onnx recipes) and how to cut it
// back into them (weight_map.json, its to_onnx recipes). The manifest names
// both (`torch_params`, and `weight_map` as a path where a 2D folder says
// "identity").
import type { FullFinetuneSpec } from "../../contracts";
import { type FetchLike, fetchJson, rawUrl } from "../../models/hf";
import type { InitializerOffsets, readTorchParams, toOnnxWeights } from "./cellmap-tfjs-train";
import { parseTrainGraph, type TrainGraph } from "./graph";

/** torch_params.json: per PyTorch param, its shape and its recipe from the initializers. */
export type TorchParams = Parameters<typeof readTorchParams>[2];
/** weight_map.json: per ONNX initializer, the PyTorch param it is cut from and how. */
export type WeightMap = Parameters<typeof toOnnxWeights>[0];

export interface FullFiles {
  graph: TrainGraph;
  /** Per .onnx file of the folder, its initializers' offsets. */
  initializers: Record<string, InitializerOffsets>;
  /** A 3D folder's (the manifest's torch_params and weight_map); null for a 2D one's. */
  torchParams: TorchParams | null;
  weightMap: WeightMap | null;
}

export type FetchFullFiles = (source: { repo: string; folder: string }, spec: FullFinetuneSpec) => Promise<FullFiles>;

const cache = new Map<string, Promise<FullFiles>>();

/** The graph's path among the spec's files. */
export function graphPath(spec: FullFinetuneSpec): string {
  const path = spec.files.find((f) => /(^|\/)train_graph\.json$/.test(f));
  if (!path) throw new Error(`the manifest's whole-network training names no train_graph.json among its files (${spec.files.join(", ")})`);
  return path;
}

/** Whether `spec` names a structured weight map, a 3D folder's: torch_params
 * and weight_map as files, where a 2D folder's weight_map is "identity". */
export function hasStructuredWeights(spec: FullFinetuneSpec): boolean {
  return !!spec.torch_params && !!spec.weight_map && spec.weight_map !== "identity";
}

/** The files of `source`'s folder, from the cache once fetched; a failed
 * fetch is not kept, so the next call tries again. */
export function fetchFullFiles(source: { repo: string; folder: string }, spec: FullFinetuneSpec, fetchImpl: FetchLike = (input, init) => globalThis.fetch(input, init)): Promise<FullFiles> {
  const key = `${source.repo}/${source.folder}`;
  let files = cache.get(key);
  if (!files) {
    files = load(source, spec, fetchImpl);
    cache.set(key, files);
    files.catch(() => cache.delete(key));
  }
  return files;
}

async function load(source: { repo: string; folder: string }, spec: FullFinetuneSpec, fetchImpl: FetchLike): Promise<FullFiles> {
  const path = `${source.folder}/${graphPath(spec)}`, initializers = `${source.folder}/${spec.initializers}`;
  const structured = hasStructuredWeights(spec);
  const torchPath = structured ? `${source.folder}/${spec.torch_params}` : null, mapPath = structured ? `${source.folder}/${spec.weight_map}` : null;
  const get = (file: string | null) => (file ? fetchJson(fetchImpl, rawUrl(file, source.repo), file) : Promise.resolve(null));
  const [rawGraph, rawInitializers, rawTorch, rawMap] = await Promise.all([get(path), get(initializers), get(torchPath), get(mapPath)]);
  return {
    graph: parseTrainGraph(rawGraph, path),
    initializers: objectOf(rawInitializers, initializers, "files") as Record<string, InitializerOffsets>,
    torchParams: torchPath ? (objectOf(rawTorch, torchPath, "params") as TorchParams) : null,
    weightMap: mapPath ? (objectOf(rawMap, mapPath, "initializers") as WeightMap) : null,
  };
}

/** `raw` as a JSON object, else an Error naming the file and what it should hold. */
function objectOf(raw: unknown, file: string, of: string): Record<string, unknown> {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`${file} is not a JSON object of ${of}`);
  return raw as Record<string, unknown>;
}
