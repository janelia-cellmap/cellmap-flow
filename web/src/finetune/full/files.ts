// A 2D folder's whole-network finetune files, fetched from Hugging Face as
// the catalog fetches the manifests (models/hf.ts rawUrl, fetchJson): the
// training graph (finetune/train_graph.json, checked by graph.ts) and the
// initializers' offsets in each .onnx file (finetune/initializers.json),
// once per folder and page. Both are a few tens of KB. Marwan's runtime,
// the folder's third file, is vendored (cellmap-tfjs-train.ts), not fetched.
import type { FullFinetuneSpec } from "../../contracts";
import { type FetchLike, fetchJson, rawUrl } from "../../models/hf";
import type { InitializerOffsets } from "./cellmap-tfjs-train";
import { parseTrainGraph, type TrainGraph } from "./graph";

export interface FullFiles {
  graph: TrainGraph;
  /** Per .onnx file of the folder, its initializers' offsets. */
  initializers: Record<string, InitializerOffsets>;
}

export type FetchFullFiles = (source: { repo: string; folder: string }, spec: FullFinetuneSpec) => Promise<FullFiles>;

const cache = new Map<string, Promise<FullFiles>>();

/** The graph's path among the spec's files. */
export function graphPath(spec: FullFinetuneSpec): string {
  const path = spec.files.find((f) => /(^|\/)train_graph\.json$/.test(f));
  if (!path) throw new Error(`the manifest's whole-network training names no train_graph.json among its files (${spec.files.join(", ")})`);
  return path;
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
  const [rawGraph, rawInitializers] = await Promise.all([
    fetchJson(fetchImpl, rawUrl(path, source.repo), path),
    fetchJson(fetchImpl, rawUrl(initializers, source.repo), initializers),
  ]);
  if (!rawInitializers || typeof rawInitializers !== "object" || Array.isArray(rawInitializers)) throw new Error(`${initializers} is not a JSON object of files`);
  return { graph: parseTrainGraph(rawGraph, path), initializers: rawInitializers as Record<string, InitializerOffsets> };
}
