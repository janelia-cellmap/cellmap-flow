// A link to the page that opens a volume and runs models on it: the query
// ?dataset=<url>&model=<ref>&model=<ref> (a model's ref is a name such as
// setup_53, or the repository path cellmap/onnx-web-models/setup_53), as the
// README's example link has it. main.ts reads it once at boot, after the
// service worker's first-visit reload (which keeps the URL): the dataset and
// the models go into the tab's session, where Connect and Submit would have
// put them, so the session's restore opens the volume and serves the models
// as after any reload; and the query is taken off the address bar, so a
// later tab completed from this URL starts fresh, as a `#!` state link's is
// (viewer/index.ts). A model the catalog does not know is logged and left
// out; the rest run.
import type { ModelEntry } from "./contracts";

export const DATASET_PARAM = "dataset";
export const MODEL_PARAM = "model";

export interface LinkRequest {
  /** The volume to open, or null when the link names none. */
  datasetUrl: string | null;
  /** The models to run, as given, each once. */
  models: string[];
}

/** What `search` (location.search) asks for: repeated `model` parameters
 * and comma-separated lists both work. */
export function parseLink(search: string): LinkRequest {
  const params = new URLSearchParams(search);
  const datasetUrl = params.get(DATASET_PARAM)?.trim() || null;
  const models = params.getAll(MODEL_PARAM).flatMap((m) => m.split(",")).map((m) => m.trim()).filter(Boolean);
  return { datasetUrl, models: [...new Set(models)] };
}

/** `href` without the link's parameters; anything else in the URL stays. */
export function withoutLink(href: string): string {
  const url = new URL(href);
  url.searchParams.delete(DATASET_PARAM);
  url.searchParams.delete(MODEL_PARAM);
  return url.href;
}

/** The catalog's entries for `refs`, in their order and each once; a ref the
 * catalog does not know, or cannot look up, is logged and left out. */
export async function resolveLinkedModels(
  catalog: { resolve(ref: string): Promise<ModelEntry | null> },
  refs: string[],
  log: (line: string) => void,
): Promise<ModelEntry[]> {
  const entries: ModelEntry[] = [];
  for (const ref of refs) {
    const entry = await catalog.resolve(ref).catch((e: unknown) => {
      log(`The link's model ${ref} could not be looked up: ${e instanceof Error ? e.message : String(e)}`);
      return null;
    });
    if (!entry) {
      log(`The link names ${ref}, which is not a model the browser can run; it is left out`);
      continue;
    }
    if (!entries.some((x) => x.id === entry.id)) entries.push(entry);
  }
  return entries;
}
