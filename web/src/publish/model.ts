// What a model finetuned in this browser looks like on Hugging Face once
// published (the files the publish route uploads, in one commit):
//
//   README.md                 the model card (modelCard)
//   <folder>/manifest.json    the model's own manifest, the one its catalog
//                             entry holds (one variant: the trained file), with
//                             a "published" block on top (publishedManifest)
//   <folder>/<file>.onnx      the bytes the browser serves for it
//
// <folder> is the model's name (its record's, `<base name>-ft<n>`). The
// manifest is the catalog's format, so the folder is a model like any of
// cellmap/onnx-web-models': the browser's catalog loads it by its ref,
// `<owner>/<repo>/<folder>`, and the server takes the same ref as a
// `type: onnx` model. The "published" block says what the manifest alone
// would not: the model's own name (the base's manifest under it names the
// base's task and model, from which entryName would give the base's name),
// what it was finetuned from (a ref) and how, and when.
import { FINETUNED_REPO } from "../contracts";
import type { FinetunedRecord, ModelManifest } from "../contracts";
import { PYTHON } from "../generated/python-data";
import { MODEL_PARAM } from "../link";
import { voxelLabel } from "../models/manifest";
import { shareLink } from "../share-link";

/** The app a published model's card links to: the public site's, wherever
 * the model was published from (a development server's address means
 * nothing to the card's readers). */
export const PUBLIC_APP = "https://janelia-cellmap.github.io/cellmap-flow/app/";

/** Hugging Face's site, where a repository's page is `<HUB>/<owner>/<repo>`. */
export const HUB = "https://huggingface.co";

/** The manifest's "published" block. `dataset` only when the user asked
 * for it: a dataset's URL can be internal. */
export interface PublishedBlock {
  name: string;
  finetuned_from: string;
  finetune_mode: "head" | "full";
  epochs: number;
  final_loss: number | null;
  trained_at: string;
  published_at: string;
  app: string;
  dataset?: string;
}

/** What a publish is: the record of the model, the repository (`owner/name`)
 * and the folder in it, when, what the model was finetuned from (a ref:
 * finetunedFrom), and the dataset's URL to name, or null to name none. */
export interface Publication {
  record: FinetunedRecord;
  repo: string;
  folder: string;
  publishedAt: string;
  finetunedFrom: string;
  dataset: string | null;
}

/** The ref of `folder` in `repo`, as links and *Add a model* take it. */
export function refOf(repo: string, folder: string): string {
  return `${repo}/${folder}`;
}

/** The ref of the model `record` was finetuned from: its base's id (a
 * Hugging Face model's id is its ref), or, for a base finetuned in this
 * browser too, that model's ref where it was published, and failing that
 * the ref of the model it came from, as far back as `records` go. A model
 * that was neither published nor is kept any more leaves its id. */
export function finetunedFrom(record: FinetunedRecord, records: readonly FinetunedRecord[]): string {
  let base = record.base;
  const seen = new Set<string>();
  while (base.id.startsWith(`${FINETUNED_REPO}/`) && !seen.has(base.id)) {
    seen.add(base.id);
    const parent = records.find((r) => r.id === base.id);
    if (!parent) break;
    if (parent.published) return refOf(parent.published.repo, parent.published.folder);
    base = parent.base;
  }
  return base.id;
}

/** The manifest uploaded as `<folder>/manifest.json`. */
export function publishedManifest(p: Publication): ModelManifest & { published: PublishedBlock } {
  const { record } = p;
  const published: PublishedBlock = {
    name: record.name,
    finetuned_from: p.finetunedFrom,
    finetune_mode: record.params.finetune_mode === "full" ? "full" : "head",
    epochs: record.epochs,
    final_loss: record.final_loss,
    trained_at: record.created_at,
    published_at: p.publishedAt,
    app: `cellmap-flow ${PYTHON.version} (browser)`,
    ...(p.dataset ? { dataset: p.dataset } : {}),
  };
  return { ...record.entry.manifest, published };
}

/** The links that open the model in the app at `app` (the page's own
 * address, or PUBLIC_APP in a card): "Try it in the browser", the model
 * alone, which the app opens with the model chosen for whatever volume the
 * user connects; and "Open on this dataset", the Share button's link
 * (share-link.ts) of `dataset` with the model, null without a dataset. A ref
 * is letters, digits, `_.-` and slashes: nothing in it needs escaping. */
export function appLinks(app: string, ref: string, dataset: string | null): { tryLink: string; datasetLink: string | null } {
  return {
    tryLink: `${app.replace(/[?#].*$/, "")}?${MODEL_PARAM}=${ref}`,
    datasetLink: dataset ? shareLink(app, dataset, [ref], null) : null,
  };
}

/** The model card, README.md: Hugging Face's front matter (the library,
 * and tags that find it), then what the model predicts and at what voxel
 * size, what it was finetuned from and how, and how to open it. */
export function modelCard(p: Publication): string {
  const m = p.record.entry.manifest;
  const v = p.record.entry.variant;
  const ref = refOf(p.repo, p.folder);
  const links = appLinks(PUBLIC_APP, ref, p.dataset);
  const mode = p.record.params.finetune_mode === "full" ? "its whole network" : "its last layer";
  const loss = p.record.final_loss === null ? "" : `, to a final loss of ${Number(p.record.final_loss.toPrecision(3))}`;
  const epochs = p.record.epochs ? ` for ${p.record.epochs} epoch(s)${loss}` : "";
  const trained = p.record.from_link ? "rebuilt in the browser from a shared link" : `trained ${p.record.created_at.slice(0, 10)}`;
  const tags = ["cellmap-flow", "onnx", "electron-microscopy", m.task.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")];
  return [
    "---",
    "library_name: onnx",
    "tags:",
    ...tags.filter(Boolean).map((tag) => `- ${tag}`),
    "---",
    "",
    `# ${p.folder}`,
    "",
    `A ${m.task} model for electron microscopy: it predicts ${m.channels.join(", ")}` +
      `${m.output ? ` (${m.output})` : ""}, from ${voxelLabel(m.input_voxel_size_nm)} nm voxels to ${voxelLabel(m.output_voxel_size_nm)} nm voxels, ` +
      `in blocks of ${v.input_size.join("×")} voxels in and ${v.output_size.join("×")} out (${v.variant}).`,
    "",
    `Finetuned in the browser with [cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow) ${PYTHON.version} from ` +
      `\`${p.finetunedFrom}\`${m.architecture ? ` (${m.architecture})` : ""}: ${mode} was trained on annotations painted in the browser${epochs} (${trained}).` +
      (p.dataset ? ` The annotations were painted on ${p.dataset}.` : ""),
    "",
    "## Open it",
    "",
    `- [Try it in the browser](${links.tryLink}): cellmap-flow opens with the model chosen; connect any volume to run it.`,
    ...(links.datasetLink ? [`- [Open it on the dataset it was trained on](${links.datasetLink}).`] : []),
    `- Anywhere cellmap-flow takes a model (*Add a model* in the dashboard's Models tab, in the browser or on a server): \`${ref}\`.`,
    "",
    `\`${p.folder}/manifest.json\` says the model's voxel sizes, blocks, channels and input normalization in cellmap-flow's ` +
      `format, and \`${p.folder}/${v.file}\` is the ONNX graph.`,
    "",
  ].join("\n");
}

/** The files of a publish, by their path in the repository, in the order
 * they are uploaded. */
export function publishedFiles(p: Publication, bytes: ArrayBuffer): { path: string; content: Blob }[] {
  return [
    { path: "README.md", content: new Blob([modelCard(p)], { type: "text/markdown" }) },
    { path: `${p.folder}/manifest.json`, content: new Blob([JSON.stringify(publishedManifest(p), null, 2) + "\n"], { type: "application/json" }) },
    { path: `${p.folder}/${p.record.entry.variant.file}`, content: new Blob([bytes], { type: "application/octet-stream" }) },
  ];
}
