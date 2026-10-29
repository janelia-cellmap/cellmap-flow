// A model opened from a file on this machine, for one that was sent rather
// than published: the Models tab's *Open an .onnx file…* (templates/
// _models_tab.html, in its browser-only block). The file is an .onnx this
// app downloaded (Download finetuned model writes the model's manifest into
// its metadata_props under MANIFEST_METADATA_KEY), or any .onnx picked
// together with its manifest.json, which then wins over one inside it. The
// model is kept in this browser as a finetuned one is (the registry and the
// byte cache), so it survives a reload and is served, finetuned and
// published like one, its record saying where it came from
// (FinetunedRecord.from_file); nothing is uploaded. A link cannot name it
// until it is published (share-link.ts says so).
//
// The dashboard's own add-model.js knows nothing of files: once the model is
// kept, this puts its name in Add a model's field and presses Resolve, so
// the tab shows it as any other resolved model and Run serves it, as the
// connect panel's Browse button (src/local-folder) fills the dataset field
// and presses Connect.
import { type Catalog, FINETUNED_REPO, type FinetunedRecord, MANIFEST_METADATA_KEY, type ModelEntry, type ModelManifest, type ModelVariant, type SessionStore } from "./contracts";
import { readMetadata } from "./finetune/onnx-proto";
import { entryFor, entryName, freeName, isLayerName, parseManifest } from "./models/manifest";
import { message } from "./shared/message";
import { keepInTab } from "./state";

/** The browser-only block's ids, and Add a model's (add-model.js looks those up). */
const OPEN_ID = "openModelFileBtn";
const INPUT_ID = "openModelFileInput";
const REF_ID = "addModelRef";
const RESOLVE_ID = "addModelResolveBtn";

/** A picked file as the opener reads it: its name and its bytes. */
export interface PickedFile {
  name: string;
  bytes: ArrayBuffer;
}

/** What opening a file needs of the catalog: the names taken, the models
 * kept (one opened before may be this file), and keeping the new one. */
export type FileCatalog = Pick<Catalog, "list" | "finetuned" | "modelBytes" | "addFinetuned">;

/** The entry of the model in `onnx`, kept in `catalog` as a model of this
 * browser unless the same bytes were opened before (then that one).
 *
 * Its manifest is `manifestFile`'s when one was picked, else the one in the
 * file's metadata_props, and must name the file among its variants (or have
 * one variant: the file was renamed). Its name is the manifest's published
 * name, else the file's own without `.onnx`, else entryName's, whichever
 * first is a layer name (isLayerName); one any model in the list has is
 * followed by the first free `_<n>` (freeName), as an added repo's is in
 * the list, except that this name is the record's and kept. The entry is
 * the manifest with that one variant under the name, as finetunedEntry
 * makes a finetuned model's. Throws, keeping nothing, for a file that is
 * not an ONNX model or has no manifest to go by. */
export async function openModelFile(catalog: FileCatalog, onnx: PickedFile, manifestFile: PickedFile | null, now: () => Date = () => new Date()): Promise<ModelEntry> {
  const manifest = manifestOf(onnx, manifestFile);
  const variant = variantOf(manifest, onnx.name);
  const wanted = [manifest.published?.name, onnx.name.replace(/\.onnx$/i, "")].find((n) => n && isLayerName(n)) ?? entryName(manifest);
  const records = await catalog.finetuned();
  for (const record of records) {
    if (!record.from_file || (record.name !== wanted && !record.name.startsWith(`${wanted}_`))) continue;
    // Its bytes gone or unreadable: not this file to serve, its name taken all the same.
    const kept = await catalog.modelBytes(record.entry).catch(() => null);
    if (kept && sameBytes(kept, onnx.bytes)) return record.entry;
  }
  const name = freeName(wanted, new Set([...(await catalog.list()).map((e) => e.name), ...records.map((r) => r.name)]));
  const file: ModelVariant = { ...variant, file: `${name}.onnx`, file_mb: Math.round(onnx.bytes.byteLength / 1e4) / 100 };
  const entry = entryFor(name, { ...manifest, model: name, variants: [file] }, file, FINETUNED_REPO);
  const record: FinetunedRecord = {
    id: entry.id,
    name,
    base: { id: "", name: onnx.name, file: onnx.name },
    volume: "",
    created_at: now().toISOString(),
    params: {},
    epochs: 0,
    final_loss: null,
    entry,
    from_file: { name: onnx.name },
  };
  await catalog.addFinetuned(record, onnx.bytes);
  return entry;
}

function manifestOf(onnx: PickedFile, manifestFile: PickedFile | null): ModelManifest {
  let embedded: string | undefined;
  try {
    embedded = readMetadata(new Uint8Array(onnx.bytes))[MANIFEST_METADATA_KEY];
  } catch (e) {
    throw new Error(`${onnx.name} is not an ONNX model (${message(e)})`);
  }
  const [text, where] = manifestFile ? [new TextDecoder().decode(manifestFile.bytes), manifestFile.name] : [embedded, `${onnx.name}'s manifest`];
  if (text === undefined) throw new Error(`${onnx.name} carries no cellmap-flow manifest (a model this app downloads does): pick its manifest.json with it`);
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new Error(`${where} is not JSON (${message(e)})`);
  }
  return parseManifest(raw, where);
}

// The manifest's variant that is the file: by its name, or the one there is.
function variantOf(manifest: ModelManifest, fileName: string): ModelVariant {
  const variant = manifest.variants.find((v) => v.file === fileName) ?? (manifest.variants.length === 1 ? manifest.variants[0] : undefined);
  if (!variant) throw new Error(`${fileName} is none of the files its manifest lists (${manifest.variants.map((v) => v.file).join(", ")}): pick the one it names, or rename it so`);
  return variant;
}

function sameBytes(a: ArrayBuffer, b: ArrayBuffer): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const x = new Uint8Array(a), y = new Uint8Array(b);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}

/** The Open button: a file picker for the .onnx and, when it carries no
 * manifest, its manifest.json; the model kept (openModelFile), made the
 * tab's (src/kept: listed here from now on, served or not) and handed to
 * Add a model, or why not in an alert, as add-model.js says a ref it cannot
 * resolve. Nothing on a page without the block (the Flask page). */
export function installOpenFileUi(opts: { catalog: FileCatalog; session: SessionStore; log: (line: string) => void }): void {
  const button = document.getElementById(OPEN_ID);
  const input = document.getElementById(INPUT_ID) as HTMLInputElement | null;
  if (!button || !input) return;
  button.addEventListener("click", () => input.click());
  input.addEventListener("change", async () => {
    const picked = Array.from(input.files ?? []);
    input.value = "";
    const onnx = picked.filter((f) => /\.onnx$/i.test(f.name));
    const json = picked.filter((f) => /\.json$/i.test(f.name));
    if (onnx.length !== 1 || json.length > 1) {
      alert("Pick one .onnx file, and with it its manifest.json if it carries none.");
      return;
    }
    try {
      const read = async (file: File): Promise<PickedFile> => ({ name: file.name, bytes: await file.arrayBuffer() });
      const entry = await openModelFile(opts.catalog, await read(onnx[0]), json.length ? await read(json[0]) : null);
      await keepInTab(opts.session, { finetuned: [entry.id] });
      opts.log(`${onnx[0].name} opened as ${entry.name}, kept in this browser`);
      const ref = document.getElementById(REF_ID) as HTMLInputElement | null;
      if (ref) ref.value = entry.name;
      document.getElementById(RESOLVE_ID)?.click();
    } catch (e) {
      alert(`Could not open that model: ${message(e)}`);
    }
  });
}
