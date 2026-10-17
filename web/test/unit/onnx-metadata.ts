// Shared by the tests that read back what withMetadata writes
// (finetune-onnx.test.ts, backend-finetune.test.ts): an ONNX model's
// metadata_props, read with the page's own protobuf reader. The page only
// writes them (a finetuned model's download); the server reads them.
import { parseFields, readModel, WIRE_LENGTH } from "../../src/finetune/onnx-proto";

const utf8 = new TextDecoder();

/** The model's metadata_props (ModelProto field 14, each entry a
 * StringStringEntryProto: field 1 its key, 2 its value), by key; a key
 * written twice is the later entry's, as onnx's own readers take it. */
export function readMetadata(modelBytes: Uint8Array): Record<string, string> {
  const metadata: Record<string, string> = {};
  for (const f of readModel(modelBytes).fields) {
    if (f.number !== 14 || f.wireType !== WIRE_LENGTH) continue;
    const entry = parseFields(f.value, f.start);
    const key = entry.find((e) => e.number === 1 && e.wireType === WIRE_LENGTH);
    const value = entry.find((e) => e.number === 2 && e.wireType === WIRE_LENGTH);
    metadata[key ? utf8.decode(key.value) : ""] = value ? utf8.decode(value.value) : "";
  }
  return metadata;
}
