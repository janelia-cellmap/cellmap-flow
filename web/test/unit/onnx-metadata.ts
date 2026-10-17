import { type Field, parseFields, readModel, WIRE_LENGTH } from "../../src/finetune/onnx-proto";

/** A model's metadata_props (ModelProto field 14; each entry's field 1 is its key, 2 its value), by key. */
export function readMetadata(modelBytes: Uint8Array): Record<string, string> {
  const text = (entry: Field[], number: number) => new TextDecoder().decode(entry.find((f) => f.number === number && f.wireType === WIRE_LENGTH)?.value);
  const entries = readModel(modelBytes).fields.filter((f) => f.number === 14 && f.wireType === WIRE_LENGTH).map((f) => parseFields(f.value, f.start));
  return Object.fromEntries(entries.map((entry) => [text(entry, 1), text(entry, 2)]));
}
