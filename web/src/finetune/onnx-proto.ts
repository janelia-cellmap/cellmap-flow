// A field-level reader and writer of protobuf messages, and on top of it the
// little of ONNX's schema that finetuning the head needs: the graph's nodes,
// its initializers (where their bytes lie in the file) and its inputs and
// outputs by name, and two edits, a graph output added and an initializer's
// values replaced; and the model's metadata_props written and read, where a
// finetuned model's download carries its manifest (routes/finetune.ts; the
// server's onnx model type reads it, cellmap_flow/models/configs/onnx.py;
// so does the Models tab's Open an .onnx file…, src/model-file.ts).
// Protobuf is a sequence of (field number, wire type) tags, each followed by
// a varint, a fixed-size value or a length-prefixed run of bytes, and a
// message is such a sequence inside a length-prefixed field; so a model is
// read without its schema by walking the tags, and edited by re-encoding the
// one field that changes and copying every other byte for byte. A protobuf
// library would do the same for a hundred times the code shipped and a full
// decode of 60 MB of weights into objects; here the weights stay where they
// are, as views of the file's bytes.
//
// The field numbers, from onnx/onnx.proto (checked against the exports on
// 2026-10-15): ModelProto 7 graph, 14 metadata_props; StringStringEntryProto
// 1 key, 2 value; GraphProto 1 node, 5 initializer, 11 input, 12 output;
// NodeProto 1 input, 2 output, 3 name, 4 op_type, 5 attribute, 7 domain;
// AttributeProto 1 name, 3 i, 4 s, 8 ints, 20 type; TensorProto 1 dims, 2
// data_type, 8 name, 9 raw_data (4 float_data, 5 int32_data, 7 int64_data,
// 10 double_data, 11 uint64_data are the typed stores it is not in; 13
// external_data, 14 data_location); ValueInfoProto 1 name, 2 type; TypeProto
// 1 tensor_type; TypeProto.Tensor 1 elem_type.

export const WIRE_VARINT = 0, WIRE_FIXED64 = 1, WIRE_LENGTH = 2, WIRE_FIXED32 = 5;

/** One field of a message as it lies in the bytes: its tag's and its value's
 * offsets, absolute in the buffer the message was read from, the value's
 * bytes (a view, not a copy) and, for a varint field, its value. */
export interface Field {
  number: number;
  wireType: number;
  /** Where the tag begins; `[tagStart, end)` is the whole field. */
  tagStart: number;
  /** The value's bytes, `[start, end)`. */
  start: number;
  end: number;
  /** The value's bytes: the message or string of a length-delimited field, the 4 or 8 bytes of a fixed one, the varint's own. */
  value: Uint8Array;
  /** A varint field's value: a number when it is a safe integer, else a bigint (an int64 below zero is one). 0 for other wire types. */
  varint: number | bigint;
}

const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);

/** The varint at `pos`, and where the next byte is. */
export function readVarint(bytes: Uint8Array, pos: number): { value: number | bigint; next: number } {
  // Up to 49 bits in a number, which covers every length and dimension;
  // beyond that (an int64 attribute below zero is ten bytes) in a bigint.
  let value = 0, shift = 0, i = pos;
  while (true) {
    if (i >= bytes.length) throw new Error(`a varint at byte ${pos} runs past the end of the message`);
    const b = bytes[i++];
    if (shift < 49) {
      value += (b & 0x7f) * 2 ** shift;
    } else {
      let big = BigInt(value) | (BigInt(b & 0x7f) << BigInt(shift));
      while (b & 0x80) {
        if (i >= bytes.length) throw new Error(`a varint at byte ${pos} runs past the end of the message`);
        shift += 7;
        if (shift > 63) throw new Error(`a varint at byte ${pos} is longer than ten bytes`);
        const c = bytes[i++];
        big |= BigInt(c & 0x7f) << BigInt(shift);
        if (!(c & 0x80)) break;
      }
      return { value: big <= MAX_SAFE ? Number(big) : big, next: i };
    }
    if (!(b & 0x80)) return { value, next: i };
    shift += 7;
  }
}

/** `value` as a varint; a negative number or bigint is an int64, two's complement over 64 bits. */
export function encodeVarint(value: number | bigint): Uint8Array {
  let v = typeof value === "bigint" ? value : BigInt(Math.trunc(value));
  if (v < 0n) v = BigInt.asUintN(64, v);
  const out: number[] = [];
  do {
    const b = Number(v & 0x7fn);
    v >>= 7n;
    out.push(v ? b | 0x80 : b);
  } while (v);
  return Uint8Array.from(out);
}

export function encodeTag(number: number, wireType: number): Uint8Array {
  return encodeVarint(number * 8 + wireType);
}

/** A varint field. */
export function encodeVarintField(number: number, value: number | bigint): Uint8Array {
  return concatBytes([encodeTag(number, WIRE_VARINT), encodeVarint(value)]);
}

/** A length-delimited field: a string's bytes, or a nested message. */
export function encodeLengthDelimited(number: number, value: Uint8Array): Uint8Array {
  return concatBytes([encodeTag(number, WIRE_LENGTH), encodeVarint(value.length), value]);
}

export function concatBytes(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** The fields of the message in `bytes`, in order; `baseOffset` is where
 * `bytes` begins in the buffer the offsets are reported in (a nested
 * message's `start`). Repeated fields are one Field each. */
export function parseFields(bytes: Uint8Array, baseOffset = 0): Field[] {
  const fields: Field[] = [];
  let pos = 0;
  while (pos < bytes.length) {
    const tagStart = pos;
    const tag = readVarint(bytes, pos);
    if (typeof tag.value !== "number" || tag.value < 8) throw new Error(`a bad field tag at byte ${baseOffset + pos}`);
    const number = Math.floor(tag.value / 8), wireType = tag.value % 8;
    pos = tag.next;
    let start = pos, end: number, varint: number | bigint = 0;
    switch (wireType) {
      case WIRE_VARINT: {
        const v = readVarint(bytes, pos);
        varint = v.value;
        end = v.next;
        break;
      }
      case WIRE_FIXED64:
        end = pos + 8;
        break;
      case WIRE_FIXED32:
        end = pos + 4;
        break;
      case WIRE_LENGTH: {
        const len = readVarint(bytes, pos);
        if (typeof len.value !== "number") throw new Error(`a field of ${len.value} bytes at byte ${baseOffset + pos}`);
        start = len.next;
        end = start + len.value;
        break;
      }
      default:
        throw new Error(`wire type ${wireType} at byte ${baseOffset + tagStart} (groups are not protobuf 3; is this an ONNX file?)`);
    }
    if (end > bytes.length) throw new Error(`field ${number} at byte ${baseOffset + tagStart} runs past the end of the message`);
    fields.push({ number, wireType, tagStart: baseOffset + tagStart, start: baseOffset + start, end: baseOffset + end, value: bytes.subarray(start, end), varint });
    pos = end;
  }
  return fields;
}

/** The fields encoded again, in order: what `parseFields` read, byte for byte
 * (protobuf writers encode varints in the fewest bytes, as this does). */
export function encodeFields(fields: Field[]): Uint8Array<ArrayBuffer> {
  const parts: Uint8Array[] = [];
  for (const f of fields) {
    parts.push(encodeTag(f.number, f.wireType));
    if (f.wireType === WIRE_VARINT) parts.push(encodeVarint(f.varint));
    else if (f.wireType === WIRE_LENGTH) parts.push(encodeVarint(f.value.length), f.value);
    else parts.push(f.value);
  }
  return concatBytes(parts);
}

const utf8 = new TextDecoder(), utf8Encoder = new TextEncoder();

function text(f: Field): string {
  return utf8.decode(f.value);
}

/** A varint field's value as a signed 64-bit integer, as a number: protobuf
 * writes an int64 below zero as its two's complement, ten bytes. */
export function int64(value: number | bigint): number {
  return typeof value === "bigint" ? Number(BigInt.asIntN(64, value)) : value;
}

/** The int64 values of a repeated field: one per unpacked occurrence, or several in one packed one. */
function int64s(f: Field): number[] {
  if (f.wireType === WIRE_VARINT) return [int64(f.varint)];
  const values: number[] = [];
  let pos = 0;
  while (pos < f.value.length) {
    const v = readVarint(f.value, pos);
    values.push(int64(v.value));
    pos = v.next;
  }
  return values;
}

// ---------------------------------------------------------------------------
// ONNX

export interface Attribute {
  name: string;
  /** AttributeProto.type when written (1 FLOAT, 2 INT, 3 STRING, 7 INTS, ...). */
  type: number | null;
  int: number | null;
  ints: number[] | null;
  string: string | null;
}

export interface Node {
  name: string;
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
  attributes: Attribute[];
}

/** An initializer: its name, dims and element type, and where its raw_data
 * lies in the model bytes (null when it is stored another way). */
export interface Initializer {
  name: string;
  dims: number[];
  dataType: number;
  rawData: { start: number; end: number } | null;
  /** Stored in one of the typed fields (float_data, int32_data, ...) instead of raw_data. */
  typedData: boolean;
  /** data_location EXTERNAL: the bytes are in another file. */
  external: boolean;
}

export interface Graph {
  nodes: Node[];
  initializers: Initializer[];
  /** The graph inputs' names, in order. */
  inputs: string[];
  /** The graph outputs' names, in order. */
  outputs: string[];
}

/** The ModelProto's fields and, among them, the graph (field 7). */
export function readModel(bytes: Uint8Array): { fields: Field[]; graph: Field } {
  const fields = parseFields(bytes, 0);
  const graphs = fields.filter((f) => f.number === 7 && f.wireType === WIRE_LENGTH);
  if (graphs.length !== 1) throw new Error(graphs.length ? "the model has more than one graph field" : "no graph in the model (is this an ONNX file?)");
  return { fields, graph: graphs[0] };
}

/** The graph's nodes, initializers and I/O names, read from a graph field of `readModel`. */
export function readGraph(graph: Field): Graph {
  const g: Graph = { nodes: [], initializers: [], inputs: [], outputs: [] };
  for (const f of parseFields(graph.value, graph.start)) {
    if (f.wireType !== WIRE_LENGTH) continue;
    switch (f.number) {
      case 1:
        g.nodes.push(readNode(f));
        break;
      case 5:
        g.initializers.push(readInitializer(f));
        break;
      case 11:
        g.inputs.push(valueInfoName(f));
        break;
      case 12:
        g.outputs.push(valueInfoName(f));
        break;
    }
  }
  return g;
}

function readNode(node: Field): Node {
  const n: Node = { name: "", opType: "", domain: "", inputs: [], outputs: [], attributes: [] };
  for (const f of parseFields(node.value, node.start)) {
    if (f.wireType !== WIRE_LENGTH) continue;
    switch (f.number) {
      case 1: n.inputs.push(text(f)); break;
      case 2: n.outputs.push(text(f)); break;
      case 3: n.name = text(f); break;
      case 4: n.opType = text(f); break;
      case 5: n.attributes.push(readAttribute(f)); break;
      case 7: n.domain = text(f); break;
    }
  }
  return n;
}

function readAttribute(attr: Field): Attribute {
  const a: Attribute = { name: "", type: null, int: null, ints: null, string: null };
  for (const f of parseFields(attr.value, attr.start)) {
    switch (f.number) {
      case 1: if (f.wireType === WIRE_LENGTH) a.name = text(f); break;
      case 3: if (f.wireType === WIRE_VARINT) a.int = int64(f.varint); break;
      case 4: if (f.wireType === WIRE_LENGTH) a.string = text(f); break;
      case 8: a.ints = [...(a.ints ?? []), ...int64s(f)]; break;
      case 20: if (f.wireType === WIRE_VARINT) a.type = int64(f.varint); break;
    }
  }
  return a;
}

function readInitializer(tensor: Field): Initializer {
  const t: Initializer = { name: "", dims: [], dataType: 0, rawData: null, typedData: false, external: false };
  for (const f of parseFields(tensor.value, tensor.start)) {
    switch (f.number) {
      case 1: t.dims.push(...int64s(f)); break;
      case 2: if (f.wireType === WIRE_VARINT) t.dataType = int64(f.varint); break;
      case 4: case 5: case 6: case 7: case 10: case 11: t.typedData = true; break;
      case 8: if (f.wireType === WIRE_LENGTH) t.name = text(f); break;
      case 9: if (f.wireType === WIRE_LENGTH) t.rawData = { start: f.start, end: f.end }; break;
      case 13: t.external = true; break;
      case 14: if (f.wireType === WIRE_VARINT && int64(f.varint) === 1) t.external = true; break;
    }
  }
  return t;
}

function valueInfoName(info: Field): string {
  const name = parseFields(info.value, info.start).find((f) => f.number === 1 && f.wireType === WIRE_LENGTH);
  return name ? text(name) : "";
}

/** The model with the tensor `name` added as a graph output of element type
 * `elemType` (1 float32, 10 float16), without a shape: a ValueInfoProto
 * appended to the graph as another field 12 (a repeated field's entries may
 * lie anywhere in the message), the graph field re-encoded with its new
 * length, every other field of the model copied as it was. */
export function withGraphOutput(modelBytes: Uint8Array, name: string, elemType: number): Uint8Array<ArrayBuffer> {
  const { fields, graph } = readModel(modelBytes);
  const tensorType = encodeLengthDelimited(1, encodeVarintField(1, elemType));  // TypeProto.tensor_type { elem_type }
  const valueInfo = concatBytes([encodeLengthDelimited(1, utf8Encoder.encode(name)), encodeLengthDelimited(2, tensorType)]);
  const graphBytes = concatBytes([graph.value, encodeLengthDelimited(12, valueInfo)]);
  return concatBytes(fields.map((f) => (f === graph ? encodeLengthDelimited(7, graphBytes) : modelBytes.subarray(f.tagStart, f.end))));
}

/** The model's metadata_props (ModelProto field 14; each entry's field 1 is
 * its key, 2 its value), by key. Throws as readModel does for bytes that
 * are not an ONNX model. */
export function readMetadata(modelBytes: Uint8Array): Record<string, string> {
  const entries = readModel(modelBytes).fields.filter((f) => f.number === 14 && f.wireType === WIRE_LENGTH).map((f) => parseFields(f.value, f.start));
  const value = (entry: Field[], number: number) => {
    const field = entry.find((f) => f.number === number && f.wireType === WIRE_LENGTH);
    return field ? text(field) : "";
  };
  return Object.fromEntries(entries.map((entry) => [value(entry, 1), value(entry, 2)]));
}

/** The model with `entries` in its metadata_props: an entry already there
 * under one of their keys is dropped (onnx's checker refuses a key written
 * twice), each new one appended as another field 14 (a repeated field's
 * entries may lie anywhere in the message, after the graph too), every other
 * field of the model copied as it was. */
export function withMetadata(modelBytes: Uint8Array, entries: Record<string, string>): Uint8Array<ArrayBuffer> {
  const { fields } = readModel(modelBytes);
  const replaced = new Set(Object.keys(entries));
  const kept = fields.filter((f) => {
    if (f.number !== 14 || f.wireType !== WIRE_LENGTH) return true;
    const key = parseFields(f.value, f.start).find((e) => e.number === 1 && e.wireType === WIRE_LENGTH);
    return !replaced.has(key ? text(key) : "");
  });
  const added = Object.entries(entries).map(([key, value]) =>
    encodeLengthDelimited(14, concatBytes([encodeLengthDelimited(1, utf8Encoder.encode(key)), encodeLengthDelimited(2, utf8Encoder.encode(value))])),
  );
  return concatBytes([...kept.map((f) => modelBytes.subarray(f.tagStart, f.end)), ...added]);
}
