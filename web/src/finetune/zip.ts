// A store-only ZIP writer, for downloading an annotation volume as a zarr
// (GET /api/finetune/volume/:id/download, backend/routes/finetune.ts): the
// volume's files from the paint store, each at `<id>.zarr/<path>`, so the
// archive unpacks to the directory the server would have written
// (finetune/session/volume.py's create_volume_zarr), which zarr-python and
// the server's crops import read as they read the server's own.
//
// Store only (method 0, no compression). The paint is sparse chunks of small
// integers and a few JSON files; the browser's CompressionStream would make
// a Deflate stream the reader must inflate, for a saving that does not
// matter at these sizes. Written by hand from the ZIP specification
// (APPNOTE 4.4): a local file header before each file's bytes, the central
// directory after them all, the end-of-central-directory record last;
// CRC-32 by table; UTF-8 names (general purpose bit 11). No ZIP64: a volume
// of more than 4 GB, or of 65535 files, is refused with a message, which no
// painted volume comes near.
import type { PaintStore } from "../contracts";

export interface ZipEntry {
  /** The path within the archive, "/" separated, UTF-8. */
  name: string;
  bytes: Uint8Array;
  /** The file's modification time; the archive's `now` when absent. */
  mtime?: Date;
}

/** The suffix of a volume's download (`<id>.zarr.zip`): the zarr, zipped. */
export const VOLUME_ZIP_SUFFIX = ".zarr.zip";

// CRC-32 (IEEE 802.3, the polynomial 0xEDB88320 reflected), as every ZIP
// reader checks it: a 256-entry table, one lookup per byte.
const CRC_TABLE: Uint32Array = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** The CRC-32 of `bytes`, continued from `crc` when given (the CRC of what came before). */
export function crc32(bytes: Uint8Array, crc = 0): number {
  let c = (crc ^ 0xffffffff) >>> 0;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** `date` as the MS-DOS time and date fields of a ZIP header: local time,
 * 2-second resolution, years 1980 to 2107 (clamped). */
export function dosDateTime(date: Date): { time: number; date: number } {
  const year = Math.min(2107, Math.max(1980, date.getFullYear()));
  const clamped = year === date.getFullYear();
  const month = clamped ? date.getMonth() + 1 : 1;
  const day = clamped ? date.getDate() : 1;
  const hours = clamped ? date.getHours() : 0;
  const minutes = clamped ? date.getMinutes() : 0;
  const seconds = clamped ? date.getSeconds() : 0;
  return {
    time: (hours << 11) | (minutes << 5) | (seconds >> 1),
    date: ((year - 1980) << 9) | (month << 4) | day,
  };
}

const LOCAL_HEADER = 0x04034b50;
const CENTRAL_HEADER = 0x02014b50;
const END_RECORD = 0x06054b50;
/** Version 2.0: what a store-only archive with a central directory needs. */
const VERSION = 20;
/** General purpose bit 11: the names are UTF-8. */
const UTF8_NAMES = 0x0800;
const MAX_32 = 0xffffffff;
const MAX_ENTRIES = 0xffff;

/** `entries` as a store-only ZIP archive, in their order. `now` stamps the
 * entries that carry no mtime. Throws when the archive would need ZIP64. */
export function zipStore(entries: ZipEntry[], now: Date = new Date()): Uint8Array {
  if (entries.length > MAX_ENTRIES) throw new Error(`a ZIP archive holds at most ${MAX_ENTRIES} files (this one would have ${entries.length})`);
  const encoder = new TextEncoder();
  const prepared = entries.map((entry) => {
    if (entry.bytes.byteLength > MAX_32) throw new Error(`${entry.name} is ${entry.bytes.byteLength} bytes, more than a ZIP entry can hold without ZIP64 (4 GB)`);
    const name = encoder.encode(entry.name);
    if (name.byteLength > 0xffff) throw new Error(`${entry.name.slice(0, 40)}…: a ZIP entry's name is at most 65535 bytes`);
    return { name, bytes: entry.bytes, crc: crc32(entry.bytes), ...dosDateTime(entry.mtime ?? now) };
  });
  const localSize = prepared.reduce((sum, e) => sum + 30 + e.name.byteLength + e.bytes.byteLength, 0);
  const centralSize = prepared.reduce((sum, e) => sum + 46 + e.name.byteLength, 0);
  if (localSize > MAX_32 || localSize + centralSize > MAX_32) throw new Error("the archive would be more than 4 GB, which needs ZIP64");

  const out = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(out.buffer);
  let at = 0;
  const offsets: number[] = [];
  for (const e of prepared) {
    offsets.push(at);
    view.setUint32(at, LOCAL_HEADER, true);
    view.setUint16(at + 4, VERSION, true);
    view.setUint16(at + 6, UTF8_NAMES, true);
    view.setUint16(at + 8, 0, true); // method: store
    view.setUint16(at + 10, e.time, true);
    view.setUint16(at + 12, e.date, true);
    view.setUint32(at + 14, e.crc, true);
    view.setUint32(at + 18, e.bytes.byteLength, true); // compressed = uncompressed, stored
    view.setUint32(at + 22, e.bytes.byteLength, true);
    view.setUint16(at + 26, e.name.byteLength, true);
    view.setUint16(at + 28, 0, true); // no extra field
    out.set(e.name, at + 30);
    out.set(e.bytes, at + 30 + e.name.byteLength);
    at += 30 + e.name.byteLength + e.bytes.byteLength;
  }
  const centralStart = at;
  prepared.forEach((e, i) => {
    view.setUint32(at, CENTRAL_HEADER, true);
    view.setUint16(at + 4, VERSION, true); // made by: version 2.0, MS-DOS attributes (the upper byte 0)
    view.setUint16(at + 6, VERSION, true);
    view.setUint16(at + 8, UTF8_NAMES, true);
    view.setUint16(at + 10, 0, true);
    view.setUint16(at + 12, e.time, true);
    view.setUint16(at + 14, e.date, true);
    view.setUint32(at + 16, e.crc, true);
    view.setUint32(at + 20, e.bytes.byteLength, true);
    view.setUint32(at + 24, e.bytes.byteLength, true);
    view.setUint16(at + 28, e.name.byteLength, true);
    view.setUint16(at + 30, 0, true); // extra
    view.setUint16(at + 32, 0, true); // comment
    view.setUint16(at + 34, 0, true); // disk
    view.setUint16(at + 36, 0, true); // internal attributes
    view.setUint32(at + 38, 0, true); // external attributes
    view.setUint32(at + 42, offsets[i], true);
    out.set(e.name, at + 46);
    at += 46 + e.name.byteLength;
  });
  view.setUint32(at, END_RECORD, true);
  view.setUint16(at + 4, 0, true);
  view.setUint16(at + 6, 0, true);
  view.setUint16(at + 8, prepared.length, true);
  view.setUint16(at + 10, prepared.length, true);
  view.setUint32(at + 12, centralSize, true);
  view.setUint32(at + 16, centralStart, true);
  view.setUint16(at + 20, 0, true);
  return out;
}

/** The annotation volume `id` as a ZIP of its zarr: every file the store
 * holds for it, at `<id>.zarr/<path>`, in the store's (sorted) order. */
export async function volumeZip(store: PaintStore, id: string, now: Date = new Date()): Promise<Uint8Array> {
  const entries: ZipEntry[] = [];
  for (const path of await store.list(id)) {
    const bytes = await store.get(id, path);
    if (bytes) entries.push({ name: `${id}.zarr/${path}`, bytes });
  }
  return zipStore(entries, now);
}
