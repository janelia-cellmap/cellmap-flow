// The store-only ZIP writer: the CRC, the DOS stamps, and archives read back
// by a parser of the specification's structures here (local headers, the
// central directory, the end record) with the CRCs checked against Node's
// own, and by Python's zipfile where python3 is on the PATH.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { crc32 as nodeCrc32 } from "node:zlib";
import { describe, expect, it } from "vitest";

import { crc32, dosDateTime, volumeZip, zipStore, type ZipEntry } from "../../src/finetune/zip";
import { MemoryPaintStore } from "../../src/state/paint-store";

const encode = (text: string) => new TextEncoder().encode(text);
const decode = (bytes: Uint8Array) => new TextDecoder().decode(bytes);

interface ParsedEntry {
  name: string;
  size: number;
  crc: number;
  flags: number;
  method: number;
  time: number;
  date: number;
  offset: number;
  data: Uint8Array;
}

/** An archive's entries from its central directory, each checked against
 * its local header (APPNOTE 4.3.7, 4.3.12, 4.3.16). */
function readZip(bytes: Uint8Array): { entries: ParsedEntry[]; centralOffset: number; centralSize: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const end = bytes.length - 22;
  expect(view.getUint32(end, true)).toBe(0x06054b50);
  const count = view.getUint16(end + 10, true);
  expect(view.getUint16(end + 8, true)).toBe(count);
  const centralSize = view.getUint32(end + 12, true);
  const centralOffset = view.getUint32(end + 16, true);
  expect(view.getUint16(end + 20, true)).toBe(0); // no comment
  expect(centralOffset + centralSize).toBe(end);
  const entries: ParsedEntry[] = [];
  let at = centralOffset;
  for (let i = 0; i < count; i++) {
    expect(view.getUint32(at, true)).toBe(0x02014b50);
    const flags = view.getUint16(at + 8, true);
    const method = view.getUint16(at + 10, true);
    const time = view.getUint16(at + 12, true);
    const date = view.getUint16(at + 14, true);
    const crc = view.getUint32(at + 16, true);
    const compressed = view.getUint32(at + 20, true);
    const size = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    const extraLength = view.getUint16(at + 30, true);
    const commentLength = view.getUint16(at + 32, true);
    const offset = view.getUint32(at + 42, true);
    const name = decode(bytes.subarray(at + 46, at + 46 + nameLength));
    expect(compressed).toBe(size); // stored
    // the local header says the same
    expect(view.getUint32(offset, true)).toBe(0x04034b50);
    expect(view.getUint16(offset + 6, true)).toBe(flags);
    expect(view.getUint16(offset + 8, true)).toBe(method);
    expect(view.getUint32(offset + 14, true)).toBe(crc);
    expect(view.getUint32(offset + 18, true)).toBe(size);
    expect(view.getUint32(offset + 22, true)).toBe(size);
    const localNameLength = view.getUint16(offset + 26, true);
    const localExtraLength = view.getUint16(offset + 28, true);
    expect(decode(bytes.subarray(offset + 30, offset + 30 + localNameLength))).toBe(name);
    const start = offset + 30 + localNameLength + localExtraLength;
    entries.push({ name, size, crc, flags, method, time, date, offset, data: bytes.subarray(start, start + size) });
    at += 46 + nameLength + extraLength + commentLength;
  }
  expect(at).toBe(centralOffset + centralSize);
  return { entries, centralOffset, centralSize };
}

describe("crc32", () => {
  it("matches the check values and Node's implementation, in one go or in pieces", () => {
    expect(crc32(new Uint8Array())).toBe(0);
    expect(crc32(encode("123456789"))).toBe(0xcbf43926);
    const bytes = encode("The quick brown fox jumps over the lazy dog, héllo wörld.");
    expect(crc32(bytes)).toBe(nodeCrc32(bytes));
    expect(crc32(bytes.subarray(10), crc32(bytes.subarray(0, 10)))).toBe(crc32(bytes));
  });
});

describe("dosDateTime", () => {
  it("packs the local date and time at two-second resolution", () => {
    const { time, date } = dosDateTime(new Date(2026, 9, 15, 14, 30, 7));
    expect(time).toBe((14 << 11) | (30 << 5) | 3);
    expect(date).toBe((46 << 9) | (10 << 4) | 15);
  });

  it("clamps a date before 1980 to the epoch's first day", () => {
    expect(dosDateTime(new Date(1970, 0, 1, 12, 0, 0))).toEqual({ time: 0, date: (0 << 9) | (1 << 4) | 1 });
  });
});

describe("zipStore", () => {
  const ENTRIES: ZipEntry[] = [
    { name: "vol.zarr/.zattrs", bytes: encode('{"type": "annotation_volume"}') },
    { name: "vol.zarr/annotation/s0/0.0.0", bytes: new Uint8Array([0, 1, 2, 2, 1, 0, 0, 0]) },
    { name: "héllo/ü.txt", bytes: encode("ünïcödé"), mtime: new Date(2020, 0, 2, 3, 4, 6) },
    { name: "empty", bytes: new Uint8Array() },
  ];
  const NOW = new Date(2026, 9, 15, 14, 30, 6);

  it("writes each file stored, under its UTF-8 name, with the right CRC, then the directory and the end record", () => {
    const zip = zipStore(ENTRIES, NOW);
    const { entries, centralOffset } = readZip(zip);
    expect(entries.map((e) => e.name)).toEqual(ENTRIES.map((e) => e.name));
    entries.forEach((e, i) => {
      expect(Array.from(e.data)).toEqual(Array.from(ENTRIES[i].bytes));
      expect(e.size).toBe(ENTRIES[i].bytes.length);
      expect(e.crc).toBe(nodeCrc32(ENTRIES[i].bytes));
      expect(e.method).toBe(0);
      expect(e.flags & 0x0800).toBe(0x0800);
    });
    // the files lie one after another from the start, and the directory after the last
    expect(entries[0].offset).toBe(0);
    expect(entries[1].offset).toBe(30 + encode(ENTRIES[0].name).length + ENTRIES[0].bytes.length);
    const files = ENTRIES.reduce((sum, e) => sum + 30 + encode(e.name).length + e.bytes.length, 0);
    expect(centralOffset).toBe(files);
    expect(zip.length).toBe(files + ENTRIES.reduce((sum, e) => sum + 46 + encode(e.name).length, 0) + 22);
  });

  it("stamps each file with its own time, else the archive's", () => {
    const { entries } = readZip(zipStore(ENTRIES, NOW));
    expect({ time: entries[0].time, date: entries[0].date }).toEqual(dosDateTime(NOW));
    expect({ time: entries[2].time, date: entries[2].date }).toEqual(dosDateTime(new Date(2020, 0, 2, 3, 4, 6)));
  });

  it("is an empty archive for no entries", () => {
    const zip = zipStore([], NOW);
    expect(zip.length).toBe(22);
    expect(readZip(zip).entries).toEqual([]);
  });

  it("refuses what would need ZIP64, saying so", () => {
    const many = Array.from({ length: 65536 }, (_, i) => ({ name: `f${i}`, bytes: new Uint8Array() }));
    expect(() => zipStore(many, NOW)).toThrow(/at most 65535 files/);
    expect(() => zipStore([{ name: "x".repeat(70000), bytes: new Uint8Array() }], NOW)).toThrow(/at most 65535 bytes/);
  });

  const python = (() => {
    try {
      execFileSync("python3", ["-c", "import zipfile"], { stdio: "ignore" });
      return true;
    } catch {
      return false;
    }
  })();

  it.skipIf(!python)("is read by Python's zipfile, every CRC good, the names as written", () => {
    const dir = mkdtempSync(join(tmpdir(), "cellmap-flow-zip-"));
    try {
      const path = join(dir, "test.zip");
      writeFileSync(path, zipStore(ENTRIES, NOW));
      const out = execFileSync(
        "python3",
        ["-I", "-c", "import sys, zipfile\nz = zipfile.ZipFile(sys.argv[1])\nassert z.testzip() is None\nprint('\\n'.join(z.namelist()))\nprint(z.read('héllo/ü.txt').decode())", path],
        { encoding: "utf-8" },
      );
      expect(out.trim().split("\n")).toEqual([...ENTRIES.map((e) => e.name), "ünïcödé"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("volumeZip", () => {
  it("holds one volume's files, and no other's, at <id>.zarr/<path> in the store's order", async () => {
    const store = new MemoryPaintStore();
    await store.put("vol-a", ".zattrs", encode("{}"));
    await store.put("vol-a", "annotation/s0/.zarray", encode("[]"));
    await store.put("vol-a", "annotation/s0/1.2.3", new Uint8Array([7, 7]));
    await store.put("vol-b", ".zattrs", encode("{}"));
    const { entries } = readZip(await volumeZip(store, "vol-a", new Date(2026, 9, 15)));
    expect(entries.map((e) => e.name)).toEqual(["vol-a.zarr/.zattrs", "vol-a.zarr/annotation/s0/.zarray", "vol-a.zarr/annotation/s0/1.2.3"]);
    expect(Array.from(entries[2].data)).toEqual([7, 7]);
  });
});
