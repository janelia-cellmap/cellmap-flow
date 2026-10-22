// The store-only ZIP writer: the CRC against the check value and Node's own,
// and archives read back by Python's zipfile where python3 is on the PATH
// (as on CI's web job): every CRC good, each file stored, under its UTF-8
// name (zipfile reads a name as UTF-8 only when the flag says it is), with
// its bytes as written.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { crc32 as nodeCrc32 } from "node:zlib";
import { describe, expect, it } from "vitest";

import { crc32, volumeZip, zipStore, type ZipEntry } from "../../src/finetune/zip";
import { MemoryPaintStore } from "../../src/state/paint-store";

const encode = (text: string) => new TextEncoder().encode(text);

const python = (() => {
  try {
    execFileSync("python3", ["-c", "import zipfile"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
})();

interface ReadEntry {
  name: string;
  size: number;
  crc: number;
  method: number;
  hex: string;
}

/** What Python's zipfile reads in `zip`, once testzip has checked every
 * CRC: each entry's name, size, CRC, compression (0 is stored) and bytes. */
function pythonReads(zip: Uint8Array): ReadEntry[] {
  const dir = mkdtempSync(join(tmpdir(), "cellmap-flow-zip-"));
  try {
    const path = join(dir, "test.zip");
    writeFileSync(path, zip);
    const script = [
      "import json, sys, zipfile",
      "z = zipfile.ZipFile(sys.argv[1])",
      "assert z.testzip() is None",
      "print(json.dumps([{'name': i.filename, 'size': i.file_size, 'crc': i.CRC, 'method': i.compress_type, 'hex': z.read(i).hex()} for i in z.infolist()]))",
    ].join("\n");
    return JSON.parse(execFileSync("python3", ["-I", "-c", script, path], { encoding: "utf-8" })) as ReadEntry[];
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** The entry Python should read for `name` holding `bytes`. */
const stored = (name: string, bytes: Uint8Array): ReadEntry => ({ name, size: bytes.length, crc: nodeCrc32(bytes), method: 0, hex: Buffer.from(bytes).toString("hex") });

describe("crc32", () => {
  it("matches the check values and Node's implementation, in one go or in pieces", () => {
    expect(crc32(new Uint8Array())).toBe(0);
    expect(crc32(encode("123456789"))).toBe(0xcbf43926);
    const bytes = encode("The quick brown fox jumps over the lazy dog, héllo wörld.");
    expect(crc32(bytes)).toBe(nodeCrc32(bytes));
    expect(crc32(bytes.subarray(10), crc32(bytes.subarray(0, 10)))).toBe(crc32(bytes));
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

  it.skipIf(!python)("is read by Python's zipfile, every file stored under its UTF-8 name with its bytes and CRC", () => {
    expect(pythonReads(zipStore(ENTRIES, NOW))).toEqual(ENTRIES.map((e) => stored(e.name, e.bytes)));
  });

  it("refuses what would need ZIP64: more than 65535 files, or a longer name", () => {
    const many = Array.from({ length: 65536 }, (_, i) => ({ name: `f${i}`, bytes: new Uint8Array() }));
    expect(() => zipStore(many, NOW)).toThrow(/65535/);
    expect(() => zipStore([{ name: "x".repeat(70000), bytes: new Uint8Array() }], NOW)).toThrow(/65535/);
  });
});

describe("volumeZip", () => {
  it.skipIf(!python)("holds one volume's files, and no other's, at <id>.zarr/<path> in the store's order", async () => {
    const store = new MemoryPaintStore();
    await store.put("vol-a", ".zattrs", encode("{}"));
    await store.put("vol-a", "annotation/s0/.zarray", encode("[]"));
    await store.put("vol-a", "annotation/s0/1.2.3", new Uint8Array([7, 7]));
    await store.put("vol-b", ".zattrs", encode("{}"));
    expect(pythonReads(await volumeZip(store, "vol-a", new Date(2026, 9, 15)))).toEqual([
      stored("vol-a.zarr/.zattrs", encode("{}")),
      stored("vol-a.zarr/annotation/s0/.zarray", encode("[]")),
      stored("vol-a.zarr/annotation/s0/1.2.3", new Uint8Array([7, 7])),
    ]);
  });
});
