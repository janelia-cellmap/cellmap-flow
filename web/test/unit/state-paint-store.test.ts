// The paint store (src/state/paint-store.ts): the PaintStore semantics as a
// suite over any implementation, run on MemoryPaintStore, the reference.
// IndexedDbPaintStore has the same semantics on db.ts's "paint" store, but
// Node has no IndexedDB and the project takes no fake of it as a dependency,
// so it runs only in a browser (the e2e test paints through the worker);
// what is checked here of it is the key scheme both share.
import { describe, expect, it } from "vitest";

import type { PaintStore } from "../../src/contracts";
import { MemoryPaintStore, openPaintStore, paintKey } from "../../src/state/paint-store";

const bytes = (...values: number[]) => new Uint8Array(values);
const VOL = "vol-0123abcd-20261015-120000";

/** The contract, on a fresh store from `make` per test. */
function paintStoreSuite(name: string, make: () => PaintStore) {
  describe(`${name} is a PaintStore`, () => {
    it("has nothing for a file never put (a chunk never painted)", async () => {
      const store = make();
      expect(await store.get(VOL, "annotation/s0/0.0.0")).toBeUndefined();
      expect(await store.list(VOL)).toEqual([]);
      expect(await store.volumes()).toEqual([]);
    });

    it("gives back the bytes put, and the last bytes put under a path", async () => {
      const store = make();
      await store.put(VOL, "annotation/s0/0.0.0", bytes(1, 2, 3));
      expect(Array.from((await store.get(VOL, "annotation/s0/0.0.0"))!)).toEqual([1, 2, 3]);
      await store.put(VOL, "annotation/s0/0.0.0", bytes(9));
      expect(Array.from((await store.get(VOL, "annotation/s0/0.0.0"))!)).toEqual([9]);
    });

    it("takes a path with or without a leading slash as the same file", async () => {
      const store = make();
      await store.put(VOL, "/.zattrs", bytes(1));
      expect(await store.get(VOL, ".zattrs")).toBeDefined();
      expect(await store.list(VOL)).toEqual([".zattrs"]);
    });

    it("deletes a file, and deleting one that is not there is nothing", async () => {
      const store = make();
      await store.put(VOL, ".zgroup", bytes(1));
      await store.delete(VOL, ".zgroup");
      expect(await store.get(VOL, ".zgroup")).toBeUndefined();
      await expect(store.delete(VOL, ".zgroup")).resolves.toBeUndefined();
    });

    it("lists a volume's paths sorted, and only that volume's", async () => {
      const store = make();
      await store.put(VOL, "annotation/s0/1.0.0", bytes(1));
      await store.put(VOL, ".zgroup", bytes(1));
      await store.put(VOL, "annotation/s0/.zarray", bytes(1));
      await store.put(VOL, ".zattrs", bytes(1));
      await store.put(`${VOL}-other`, ".zattrs", bytes(1));  // an id this one is a prefix of
      await store.put("vol-other", "annotation/s0/1.0.0", bytes(1));
      expect(await store.list(VOL)).toEqual([".zattrs", ".zgroup", "annotation/s0/.zarray", "annotation/s0/1.0.0"]);
    });

    it("lists the paths under a prefix (a directory, with its slash), the prefix kept", async () => {
      const store = make();
      await store.put(VOL, ".zattrs", bytes(1));
      await store.put(VOL, "annotation/.zattrs", bytes(1));
      await store.put(VOL, "annotation/s0/.zarray", bytes(1));
      await store.put(VOL, "annotation/s0/0.0.0", bytes(1));
      await store.put(VOL, "annotation/s0/0.0.1", bytes(1));
      expect(await store.list(VOL, "annotation/s0/")).toEqual(["annotation/s0/.zarray", "annotation/s0/0.0.0", "annotation/s0/0.0.1"]);
      expect(await store.list(VOL, "annotation/")).toHaveLength(4);
      expect(await store.list(VOL, "nothing/")).toEqual([]);
    });

    it("names the volumes with any file, sorted, once each", async () => {
      const store = make();
      await store.put("vol-b", ".zattrs", bytes(1));
      await store.put("vol-a", ".zattrs", bytes(1));
      await store.put("vol-a", ".zgroup", bytes(1));
      expect(await store.volumes()).toEqual(["vol-a", "vol-b"]);
      await store.delete("vol-b", ".zattrs");
      expect(await store.volumes()).toEqual(["vol-a"]);
    });

    it("removes a volume whole, and no other", async () => {
      const store = make();
      await store.put(VOL, ".zattrs", bytes(1));
      await store.put(VOL, "annotation/s0/0.0.0", bytes(1));
      await store.put(`${VOL}-other`, ".zattrs", bytes(1));
      await store.removeVolume(VOL);
      expect(await store.list(VOL)).toEqual([]);
      expect(await store.volumes()).toEqual([`${VOL}-other`]);
      await expect(store.removeVolume("vol-never")).resolves.toBeUndefined();
    });
  });
}

paintStoreSuite("MemoryPaintStore", () => new MemoryPaintStore());

describe("paintKey", () => {
  it("is <volume>/<path>, the path's leading slashes dropped", () => {
    expect(paintKey(VOL, "annotation/s0/.zarray")).toBe(`${VOL}/annotation/s0/.zarray`);
    expect(paintKey(VOL, "/.zattrs")).toBe(`${VOL}/.zattrs`);
    expect(paintKey(VOL, "")).toBe(`${VOL}/`);  // a volume's whole range starts here
  });
});

describe("openPaintStore", () => {
  it("falls back to memory where there is no IndexedDB (Node), and says so", () => {
    const notes: string[] = [];
    const store = openPaintStore((line) => void notes.push(line));
    expect(store).toBeInstanceOf(MemoryPaintStore);
    expect(notes).toEqual(["paint store: no IndexedDB here, the paint lasts only as long as the page"]);
  });
});
