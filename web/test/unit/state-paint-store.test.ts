// The paint store (src/state/paint-store.ts): the PaintStore semantics as a
// suite over any implementation, run on MemoryPaintStore, the reference.
// IndexedDbPaintStore has the same semantics on db.ts's "paint" store, but
// Node has no IndexedDB and the project takes no fake of it as a dependency,
// so it runs only in a browser (the e2e test paints through the worker).
// A file never put and a deletion are paint-route.test.ts's (404, DELETE).
import { describe, expect, it } from "vitest";

import type { PaintStore } from "../../src/contracts";
import { MemoryPaintStore } from "../../src/state/paint-store";

const bytes = (...values: number[]) => new Uint8Array(values);
const VOL = "vol-0123abcd-20261015-120000";

/** The contract, on a fresh store from `make` per test. */
function paintStoreSuite(name: string, make: () => PaintStore) {
  describe(`${name} is a PaintStore`, () => {
    it("gives back the bytes put, and the last bytes put under a path", async () => {
      const store = make();
      await store.put(VOL, "annotation/s0/0.0.0", bytes(1, 2, 3));
      expect(Array.from((await store.get(VOL, "annotation/s0/0.0.0"))!)).toEqual([1, 2, 3]);
      await store.put(VOL, "annotation/s0/0.0.0", bytes(9));
      expect(Array.from((await store.get(VOL, "annotation/s0/0.0.0"))!)).toEqual([9]);
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
