// Good regions on a volume's grid: a mark is a box of one output block
// centred where the viewer looks; the regions round-trip through the
// volume's good_regions.json, and anything else there reads as no regions;
// a region anchors the server's rehearsal patch, one output block centred
// on it and snapped to whole voxels (voxel 0's centre is the volume's
// dataset_offset_nm), clipped to the volume, and a region centred outside
// it or not a region at all is set aside; the boxes become each chunk's
// voxels inside them, counted once however many boxes cover them.
import { describe, expect, it } from "vitest";

import type { PaintVolume } from "../../src/contracts";
import { anchorBoxes, anchorChunks, compareIndex, defaultRegionSizeNm, GOOD_REGIONS_FILE, isGoodRegion, loadGoodRegions, newRegionId, regionAt, saveGoodRegions } from "../../src/finetune/good-regions";
import { MemoryPaintStore } from "../../src/state/paint-store";

/** The grid of a volume: 2×4×4 chunks of 8 nm voxels over 4×8×8, voxel 0 centred at 4 nm (its corner at 0). */
const VOLUME: Pick<PaintVolume, "chunk_size" | "output_voxel_size" | "dataset_offset_nm" | "dataset_shape_voxels"> = {
  chunk_size: [2, 4, 4], output_voxel_size: [8, 8, 8], dataset_offset_nm: [4, 4, 4], dataset_shape_voxels: [4, 8, 8],
};
/** A position in nm at the centre of voxel (z, y, x). */
const centreOf = (v: number[]) => v.map((i) => 4 + 8 * i);

describe("a mark", () => {
  it("is one output block, centred where the viewer looks, labelled in turn", () => {
    expect(defaultRegionSizeNm(VOLUME)).toEqual([16, 32, 32]);
    const region = regionAt([16, 32, 32], [16, 32, 32], 2, undefined, "abcd1234");
    expect(region).toEqual({ id: "abcd1234", label: "good-3", offset_nm: [8, 16, 16], shape_nm: [16, 32, 32] });
    expect(regionAt([0, 0, 0], [8, 8, 8], 0, "mito edge").label).toBe("mito edge");
    expect(newRegionId()).toMatch(/^[0-9a-f]{8}$/);
    expect(isGoodRegion(region)).toBe(true);
    for (const bad of [null, "good", { ...region, id: 1 }, { ...region, offset_nm: [1, 2] }, { ...region, shape_nm: [1, 0, 1] }, { ...region, offset_nm: [1, NaN, 2] }]) expect(isGoodRegion(bad)).toBe(false);
  });

  it("is kept in the volume's good_regions.json, and anything but a list there is no regions", async () => {
    const store = new MemoryPaintStore();
    expect(await loadGoodRegions(store, "vol")).toEqual([]);
    const regions = [regionAt([16, 32, 32], [16, 32, 32], 0), { kept: "as it is" }];
    await saveGoodRegions(store, "vol", regions);
    expect(await store.list("vol")).toEqual([GOOD_REGIONS_FILE]);
    expect(await loadGoodRegions(store, "vol")).toEqual(regions);
    await store.put("vol", GOOD_REGIONS_FILE, new TextEncoder().encode('{"not": "a list"}'));
    expect(await loadGoodRegions(store, "vol")).toEqual([]);
    await store.put("vol", GOOD_REGIONS_FILE, new TextEncoder().encode("not json"));
    expect(await loadGoodRegions(store, "vol")).toEqual([]);
  });
});

describe("anchorBoxes", () => {
  it("anchors one output block centred on the region, snapped to whole voxels", () => {
    // Marked at voxel (2, 4, 4)'s corner, the meeting point of eight voxels: the block is [1, 3) × [2, 6) × [2, 6).
    const { boxes } = anchorBoxes(VOLUME, [regionAt([16, 32, 32], [16, 32, 32], 0)]);
    expect(boxes.map(({ lo, hi }) => ({ lo, hi }))).toEqual([{ lo: [1, 2, 2], hi: [3, 6, 6] }]);
    // Marked at a voxel's centre, an even block takes the voxel and the one after it.
    expect(anchorBoxes(VOLUME, [regionAt(centreOf([1, 3, 3]), [16, 32, 32], 0)]).boxes[0]).toMatchObject({ lo: [1, 2, 2], hi: [3, 6, 6] });
    // Whatever the region's own size: the server's patch is one output block.
    expect(anchorBoxes(VOLUME, [regionAt([16, 32, 32], [80, 80, 80], 0)]).boxes[0]).toMatchObject({ lo: [1, 2, 2], hi: [3, 6, 6] });
  });

  it("anchors the slice the viewer is on for a one-slice block, the float noise of nm aside", () => {
    const flat = { ...VOLUME, chunk_size: [1, 4, 4] };
    for (const z of [0, 1, 2, 3]) {
      const { boxes } = anchorBoxes(flat, [regionAt(centreOf([z, 3, 3]), [8, 32, 32], 0)]);
      expect(boxes[0]).toMatchObject({ lo: [z, 2, 2], hi: [z + 1, 6, 6] });
    }
    // 4 + 8 × 2.5 in nm comes back from offset + shape / 2 as a hair under or over: still slice 2.
    const noisy = { id: "n", label: "n", offset_nm: [20 - 4.000000000000001, 0, 0], shape_nm: [8, 32, 32] };
    expect(anchorBoxes(flat, [noisy]).boxes[0].lo[0]).toBe(2);
  });

  it("clips the block to the volume, and sets aside a region centred outside it or not a region", () => {
    const edge = regionAt(centreOf([3, 7, 0]), [16, 32, 32], 0);
    const outside = regionAt([-10, 32, 32], [16, 32, 32], 1);
    const far = regionAt([16, 32, 1000], [16, 32, 32], 2);
    const { boxes, malformed, outside: skipped } = anchorBoxes(VOLUME, [edge, { id: "x" }, outside, far]);
    expect(boxes).toHaveLength(1);
    // z [3, 5), y [6, 10) and x [-1, 3), clipped to the volume's 4 × 8 × 8.
    expect(boxes[0]).toMatchObject({ region: edge, lo: [3, 6, 0], hi: [4, 8, 3] });
    expect(malformed).toEqual([{ id: "x" }]);
    expect(skipped).toEqual([outside, far]);
  });
});

describe("anchorChunks", () => {
  it("gives each chunk a box reaches its voxels inside, counted once, the chunks in grid order", () => {
    const one = anchorBoxes(VOLUME, [regionAt([16, 32, 32], [16, 32, 32], 0)]).boxes;
    const chunks = anchorChunks(VOLUME, one);
    expect(chunks.map((c) => c.index)).toEqual([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]);
    expect(chunks.map((c) => c.voxels)).toEqual(new Array(8).fill(4));
    // Chunk (0, 0, 0) holds z 1, y 2-3, x 2-3 of the box: its voxels (1, 2..3, 2..3) in C order.
    const inside = Array.from(chunks[0].inRegion).flatMap((v, i) => (v ? [i] : []));
    expect(inside).toEqual([1 * 16 + 2 * 4 + 2, 1 * 16 + 2 * 4 + 3, 1 * 16 + 3 * 4 + 2, 1 * 16 + 3 * 4 + 3]);
    // The same box twice covers the same voxels.
    expect(anchorChunks(VOLUME, [...one, ...one]).map((c) => c.voxels)).toEqual(new Array(8).fill(4));
    expect(compareIndex([0, 1, 10], [0, 2, 0])).toBeLessThan(0);
  });
});
