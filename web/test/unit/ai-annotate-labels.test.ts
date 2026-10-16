// The ids a box gets from a segmentation (src/ai-annotate/labels.ts) against
// cellmap_flow/finetune/session/fill.py's _fresh_ids and seed_labels, and
// scipy.ndimage.label's default (4-connected) labelling of a plane. The
// parity fixture checks the same against the Python's own numbers; here the
// rules are spelled out case by case.
import { describe, expect, it } from "vitest";

import { freshIds, label2d, seedLabels } from "../../src/ai-annotate/labels";
import { AIAnnotateError } from "../../src/ai-annotate/types";

describe("label2d", () => {
  it("numbers 4-connected objects in raster order of their first pixel", () => {
    // Two blocks and a lone pixel touching the second only at a corner.
    const mask = Uint8Array.from([
      1, 1, 0, 0, 0,
      1, 1, 0, 1, 1,
      0, 0, 0, 1, 1,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
    ]);
    mask[3 * 5 + 2] = 1;  // diagonal from (2, 3): apart under 4-connectivity
    const { objects, count } = label2d(mask, 5, 5);
    expect(count).toBe(3);
    expect(objects).toBeInstanceOf(Int32Array);
    expect(Array.from(objects)).toEqual([
      1, 1, 0, 0, 0,
      1, 1, 0, 2, 2,
      0, 0, 0, 2, 2,
      0, 0, 3, 0, 0,
      0, 0, 0, 0, 0,
    ]);
  });

  it("joins pieces that meet later in the scan (a U shape is one object)", () => {
    const mask = Uint8Array.from([
      1, 0, 1,
      1, 0, 1,
      1, 1, 1,
    ]);
    const { objects, count } = label2d(mask, 3, 3);
    expect(count).toBe(1);
    expect(Array.from(objects)).toEqual([1, 0, 1, 1, 0, 1, 1, 1, 1]);
  });

  it("an empty mask has no objects; any nonzero byte is foreground", () => {
    expect(label2d(new Uint8Array(6), 2, 3)).toEqual({ objects: new Int32Array(6), count: 0 });
    expect(label2d(Uint8Array.from([255, 0, 7]), 1, 3).count).toBe(2);
    expect(() => label2d(new Uint8Array(5), 2, 3)).toThrow(/2×3/);
  });
});

describe("freshIds", () => {
  it("takes the lowest ids the box does not use, from 2", () => {
    expect(freshIds(Uint8Array.from([0, 0, 1, 2, 3, 5]), 3, false)).toEqual([4, 6, 7]);
    expect(freshIds(new Uint8Array(4), 2, false)).toEqual([2, 3]);
    expect(freshIds(Uint16Array.from([0, 300, 7]), 2, false)).toEqual([2, 3]);
    expect(freshIds(Uint8Array.from([0, 2]), 0, false)).toEqual([]);
  });

  it("counts up from the box's largest id when asked, from 2 in an empty box", () => {
    expect(freshIds(Uint8Array.from([0, 0, 1, 2, 3, 5]), 3, true)).toEqual([6, 7, 8]);
    expect(freshIds(new Uint8Array(4), 2, true)).toEqual([2, 3]);
    expect(freshIds(Uint16Array.from([0, 300, 7]), 2, true)).toEqual([301, 302]);
  });

  it("refuses when the dtype has no room, with fill.py's message, as a 409", () => {
    const full = new Uint8Array(254);
    for (let i = 0; i < 254; i++) full[i] = i + 2;
    expect(freshIds(full.subarray(0, 253), 1, false)).toEqual([255]);
    for (const [existing, count, countUp] of [[full, 1, false], [Uint8Array.from([255]), 1, true], [Uint16Array.from([65534]), 2, true]] as const) {
      let caught: unknown;
      try {
        freshIds(existing, count, countUp);
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(AIAnnotateError);
      const e = caught as AIAnnotateError;
      expect(e.category).toBe("refused");
      expect(e.httpStatus).toBe(409);
      expect(e.userMessage).toBe(
        `${count} new objects do not fit in the volume's ${existing instanceof Uint16Array ? "uint16" : "uint8"} labels: make a new annotation volume, ` +
          "which for an instance model (Cellpose, affinities) now holds uint16 ids, or raise Min size or the threshold so the seed makes fewer objects",
      );
    }
  });
});

describe("seedLabels", () => {
  // Two objects in a 4 × 6 box: 1 at the top left, 2 at the bottom right.
  const objects = Int32Array.from([
    1, 1, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0,
    0, 0, 0, 2, 2, 2,
    0, 0, 0, 2, 2, 2,
  ]);

  it("gives each object a fresh id and 1 elsewhere, in the box's dtype", () => {
    const labels = seedLabels(objects, new Uint8Array(24), false);
    expect(labels).toBeInstanceOf(Uint8Array);
    expect(Array.from(labels)).toEqual(Array.from(objects, (o) => (o ? o + 1 : 1)));
    const wide = seedLabels(objects, new Uint16Array(24), true);
    expect(wide).toBeInstanceOf(Uint16Array);
    expect(Array.from(wide)).toEqual(Array.from(objects, (o) => (o ? o + 1 : 1)));
  });

  it("an object the user painted part of keeps the id painted most; others get fresh ids past the painted ones", () => {
    const existing = new Uint8Array(24);
    existing[0] = 7;  // one voxel of object 1 painted 7
    existing[3 * 6] = 9;  // paint outside both objects
    const labels = seedLabels(objects, existing, false);
    expect(labels[0]).toBe(7);
    expect(labels[1 * 6 + 1]).toBe(7);
    expect(labels[2 * 6 + 3]).toBe(2);  // the lowest free id: 7 and 9 are in use
    expect(labels[3 * 6]).toBe(1);  // what the paint held is not written by the seed: the labels say background there
  });

  it("the id with more voxels wins; on a tie the larger id, as the Python's stable sort leaves it", () => {
    const majority = new Uint8Array(24);
    majority[0] = majority[1] = 4;
    majority[6] = 6;
    expect(seedLabels(objects, majority, false)[7]).toBe(4);
    const tie = new Uint8Array(24);
    tie[0] = 4;
    tie[6] = 6;
    expect(seedLabels(objects, tie, false)[7]).toBe(6);
  });

  it("an object id the segmenter skipped is no object; nothing painted means all background", () => {
    const sparse = Int32Array.from([1, 0, 0, 0, 3]);
    expect(Array.from(seedLabels(sparse, new Uint8Array(5), false))).toEqual([2, 1, 1, 1, 3]);
    expect(Array.from(seedLabels(new Int32Array(4), Uint8Array.from([0, 4, 0, 0]), false))).toEqual([1, 1, 1, 1]);
    expect(() => seedLabels(new Int32Array(3), new Uint8Array(4), false)).toThrow(/3 object labels/);
  });
});
