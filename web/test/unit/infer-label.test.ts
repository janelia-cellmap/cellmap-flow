// The connected-components labeller (src/infer/label.ts): its union-find,
// the neighbour offsets of each connectivity, scipy's numbering on small
// hand-made masks, per-slice labelling, remove_small, the parameter
// parsing of the Python constructor, and the step as a whole. Parity with
// scipy on a random block is test/parity/label.test.ts.
import { describe, expect, it } from "vitest";

import {
  asBoolLike, asConnectivity, asInt, connectedComponents, earlierOffsets, foregroundOf, labelComponents, labelPostprocessor, toUint32, UnionFind,
} from "../../src/infer/label";
import type { PostContext } from "../../src/infer/post";

const context: PostContext = { chunkIndex: [0, 0, 0], chunkNumVoxels: 0 };

/** A (z, y, x) mask from nested arrays, as a uint8 foreground mask. */
function mask(planes: number[][][]): { data: Uint8Array; shape: number[] } {
  const shape = [planes.length, planes[0].length, planes[0][0].length];
  return { data: Uint8Array.from(planes.flat(2)), shape };
}

function labels(planes: number[][][], connectivity: number, minSize = 0, perSlice = false): number[] {
  const m = mask(planes);
  return Array.from(connectedComponents(m.data, m.shape, { connectivity, minSize, perSlice }));
}

describe("UnionFind", () => {
  it("adds elements in sets of their own and joins them", () => {
    const sets = new UnionFind(2);
    const a = sets.add(), b = sets.add(), c = sets.add();  // grows past its capacity
    expect([a, b, c]).toEqual([0, 1, 2]);
    expect(sets.count).toBe(3);
    expect(sets.find(a)).not.toBe(sets.find(b));
    expect(sets.union(a, b)).toBe(true);
    expect(sets.find(a)).toBe(sets.find(b));
    expect(sets.union(b, a), "already joined").toBe(false);
    expect(sets.find(c)).not.toBe(sets.find(a));
    expect(sets.union(c, a)).toBe(true);
    expect(new Set([sets.find(a), sets.find(b), sets.find(c)]).size).toBe(1);
  });
  it("grows far past its first capacity", () => {
    const sets = new UnionFind(1);
    for (let i = 0; i < 1000; i++) sets.add();
    for (let i = 1; i < 1000; i++) sets.union(0, i);
    expect(sets.count).toBe(1000);
    expect(sets.find(999)).toBe(sets.find(0));
  });
});

describe("earlierOffsets (generate_binary_structure, the visited half)", () => {
  it("gives the face, edge and corner neighbours before a voxel in raster order", () => {
    expect(earlierOffsets(3, 1)).toEqual([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]);
    expect(earlierOffsets(3, 2)).toHaveLength(9);   // 3 faces + 6 edges
    expect(earlierOffsets(3, 3)).toHaveLength(13);  // + 4 corners
    for (const [dz, dy, dx] of earlierOffsets(3, 3)) {
      expect(dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0))), `${[dz, dy, dx]} is earlier`).toBe(true);
    }
  });
  it("in 2-D stays in the plane", () => {
    expect(earlierOffsets(2, 1)).toEqual([[0, -1, 0], [0, 0, -1]]);
    expect(earlierOffsets(2, 2)).toEqual([[0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1]]);
  });
});

describe("labelComponents numbers objects as scipy does", () => {
  it("by the raster order of their first voxel", () => {
    // Four separate voxels: 1..4 row by row.
    expect(labels([[[1, 0, 1], [0, 0, 0], [1, 0, 1]]], 1)).toEqual([1, 0, 2, 0, 0, 0, 3, 0, 4]);
    // A U whose first voxel is the left arm's top: one object, id 1.
    expect(labels([[[1, 0, 1], [1, 1, 1]]], 1)).toEqual([1, 0, 1, 1, 1, 1]);
    // An object met later than a smaller one gets the later id, whatever its size.
    expect(labels([[[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]], 1)).toEqual([1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2]);
  });
  it("joins objects that only meet later in the scan", () => {
    // Two arms that meet at the bottom: the right arm is met first on row
    // 0 after the left? No: the left arm (column 0) comes first in raster
    // order, the right arm is provisionally another label until row 2 joins
    // them; the object keeps the left arm's id, 1.
    expect(labels([[[1, 0, 1], [1, 0, 1], [1, 1, 1]]], 1)).toEqual([1, 0, 1, 1, 0, 1, 1, 1, 1]);
  });
  it("a diagonal in a plane is one object from connectivity 2", () => {
    const diagonal = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]];
    expect(labels(diagonal, 1)).toEqual([1, 0, 0, 0, 2, 0, 0, 0, 3]);
    expect(labels(diagonal, 2)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(labels(diagonal, 3)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
  it("two voxels touching only at a corner are one object at connectivity 3 only", () => {
    const corner = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]];
    expect(labels(corner, 1)).toEqual([1, 0, 0, 0, 0, 0, 0, 2]);
    expect(labels(corner, 2)).toEqual([1, 0, 0, 0, 0, 0, 0, 2]);
    expect(labels(corner, 3)).toEqual([1, 0, 0, 0, 0, 0, 0, 1]);
  });
  it("two voxels touching along an edge are one object from connectivity 2", () => {
    const edge = [[[1, 0], [0, 0]], [[0, 1], [0, 0]]];
    expect(labels(edge, 1)).toEqual([1, 0, 0, 0, 0, 2, 0, 0]);
    expect(labels(edge, 2)).toEqual([1, 0, 0, 0, 0, 1, 0, 0]);
  });
  it("counts nonzero as foreground, whatever the value", () => {
    const m = { data: Uint8Array.from([0, 7, 0, 255]), shape: [1, 1, 4] };
    expect(Array.from(connectedComponents(m.data, m.shape, { connectivity: 1, minSize: 0, perSlice: false }))).toEqual([0, 1, 0, 2]);
    expect(Array.from(foregroundOf(new Float32Array([0, 0.5, -1, 0])))).toEqual([0, 1, 1, 0]);
    expect(Array.from(foregroundOf(new BigUint64Array([0n, 3n])))).toEqual([0, 1]);
  });
  it("refuses a mask that is not of the given shape", () => {
    expect(() => labelComponents(new Uint8Array(5), [1, 2, 2], earlierOffsets(3, 1))).toThrow(/shape/);
  });
});

describe("per_slice", () => {
  it("labels each plane on its own, ids following the planes before", () => {
    // One object per plane, stacked: in 3-D one object; per slice, 1 then 2.
    const stacked = [[[1, 1], [0, 0]], [[1, 1], [0, 0]]];
    expect(labels(stacked, 1)).toEqual([1, 1, 0, 0, 1, 1, 0, 0]);
    expect(labels(stacked, 1, 0, true)).toEqual([1, 1, 0, 0, 2, 2, 0, 0]);
  });
  it("uses the 2-D structure: connectivity 3 is 2, and nothing joins across planes", () => {
    const planes = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]];
    expect(labels(planes, 1, 0, true)).toEqual([1, 0, 0, 2, 0, 3, 4, 0]);
    expect(labels(planes, 3, 0, true)).toEqual([1, 0, 0, 1, 0, 2, 2, 0]);
  });
  it("counts an object's size per slice for min_size", () => {
    const stacked = [[[1, 1], [0, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]];
    expect(labels(stacked, 1, 2, true)).toEqual([1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0]);
  });
});

describe("min_size (segment.remove_small, affinity.ts's removeSmall)", () => {
  it("drops objects under min_size and renumbers the rest in order, as uint32", () => {
    // Objects of 1, 2 and 3 voxels in a row.
    const row = [[[1, 0, 1, 1, 0, 1, 1, 1]]];
    expect(labels(row, 1, 2)).toEqual([0, 0, 1, 1, 0, 2, 2, 2]);
    expect(labels(row, 1, 3)).toEqual([0, 0, 0, 0, 0, 1, 1, 1]);
    expect(labels(row, 1, 4)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(labels(row, 1, 1)).toEqual([1, 0, 2, 2, 0, 3, 3, 3]);
  });
});

describe("the parameters, as the Python constructor reads them", () => {
  it("asInt is Python's int()", () => {
    expect(asInt("4", 0, "n")).toBe(4);
    expect(asInt(" +7 ", 0, "n")).toBe(7);
    expect(asInt("-2", 0, "n")).toBe(-2);
    expect(asInt(4.7, 0, "n")).toBe(4);
    expect(asInt(true, 0, "n")).toBe(1);
    expect(asInt(undefined, 9, "n")).toBe(9);
    expect(asInt("", 9, "n")).toBe(9);
    expect(asInt(null, 9, "n")).toBe(9);
    expect(() => asInt("4.5", 0, "n")).toThrow(/n must be an integer/);
    expect(() => asInt("four", 0, "n")).toThrow(/integer/);
    expect(() => asInt(Number.NaN, 0, "n")).toThrow(/integer/);
  });
  it("asConnectivity accepts 1, 2, 3 as numbers or strings, capped at the dimension", () => {
    expect(asConnectivity(undefined)).toBe(1);
    expect(asConnectivity("2")).toBe(2);
    expect(asConnectivity(3)).toBe(3);
    expect(asConnectivity(3, 2)).toBe(2);
    expect(asConnectivity("3", 2)).toBe(2);
    expect(() => asConnectivity("4")).toThrow(/connectivity must be 1, 2 or 3, got "4"/);
    expect(() => asConnectivity(0)).toThrow(/connectivity must be 1, 2 or 3/);
    expect(() => asConnectivity("faces")).toThrow(/connectivity must be 1, 2 or 3/);
  });
  it("asBoolLike reads a form's true words, else Python's truth", () => {
    for (const yes of ["true", "True", " TRUE ", "1", "yes", "on"]) expect(asBoolLike(yes), yes).toBe(true);
    for (const no of ["false", "0", "", "no", "off", "anything"]) expect(asBoolLike(no), no).toBe(false);
    expect(asBoolLike(1)).toBe(true);
    expect(asBoolLike(0)).toBe(false);
    expect(asBoolLike(true)).toBe(true);
    expect(asBoolLike(undefined)).toBe(false);
    expect(asBoolLike(null)).toBe(false);
  });
});

describe("toUint32", () => {
  it("casts as numpy's astype(uint32): truncation, modulo 2^32", () => {
    expect(Array.from(toUint32(new Float32Array([2.7, 0, 1e10])))).toEqual([2, 0, 1e10 % 2 ** 32]);
    expect(Array.from(toUint32(new BigUint64Array([5n, 2n ** 32n + 1n])))).toEqual([5, 1]);
    expect(Array.from(toUint32(Uint8Array.from([255, 0])))).toEqual([255, 0]);
  });
});

describe("labelPostprocessor, the step", () => {
  it("replaces the channel by its labels and casts the others to uint32", () => {
    const step = labelPostprocessor.make({ name: "LabelPostprocessor", channel: "1" }, {});
    // Two channels of a 1x1x3 block: channel 0 is passed through (truncated), channel 1 labelled.
    const out = step({ data: new Float32Array([0.2, 2.7, -0, 1, 0, 1]), shape: [2, 1, 1, 3] }, context);
    expect(out.shape).toEqual([2, 1, 1, 3]);
    expect(out.data).toBeInstanceOf(Uint32Array);
    expect(Array.from(out.data as Uint32Array)).toEqual([0, 2, 0, 1, 0, 2]);
  });
  it("declares uint32 labels", () => {
    expect(labelPostprocessor.declares({ name: "LabelPostprocessor" })).toEqual({ dtype: "uint32", isSegmentation: true });
  });
  it("takes a negative channel from the end, as numpy indexes", () => {
    const step = labelPostprocessor.make({ name: "LabelPostprocessor", channel: -1 }, {});
    const out = step({ data: Uint8Array.from([9, 9, 1, 1]), shape: [2, 1, 1, 2] }, context);
    expect(Array.from(out.data as Uint32Array)).toEqual([9, 9, 1, 1]);
  });
  it("refuses a bad parameter when built, and a missing channel when run", () => {
    expect(() => labelPostprocessor.make({ name: "LabelPostprocessor", connectivity: "4" }, {})).toThrow(/connectivity/);
    expect(() => labelPostprocessor.make({ name: "LabelPostprocessor", min_size: "some" }, {})).toThrow(/min_size/);
    const step = labelPostprocessor.make({ name: "LabelPostprocessor", channel: 2 }, {});
    expect(() => step({ data: new Uint8Array(4), shape: [2, 1, 1, 2] }, context)).toThrow(/channel 2 of an output with 2 channel/);
    expect(() => step({ data: new Uint8Array(4), shape: [2, 2] }, context)).toThrow(/\(c, z, y, x\)/);
  });
  it("labels a 60^3 block quickly", () => {
    const n = 60;
    let seed = 12345;
    const data = new Uint8Array(n * n * n);
    for (let i = 0; i < data.length; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;  // a fixed pseudo-random mask, about half foreground
      data[i] = seed & 0x40000000 ? 1 : 0;
    }
    const step = labelPostprocessor.make({ name: "LabelPostprocessor", connectivity: 3, min_size: 2 }, {});
    const began = performance.now();
    const out = step({ data, shape: [1, n, n, n] }, context);
    const ms = performance.now() - began;
    expect(out.data.length).toBe(n * n * n);
    // Well under 100 ms on a workstation; the bound is loose for a busy CI machine.
    expect(ms, `took ${ms.toFixed(1)} ms`).toBeLessThan(500);
  });
});
