// The blockwise merger (src/infer/merger.ts): its equivalence map against
// neuroglancer's to_json(), the face extraction and erosion, the matching
// of faces across chunks (the Python's bookkeeping included), the throttled
// reporting, and the step as the registry builds it. Parity on a sequence
// of chunks is test/parity/merger.test.ts.
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  BlockwiseMerger, blockwiseMerger, EquivalenceMap, erodeLabels2d, FACE_OFFSETS, faceIds, faceOf, makeMerger, REPORT_EVERY_MS, throttled,
} from "../../src/infer/merger";
import type { NdArray, PostContext } from "../../src/infer/post";

/** A one-channel (1, z, y, x) block of labels from nested arrays. */
function block(planes: number[][][]): NdArray {
  return { data: Uint32Array.from(planes.flat(2)), shape: [1, planes.length, planes[0].length, planes[0][0].length] };
}

/** A (1, z, y, x) block filled with one id. */
function filled(shape: [number, number, number], id: number): NdArray {
  const n = shape[0] * shape[1] * shape[2];
  return { data: new Uint32Array(n).fill(id), shape: [1, ...shape] };
}

function at(chunkIndex: number[]): PostContext {
  return { chunkIndex, chunkNumVoxels: 0 };
}

const big = (values: number[]) => BigUint64Array.from(values.map(BigInt));

describe("EquivalenceMap (neuroglancer.equivalence_map.EquivalenceMap.to_json)", () => {
  it("joins ids and says whether anything changed", () => {
    const map = new EquivalenceMap();
    expect(map.union(1n, 2n)).toBe(true);
    expect(map.union(2n, 1n), "already one set").toBe(false);
    expect(map.union(3n, 4n)).toBe(true);
    expect(map.union(4n, 1n)).toBe(true);
    expect(map.find(3n)).toBe(map.find(2n));
    expect(map.count).toBe(4);
  });
  it("writes the sets sorted by value, not by text", () => {
    const map = new EquivalenceMap();
    map.union(10n, 9n);
    map.union(100n, 2n);
    expect(map.toJSON()).toEqual([["2", "100"], ["9", "10"]]);
  });
  it("sorts the sets lexicographically, a shorter prefix first", () => {
    const map = new EquivalenceMap();
    map.union(1n, 5n);
    map.union(1n, 9n);
    map.union(2n, 3n);
    map.union(1n, 12n);
    expect(map.toJSON()).toEqual([["1", "5", "9", "12"], ["2", "3"]]);
  });
  it("keeps an id joined only with itself as a set of one, as the Python does", () => {
    const map = new EquivalenceMap();
    expect(map.union(7n, 7n)).toBe(false);
    map.union(1n, 2n);
    expect(map.toJSON()).toEqual([["1", "2"], ["7"]]);
    expect(map.count).toBe(3);
  });
  it("find() makes an id known, in a set of its own", () => {
    const map = new EquivalenceMap();
    expect(map.find(5n)).toBe(5n);
    expect(map.toJSON()).toEqual([["5"]]);
  });
  it("handles ids past 2^53 exactly", () => {
    const map = new EquivalenceMap();
    const a = 2n ** 60n + 1n, b = 2n ** 60n + 2n;
    map.union(b, a);
    expect(map.toJSON()).toEqual([[String(a), String(b)]]);
  });
});

describe("faceOf (self.slices)", () => {
  // A 2x3x4 block whose labels are flat index + 1, so a value says where it came from.
  const shape = [2, 3, 4];
  const labels = Uint32Array.from({ length: 24 }, (_, i) => i + 1);
  const value = (z: number, y: number, x: number) => BigInt((z * 3 + y) * 4 + x + 1);

  it("lists the six faces in the Python's order", () => {
    expect(FACE_OFFSETS).toEqual([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]);
  });
  it("a z face is the (y, x) plane at z = 0 or the last z", () => {
    const first = faceOf(labels, shape, [-1, 0, 0]);
    expect([first.height, first.width]).toEqual([3, 4]);
    expect(first.values[1 * 4 + 2]).toBe(value(0, 1, 2));
    const last = faceOf(labels, shape, [1, 0, 0]);
    expect(last.values[1 * 4 + 2]).toBe(value(1, 1, 2));
  });
  it("a y face is the (z, x) plane", () => {
    const face = faceOf(labels, shape, [0, 1, 0]);
    expect([face.height, face.width]).toEqual([2, 4]);
    expect(face.values[1 * 4 + 3]).toBe(value(1, 2, 3));
  });
  it("an x face is the (z, y) plane", () => {
    const face = faceOf(labels, shape, [0, 0, -1]);
    expect([face.height, face.width]).toEqual([2, 3]);
    expect(face.values[1 * 3 + 2]).toBe(value(1, 2, 0));
  });
  it("reads labels from any array type", () => {
    expect(faceOf(new Float32Array([1.9, 2]), [1, 1, 2], [-1, 0, 0]).values).toEqual(big([1, 2]));
    expect(faceOf(new BigUint64Array([2n ** 40n, 0n]), [1, 1, 2], [-1, 0, 0]).values[0]).toBe(2n ** 40n);
  });
});

describe("erodeLabels2d (fastmorph.erode on a face)", () => {
  it("drops the border and keeps an id whose eight neighbours are the same", () => {
    const ones = new BigUint64Array(5 * 6).fill(1n);
    const once = erodeLabels2d(ones, 5, 6, 1);
    const expected = new BigUint64Array(5 * 6);
    for (let i = 1; i < 4; i++) for (let j = 1; j < 5; j++) expected[i * 6 + j] = 1n;
    expect(once).toEqual(expected);
    // A second erosion shrinks the 3x4 interior to 1x2.
    const twice = erodeLabels2d(ones, 5, 6, 2);
    expect(Array.from(twice).filter((v) => v !== 0n)).toHaveLength(2);
    expect(twice[2 * 6 + 2]).toBe(1n);
    expect(twice[2 * 6 + 3]).toBe(1n);
  });
  it("erodes the boundary between two ids, as multilabel erosion does", () => {
    // The 7x9 plane of the fastmorph probe: ids 1 (columns 1-3) and 2 (columns 4-7) in rows 1-5.
    const plane = new BigUint64Array(7 * 9);
    for (let i = 1; i < 6; i++) {
      for (let j = 1; j < 4; j++) plane[i * 9 + j] = 1n;
      for (let j = 4; j < 8; j++) plane[i * 9 + j] = 2n;
    }
    const eroded = erodeLabels2d(plane, 7, 9, 1);
    const kept: [number, number, bigint][] = [];
    for (let i = 0; i < 7; i++) for (let j = 0; j < 9; j++) if (eroded[i * 9 + j]) kept.push([i, j, eroded[i * 9 + j]]);
    // Rows 2-4: id 1 survives in column 2 only, id 2 in columns 5 and 6 (fastmorph's output).
    expect(kept).toEqual([
      [2, 2, 1n], [2, 5, 2n], [2, 6, 2n],
      [3, 2, 1n], [3, 5, 2n], [3, 6, 2n],
      [4, 2, 1n], [4, 5, 2n], [4, 6, 2n],
    ]);
  });
  it("leaves the face alone for zero iterations and erodes a thin face away", () => {
    const values = big([1, 1, 1]);
    expect(erodeLabels2d(values, 1, 3, 0)).toBe(values);
    expect(erodeLabels2d(values, 1, 3, 1)).toEqual(big([0, 0, 0]));
  });
});

describe("faceIds", () => {
  it("keys the nonzero ids by their face position, as np.where gives it", () => {
    const labels = Uint32Array.from([0, 3, 0, 0, 0, 5]);  // (1, 2, 3): one z plane
    const ids = faceIds(labels, [1, 2, 3], [-1, 0, 0], 0);
    expect([...ids]).toEqual([["0,1", 3n], ["1,2", 5n]]);
  });
  it("applies the erosion first", () => {
    const labels = new Uint32Array(5 * 6).fill(4);
    const ids = faceIds(labels, [1, 5, 6], [1, 0, 0], 1);
    expect(ids.size).toBe(3 * 4);
    expect(ids.get("0,0")).toBeUndefined();
    expect(ids.get("1,1")).toBe(4n);
  });
});

describe("BlockwiseMerger", () => {
  it("joins the ids that meet across a shared face, and marks both faces done", () => {
    const merger = new BlockwiseMerger(0, 0);
    expect(merger.addChunk(filled([2, 2, 2], 1), [0, 0, 0])).toBe(false);
    expect(merger.waiting, "six faces recorded, none matched").toBe(6);
    expect(merger.equivalences.toJSON()).toEqual([]);
    // The x neighbour: its -x face meets the first chunk's +x face.
    expect(merger.addChunk(filled([2, 2, 2], 2), [0, 0, 1])).toBe(true);
    expect(merger.equivalences.toJSON()).toEqual([["1", "2"]]);
    // The matched pair is dropped when the next chunk comes, as the Python
    // pops its keys_to_skip before it calculates: twelve now, then
    // twelve plus six less the pair.
    expect(merger.waiting).toBe(12);
    expect(merger.addChunk(filled([2, 2, 2], 3), [5, 5, 5])).toBe(false);
    expect(merger.waiting, "the matched pair gone, a far chunk's six faces added").toBe(16);
  });
  it("joins by position: different ids at different positions give different sets", () => {
    const merger = new BlockwiseMerger(0, 0);
    // Chunk at the origin: id 1 in the top row, 2 in the bottom row, along x.
    merger.addChunk(block([[[1, 1], [2, 2]]]), [0, 0, 0]);
    // Its x neighbour carries 3 and 4 in the same rows.
    expect(merger.addChunk(block([[[3, 3], [4, 4]]]), [0, 0, 1])).toBe(true);
    expect(merger.equivalences.toJSON()).toEqual([["1", "3"], ["2", "4"]]);
  });
  it("joins nothing between chunks that are not neighbours, and nothing against background", () => {
    const merger = new BlockwiseMerger(0, 0);
    merger.addChunk(filled([2, 2, 2], 1), [0, 0, 0]);
    expect(merger.addChunk(filled([2, 2, 2], 2), [0, 0, 2])).toBe(false);
    // A y neighbour whose shared face is background where chunk 0 has ids.
    expect(merger.addChunk(block([[[0, 0], [3, 3]], [[0, 0], [3, 3]]]), [0, 1, 0])).toBe(false);
    expect(merger.equivalences.toJSON()).toEqual([]);
  });
  it("leaves a pair of faces alone while the one looked across to is empty, as the Python does", () => {
    const merger = new BlockwiseMerger(0, 0);
    // Both chunks empty: neither face is matched, both keep waiting.
    merger.addChunk(filled([1, 1, 1], 0), [0, 0, 0]);
    merger.addChunk(filled([1, 1, 1], 0), [0, 0, 1]);
    merger.addChunk(filled([1, 1, 1], 0), [5, 5, 5]);  // a far chunk, so the done faces (none) would be dropped now
    expect(merger.waiting).toBe(18);
    // One side empty, the other with an id: the pair is matched (the empty
    // face looks across at one with ids), nothing joined, both faces done
    // and dropped at the next chunk.
    const other = new BlockwiseMerger(0, 0);
    other.addChunk(filled([1, 1, 1], 0), [0, 0, 0]);
    other.addChunk(filled([1, 1, 1], 5), [0, 0, 1]);
    other.addChunk(filled([1, 1, 1], 0), [5, 5, 5]);
    expect(other.waiting).toBe(16);
    expect(other.equivalences.toJSON()).toEqual([]);
  });
  it("a chunk seen again changes nothing", () => {
    const merger = new BlockwiseMerger(0, 0);
    merger.addChunk(filled([2, 2, 2], 1), [0, 0, 0]);
    merger.addChunk(filled([2, 2, 2], 2), [0, 0, 1]);
    expect(merger.addChunk(filled([2, 2, 2], 1), [0, 0, 0])).toBe(false);
    expect(merger.waiting).toBe(10);
    expect(merger.equivalences.toJSON()).toEqual([["1", "2"]]);
  });
  it("erodes the faces first: a one-voxel contact does not join, a 3x3 patch does", () => {
    const thin = new BlockwiseMerger(0, 1);
    thin.addChunk(block([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]), [0, 0, 0]);  // a z plane of 3x3 with the centre set
    thin.addChunk(block([[[0, 0, 0], [0, 2, 0], [0, 0, 0]]]), [1, 0, 0]);
    expect(thin.equivalences.toJSON()).toEqual([]);
    const patch = new BlockwiseMerger(0, 1);
    patch.addChunk(filled([1, 3, 3], 1), [0, 0, 0]);
    patch.addChunk(filled([1, 3, 3], 2), [1, 0, 0]);
    expect(patch.equivalences.toJSON()).toEqual([["1", "2"]]);
  });
  it("reads the labels from the given channel", () => {
    const merger = new BlockwiseMerger(1, 0);
    const two = (id: number) => ({ data: Uint32Array.from([9, 9, id, id]), shape: [2, 1, 1, 2] });
    merger.addChunk(two(1), [0, 0, 0]);
    merger.addChunk(two(2), [0, 0, 1]);
    expect(merger.equivalences.toJSON()).toEqual([["1", "2"]]);
  });
  it("refuses a block or index of the wrong shape", () => {
    const merger = new BlockwiseMerger(0, 0);
    expect(() => merger.addChunk({ data: new Uint32Array(4), shape: [2, 2] }, [0, 0, 0])).toThrow(/\(c, z, y, x\)/);
    expect(() => merger.addChunk(filled([1, 1, 1], 1), [0, 0])).toThrow(/three axes/);
    expect(() => new BlockwiseMerger(1, 0).addChunk(filled([1, 1, 1], 1), [0, 0, 0])).toThrow(/channel 1 of an output with 1 channel/);
  });
});

describe("throttled", () => {
  afterEach(() => vi.useRealTimers());

  it("calls at once the first time, then once more when the interval has passed", () => {
    vi.useFakeTimers();
    const calls: number[] = [];
    const t = throttled(() => calls.push(Date.now()), 5000);
    t.request();
    expect(calls).toEqual([Date.now()]);
    expect(t.pending()).toBe(false);
    // A burst inside the interval: one trailing call, at the interval's end.
    vi.advanceTimersByTime(1000);
    for (let i = 0; i < 10; i++) t.request();
    expect(calls).toHaveLength(1);
    expect(t.pending()).toBe(true);
    vi.advanceTimersByTime(3999);
    expect(calls).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(calls).toHaveLength(2);
    expect(calls[1] - calls[0]).toBe(5000);
    expect(t.pending()).toBe(false);
  });
  it("calls at once again after a quiet interval", () => {
    vi.useFakeTimers();
    let calls = 0;
    const t = throttled(() => calls++, 5000);
    t.request();
    vi.advanceTimersByTime(5000);
    t.request();
    expect(calls).toBe(2);
  });
  it("the trailing call reads the state as it is then", () => {
    vi.useFakeTimers();
    let state = 0;
    const seen: number[] = [];
    const t = throttled(() => seen.push(state), 5000);
    state = 1; t.request();
    state = 2; t.request();
    state = 3; t.request();
    vi.runAllTimers();
    expect(seen).toEqual([1, 3]);
  });
});

describe("blockwiseMerger, the step", () => {
  afterEach(() => vi.useRealTimers());

  it("declares uint64 labels and casts the whole block to uint64", () => {
    expect(blockwiseMerger.declares({ name: "SimpleBlockwiseMerger" })).toEqual({ dtype: "uint64", isSegmentation: true });
    const step = blockwiseMerger.make({ name: "SimpleBlockwiseMerger" }, {});
    const out = step({ data: new Float32Array([1.5, 0, 7, 2]), shape: [2, 1, 1, 2] }, at([0, 0, 0]));
    expect(out.shape).toEqual([2, 1, 1, 2]);
    expect(out.data).toEqual(big([1, 0, 7, 2]));
  });
  it("reports the joined ids as sets, at once the first time and at most every 5 s after", () => {
    vi.useFakeTimers();
    const reports: string[][][] = [];
    const step = blockwiseMerger.make({ name: "SimpleBlockwiseMerger", channel: "0", face_erosion_iterations: "0" }, { onEquivalences: (p) => reports.push(p) });
    step(filled([1, 1, 1], 1), at([0, 0, 0]));
    expect(reports, "nothing joined yet").toEqual([]);
    step(filled([1, 1, 1], 2), at([0, 0, 1]));
    expect(reports).toEqual([[["1", "2"]]]);
    step(filled([1, 1, 1], 3), at([0, 0, 2]));
    expect(reports, "within the interval: held back").toHaveLength(1);
    vi.advanceTimersByTime(REPORT_EVERY_MS);
    expect(reports).toEqual([[["1", "2"]], [["1", "2", "3"]]]);
  });
  it("does not report when nothing changed, and runs without a hook", () => {
    vi.useFakeTimers();
    let reports = 0;
    const step = blockwiseMerger.make({ name: "SimpleBlockwiseMerger" }, { onEquivalences: () => reports++ });
    step(filled([1, 1, 1], 1), at([0, 0, 0]));
    step(filled([1, 1, 1], 1), at([0, 0, 1]));  // the same id meets itself: the sets do not change
    vi.runAllTimers();
    expect(reports).toBe(0);
    const silent = blockwiseMerger.make({ name: "SimpleBlockwiseMerger" }, {});
    silent(filled([1, 1, 1], 1), at([0, 0, 0]));
    expect(silent(filled([1, 1, 1], 2), at([0, 0, 1])).data).toEqual(big([2]));
  });
  it("a failing listener does not fail the chunk", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const step = blockwiseMerger.make({ name: "SimpleBlockwiseMerger" }, { onEquivalences: () => { throw new Error("viewer gone"); } });
    step(filled([1, 1, 1], 1), at([0, 0, 0]));
    expect(() => step(filled([1, 1, 1], 2), at([0, 0, 1]))).not.toThrow();
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });
  it("parses its parameters as the Python constructor does, and refuses bad ones", () => {
    const { merger } = makeMerger({ name: "SimpleBlockwiseMerger", channel: "1", face_erosion_iterations: "2" }, {});
    expect([merger.channel, merger.erosion]).toEqual([1, 2]);
    expect(() => makeMerger({ name: "SimpleBlockwiseMerger", face_erosion_iterations: "many" }, {})).toThrow(/face_erosion_iterations/);
    expect(() => makeMerger({ name: "SimpleBlockwiseMerger", face_erosion_iterations: -1 }, {})).toThrow(/not be negative/);
    expect(() => makeMerger({ name: "SimpleBlockwiseMerger", channel: "0.5" }, {})).toThrow(/channel/);
  });
});
