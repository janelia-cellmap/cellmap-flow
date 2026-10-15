// AffinityPostprocessor and the mutex watershed behind it, on hand-made
// cases small enough to follow: the neighborhood literal, the greedy pass
// over the edges (merges, mutexes, a mutex blocking a later weaker merge,
// the weight-0 edge that does nothing), mwatershed's background rules, the
// mean-affinity background, the first-appearance numbering, remove_small,
// the 1/255 scaling of integer input, the Morton offset, and the time a
// real block takes. Where a case says what mwatershed gives, that value was
// printed by mwatershed.agglom on the pixi environment.
import { describe, expect, it } from "vitest";

import {
  DEFAULT_NEIGHBORHOOD, affinityPostprocessor, agglom, mutexWatershed, parseNeighborhood, removeSmall,
} from "../../src/infer/affinity";
import { type NdArray, applyPostprocessors, isSegmentation, outputChannels, outputDtype, validatePostprocessors } from "../../src/infer/post";

const AFFINITY = { name: "AffinityPostprocessor" };
const UNIT_OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
/** Along x only: offsets of 1 and 2 voxels. */
const X1 = [0, 0, 1], X2 = [0, 0, 2];
const origin = { chunkIndex: [0, 0, 0], chunkNumVoxels: 1 };

/** A one-row block of `channels` channels over `values[c]`, shape [C, 1, 1, n]. */
function row(...channels: number[][]): { aff: Float64Array; spatial: number[]; nd: NdArray } {
  const n = channels[0]!.length;
  const aff = Float64Array.from(channels.flat());
  return { aff, spatial: [1, 1, n], nd: { data: aff, shape: [channels.length, 1, 1, n] } };
}

const ids = (labels: ArrayLike<number> | ArrayLike<bigint>) => Array.from(labels as ArrayLike<number | bigint>, Number);

describe("parseNeighborhood (ast.literal_eval of the parameter)", () => {
  const DEFAULT = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [9, 0, 0], [0, 9, 0], [0, 0, 9]];

  it("is the class default without a value", () => {
    expect(parseNeighborhood(undefined)).toEqual(DEFAULT);
    expect(parseNeighborhood(null)).toEqual(DEFAULT);
    expect(parseNeighborhood("")).toEqual(DEFAULT);
    expect(parseNeighborhood(DEFAULT_NEIGHBORHOOD)).toEqual(DEFAULT);
  });

  it("reads the Python class's own literal: several lines, trailing commas", () => {
    const literal = `[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
                [9, 0, 0],
                [0, 9, 0],
                [0, 0, 9],
            ]`;
    expect(parseNeighborhood(literal)).toEqual(DEFAULT);
  });

  it("reads tuples, negative offsets and odd spacing", () => {
    expect(parseNeighborhood("[(0, 0, 1), (0, -1, 0)]")).toEqual([[0, 0, 1], [0, -1, 0]]);
    expect(parseNeighborhood(" [[1,0,0],[0,1,0] , [0,0,+2]] ")).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 2]]);
  });

  it("takes a list (from a YAML) as it is", () => {
    expect(parseNeighborhood([[0, 0, 1], [0, 1, 0]])).toEqual([[0, 0, 1], [0, 1, 0]]);
  });

  it("refuses what is not a list of three-integer offsets", () => {
    expect(() => parseNeighborhood("[[1, 0], [0, 1]]")).toThrow(/three integers/);
    expect(() => parseNeighborhood("[[1.5, 0, 0]]")).toThrow(/neighborhood/);
    expect(() => parseNeighborhood("[]")).toThrow(/list of offsets/);
    expect(() => parseNeighborhood("[[1, 0, 0]")).toThrow(/missing \]/);
    expect(() => parseNeighborhood("[[1, 0, 0]] extra")).toThrow(/unexpected text/);
    expect(() => parseNeighborhood("[1, 2, 3]")).toThrow(/three integers/);
    expect(() => validatePostprocessors([{ ...AFFINITY, neighborhood: "nonsense" }])).toThrow(/neighborhood/);
  });
});

describe("agglom (mwatershed.agglom's partition and background)", () => {
  it("joins two voxels across a positive edge", () => {
    const { aff, spatial } = row([0.5, 0]);  // the edge (0, 1) has weight 0.5; voxel 1's own value has no partner
    const labels = ids(agglom(aff, spatial, [X1], 0));
    expect(labels[0]).toBe(labels[1]);
    expect(labels[0]).not.toBe(0);
  });

  it("files voxels without any positive edge as background, as mwatershed does", () => {
    const { aff, spatial } = row([-0.5, 0]);
    expect(ids(agglom(aff, spatial, [X1], 0))).toEqual([0, 0]);  // mwatershed: [0 0]
    const line = row([-0.5, -0.5, -0.5, -0.5, 0]);
    expect(ids(agglom(line.aff, line.spatial, [X1], 0))).toEqual([0, 0, 0, 0, 0]);  // mwatershed: [0 0 0 0 0]
  });

  it("lets a mutex block a later, weaker merge", () => {
    // edges (0,1) = +0.9, (1,2) = +0.4 along X1; (0,2) = -0.6 along X2:
    // 0 and 1 merge, then the mutex 0|2 is set, then (1,2) is blocked
    const { aff, spatial } = row([0.9, 0.4, 0], [-0.6, 0, 0]);
    const labels = ids(agglom(aff, spatial, [X1, X2], 0));
    expect(labels[0]).toBe(labels[1]);
    expect(labels[2]).not.toBe(labels[0]);
    expect(labels[2]).not.toBe(0);  // a singleton with a positive edge is an object (mwatershed: [1 1 2])
  });

  it("files a blocked singleton at the block's first voxel as background, as mwatershed does", () => {
    // the mirror image: (0,1) = +0.4, (1,2) = +0.9, (0,2) = -0.6; voxel 0 is left alone
    const { aff, spatial } = row([0.4, 0.9, 0], [-0.6, 0, 0]);
    const labels = ids(agglom(aff, spatial, [X1, X2], 0));
    expect(labels[1]).toBe(labels[2]);
    expect(labels[1]).not.toBe(0);
    expect(labels[0]).toBe(0);  // mwatershed: [0 2 2]
  });

  it("does nothing on an edge of weight exactly 0", () => {
    const { aff, spatial } = row([0, 0]);
    expect(ids(agglom(aff, spatial, [X1], 0))).toEqual([0, 0]);  // mwatershed: [0 0]
    // a zero edge beside a real merge: it neither joins nor keeps anything apart
    const three = row([0.5, 0, 0]);
    const labels = ids(agglom(three.aff, three.spatial, [X1], 0));
    expect(labels[0]).toBe(labels[1]);
    expect(labels[2]).toBe(0);
  });

  it("measures edges against the bias", () => {
    const { aff, spatial } = row([0.5, 0]);
    expect(ids(agglom(aff, spatial, [X1], 0.6))).toEqual([0, 0]);  // 0.5 - 0.6 < 0: a split, and no positive edge
    const labels = ids(agglom(aff, spatial, [X1], 0.4));
    expect(labels[0]).toBe(labels[1]);
  });

  it("keeps clusters apart across a negative edge", () => {
    const { aff, spatial } = row([0.5, -0.5, 0.5, 0]);
    const labels = ids(agglom(aff, spatial, [X1], 0));  // mwatershed: [1 1 3 3]
    expect(labels[0]).toBe(labels[1]);
    expect(labels[2]).toBe(labels[3]);
    expect(labels[0]).not.toBe(labels[2]);
    expect(labels.every((l) => l !== 0)).toBe(true);
  });

  it("takes a negative offset, and skips an offset longer than the block", () => {
    const { aff, spatial } = row([0, 0.5]);  // along -x: voxel 1's edge goes to voxel 0
    const labels = ids(agglom(aff, spatial, [[0, 0, -1]], 0));
    expect(labels[0]).toBe(labels[1]);
    expect(labels[0]).not.toBe(0);
    expect(ids(agglom(aff, spatial, [[0, 0, 9]], 0))).toEqual([0, 0]);  // no edge fits (mwatershed would panic)
  });

  it("works in 3-D with the unit offsets", () => {
    // a 2×2×2 block, channel per axis, every edge positive: one object
    const aff = new Float64Array(3 * 8).fill(0.8);
    const labels = ids(agglom(aff, [2, 2, 2], UNIT_OFFSETS, 0));
    expect(new Set(labels).size).toBe(1);
    expect(labels[0]).not.toBe(0);
    // the z edges negative, the rest positive: two slabs
    for (let i = 0; i < 8; i++) aff[i] = -0.8;
    const slabs = ids(agglom(aff, [2, 2, 2], UNIT_OFFSETS, 0));
    expect(new Set(slabs.slice(0, 4)).size).toBe(1);
    expect(new Set(slabs.slice(4)).size).toBe(1);
    expect(slabs[0]).not.toBe(slabs[4]);
  });

  it("refuses NaN affinities, as mwatershed does", () => {
    const { aff, spatial } = row([Number.NaN, 0]);
    expect(() => agglom(aff, spatial, [X1], 0)).toThrow(/NaN/);
  });
});

describe("mutexWatershed (segment.mutex_watershed)", () => {
  it("numbers the objects 1..n in order of first appearance, 0 for background", () => {
    // {2,3} merge first (+0.9), {0,1} after (+0.3); the numbering follows the voxels, not the merges
    const { nd } = row([0.3, -0.1, 0.9, 0.9]);
    expect(ids(mutexWatershed(nd, [X1], 0))).toEqual([1, 1, 2, 2]);
  });

  it("makes a fragment background when its mean affinity is below the bias", () => {
    // channel 0 (offset 1) joins {0,1} and {2,3}; channel 1 (offset 2) is very negative at 0 and 1,
    // so {0,1}'s mean over both channels is below 0 while {2,3}'s is not
    const { nd } = row([0.9, -0.5, 0.8, 0.8], [-0.9, -0.9, 0.9, 0.9]);
    expect(ids(mutexWatershed(nd, [X1, X2], 0))).toEqual([0, 0, 1, 1]);
    // the mean is of the affinities before the bias is subtracted, compared with the bias
    const { nd: weak } = row([0.5, 0.5]);
    expect(ids(mutexWatershed(weak, [X1], 0.4))).toEqual([1, 1]);  // mean 0.5 >= 0.4
    expect(ids(mutexWatershed(weak, [X1], 0.5))).toEqual([0, 0]);  // the edge is 0: no positive edge at all
  });

  it("cuts the offsets to the channels, and uses every channel for the mean", () => {
    const { nd } = row([0.5, 0.5]);
    expect(ids(mutexWatershed(nd, [X1, X2, [1, 0, 0]], 0))).toEqual([1, 1]);
  });

  it("returns uint64 labels of the block's spatial shape", () => {
    const { nd } = row([0.5, 0.5]);
    const labels = mutexWatershed(nd, [X1], 0);
    expect(labels).toBeInstanceOf(BigUint64Array);
    expect(labels.length).toBe(2);
  });

  it("removes small objects when asked", () => {
    const { nd } = row([0.3, -0.1, 0.9, 0.9, 0.9]);  // {0,1} and {2,3,4}
    expect(ids(mutexWatershed(nd, [X1], 0))).toEqual([1, 1, 2, 2, 2]);
    expect(ids(mutexWatershed(nd, [X1], 0, 3))).toEqual([0, 0, 1, 1, 1]);
  });
});

describe("removeSmall (segment.remove_small)", () => {
  it("drops objects under min_size and renumbers the rest in the order of their ids", () => {
    expect(ids(removeSmall([1, 1, 2, 0, 3, 3, 3], 2))).toEqual([1, 1, 0, 0, 2, 2, 2]);
    expect(ids(removeSmall([1, 1, 2, 0, 3, 3, 3], 3))).toEqual([0, 0, 0, 0, 1, 1, 1]);
    expect(ids(removeSmall([2, 2, 1], 2))).toEqual([1, 1, 0]);
  });

  it("removes nothing for min_size 1 or less, or an empty block, and returns uint64", () => {
    const out = removeSmall(new Uint32Array([5, 0, 3]), 1);
    expect(out).toBeInstanceOf(BigUint64Array);
    expect(ids(out)).toEqual([5, 0, 3]);
    expect(ids(removeSmall([5, 0, 3], 0))).toEqual([5, 0, 3]);
    expect(ids(removeSmall([5, 0, 3], 1.9))).toEqual([5, 0, 3]);  // int(1.9) is 1
    expect(ids(removeSmall([0, 0], 10))).toEqual([0, 0]);
  });

  it("takes uint64 labels too", () => {
    expect(ids(removeSmall(new BigUint64Array([1n, 2n, 2n]), 2))).toEqual([0, 1, 1]);
  });
});

describe("AffinityPostprocessor", () => {
  it("declares one channel of uint64 labels", () => {
    expect(affinityPostprocessor.declares(AFFINITY).dtype).toBe("uint64");
    expect(affinityPostprocessor.declares(AFFINITY).channels!(9)).toBe(1);
    expect(affinityPostprocessor.declares(AFFINITY).isSegmentation).toBe(true);
    expect(outputDtype([AFFINITY], "float32")).toBe("uint64");
    expect(outputChannels([AFFINITY], 9)).toBe(1);
    expect(isSegmentation([AFFINITY])).toBe(true);
  });

  it("returns [1, z, y, x] uint64 labels", () => {
    const { nd } = row([0.5, 0.5]);
    const out = applyPostprocessors(nd, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]" }], origin);
    expect(out.shape).toEqual([1, 1, 1, 2]);
    expect(out.data).toBeInstanceOf(BigUint64Array);
    expect(ids(out.data)).toEqual([1, 1]);
  });

  it("scales integer input by 1/255 and takes float input as it is", () => {
    // 255/255 - 0.5 > 0 joins the pair; the mean (1 + 0) / 2 = 0.5 is not below the bias
    const bytes: NdArray = { data: new Uint8Array([255, 0]), shape: [1, 1, 1, 2] };
    expect(ids(applyPostprocessors(bytes, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]", bias: 0.5 }], origin).data)).toEqual([1, 1]);
    // the same values as floats are affinities of 255 and 0, not 1 and 0: still one object (255 - 0.5 > 0)
    const floats: NdArray = { data: new Float32Array([255, 0]), shape: [1, 1, 1, 2] };
    expect(ids(applyPostprocessors(floats, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]", bias: 0.5 }], origin).data)).toEqual([1, 1]);
    // ... and 1 and 0 as bytes would be 1/255 - 0.5 < 0: no object
    const faint: NdArray = { data: new Uint8Array([1, 0]), shape: [1, 1, 1, 2] };
    expect(ids(applyPostprocessors(faint, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]", bias: 0.5 }], origin).data)).toEqual([0, 0]);
  });

  it("takes the bias from the form's string", () => {
    const { nd } = row([0.5, 0.5]);
    expect(ids(applyPostprocessors(nd, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]", bias: "0.6" }], origin).data)).toEqual([0, 0]);
    expect(() => validatePostprocessors([{ ...AFFINITY, bias: "high" }])).toThrow(/bias/);
  });

  it("offsets the ids by the chunk's Morton increment", () => {
    const { nd } = row([0.3, -0.1, 0.9, 0.9]);
    const out = applyPostprocessors(nd, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]" }], { chunkIndex: [2, 1, 3], chunkNumVoxels: 336 });
    // interleave(2, 1, 3) = 46: 336 * 46 = 15456 on every object, background stays 0
    expect(Array.from(out.data as BigUint64Array)).toEqual([15457n, 15457n, 15458n, 15458n]);
    const { nd: none } = row([-0.5, 0]);
    expect(Array.from(applyPostprocessors(none, [{ ...AFFINITY, neighborhood: "[[0, 0, 1]]" }], { chunkIndex: [2, 1, 3], chunkNumVoxels: 336 }).data as BigUint64Array)).toEqual([0n, 0n]);
  });

  it("segments a 60³ block of 3 and of 9 channels in about the time the Rust mwatershed takes", () => {
    // random logits as a model outputs them: ~650 k edges for 3 channels, 1.8 M for 9
    const seeded = (seed: number) => () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
    const timed = (channels: number, neighborhood?: string) => {
      const rnd = seeded(channels);
      const data = Float32Array.from({ length: channels * 60 ** 3 }, () => rnd() * 2 - 1);
      const step = neighborhood ? { ...AFFINITY, neighborhood } : AFFINITY;
      const start = performance.now();
      const out = applyPostprocessors({ data, shape: [channels, 60, 60, 60] }, [step], { chunkIndex: [1, 2, 3], chunkNumVoxels: 60 ** 3 });
      const ms = performance.now() - start;
      expect(out.shape).toEqual([1, 60, 60, 60]);
      expect((out.data as BigUint64Array).some((v) => v !== 0n)).toBe(true);
      return ms;
    };
    const three = timed(3, "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]");
    const nine = timed(9);
    console.log(`AffinityPostprocessor on 60³: 3 channels ${three.toFixed(0)} ms, 9 channels ${nine.toFixed(0)} ms`);
    // About 420 and 770 ms in node on the workstation this was written on,
    // where mwatershed.agglom (Rust, what the Python runs) took 350 and 620 ms
    // on the same blocks. The bounds are loose so that a slow CI machine does
    // not fail the test; they catch a pathological slowdown, not a regression.
    expect(three).toBeLessThan(2500);
    expect(nine).toBeLessThan(5000);
  });
});
