// MortonSegmentationRelabeling and pymorton's interleave: the values below
// were printed by pymorton on the pixi environment
// (pymorton.interleave(z, y, x)); the step's behaviour follows the Python
// class (data.astype(np.uint64); the channel's nonzero ids += chunk_num_voxels
// * interleave(*chunk_corner)).
import { describe, expect, it } from "vitest";

import { interleave3, mortonIncrement, mortonRelabeling, toUint64 } from "../../src/infer/morton";
import { type NdArray, applyPostprocessors, isSegmentation, outputChannels, outputDtype } from "../../src/infer/post";

const MORTON = { name: "MortonSegmentationRelabeling" };

describe("interleave3 (pymorton.interleave)", () => {
  it("gives pymorton's values", () => {
    expect(interleave3(0, 0, 0)).toBe(0);
    expect(interleave3(1, 0, 0)).toBe(1);  // the first coordinate takes bits 0, 3, 6, ...
    expect(interleave3(0, 1, 0)).toBe(2);
    expect(interleave3(0, 0, 1)).toBe(4);
    expect(interleave3(1, 2, 3)).toBe(53);
    expect(interleave3(5, 0, 7)).toBe(357);
    expect(interleave3(2, 1, 3)).toBe(46);
    expect(interleave3(7, 7, 7)).toBe(511);
    expect(interleave3(100, 200, 300)).toBe(72256832);
  });

  it("keeps 10 bits per coordinate, as pymorton does", () => {
    expect(interleave3(1023, 1023, 1023)).toBe(2 ** 30 - 1);
    expect(interleave3(1024, 0, 0)).toBe(0);  // 1024 is 0 modulo 1024
    expect(interleave3(1025, 1, 2)).toBe(interleave3(1, 1, 2));
    expect(interleave3(2 ** 21 - 1, 2 ** 21 - 1, 2 ** 21 - 1)).toBe(2 ** 30 - 1);
  });

  it("takes integers only", () => {
    expect(() => interleave3(1.5, 0, 0)).toThrow(/integers/);
    expect(() => interleave3(0, Number.NaN, 0)).toThrow(/integers/);
  });
});

describe("mortonIncrement", () => {
  it("is chunk_num_voxels * interleave(z, y, x), a bigint", () => {
    expect(mortonIncrement([2, 1, 3], 336)).toBe(336n * 46n);
    expect(mortonIncrement([0, 0, 0], 216000)).toBe(0n);
    // past 2^53 as a product: 60³ voxels times the largest index
    expect(mortonIncrement([1023, 1023, 1023], 216000)).toBe(216000n * (2n ** 30n - 1n));
  });

  it("wants a three-axis index and a whole number of voxels", () => {
    expect(() => mortonIncrement([1, 2], 10)).toThrow(/three axes/);
    expect(() => mortonIncrement([1, 2, 3], 10.5)).toThrow(/chunk_num_voxels/);
  });
});

describe("toUint64 (astype(np.uint64))", () => {
  it("keeps integers, truncates floats toward zero, zeroes non-finite values, wraps negatives", () => {
    expect(Array.from(toUint64(new Uint8Array([0, 7, 255])))).toEqual([0n, 7n, 255n]);
    expect(Array.from(toUint64(new Uint32Array([4000000000])))).toEqual([4000000000n]);
    expect(Array.from(toUint64(new Float32Array([0.5, 2.7, 300.2, Number.NaN, Number.POSITIVE_INFINITY])))).toEqual([0n, 2n, 300n, 0n, 0n]);
    expect(Array.from(toUint64(new Float64Array([-1])))).toEqual([2n ** 64n - 1n]);
    const big = new BigUint64Array([1n, 2n ** 63n]);
    const copy = toUint64(big);
    expect(Array.from(copy)).toEqual([1n, 2n ** 63n]);
    expect(copy).not.toBe(big);
  });
});

describe("MortonSegmentationRelabeling", () => {
  // two channels of a 1×2×2 block: channel 0 holds ids 0, 1, 2, 2; channel 1 holds 5, 0, 6, 7
  const labels = (): NdArray => ({ data: new Uint8Array([0, 1, 2, 2, 5, 0, 6, 7]), shape: [2, 1, 2, 2] });
  const at = (chunkIndex: number[], chunkNumVoxels: number) => ({ chunkIndex, chunkNumVoxels });

  it("adds the chunk's increment to the nonzero ids of channel 0, casting the whole block to uint64", () => {
    const out = applyPostprocessors(labels(), [MORTON], at([1, 2, 3], 1000));
    expect(out.data).toBeInstanceOf(BigUint64Array);
    expect(out.shape).toEqual([2, 1, 2, 2]);
    // interleave(1, 2, 3) = 53: channel 0's ids become 53000 + id; channel 1 is cast only
    expect(Array.from(out.data as BigUint64Array)).toEqual([0n, 53001n, 53002n, 53002n, 5n, 0n, 6n, 7n]);
  });

  it("leaves chunk (0, 0, 0) as it is", () => {
    const out = applyPostprocessors(labels(), [MORTON], at([0, 0, 0], 1000));
    expect(Array.from(out.data as BigUint64Array)).toEqual([0n, 1n, 2n, 2n, 5n, 0n, 6n, 7n]);
  });

  it("takes the channel from the form's string", () => {
    const out = applyPostprocessors(labels(), [{ ...MORTON, channel: "1" }], at([1, 2, 3], 1000));
    expect(Array.from(out.data as BigUint64Array)).toEqual([0n, 1n, 2n, 2n, 53005n, 0n, 53006n, 53007n]);
    expect(() => applyPostprocessors(labels(), [{ ...MORTON, channel: 2 }], at([0, 0, 0], 4))).toThrow(/channel 2 of an output with 2/);
    expect(() => applyPostprocessors(labels(), [{ ...MORTON, channel: "x" }], at([0, 0, 0], 4))).toThrow(/channel/);
  });

  it("casts float ids by truncation before testing them for zero", () => {
    const floats: NdArray = { data: new Float32Array([0.5, 1.0, 2.7, 0]), shape: [1, 1, 2, 2] };
    const out = applyPostprocessors(floats, [MORTON], at([1, 0, 0], 100));
    expect(Array.from(out.data as BigUint64Array)).toEqual([0n, 101n, 102n, 0n]);
  });

  it("keeps uint64 ids from an earlier step and sums in uint64", () => {
    const big: NdArray = { data: new BigUint64Array([0n, 2n ** 40n]), shape: [1, 1, 1, 2] };
    const out = applyPostprocessors(big, [MORTON], at([1023, 1023, 1023], 216000));
    expect(Array.from(out.data as BigUint64Array)).toEqual([0n, 2n ** 40n + 216000n * (2n ** 30n - 1n)]);
  });

  it("declares uint64 labels and keeps the channels", () => {
    expect(mortonRelabeling.declares(MORTON)).toEqual({ dtype: "uint64", isSegmentation: true });
    expect(outputDtype([MORTON], "float32")).toBe("uint64");
    expect(outputChannels([MORTON], 3)).toBe(3);
    expect(isSegmentation([MORTON])).toBe(true);
  });
});
