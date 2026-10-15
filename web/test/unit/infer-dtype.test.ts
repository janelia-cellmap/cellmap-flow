// The casts between a chain's dtypes (numpy's astype), which the postprocess
// stage uses to hand the engine the typed array it declares.
import { describe, expect, it } from "vitest";

import { asDtype, castInto, TYPED } from "../../src/infer/dtype";

describe("castInto (numpy's astype between number and bigint arrays)", () => {
  it("truncates numbers toward zero into uint64, zeroes non-finite ones, wraps negatives", () => {
    const big = new BigUint64Array(5);
    castInto(big, new Float32Array([1.9, 300, -1, Number.NaN, Number.POSITIVE_INFINITY]));
    expect(Array.from(big)).toEqual([1n, 300n, 2n ** 64n - 1n, 0n, 0n]);
  });

  it("takes bigints into a number array through Number, and copies like for like", () => {
    const u32 = new Uint32Array(2);
    castInto(u32, new BigUint64Array([5n, 2n ** 32n + 1n]));
    expect(Array.from(u32)).toEqual([5, 1]);  // the store wraps, as astype(uint32) does
    const f64 = new Float64Array(1);
    castInto(f64, new BigUint64Array([2n ** 53n + 1n]));
    expect(f64[0]).toBe(2 ** 53);  // rounded above 2^53
    const copy = new BigUint64Array(2);
    castInto(copy, new BigUint64Array([1n, 2n]));
    expect(Array.from(copy)).toEqual([1n, 2n]);
    expect(() => castInto(new Uint8Array(1), new Uint8Array(2))).toThrow(/cannot cast 2 values into 1/);
  });

  it("stores into a narrower integer array as the array does", () => {
    const u8 = new Uint8Array(2);
    castInto(u8, new Float32Array([1.9, 300]));
    expect(Array.from(u8)).toEqual([1, 44]);
  });
});

describe("asDtype", () => {
  it("returns an array of the dtype as it is", () => {
    const data = new Uint8Array([1, 2]);
    expect(asDtype(data, "uint8")).toBe(data);
  });

  it("casts into another dtype, uint64 included", () => {
    const u8 = asDtype(new Float32Array([1.9, 300]), "uint8");
    expect(u8).toBeInstanceOf(Uint8Array);
    expect(Array.from(u8 as Uint8Array)).toEqual([1, 44]);
    const ids = asDtype(new Uint32Array([7, 8]), "uint64");
    expect(ids).toBeInstanceOf(BigUint64Array);
    expect(Array.from(ids as BigUint64Array)).toEqual([7n, 8n]);
    const narrowed = asDtype(new BigUint64Array([7n]), "uint32");
    expect(Array.from(narrowed as Uint32Array)).toEqual([7]);
  });

  it("refuses a dtype it does not know", () => {
    expect(() => asDtype(new Uint8Array(1), "complex64")).toThrow(/dtype complex64/);
    expect(Object.keys(TYPED)).toContain("uint64");
  });
});
