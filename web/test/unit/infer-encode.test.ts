// Layouts and bytes: the raw block into float32, the padded copy, the chunk
// layout (channel first, as chunkmirage's engine declares an added axis), and
// the little-endian bytes of a whole chunk in each dtype.
import { describe, expect, it } from "vitest";

import { castInto, chunkLayout, encodeChunk, placeBlock, reorderToZarr, toFloat32 } from "../../src/infer/encode";

describe("toFloat32", () => {
  it("converts the reader's typed bytes", () => {
    expect(Array.from(toFloat32(new Uint8Array([0, 255]).buffer, "uint8"))).toEqual([0, 255]);
    expect(Array.from(toFloat32(new Uint16Array([0, 65535]).buffer, "uint16"))).toEqual([0, 65535]);
    expect(Array.from(toFloat32(new Int16Array([-5]).buffer, "int16"))).toEqual([-5]);
    expect(Array.from(toFloat32(new Float64Array([0.5]).buffer, "float64"))).toEqual([0.5]);
  });

  it("returns float32 bytes as they are", () => {
    const f = new Float32Array([1.5, 2.5]);
    const out = toFloat32(f.buffer, "float32");
    expect(out.buffer).toBe(f.buffer);
  });

  it("converts uint64 ids through Number", () => {
    expect(Array.from(toFloat32(new BigUint64Array([0n, 7n, 2n ** 40n]).buffer, "uint64"))).toEqual([0, 7, 2 ** 40]);
  });

  it("refuses a dtype it does not know", () => {
    expect(() => toFloat32(new ArrayBuffer(8), "complex64")).toThrow(/dtype complex64/);
  });
});

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
});

describe("placeBlock", () => {
  it("copies a block into the window at an offset, leaving the padding", () => {
    const out = new Float32Array(3 * 3 * 3);
    placeBlock(out, [3, 3, 3], new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2], [1, 1, 1]);
    const at = (z: number, y: number, x: number) => out[(z * 3 + y) * 3 + x];
    expect(at(0, 0, 0)).toBe(0);
    expect(at(1, 1, 1)).toBe(1);
    expect(at(1, 1, 2)).toBe(2);
    expect(at(1, 2, 1)).toBe(3);
    expect(at(2, 1, 1)).toBe(5);
    expect(at(2, 2, 2)).toBe(8);
    expect(Array.from(out).filter((v) => v !== 0).length).toBe(8);
  });

  it("refuses a block that does not fit", () => {
    expect(() => placeBlock(new Float32Array(8), [2, 2, 2], new Float32Array(8), [2, 2, 2], [1, 0, 0])).toThrow(/does not fit/);
  });
});

describe("chunkLayout", () => {
  it("keeps [C, z, y, x] for a view with a channel axis", () => {
    const nd = { data: new Float32Array(24), shape: [3, 2, 2, 2] };
    expect(chunkLayout(nd, true)).toBe(nd);
  });

  it("squeezes the one channel without one", () => {
    const nd = { data: new Float32Array(8), shape: [1, 2, 2, 2] };
    expect(chunkLayout(nd, false)).toEqual({ data: nd.data, shape: [2, 2, 2] });
    expect(() => chunkLayout({ data: new Float32Array(16), shape: [2, 2, 2, 2] }, false)).toThrow(/cannot serve 2 channels/);
  });
});

describe("reorderToZarr (the Flask server's channels-last layout)", () => {
  it("transposes [C, z, y, x] to [z, y, x, C]", () => {
    const nd = { data: Float32Array.from({ length: 2 * 1 * 2 * 2 }, (_, i) => i), shape: [2, 1, 2, 2] };
    const out = reorderToZarr(nd);
    expect(out.shape).toEqual([1, 2, 2, 2]);
    // voxel (0,0,0): channels 0 and 1 are 0 and 4; voxel (0,0,1): 1 and 5
    expect(Array.from(out.data as ArrayLike<number>)).toEqual([0, 4, 1, 5, 2, 6, 3, 7]);
  });

  it("reuses the data for one channel", () => {
    const nd = { data: new Uint8Array([1, 2]), shape: [1, 1, 1, 2] };
    expect(reorderToZarr(nd).data).toBe(nd.data);
  });
});

describe("encodeChunk", () => {
  it("writes little-endian bytes of the dtype asked for", () => {
    const buf = encodeChunk({ data: new Float32Array([1]), shape: [1, 1, 1, 1] }, "float32");
    expect(buf.byteLength).toBe(4);
    expect(Array.from(new Uint8Array(buf))).toEqual([0, 0, 0x80, 0x3f]);
    const u16 = encodeChunk({ data: new Uint8Array([258 & 0xff, 7]), shape: [1, 1, 1, 2] }, "uint16");
    expect(Array.from(new Uint8Array(u16))).toEqual([2, 0, 7, 0]);
  });

  it("casts when the data is another dtype", () => {
    const buf = encodeChunk({ data: new Float32Array([1.9, 300]), shape: [1, 1, 1, 2] }, "uint8");
    expect(Array.from(new Uint8Array(buf))).toEqual([1, 44]);
    const u32 = encodeChunk({ data: new Uint8Array([200]), shape: [1, 1, 1, 1] }, "uint32");
    expect(Array.from(new Uint32Array(u32))).toEqual([200]);
  });

  it("copies the data, not a view on it", () => {
    const data = new Float32Array([1, 2]);
    const buf = encodeChunk({ data, shape: [1, 1, 1, 2] }, "float32");
    expect(buf).not.toBe(data.buffer);
    expect(buf.byteLength).toBe(8);
  });

  it("fills a whole chunk, zero past the array's end", () => {
    // a [1, 2, 2] array in a [1, 2, 3] chunk: each row gets a zero
    const buf = encodeChunk({ data: new Uint8Array([1, 2, 3, 4]), shape: [1, 2, 2] }, "uint8", [1, 2, 3]);
    expect(Array.from(new Uint8Array(buf))).toEqual([1, 2, 0, 3, 4, 0]);
    // with a channel axis: [2, 1, 1, 2] in [2, 1, 2, 2]
    const c = encodeChunk({ data: new Uint8Array([1, 2, 3, 4]), shape: [2, 1, 1, 2] }, "uint8", [2, 1, 2, 2]);
    expect(Array.from(new Uint8Array(c))).toEqual([1, 2, 0, 0, 3, 4, 0, 0]);
  });

  it("writes uint64 chunks: little-endian 8-byte ids, from bigints or from numbers", () => {
    const ids = encodeChunk({ data: new BigUint64Array([1n, 2n ** 40n + 3n]), shape: [1, 1, 1, 2] }, "uint64");
    expect(ids.byteLength).toBe(16);
    expect(Array.from(new Uint8Array(ids))).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0]);
    const cast = encodeChunk({ data: new Float32Array([1.9, 300]), shape: [1, 1, 1, 2] }, "uint64");
    expect(Array.from(new BigUint64Array(cast))).toEqual([1n, 300n]);
    // and the other way: uint64 ids into a uint32 chunk
    const narrowed = encodeChunk({ data: new BigUint64Array([7n]), shape: [1, 1, 1, 1] }, "uint32");
    expect(Array.from(new Uint32Array(narrowed))).toEqual([7]);
  });

  it("fills a whole uint64 chunk, zero past the array's end", () => {
    const buf = encodeChunk({ data: new BigUint64Array([1n, 2n, 3n, 4n]), shape: [1, 2, 2] }, "uint64", [1, 2, 3]);
    expect(Array.from(new BigUint64Array(buf))).toEqual([1n, 2n, 0n, 3n, 4n, 0n]);
  });

  it("refuses a chunk of another rank or an unknown dtype", () => {
    expect(() => encodeChunk({ data: new Uint8Array(1), shape: [1] }, "uint8", [1, 1])).toThrow(/cannot hold/);
    expect(() => encodeChunk({ data: new Uint8Array(1), shape: [1] }, "complex64")).toThrow(/dtype complex64/);
  });
});
