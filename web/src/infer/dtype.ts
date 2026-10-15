// The dtypes a chain's output may have and the casts between them: numpy's
// astype, for the postprocess stage, which must hand chunkmirage's engine a
// typed array of the dtype it declares (stages.ts there checks `instanceof`),
// while a step may compute in another (DefaultPostprocessor clips in float
// and declares uint8; a merger relabels uint32 components into uint64 ids).
// From Marwan Zouinkhi's web/legacy/pipeline.js (encodeChunk's casts).
import type { NdData } from "./post";

type TypedCtor = { new (n: number): NdData; new (buffer: ArrayBuffer, byteOffset?: number, length?: number): NdData; BYTES_PER_ELEMENT: number };

/** The zarr dtypes a chain may leave, and the typed array of each. uint64
 * is the segmentation steps' dtype (Morton ids, the affinity segmenter, the
 * blockwise merger): a BigUint64Array, whose elements are bigints, which is
 * why the casts below spell out the two kinds. int64 has no step and is
 * left out. */
export const TYPED: Record<string, TypedCtor> = {
  uint8: Uint8Array, int8: Int8Array as unknown as TypedCtor, uint16: Uint16Array, int16: Int16Array as unknown as TypedCtor,
  uint32: Uint32Array, int32: Int32Array, uint64: BigUint64Array, float32: Float32Array, float64: Float64Array,
};

/** `src`'s values into `dst`, element by element, as numpy's astype casts:
 * a number into a BigUint64Array is truncated toward zero (a value that is
 * not finite becomes 0, where numpy's result is platform-defined; a negative
 * one wraps modulo 2^64 as the array stores it), a bigint into a number
 * array goes through Number (exact below 2^53, rounded above), and a store
 * into a narrower integer array truncates and wraps as the array does. */
export function castInto(dst: NdData, src: NdData): void {
  if (dst.length !== src.length) throw new Error(`cannot cast ${src.length} values into ${dst.length}`);
  const toBig = dst instanceof BigUint64Array, fromBig = src instanceof BigUint64Array;
  if (toBig && fromBig) {
    (dst as BigUint64Array).set(src as BigUint64Array);
  } else if (toBig) {
    const big = dst as BigUint64Array, numbers = src as ArrayLike<number>;
    for (let i = 0; i < numbers.length; i++) {
      const v = numbers[i]!;
      big[i] = Number.isFinite(v) ? BigInt(Math.trunc(v)) : 0n;
    }
  } else if (fromBig) {
    const numbers = dst as Exclude<NdData, BigUint64Array>, big = src as BigUint64Array;
    for (let i = 0; i < big.length; i++) numbers[i] = Number(big[i]);
  } else {
    const d = dst as Exclude<NdData, BigUint64Array>, s = src as Exclude<NdData, BigUint64Array>;
    for (let i = 0; i < s.length; i++) d[i] = s[i]!;  // the store casts like astype
  }
}

/** `data` as the typed array of `dtype`: itself when it is one already, else
 * a cast copy (castInto). */
export function asDtype(data: NdData, dtype: string): NdData {
  const Ctor = TYPED[dtype];
  if (!Ctor) throw new Error(`cannot serve a chunk of dtype ${dtype} (one of ${Object.keys(TYPED).join(", ")})`);
  if (data instanceof Ctor) return data;
  const out = new Ctor(data.length);
  castInto(out, data);
  return out;
}
