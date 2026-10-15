// Array layout and bytes: the raw block's typed array into the model's
// float32 input, the output ndarray into the layout a served chunk has, and
// that into the bytes chunkmirage's engine serves. From Marwan Zouinkhi's
// web/legacy/pipeline.js (reorderToZarr, encodeChunk) and zarr-reader.js
// (readRoi's copy into the zero-padded window).
//
// The chunk layout is channel first. chunkmirage's engine declares a view's
// added channel axis as the LEADING axis of the zarr array: engine.ts's
// array() writes shape [...added, ...level shape] and chunk_shape [...added,
// ...chunk] (lines 237-241 of web/chunkmirage/web/src/engine.ts), its
// omeGroup() puts the "c" axis first in the multiscales (lines 204-215), and
// answer() takes the chunk key's leading channel index, requires it to be 0
// and drops it ("a channel axis is one chunk: the volume's index", lines
// 366-367). So a chunk's bytes are the C-order array [C, z, y, x]: exactly
// the model's output. The Python Flask server served channels last, and
// reorderToZarr below is that transpose, kept for a zarr v2 layout; the
// engine's zarr v3 layout never uses it.
import type { NdArray, NdData } from "./post";

type TypedCtor = { new (n: number): NdData; new (buffer: ArrayBuffer, byteOffset?: number, length?: number): NdData; BYTES_PER_ELEMENT: number };

/** The zarr dtypes served, and the typed array of each. 64-bit integers are
 * not among them: no browser step produces them. */
export const TYPED: Record<string, TypedCtor> = {
  uint8: Uint8Array, int8: Int8Array as unknown as TypedCtor, uint16: Uint16Array, int16: Int16Array as unknown as TypedCtor,
  uint32: Uint32Array, int32: Int32Array, float32: Float32Array, float64: Float64Array,
};

export const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** The raw bytes the engine's reader returns (its source's dtype, C order) as
 * float32 values for the model. */
export function toFloat32(buffer: ArrayBuffer, dtype: string): Float32Array {
  const Ctor = TYPED[dtype];
  if (!Ctor) throw new Error(`raw data of dtype ${dtype} cannot be fed to a model here (one of ${Object.keys(TYPED).join(", ")})`);
  const typed = new Ctor(buffer);
  if (typed instanceof Float32Array) return typed;
  const out = new Float32Array(typed.length);
  out.set(typed as ArrayLike<number>);
  return out;
}

/** `block` (shape `blockShape`) copied into `out` (shape `outShape`) at
 * `offset`, leaving the rest of `out` as it is (zeros, for padding past the
 * data). Three axes, C order. */
export function placeBlock(out: Float32Array, outShape: number[], block: ArrayLike<number>, blockShape: number[], offset: number[]): void {
  const [bz, by, bx] = blockShape, [, oy, ox] = outShape;
  for (let a = 0; a < 3; a++) {
    if (offset[a] < 0 || offset[a] + blockShape[a] > outShape[a]) throw new Error(`a block of ${blockShape} at ${offset} does not fit in ${outShape}`);
  }
  const rows = block as unknown as NdData;
  for (let z = 0; z < bz; z++) {
    for (let y = 0; y < by; y++) {
      const src = (z * by + y) * bx;
      const dst = ((z + offset[0]) * oy + (y + offset[1])) * ox + offset[2];
      out.set(rows.subarray(src, src + bx) as ArrayLike<number>, dst);
    }
  }
}

/** The model output [C, z, y, x] as the served array is laid out: unchanged
 * when the view has a channel axis (see the top of this file), else the one
 * channel squeezed to [z, y, x]. */
export function chunkLayout(nd: NdArray, hasChannel: boolean): NdArray {
  if (hasChannel) return nd;
  if (nd.shape[0] !== 1) throw new Error(`a view without a channel axis cannot serve ${nd.shape[0]} channels`);
  return { data: nd.data, shape: nd.shape.slice(1) };
}

/** [C, z, y, x] transposed to channels last, [z, y, x, C]: the Flask
 * server's zarr v2 layout (server.py _reorder_to_zarr_axes). Not used for the
 * engine's layout; see the top of this file. */
export function reorderToZarr(nd: NdArray): NdArray {
  const [c, z, y, x] = nd.shape;
  if (c === 1) return { data: nd.data, shape: [z, y, x, 1] };  // the byte layout is identical
  const out = new (nd.data.constructor as TypedCtor)(nd.data.length);
  const src = nd.data;
  let o = 0;
  for (let iz = 0; iz < z; iz++)
    for (let iy = 0; iy < y; iy++)
      for (let ix = 0; ix < x; ix++)
        for (let ic = 0; ic < c; ic++)
          out[o++] = src[((ic * z + iz) * y + iy) * x + ix];
  return { data: out, shape: [z, y, x, c] };
}

/** `nd` as the bytes of one whole chunk of `dtype`: little endian (what typed
 * arrays are on every platform a browser runs on), C order. With
 * `chunkShape`, the chunk is that shape and `nd` fills its leading part, the
 * rest zero, as zarr wants a chunk at the array's edge; without it the chunk
 * is `nd`'s own shape. */
export function encodeChunk(nd: NdArray, dtype: string, chunkShape?: number[]): ArrayBuffer {
  const Ctor = TYPED[dtype];
  if (!Ctor) throw new Error(`cannot encode a chunk of dtype ${dtype}`);
  const shape = chunkShape ?? nd.shape;
  if (shape.length !== nd.shape.length) throw new Error(`a chunk of shape ${shape} cannot hold an array of shape ${nd.shape}`);
  const same = shape.every((n, a) => n === nd.shape[a]);
  let typed: NdData;
  if (same && nd.data instanceof Ctor) {
    typed = nd.data;
  } else if (same) {
    typed = new Ctor(nd.data.length);
    for (let i = 0; i < nd.data.length; i++) typed[i] = nd.data[i];  // the store casts like astype
  } else {
    typed = new Ctor(prod(shape));
    copyLeading(typed, shape, nd.data, nd.shape);
  }
  return typed.buffer.slice(typed.byteOffset, typed.byteOffset + typed.byteLength) as ArrayBuffer;
}

/** `src` (shape `srcShape`) into the leading part of `dst` (shape `dstShape`),
 * any rank, whatever of it fits. */
function copyLeading(dst: NdData, dstShape: number[], src: NdData, srcShape: number[]): void {
  const n = srcShape.map((s, a) => Math.min(s, dstShape[a]));
  const rank = n.length;
  const index = new Array<number>(rank).fill(0);
  // the innermost axis is copied as a run; the outer ones by odometer
  const run = n[rank - 1];
  const count = prod(n.slice(0, -1));
  for (let k = 0; k < count; k++) {
    let s = 0, d = 0;
    for (let a = 0; a < rank - 1; a++) { s = s * srcShape[a] + index[a]; d = d * dstShape[a] + index[a]; }
    s = s * srcShape[rank - 1];
    d = d * dstShape[rank - 1];
    for (let i = 0; i < run; i++) dst[d + i] = src[s + i];
    for (let a = rank - 2; a >= 0; a--) { if (++index[a] < n[a]) break; index[a] = 0; }
  }
}
