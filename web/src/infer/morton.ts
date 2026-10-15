// Ids unique across chunks by the chunk's Morton index: the browser side of
// cellmap_flow/post/postprocessors.py's MortonSegmentationRelabeling.
//
// Every chunk is segmented on its own, so two chunks both number their
// objects from 1. The Python step makes the ids unique across the volume by
// adding, to every nonzero id of one channel, chunk_num_voxels times the
// chunk's Morton index (pymorton.interleave of its z, y, x index on the
// served grid): a chunk has at most chunk_num_voxels objects, so no two
// chunks' id ranges overlap. The output is uint64, since the sums pass 2^32
// on a large volume. AffinityPostprocessor (affinity.ts) offsets its ids the
// same way, with mortonIncrement below.
import type { StepValue } from "../contracts";
import { asNumber } from "./norms";
import type { NdData, PostprocessorDef } from "./post";

/** pymorton's `__part1by2`: the low 10 bits of `n` spread out so that two
 * zero bits follow each of them (bit k of `n` lands at bit 3k). Only 10 bits
 * per coordinate, as pymorton's first mask (0x3ff) keeps: a chunk index of
 * 1024 or more is taken modulo 1024, in the Python and so here too. The
 * result is below 2^28, so plain 32-bit bitwise operations are exact. */
function part1by2(n: number): number {
  n &= 0x000003ff;
  n = (n ^ (n << 16)) & 0xff0000ff;
  n = (n ^ (n << 8)) & 0x0300f00f;
  n = (n ^ (n << 4)) & 0x030c30c3;
  n = (n ^ (n << 2)) & 0x09249249;
  return n >>> 0;
}

/** pymorton.interleave(a, b, c) (its `interleave3`): the three coordinates'
 * bits interleaved, `a`'s at bits 0, 3, 6, ..., `b`'s one higher, `c`'s two
 * higher. The chunk's index is passed in its axis order (z, y, x), as the
 * Python does with `pymorton.interleave(*chunk_corner)`. Below 2^30, so an
 * exact JavaScript number; pymorton requires integers and so does this. */
export function interleave3(a: number, b: number, c: number): number {
  for (const n of [a, b, c]) {
    if (!Number.isInteger(n)) throw new Error(`a Morton index interleaves integers, got ${JSON.stringify(n)}`);
  }
  return (part1by2(a) | (part1by2(b) << 1) | (part1by2(c) << 2)) >>> 0;
}

/** What the chunk at `chunkIndex` adds to its nonzero ids:
 * `chunk_num_voxels * pymorton.interleave(*chunk_corner)`, as a uint64
 * value. Both steps that key their ids on the chunk use it. */
export function mortonIncrement(chunkIndex: number[], chunkNumVoxels: number): bigint {
  if (chunkIndex.length !== 3) throw new Error(`a chunk index has three axes (z, y, x), got ${JSON.stringify(chunkIndex)}`);
  if (!Number.isInteger(chunkNumVoxels) || chunkNumVoxels < 0) {
    throw new Error(`chunk_num_voxels is a whole number of voxels, got ${JSON.stringify(chunkNumVoxels)}`);
  }
  return BigInt(chunkNumVoxels) * BigInt(interleave3(chunkIndex[0], chunkIndex[1], chunkIndex[2]));
}

/** `data` as uint64, as numpy's astype(np.uint64) casts: integers as they
 * are, floats truncated toward zero. A value that is not finite becomes 0
 * (numpy's result for it is platform-defined), and a negative one wraps
 * modulo 2^64, as BigUint64Array stores it. */
export function toUint64(data: NdData): BigUint64Array {
  const out = new BigUint64Array(data.length);
  if (data instanceof BigUint64Array) {
    out.set(data);
    return out;
  }
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number;
    out[i] = Number.isFinite(v) ? BigInt(Math.trunc(v)) : 0n;
  }
  return out;
}

/** The Python constructor's `int(channel)`: a channel index, from a number
 * or the form's string. */
function asChannel(value: StepValue | undefined, what: string): number {
  const n = asNumber(value, 0, what);
  if (!Number.isInteger(n) || n < 0) throw new Error(`${what} must be a channel index, got ${JSON.stringify(value)}`);
  return n;
}

export const mortonRelabeling: PostprocessorDef = {
  // The Python class declares dtype uint64 and is_segmentation True; it
  // says nothing about channels (it keeps them all).
  declares: () => ({ dtype: "uint64", isSegmentation: true }),
  make: (step) => {
    const channel = asChannel(step.channel, "MortonSegmentationRelabeling.channel");
    return ({ data, shape }, context) => {
      const [c, z, y, x] = shape;
      if (channel >= c) throw new Error(`MortonSegmentationRelabeling: channel ${channel} of an output with ${c} channel(s)`);
      const plane = z * y * x;  // voxels per channel
      // The whole block is cast (data.astype(np.uint64)); only the chosen
      // channel's nonzero ids move. The addition wraps modulo 2^64 if it
      // ever overflowed, as numpy's would.
      const out = toUint64(data);
      const increment = mortonIncrement(context.chunkIndex, context.chunkNumVoxels);
      const end = (channel + 1) * plane;
      for (let i = channel * plane; i < end; i++) {
        if (out[i] !== 0n) out[i] += increment;
      }
      return { data: out, shape };
    };
  },
};
