// A painted chunk's labels as training targets: the port of
// cellmap_flow/finetune/target_transforms.py (BinaryTargetTransform,
// BroadcastBinaryTargetTransform, AffinityTargetTransform with its
// _offset_slices) for the browser, checked against the Python classes by
// web/test/parity/targets.test.ts.
//
// The Python works on a batch, (B, 1, Z, Y, X) in and (B, C, Z, Y, X) out;
// a chunk here is one sample without the batch axis: its labels are the
// chunk's voxels in C order (z, y, x), and a target or mask is `channels × N`
// values with the channel outermost, so channel c's voxel i is at c * N + i,
// which is the same memory order as the Python's (0, c, z, y, x) once the
// batch is gone. Labels mean what the paint layer paints: 0 unpainted (no
// loss), 1 background, 2 and up objects.
//
// Which transform a model gets (targetTransformFor) follows the channels its
// manifest names, as the server's autodetect_output_type (dashboard/routes/
// finetune/common.py) follows them when nothing else says: a model with a
// channel named `*_aff*` is an affinity model (volume.ts hasAffinityChannels,
// the same test that gives its volume uint16 ids), and every other model is
// a binary one, the one target broadcast to all its channels. The server
// takes an affinity model's offsets from its script, or the unit offsets in
// z, y, x order by default; here the channel names say them (`mito_aff_z`,
// `mito_aff_y`, `mito_aff_x`, which is how Marwan's exports name them), and
// a channel of such a model that names no offset is masked out, as the
// server masks the channels past its offsets (LSDs, say).
import type { ModelEntry } from "../contracts";
import type { PaintedChunk, Targets, TargetTransform } from "./types";
import { hasAffinityChannels } from "./volume";

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** BinaryTargetTransform for one channel, BroadcastBinaryTargetTransform for
 * more: target 1 where the voxel is an object (label 2 or more, the Python's
 * `clamp(label - 1, min=0) > 0`), mask 1 where it is painted (label > 0),
 * the same on every channel. */
export function binaryTargets(channels: number): TargetTransform {
  if (!Number.isInteger(channels) || channels < 1) throw new Error(`a binary target needs at least one channel, not ${channels}`);
  return (chunk: PaintedChunk): Targets => {
    const n = voxelsOf(chunk);
    const target = new Float32Array(channels * n), mask = new Float32Array(channels * n);
    const { labels } = chunk;
    for (let i = 0; i < n; i++) {
      const label = labels[i];
      if (label > 1) target[i] = 1;
      if (label > 0) mask[i] = 1;
    }
    // expand(...).contiguous(): the first channel's values copied to the others.
    for (let c = 1; c < channels; c++) {
      target.copyWithin(c * n, 0, n);
      mask.copyWithin(c * n, 0, n);
    }
    return { channels, target, mask };
  };
}

/** AffinityTargetTransform: channel i is the affinity along `offsets[i]`,
 * 1 where a voxel and its neighbour at the offset are the same object
 * (equal labels above 1), masked where both are painted; the pair is
 * written at the voxel (the Python's `src_slices`), and a voxel whose
 * neighbour falls outside the chunk is left unmasked, as are the channels
 * past the offsets (`num_channels` beyond `len(offsets)`: LSDs, say, which
 * then get no gradient) and any channel whose offset is null (one named no
 * offset, affinityOffsets). An offset no smaller than the chunk along its
 * axis pairs nothing: a one-slice chunk's z channel has an empty mask,
 * which is right for a 2D student (its z head keeps the model's weights). */
export function affinityTargets(offsets: (number[] | null)[], channels: number): TargetTransform {
  if (!Number.isInteger(channels) || channels < 1) throw new Error(`an affinity target needs at least one channel, not ${channels}`);
  if (offsets.length > channels) throw new Error(`${offsets.length} affinity offsets for ${channels} channel(s): a model has a channel per offset`);
  for (const offset of offsets) {
    if (offset && (offset.length !== 3 || offset.some((d) => !Number.isInteger(d)))) throw new Error(`an affinity offset is three whole numbers (dz, dy, dx), not ${JSON.stringify(offset)}`);
  }
  return (chunk: PaintedChunk): Targets => {
    const n = voxelsOf(chunk);
    const [sz, sy, sx] = chunk.shape;
    const target = new Float32Array(channels * n), mask = new Float32Array(channels * n);
    const { labels } = chunk;
    offsets.forEach((offset, c) => {
      if (!offset) return;  // a channel with no offset: masked out
      const [dz, dy, dx] = offset;
      // _offset_slices: the source voxels are those whose neighbour at the
      // offset is in the chunk, [max(0, -d), min(size, size - d)) along each
      // axis; the neighbour is d further on. Empty when |d| >= size.
      const z0 = Math.max(0, -dz), z1 = Math.min(sz, sz - dz);
      const y0 = Math.max(0, -dy), y1 = Math.min(sy, sy - dy);
      const x0 = Math.max(0, -dx), x1 = Math.min(sx, sx - dx);
      const step = (dz * sy + dy) * sx + dx;  // the neighbour's index minus the voxel's
      const base = c * n;
      for (let z = z0; z < z1; z++) for (let y = y0; y < y1; y++) {
        const row = (z * sy + y) * sx;
        for (let x = x0; x < x1; x++) {
          const i = row + x, src = labels[i], dst = labels[i + step];
          if (src === dst && src > 1) target[base + i] = 1;
          if (src > 0 && dst > 0) mask[base + i] = 1;
        }
      }
    });
    return { channels, target, mask };
  };
}

/** The affinity offset of each of `channelNames`, in the channels' order:
 * the unit offset a name ending in `_aff_z`, `_aff_y` or `_aff_x` says
 * (mito_aff_z, mito_aff_y, mito_aff_x → (1,0,0), (0,1,0), (0,0,1)), and
 * null for a channel named no offset, which the affinity target masks out. */
export function affinityOffsets(channelNames: string[]): (number[] | null)[] {
  return channelNames.map((name) => {
    const axis = /_aff_(z|y|x)$/.exec(name)?.[1];
    return axis ? UNIT_OFFSETS[axis] : null;
  });
}

const UNIT_OFFSETS: Record<string, number[]> = { z: [1, 0, 0], y: [0, 1, 0], x: [0, 0, 1] };

/** The transform a model's labels train it with: affinities along the
 * offsets its channel names say for a model with affinity channels
 * (hasAffinityChannels, the server's test), its other channels masked
 * out; the binary target on every channel otherwise. */
export function targetTransformFor(entry: ModelEntry): TargetTransform {
  const channels = entry.manifest.channels;
  if (!channels.length) throw new Error(`${entry.name}'s manifest names no output channels to train`);
  if (!hasAffinityChannels(entry)) return binaryTargets(channels.length);
  const offsets = affinityOffsets(channels);
  if (offsets.every((offset) => offset === null)) {
    throw new Error(`${entry.name}'s channels (${channels.join(", ")}) are affinities, but none is named _aff_z, _aff_y or _aff_x, so their offsets are unknown`);
  }
  return affinityTargets(offsets, channels.length);
}

/** The chunk's voxel count, checked against its labels. */
function voxelsOf(chunk: PaintedChunk): number {
  if (chunk.shape.length !== 3) throw new Error(`a chunk has three axes (z, y, x), not ${chunk.shape.length}`);
  const n = prod(chunk.shape);
  if (chunk.labels.length !== n) throw new Error(`a chunk of ${chunk.shape.join("×")} has ${n} voxels, but ${chunk.labels.length} labels`);
  return n;
}
