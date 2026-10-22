// A painted chunk's labels as training targets, made by the Python both
// modes run: cellmap_flow/portable/targets.py, the functions the server's
// finetune/target_transforms.py wraps for its trainer, called here in the
// engine's Pyodide worker through Inference.python
// (portable/targets_calls.py decodes the labels and encodes the answer). So
// the head's trainer, the whole network's and the ranking train and score
// on the targets the server's finetune would make from the same paint;
// there is no port to keep in step.
//
// What is decided here is what the browser knows of a model that the
// server learns another way: which target it takes, and an affinity
// model's offsets. A model with a channel named `*_aff*` is an affinity
// model (volume.ts hasAffinityChannels, the server's test in
// autodetect_output_type, which gives its volume uint16 ids too), and every
// other model is a binary one, the one target broadcast to all its
// channels. The server takes an affinity model's offsets from its script,
// or unit offsets for its first `_aff` channels; the browser has no script,
// and the manifests on Hugging Face say no offsets (none of the seven did
// on 2026-10-21), so the channel names say them (`mito_aff_z`,
// `mito_aff_y`, `mito_aff_x`, which is how Marwan's exports name them), and
// a channel of such a model that names no offset trains nothing, as the
// server leaves out the channels past its offsets (LSDs, say). The Python
// takes the offsets as given, so an offset however found means the same
// in both modes.
//
// A chunk is one sample without the batch axis: its labels are its voxels
// in C order (z, y, x), and a target or mask is `channels × N` values with
// the channel outermost, so channel c's voxel i is at c * N + i, the
// server's (0, c, z, y, x) once the batch is gone.
import type { Inference, ModelEntry } from "../contracts";
import type { PaintedChunk, Targets, TargetTransform } from "./types";
import { hasAffinityChannels } from "./volume";

/** The portable Python's entry point for targets, by module path and name. */
const TARGETS = "cellmap_flow.portable.targets_calls:targets";

/** The affinity offset of each of `channelNames`, in the channels' order:
 * the unit offset a name ending in `_aff_z`, `_aff_y` or `_aff_x` says
 * (mito_aff_z, mito_aff_y, mito_aff_x → (1,0,0), (0,1,0), (0,0,1)), and
 * null for a channel named no offset, which the affinity target leaves
 * out of the loss. */
export function affinityOffsets(channelNames: string[]): (number[] | null)[] {
  return channelNames.map((name) => {
    const axis = /_aff_(z|y|x)$/.exec(name)?.[1];
    return axis ? UNIT_OFFSETS[axis] : null;
  });
}

const UNIT_OFFSETS: Record<string, number[]> = { z: [1, 0, 0], y: [0, 1, 0], x: [0, 0, 1] };

/** The transform a model's labels train it with, each chunk's targets made
 * by `infer`'s Python: affinities along the offsets its channel names say
 * for a model with affinity channels (hasAffinityChannels), its other
 * channels left out of the loss; the binary target on every channel
 * otherwise. */
export function targetTransformFor(entry: ModelEntry, infer: Pick<Inference, "python">): TargetTransform {
  const channels = entry.manifest.channels;
  if (!channels.length) throw new Error(`${entry.name}'s manifest names no output channels to train`);
  const offsets = hasAffinityChannels(entry) ? affinityOffsets(channels) : null;
  if (offsets?.every((offset) => offset === null)) {
    throw new Error(`${entry.name}'s channels (${channels.join(", ")}) are affinities, but none is named _aff_z, _aff_y or _aff_x, so their offsets are unknown`);
  }
  const C = channels.length;
  return async (chunk: PaintedChunk): Promise<Targets> => {
    const n = voxelsOf(chunk);
    const labels_dtype = chunk.labels instanceof Uint16Array ? "uint16" : "uint8";
    const { arrays } = await infer.python<{ value: { channels: number }; arrays: ArrayBuffer[] }>(
      TARGETS, { shape: chunk.shape, labels_dtype, channels: C, offsets }, [bytesOf(chunk.labels)]);
    const target = new Float32Array(arrays[0]), mask = new Float32Array(arrays[1]);
    if (target.length !== C * n || mask.length !== C * n) throw new Error(`the Python's targets have ${target.length} and ${mask.length} values for ${C} channel(s) of ${n} voxels`);
    return { channels: C, target, mask };
  };
}

/** The chunk's voxel count, checked against its labels. */
function voxelsOf(chunk: PaintedChunk): number {
  if (chunk.shape.length !== 3) throw new Error(`a chunk has three axes (z, y, x), not ${chunk.shape.length}`);
  const n = chunk.shape.reduce((a, b) => a * b, 1);
  if (chunk.labels.length !== n) throw new Error(`a chunk of ${chunk.shape.join("×")} has ${n} voxels, but ${chunk.labels.length} labels`);
  return n;
}

/** A typed array's own bytes, copied: a call's arrays are handed to the worker. */
function bytesOf(a: ArrayBufferView): ArrayBuffer {
  return a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength) as ArrayBuffer;
}
