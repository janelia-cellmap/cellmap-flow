// A painted chunk's labels as training targets, made by the Python both
// modes run: cellmap_flow/portable/targets.py, the functions the server's
// finetune/target_transforms.py wraps for its trainer, called here in the
// engine's Pyodide worker through Inference.python
// (portable/targets_calls.py decodes the labels and encodes the answer). So
// the head's trainer, the whole network's and the ranking train and score
// on the targets the server's finetune would make from the same paint;
// there is no port to keep in step.
//
// Which target a model takes, and an affinity model's offsets, the Python
// decides too, from the model's channel names (label_it.affinity_offsets,
// the server's rule in autodetect_output_type: a model with a channel named
// `*_aff*` is an affinity model, its first affinity channels along z, y and
// x; every other model is a binary one, the one target broadcast to all its
// channels, or with a channel picked, that channel's alone). The server
// takes offsets from a model's script when it has one; the browser has no
// script, and the manifests on Hugging Face say no offsets (none of the
// seven did on 2026-10-21), so the names are all it has, read by the one
// rule. The channels past the offsets (LSDs, say) train nothing in both
// modes.
//
// A binary model of several channels (fly_organelles' all_mem, organelle,
// mito, er, ...) trains one channel at a time (channelChoice): the paint
// says what one class is, and broadcast to every channel it taught each of
// them that class. The run's channel (FinetuneOptions.selectChannel) is the
// one its target is made for, as the server's select_channel makes a
// one-channel target for the prediction sliced to that channel.
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

/** Whether `entry` trains one of its channels at a time, so that a run
 * needs the channel its paint is for (the models route answers it as
 * `channel_choice`, and submit refuses a run without the pick): more than
 * one output channel, and not an affinity model, each of whose channels
 * has a target of its own. The rule the targets are made by, which the
 * browser's routes already read as hasAffinityChannels (the server's
 * `*_aff*` test, the Python's label_it.affinity_offsets) rather than ask
 * the Python on every poll of the models' list; the browser makes no flow
 * targets (Cellpose has no web export), so flows need no test here. */
export function channelChoice(entry: ModelEntry): boolean {
  return entry.manifest.channels.length > 1 && !hasAffinityChannels(entry);
}

/** The transform a model's labels train it with, each chunk's targets made
 * by `infer`'s Python from the model's channel names: affinities for a
 * model with affinity channels, the binary target on every channel
 * otherwise, or with `channel` (an index into the model's channels, not for
 * an affinity model) that channel's target alone, one channel. */
export function targetTransformFor(entry: ModelEntry, infer: Pick<Inference, "python">, channel: number | null = null): TargetTransform {
  const channels = entry.manifest.channels;
  if (!channels.length) throw new Error(`${entry.name}'s manifest names no output channels to train`);
  const C = channel === null ? channels.length : 1;
  return async (chunk: PaintedChunk): Promise<Targets> => {
    const n = voxelsOf(chunk);
    const labels_dtype = chunk.labels instanceof Uint16Array ? "uint16" : "uint8";
    const { arrays } = await infer.python<{ value: { channels: number }; arrays: ArrayBuffer[] }>(
      TARGETS, { shape: chunk.shape, labels_dtype, channels: channels.length, channel_names: channels, select_channel: channel }, [bytesOf(chunk.labels)]);
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
