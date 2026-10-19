// What the parts of finetuning in the browser agree on: the types each
// module of src/finetune exports to the others and to the backend's
// finetune routes. The pieces, in the order a run uses them:
//
//   volume.ts       a new annotation volume over the open dataset for a
//                   model (planVolume), and its zarr v2 files;
//   samples.ts      the painted chunks of a volume and, for each, the
//                   model's input block around it, read and normalized as
//                   the model stage feeds the model (a Sample);
//   targets.ts      a chunk's labels as training targets per output channel
//                   with the voxels the loss counts (TargetTransform);
//   onnx-head.ts    the model's head, the last 1×1 convolution, found in
//                   the ONNX graph; the graph with the head's input tensor
//                   as a second output; the graph with new head weights;
//   features.ts     the model run for its logits and features at once;
//   head-trainer.ts the head trained on the painted voxels' features;
//   distillation.ts the voxels the distillation term holds to the starting
//                   head (unpainted, all, or inside good regions), sampled;
//   good-regions.ts the regions the user marked as already right, kept
//                   with the volume, and where they fall on its grid;
//   run.ts          all of it, one run: runFinetune;
//   modes.ts        the ways a model can be trained, from its manifest;
//   augment.ts      the whole network's augmentation of a tile;
//   full/           the other way, the whole network of a 2D model trained
//                   with TensorFlow.js (full/run-full.ts), when the
//                   manifest offers it.
//
// Why the head. The browser runs the models with onnxruntime-web, which
// trains nothing on the GPU, so the network's body stays as exported and
// its last layer, the 1×1 convolution that turns the final feature map into
// the output channels, is what trains: a few thousand weights, started from
// the model's own, on the features of the painted voxels alone, in seconds
// on any machine. It learns what the model mislabels where its features
// already tell the two apart (a membrane called mitochondrion), not new
// shapes; that is the server's finetune, whose result the browser can serve
// as any other model. Everything the trainer sees goes through the same
// normalizers and the same ONNX session as inference, so what it fits is
// what the layer shows.
//
// Since 2026-10-16 the 2D students' manifests also offer the whole network
// ("full"): Marwan Zouinkhi exported each with a training graph that
// TensorFlow.js builds and trains on WebGPU (its gradients are TF.js's, not
// onnxruntime's), and the weights trained are written back into the .onnx,
// which serves as before. The tab offers what the manifest declares
// (modes.ts) and FinetuneOptions.mode says which a run is.
import type { Chain, DatasetInfo, Inference, ModelEntry, PaintStore, PaintVolume } from "../contracts";
import type { FinetuneMode } from "./modes";

/** The labels of a painted chunk of an annotation volume: its grid index
 * (z, y, x, as the chunk key names it) and its voxels, the whole chunk in
 * C order (z, y, x): 0 unpainted, 1 background, 2 and up objects. */
export interface PaintedChunk {
  index: number[];
  shape: number[];
  labels: Uint8Array | Uint16Array;
}

/** A training sample: a painted chunk and the model's input block around
 * it, float32, normalized through the volume's input_norm and zero past the
 * volume, exactly as the model stage feeds the model (infer/stages.ts
 * modelInput), with the dims the model takes ([1, 1, z, y, x]). */
export interface Sample {
  chunk: PaintedChunk;
  input: Float32Array;
  inputDims: number[];
}

/** A chunk's training target and the voxels the loss counts, per output
 * channel: both `channels × N` where N is the chunk's voxels, C order, the
 * channel outermost (the server's target_transforms.py, less the batch). */
export interface Targets {
  channels: number;
  target: Float32Array;
  mask: Float32Array;
}

export type TargetTransform = (chunk: PaintedChunk) => Targets;

/** A model's head: the last 1×1 convolution of its ONNX graph, which makes
 * the output channels from the final feature map, and its weights as
 * float32 whatever the file stores (fp16 variants store float16). */
export interface Head {
  /** The convolution's input: the feature map, `[N, inChannels, ...]` with
   * the channel on axis 1 (N × the spatial dims = the output block). */
  featureTensor: string;
  weightName: string;
  /** null when the convolution has no bias (then the bias trained is written nowhere and stays 0). */
  biasName: string | null;
  inChannels: number;
  outChannels: number;
  /** `[outChannels, inChannels]`, the kernel's 1×1 (or 1×1×1) squeezed out. */
  weight: Float32Array;
  bias: Float32Array;
  /** The initializers' element type in the file. */
  dtype: "float32" | "float16";
}

/** The model's answer on one sample: its logits (the graph's output) and
 * the head's input, the feature map, both float32 with their dims. */
export interface ModelFeatures {
  logits: Float32Array;
  logitsDims: number[];
  features: Float32Array;
  featuresDims: number[];
}

/** The supervised losses the head trains with: the server's per-voxel ones
 * (lora_trainer.py's "bce", "mse" and "margin"), each a function of one
 * voxel's logit and target, so a batch of voxels drawn from any chunks is
 * as well defined as a patch. Dice (and "combined", which adds it) is an
 * overlap over a whole patch per sample, which a shuffled batch of voxels
 * from several chunks is not; the routes refuse it. */
export type HeadLoss = "bce" | "mse" | "margin";

/** Where the distillation term holds the head to the one it started from,
 * as the form's Distillation Scope says it (the server's
 * distillation_all_voxels): the voxels of the painted chunks that are not
 * painted, or every voxel of them. Good regions, when the volume has any,
 * decide instead (the server's "anchor" scope): run.ts. */
export type DistillationScope = "unlabeled" | "all";

/** The training parameters, as the Finetune tab's form sends them to
 * /api/finetune/submit and the routes read them. */
export interface FinetuneOptions {
  /** What trains: the head, or the whole network (full/run-full.ts), which
   * reads the same loss, smoothing, balance and distillation fields, its
   * own batch and tile below, and augmentation, but not batchPixels. */
  mode: FinetuneMode;
  epochs: number;
  learningRate: number;
  /** The supervised loss on the painted voxels (HeadLoss). */
  lossType: HeadLoss;
  /** The margin loss's margin m: a foreground voxel costs nothing once its
   * probability is over 1 - m, a background one once it is under m. */
  margin: number;
  /** The foreground's and the background's mean losses averaged, as the
   * server's balanced_mean, rather than every voxel counted alike. */
  balanceClasses: boolean;
  /** s moves the targets to s/2 and 1 - s/2 (not for the margin loss, which
   * has its own band, as the server forces it to 0 there). */
  labelSmoothing: number;
  /** Voxels per optimizer step. */
  batchPixels: number;
  /** λ: the loss is the supervised term plus λ × the mean squared distance
   * of the head's logits from the starting head's on the distillation
   * voxels (losses.py's distillation_loss); 0 trains on the paint alone. */
  distillationLambda: number;
  distillationScope: DistillationScope;
  seed: number;
  /** The whole network's batch: tiles per optimizer step (the server's Batch Size). */
  batchPatches: number;
  /** The whole network's tile side, in pixels of the input; null for the
   * default (full/graph.ts DEFAULT_TILE, the side the graph takes nearest it). */
  tileSize: number | null;
  /** Whole network only: each step's tiles get a random brightness and
   * noise, flips and quarter turns, as the server's augmentation
   * (augment.ts); the head trains on features computed once, and ignores it. */
  augment: boolean;
}

/** How a run speaks to the job that follows it: its log lines, in the words
 * the dashboard's job card and loss plot read ("Epoch 3/20 - Loss: 0.0113"
 * ends an epoch), and whether it has been cancelled, asked between steps. */
export interface RunReporter {
  log(line: string): void;
  cancelled(): boolean;
}

/** What a run made: the finetuned model's bytes (the base's graph with the
 * trained head, or every trained weight, written into its initializers)
 * and how it went. */
export interface Finetuned {
  bytes: ArrayBuffer;
  /** The head trained and its weights, after a head run (a restart starts
   * from them); absent after a whole-network run, whose weights are all in
   * `bytes` (a restart starts from those). */
  head?: Head;
  weight?: Float32Array;
  bias?: Float32Array;
  epochs: number;
  /** Each epoch's loss as its log line gives it: the supervised term plus
   * λ × the distillation term when distillation was on. */
  losses: number[];
  finalLoss: number | null;
  /** Each epoch's supervised term alone: the paint's fit, which runs with
   * different λ can be compared by (the server ranks its epochs by it). */
  supervisedLosses: number[];
  /** The painted voxels trained on, over all channels' masks, and the chunks they came from. */
  pixels: number;
  chunks: number;
  /** The voxels the distillation term held to the starting head (the
   * starting network, after a whole-network run: each epoch's); 0 without
   * distillation. */
  distilled: number;
  /** The tiles a whole-network run trained on (full/tiles.ts). */
  tiles?: number;
}

/** What a head run made: a Finetuned with the head and its trained weights. */
export interface HeadFinetuned extends Finetuned {
  head: Head;
  weight: Float32Array;
  bias: Float32Array;
}

/** A region the user marked as one the model already gets right (Mark as
 * Good): a box in nm, (z, y, x), as the server's good_regions.json keeps
 * it (dashboard/routes/finetune/good_regions.py). */
export interface GoodRegion {
  id: string;
  label: string;
  offset_nm: number[];
  shape_nm: number[];
}

/** One finetune of `entry` on the paint of `volume`: the samples read through
 * `infer` and `paint`, the features computed with the model's own bytes,
 * the head trained from `initialHead` (the model's own when absent; the
 * last run's for a restart) and written into a copy of the bytes. The
 * distillation teacher is always the head in `baseBytes`, the weights the
 * job started from, whichever head the run starts from (run.ts). The good
 * regions are the entries of the volume's good_regions.json as they are
 * when the run starts (good-regions.ts loadGoodRegions); one that is not a
 * GoodRegion is skipped with a line in the log, as the server skips it. */
export interface FinetuneRun {
  entry: ModelEntry;
  baseBytes: ArrayBuffer;
  volume: PaintVolume;
  chain: Chain;
  dataset: DatasetInfo;
  infer: Pick<Inference, "readRaw">;
  paint: PaintStore;
  options: FinetuneOptions;
  report: RunReporter;
  initialHead?: { weight: Float32Array; bias: Float32Array };
  /** A whole-network run's start: the last run's model, for a restart
   * (`baseBytes` otherwise). */
  initialBytes?: ArrayBuffer;
  goodRegions?: readonly unknown[];
}

export type RunFinetune = (run: FinetuneRun) => Promise<Finetuned>;

/** The whole network's learning rate where the tab sends none: Adam's
 * 1e-4, at which Marwan measured the steps, and the server's LoRA default.
 * The head's 1e-2 would move two million weights a hundred times as far. */
export const FULL_LEARNING_RATE = 1e-4;

/** The defaults of the form's parameters where the tab does not send one.
 *
 * λ is the server form's default, 0.01, and it means something for a head
 * alone too. The term is in squared logits: a head that moved the
 * unpainted voxels' logits by 3 on average (a probability of 0.5 taken to
 * 0.95) pays 0.01 × 9 = 0.09, about what the balanced BCE of a fitted
 * paint is (0.05 to 0.2), so it is a light hold that a correction the
 * paint asks for outweighs, while drift the paint does not ask for (the
 * bias, directions of the features no painted voxel spans) costs something
 * where before it cost nothing. The server's own reason for keeping it
 * small, that LoRA cannot move the model far, does not carry over: a head
 * moves freely at its learning rate, so a heavier λ (0.1 to 1) holds it
 * firmly; the form offers those.
 *
 * The whole network's batch is one tile, where the server's is 8 patches:
 * a step's time and memory grow with the batch (one 300² tile of
 * setup_54_2d already takes 908 MB), BatchNorm's statistics are frozen, so
 * one tile is as well defined a batch as eight, and a session's few painted
 * slices make few tiles, which a batch of one turns into as many steps. */
export const DEFAULT_OPTIONS: FinetuneOptions = {
  mode: "head",
  epochs: 20,
  learningRate: 0.01,
  lossType: "bce",
  margin: 0.3,
  balanceClasses: true,
  labelSmoothing: 0,
  batchPixels: 4096,
  distillationLambda: 0.01,
  distillationScope: "unlabeled",
  seed: 0,
  batchPatches: 1,
  tileSize: null,
  augment: false,
};
