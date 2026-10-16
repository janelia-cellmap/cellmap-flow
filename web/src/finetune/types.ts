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
//   run.ts          all of it, one run: runFinetune.
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
import type { Chain, DatasetInfo, Inference, ModelEntry, PaintStore, PaintVolume } from "../contracts";

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

/** The training parameters, as the Finetune tab's form sends them to
 * /api/finetune/submit and the routes read them. */
export interface FinetuneOptions {
  epochs: number;
  learningRate: number;
  /** The foreground's and the background's mean losses averaged, as the
   * server's balanced_mean, rather than every voxel counted alike. */
  balanceClasses: boolean;
  /** s moves the targets to s/2 and 1 - s/2. */
  labelSmoothing: number;
  /** Voxels per optimizer step. */
  batchPixels: number;
  seed: number;
}

/** How a run speaks to the job that follows it: its log lines, in the words
 * the dashboard's job card and loss plot read ("Epoch 3/20 - Loss: 0.0113"
 * ends an epoch), and whether it has been cancelled, asked between steps. */
export interface RunReporter {
  log(line: string): void;
  cancelled(): boolean;
}

/** What a run made: the finetuned model's bytes (the base's graph with the
 * trained head written into its initializers) and how it went. */
export interface Finetuned {
  bytes: ArrayBuffer;
  head: Head;
  weight: Float32Array;
  bias: Float32Array;
  epochs: number;
  losses: number[];
  finalLoss: number | null;
  /** The painted voxels trained on, over all channels' masks, and the chunks they came from. */
  pixels: number;
  chunks: number;
}

/** One finetune of `entry` on the paint of `volume`: the samples read through
 * `infer` and `paint`, the features computed with the model's own bytes,
 * the head trained from `initialHead` (the model's own when absent; the
 * last run's for a restart) and written into a copy of the bytes. */
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
}

export type RunFinetune = (run: FinetuneRun) => Promise<Finetuned>;

/** The defaults of the form's parameters where the tab does not send one. */
export const DEFAULT_OPTIONS: FinetuneOptions = {
  epochs: 20,
  learningRate: 0.01,
  balanceClasses: true,
  labelSmoothing: 0,
  batchPixels: 4096,
  seed: 0,
};
