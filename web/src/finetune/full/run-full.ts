// One whole-network finetune of a 2D model, start to finish: the folder's
// training graph and weight offsets fetched (files.ts), the paint cut into
// tiles in the planes it was drawn in and read with its raw (tiles.ts),
// TF.js loaded (tfjs.ts), the network built from the weights in the served
// .onnx (Marwan Zouinkhi's runtime, cellmap-tfjs-train.ts) and trained on
// the tiles with Adam, and the trained weights written back into a copy of
// the .onnx at their offsets, so the result serves in onnxruntime-web
// exactly as the model did.
//
// What trains: every param the graph marks trainable but the manifest's
// `frozen` (BatchNorm's running statistics, which the forward pass reads as
// constants). The loss is the head's (loss.ts): the form's supervised loss
// on the painted voxels, smoothed and class-balanced as it says, plus λ ×
// the distance of the logits from the starting network's (the base's
// weights, also on a restart) over the voxels its scope picks. Good regions
// are not used. The epochs and the learning rate are the form's; the batch
// is in tiles (the form's Batch Size), the tile's side the form's Tile
// size, checked against the sides the graph takes (graph.ts).
//
// With augmentation (augment.ts), each step's tiles are drawn anew from
// their raw as stored and their paint: brightness and noise on the raw,
// then the same flips and quarter turn of the raw and the paint, then the
// targets made from the turned paint and the starting network run on the
// turned tile, in the server's order (lora_trainer.py), so every target
// means what it did whatever the turn.
//
// The pieces are a parameter (createRunFull) so a test can run it with its
// own TF.js loader and files.
import type * as TF from "@tensorflow/tfjs-core";

import type { DatasetInfo, ModelEntry } from "../../contracts";
import { chooseRawLevel, geometryOf, inputShape, outputLevel, outputShape, ratio, sameVoxelSize } from "../../infer/geometry";
import { mulberry32, shuffle, SLICE_MS, yieldEvery } from "../head-trainer";
import { count, epochSummary, plural } from "../log-words";
import { AugmentTally, augmentIntensity, drawAugmentation, intensityRange, transformBlock } from "../augment";
import { refuseMode, sourceOf } from "../modes";
import { readGraph, readModel } from "../onnx-proto";
import { targetTransformFor } from "../targets";
import type { Finetuned, RunFinetune } from "../types";
import { buildNet, type InitializerOffsets, readOnnxWeights, writeOnnxWeights } from "./cellmap-tfjs-train";
import { type FetchFullFiles, fetchFullFiles } from "./files";
import { defaultTile, describeSides, outputSide, tileChoices, tileSides } from "./graph";
import { fullTrainStep } from "./loss";
import { type LoadedTf, loadTf } from "./tfjs";
import { normalizeTile, paintDepthAxes, paintedVoxels, planeAxes, planTiles, PLANE_NAMES, readPaint, type TileSample, tileLabels, tileRaw, tileTargets } from "./tiles";

export interface FullDeps {
  loadTf: () => Promise<LoadedTf>;
  fetchFiles: FetchFullFiles;
  targetTransformFor: typeof targetTransformFor;
}

/** The largest tile side offered or accepted. */
export const MAX_TILE = 1100;

/** The tile side `requested` (null: the default) if `graph` takes it, else an Error saying which it takes. */
export function checkTileSide(graph: Parameters<typeof outputSide>[0], requested: number | null, name: string): number {
  const sides = tileSides(graph, MAX_TILE);
  if (!sides.length) throw new Error(`${name}'s training graph takes no tile side up to ${MAX_TILE}`);
  const side = requested ?? defaultTile(tileChoices(graph))!;
  if (!Number.isInteger(side) || side > MAX_TILE || outputSide(graph, side) === null) {
    throw new Error(`${name} cannot train on tiles of ${requested}²: its convolutions and pools take sides of ${describeSides(sides)}, up to ${MAX_TILE}`);
  }
  return side;
}

/** Whether a voxel size is a cube. */
const isCube = (voxel: number[]) => sameVoxelSize(voxel, voxel.map(() => voxel[0]));

/** The model's one-slice, ratio-1 geometry, the only one the graph is for. */
function planeGeometry(entry: ModelEntry, dataset: DatasetInfo) {
  const g = geometryOf(entry);
  const inShape = inputShape(g), outShape = outputShape(g);
  if (inShape[0] !== 1 || outShape[0] !== 1) throw new Error(`whole-network training is for one-slice (2D) models; ${entry.name} reads ${inShape.join("×")} voxels`);
  if (ratio(g).some((r) => r !== 1)) throw new Error(`${entry.name} writes at another voxel size than it reads (${g.output_voxel_size.join("×")} from ${g.input_voxel_size.join("×")} nm); whole-network training takes the same size in and out`);
  const rawLevel = chooseRawLevel(dataset.levels, g.input_voxel_size);
  const raw = dataset.levels[rawLevel];
  return { g, inShape, outShape, rawLevel, raw, level: outputLevel(raw, g) };
}

/** The offsets of the ONNX bytes' float initializers stored in raw_data,
 * as initializers.json gives them, read from the file itself. */
export function onnxInitializerOffsets(bytes: Uint8Array): InitializerOffsets {
  const DTYPES: Record<number, string> = { 1: "float32", 10: "float16" };
  const out: InitializerOffsets = {};
  for (const t of readGraph(readModel(bytes).graph).initializers) {
    if (!t.rawData || t.external || !DTYPES[t.dataType]) continue;
    out[t.name] = { offset: t.rawData.start, nbytes: t.rawData.end - t.rawData.start, dtype: DTYPES[t.dataType], shape: t.dims.slice() };
  }
  return out;
}

/** The offsets for `names` (initializers.json) checked against the bytes:
 * each initializer where they say, as long, of their dtype; else the file
 * is not the one they were written for (exported again since), and writing
 * at them would corrupt it. */
export function checkOffsets(bytes: Uint8Array, offsets: InitializerOffsets, names: string[], file: string): void {
  const actual = onnxInitializerOffsets(bytes);
  for (const name of names) {
    const o = offsets[name], t = actual[name];
    const where = `finetune/initializers.json does not match ${file} as served (${name}`;
    if (!o) throw new Error(`${where} has no offset): the model's files and its training graph disagree`);
    if (!t) throw new Error(`${where} is not a float initializer with raw_data in the file)`);
    if (t.offset !== o.offset || t.nbytes !== o.nbytes || t.dtype !== o.dtype) {
      throw new Error(`${where} lies at byte ${t.offset}, ${t.nbytes} bytes of ${t.dtype}, not at ${o.offset}, ${o.nbytes} of ${o.dtype}): the file was exported again since; reload the page`);
    }
  }
}

/** A RunFinetune of the whole network over `deps` (FinetuneOptions.mode "full"). */
export function createRunFull(deps: FullDeps = { loadTf, fetchFiles: fetchFullFiles, targetTransformFor }): RunFinetune {
  return async (run): Promise<Finetuned> => {
    const { report, options, volume, entry, dataset } = run;
    const refused = refuseMode(entry, "full");
    if (refused) throw new Error(refused);
    const spec = entry.manifest.finetune!.full!;
    const source = sourceOf(entry);
    const geo = planeGeometry(entry, dataset);
    if (!sameVoxelSize(volume.output_voxel_size, geo.level.voxel) || !sameVoxelSize(volume.dataset_offset_nm, geo.level.origin)) {
      throw new Error(`the volume ${volume.id} lies on a grid of ${volume.output_voxel_size.join("×")} nm, and ${entry.name} is served on ${geo.level.voxel.join("×")} nm: it was made for another model or dataset`);
    }

    const files = await deps.fetchFiles(source, spec);
    const { graph } = files;
    if (graph.context !== geo.inShape[1] - geo.outShape[1]) {
      throw new Error(`the training graph uses ${graph.context} pixels of context, and ${source.file} ${geo.inShape[1] - geo.outShape[1]}: they are not the same network`);
    }
    const offsets = files.initializers[source.file];
    if (!offsets) throw new Error(`finetune/initializers.json has no offsets for ${source.file}`);
    const names = Object.keys(graph.params);
    const start = new Uint8Array(run.initialBytes ?? run.baseBytes);
    checkOffsets(start, offsets, names, source.file);
    // The teacher is the base's weights, the model the job started from, also
    // when a restart starts from the last run's (as the head's is).
    const lambda = options.distillationLambda;
    const base = lambda > 0 ? new Uint8Array(run.baseBytes) : null;
    if (base) checkOffsets(base, offsets, names, source.file);
    const side = checkTileSide(graph, options.tileSize, entry.name);
    const output = side - graph.context;
    const frozen = new Set(spec.frozen);
    const trainable = new Set(names.filter((n) => graph.params[n].trainable && !frozen.has(n)));
    const trainableValues = [...trainable].reduce((n, name) => n + graph.params[name].shape.reduce((a, b) => a * b, 1), 0);

    // The tiles: the paint, the plane of each painted voxel, the grid over each slice.
    const chunks = await readPaint(run.paint, volume);
    const voxels = paintedVoxels(chunks);
    const painted = voxels.length / 3;
    if (!painted) throw new Error("no painted voxels: paint background (1) and objects (2 and up) on the annotation layer first, then train");
    let depth = paintDepthAxes(voxels, volume.dataset_shape_voxels);
    const planes = [0, 0, 0];
    for (const d of depth) planes[d]++;
    if ((planes[1] || planes[2]) && !(isCube(geo.raw.voxel) && isCube(geo.g.input_voxel_size))) {
      report.log(`${count(planes[1] + planes[2])} painted voxels lie in z–x or z–y slices, but the raw level (${geo.raw.voxel.join("×")} nm) or the model's voxels are not cubes, so every tile is an x–y slice and that paint trains only where an x–y slice crosses it`);
      depth = new Int8Array(depth.length);
      planes[0] = painted; planes[1] = planes[2] = 0;
    }
    const tiles = planTiles(voxels, depth, output);
    const inPlanes = planes.map((n, d) => (n ? `${count(n)} in ${PLANE_NAMES[d]} slices` : "")).filter(Boolean).join(", ");
    report.log(`Finetuning ${entry.name}'s whole network on volume ${volume.id}: ${count(painted)} painted voxels (${inPlanes}) in ${plural(tiles.length, "tile")} of ${side}² (output ${output}²)`);
    report.log(`The network: ${count(trainableValues)} weights in ${plural(trainable.size, "tensor")} train${frozen.size ? `, ${plural(frozen.size, "frozen statistic")} held` : ""}; ${plural(options.epochs, "epoch")} of ${plural(Math.min(options.batchPatches, tiles.length), "tile")} per step at learning rate ${options.learningRate}, ${lossWords(options)}, Adam`);
    report.log(lambda > 0
      ? `Distillation, λ ${lambda}: held to the starting network on ${options.distillationScope === "all" ? "every voxel" : "the unpainted voxels"} of each tile's output`
      : "Distillation off (λ 0): trained on the paint alone");
    if (run.goodRegions?.length) report.log(`${plural(run.goodRegions.length, "good region")} marked: whole-network training does not use them (the last layer's training holds the model in them)`);
    const norm = volume.input_norm, range = intensityRange(norm, dataset.dtype);
    // A tile turns where its plane's two voxel sizes are the same, in the raw and the model.
    const turnable = (depth: number) => {
      const [u, v] = planeAxes(depth as 0 | 1 | 2);
      return [geo.g.input_voxel_size, geo.raw.voxel].every((size) => sameVoxelSize([size[u]], [size[v]]));
    };
    report.log(options.augment
      ? `Augmentation on: each step's tiles get a brightness ×0.8 to ×1.2 and noise of 1% of ${range ? `the raw's range (${range[0]} to ${range[1]})` : "each tile's largest value"} on the raw before it is normalized, then random flips and quarter turns of the raw and the paint alike, with the targets${lambda > 0 ? " and the starting network's logits" : ""} made after`
      : "Augmentation off");

    const transform = deps.targetTransformFor(entry);
    const samples: TileSample[] = [];
    let distilled = 0;
    for (const tile of tiles) {
      if (report.cancelled()) throw new Error("cancelled before training");
      const labels = tileLabels(tile, output, chunks, volume);
      const t = tileTargets(labels, output, transform);
      if (!t.masked) continue;  // paint the loss cannot count here (an affinity needs a painted pair)
      const { raw, inside } = await tileRaw(tile, side, output, run.infer, dataset, geo.rawLevel);
      samples.push(options.augment
        ? { tile, input: new Float32Array(0), target: t.target, mask: t.mask, masked: t.masked, raw, inside, labels }
        : { tile, input: normalizeTile(raw, inside, norm), target: t.target, mask: t.mask, masked: t.masked });
      if (lambda > 0) distilled += options.distillationScope === "all" ? output * output : unmaskedVoxels(t.mask, t.channels);
    }
    if (!samples.length) throw new Error("the paint gives the loss nothing to count: an affinity is between two painted neighbours, so paint strokes, not single voxels");
    if (samples.length < tiles.length) report.log(`${plural(tiles.length - samples.length, "tile")} of lone painted voxels give the loss nothing to count (an affinity needs two painted neighbours) and ${tiles.length - samples.length === 1 ? "is" : "are"} left out`);
    const channels = samples[0].target.length / (output * output);

    const { tf, backend, fallback, release } = await deps.loadTf();
    // The routes refuse the run without a GPU (tfjs.ts webgpuMissing): on
    // the CPU, it is TF.js's own WebGPU backend that did not load or start.
    report.log(backend === "webgpu"
      ? "TF.js on WebGPU"
      : `TF.js on its CPU backend: ${fallback ?? "its WebGPU backend did not start"}. It trains, but a step takes minutes here where the GPU takes a fraction of a second; reloading the page tries WebGPU again, and a smaller tile is quicker`);
    // Augmentation's draws: a stream of its own, so a run without it shuffles as before.
    const augmentRandom = mulberry32((options.seed ^ 0x9e3779b9) >>> 0);
    const tally = new AugmentTally();
    let net: ReturnType<typeof buildNet> | null = null, teacher: ReturnType<typeof buildNet> | null = null, opt: ReturnType<typeof tf.train.adam> | null = null;
    try {
      net = buildNet(tf, graph, readOnnxWeights(start, offsets, names), { trainable });
      if (base) teacher = buildNet(tf, graph, readOnnxWeights(base, offsets, names), { trainable: new Set() });
      opt = tf.train.adam(options.learningRate);
      const batch = Math.max(1, Math.min(options.batchPatches, samples.length));
      const order = new Uint32Array(samples.length).map((_, i) => i);
      const random = mulberry32(options.seed);
      const breathe = yieldEvery(SLICE_MS);
      const losses: number[] = [], supervisedLosses: number[] = [];
      const x = new Float32Array(batch * side * side), t = new Float32Array(batch * output * output * channels), m = new Float32Array(t.length);
      let steps = 0, firstStepMs = 0;
      const stepsPerEpoch = Math.ceil(samples.length / batch);
      training: for (let epoch = 1; epoch <= options.epochs; epoch++) {
        if (report.cancelled()) break;
        report.log(`Starting epoch ${epoch} of ${options.epochs}`);
        shuffle(order, random);
        let sum = 0, sumSupervised = 0, sumDistillation = 0, weight = 0;
        for (let from = 0; from < samples.length; from += batch) {
          const rows = Math.min(batch, samples.length - from);
          let masked = 0;
          for (let r = 0; r < rows; r++) {
            const s = augmented(samples[order[from + r]]);
            x.set(s.input, r * s.input.length);
            t.set(s.target, r * s.target.length);
            m.set(s.mask, r * s.mask.length);
            masked += s.masked;
          }
          // Copies: a backend may keep the array it is given as the tensor's values.
          const xs = tf.tensor4d(x.slice(0, rows * side * side), [rows, side, side, 1]);
          const ts = tf.tensor4d(t.slice(0, rows * output * output * channels), [rows, output, output, channels]);
          const ms = tf.tensor4d(m.slice(0, rows * output * output * channels), [rows, output, output, channels]);
          const t0 = performance.now();
          let value: number, supervised: number, distillation = 0;
          let zs: TF.Tensor4D | null = null;
          try {
            if (teacher) zs = tf.tidy(() => teacher!.forward(xs));
            const step = fullTrainStep(tf, net, opt, xs, ts, ms, zs, options);
            value = (await step.total.data())[0];
            supervised = (await step.supervised.data())[0];
            if (step.distillation) distillation = (await step.distillation.data())[0];
            step.total.dispose(); step.supervised.dispose(); step.distillation?.dispose();
          } finally {
            xs.dispose(); ts.dispose(); ms.dispose(); zs?.dispose();
          }
          if (!Number.isFinite(value)) throw new Error(`the loss is ${value} at epoch ${epoch}: the learning rate ${options.learningRate} is too high for the whole network; lower it`);
          sum += value * masked;
          sumSupervised += supervised * masked;
          sumDistillation += distillation * masked;
          weight += masked;
          steps++;
          // The second step's time: the first compiles the GPU's shaders.
          if (steps === 1) firstStepMs = performance.now() - t0;
          if (steps === 2 || (steps === 1 && stepsPerEpoch * options.epochs === 1)) {
            const stepMs = steps === 2 ? performance.now() - t0 : firstStepMs;
            report.log(`A step takes ${Math.round(stepMs)} ms on ${backend === "webgpu" ? "this GPU" : "the CPU"}: about ${seconds((stepMs * stepsPerEpoch) / 1000)} an epoch of ${plural(stepsPerEpoch, "step")}`);
          }
          await breathe();
          if (report.cancelled()) break training;
        }
        losses.push(sum / weight);
        supervisedLosses.push(sumSupervised / weight);
        report.log(teacher
          ? `Epoch ${epoch}/${options.epochs} - Loss: ${(sum / weight).toFixed(6)} - Supervised: ${(sumSupervised / weight).toFixed(6)} - Distillation: ${(sumDistillation / weight).toFixed(6)}`
          : `Epoch ${epoch}/${options.epochs} - Loss: ${(sum / weight).toFixed(6)}`);
      }
      if (options.augment) report.log(tally.describe([null, "rows", "columns"], samples.some((s) => turnable(s.tile.depth))));
      const bytes = writeOnnxWeights(start, offsets, net.exportWeights());
      const finalLoss = losses.length ? losses[losses.length - 1] : null;
      const finalSupervised = supervisedLosses.length ? supervisedLosses[supervisedLosses.length - 1] : null;
      report.log(epochSummary(losses, options.epochs, teacher && finalSupervised !== null ? ` (supervised ${finalSupervised.toFixed(6)})` : ""));
      return {
        bytes: bytes.buffer, epochs: losses.length, losses, finalLoss, supervisedLosses,
        pixels: painted, chunks: chunks.size, distilled, tiles: samples.length,
      };
    } finally {
      net?.dispose();
      teacher?.dispose();
      opt?.dispose();
      release();
    }

    /** `s` as this step trains on it: as read, or with augmentation a new
     * draw of its raw and paint, normalized, turned, and its targets made
     * from the turned paint. */
    function augmented(s: TileSample): Pick<TileSample, "input" | "target" | "mask" | "masked"> {
      if (!s.raw || !s.inside || !s.labels) return s;
      const draw = drawAugmentation(augmentRandom, turnable(s.tile.depth));
      tally.add(draw);
      const input = transformBlock(normalizeTile(augmentIntensity(s.raw, s.inside, draw.scale, range, augmentRandom), s.inside, norm), [1, side, side], draw);
      const t = tileTargets(transformBlock(s.labels, [1, output, output], draw), output, transform);
      return { input, target: t.target, mask: t.mask, masked: t.masked };
    }
  };
}

/** The voxels of a tile's output that some channel's mask leaves out: what "unlabeled" distillation holds. */
function unmaskedVoxels(mask: Uint8Array, channels: number): number {
  let n = 0;
  for (let p = 0; p < mask.length; p += channels) {
    let all = true;
    for (let c = 0; c < channels; c++) if (!mask[p + c]) { all = false; break; }
    if (!all) n++;
  }
  return n;
}

/** The supervised loss as the log names it: "BCE on the logits, class-balanced, smoothed 0.1". */
function lossWords(o: { lossType: string; margin: number; labelSmoothing: number; balanceClasses: boolean }): string {
  const name = o.lossType === "mse" ? "MSE on the probabilities" : o.lossType === "margin" ? `the margin loss (margin ${o.margin})` : "BCE on the logits";
  const smoothing = o.lossType !== "margin" && o.labelSmoothing > 0 ? `, smoothed ${o.labelSmoothing}` : "";
  return `${name} on the painted voxels${o.balanceClasses ? ", class-balanced" : ""}${smoothing}`;
}

/** "12 s", "3.4 min". */
function seconds(s: number): string {
  return s < 90 ? `${s < 10 ? s.toFixed(1) : Math.round(s)} s` : `${(s / 60).toFixed(1)} min`;
}

export const runFull: RunFinetune = createRunFull();
