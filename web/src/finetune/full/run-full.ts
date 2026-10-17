// One whole-network finetune of a 2D model, start to finish: the folder's
// training graph and weight offsets fetched (files.ts), the paint cut into
// tiles in the planes it was drawn in and read with its raw (tiles.ts),
// TF.js loaded (tfjs.ts), the network built from the weights in the served
// .onnx and trained on the tiles with masked BCE on its logits and Adam
// (Marwan Zouinkhi's runtime, cellmap-tfjs-train.ts), and the trained
// weights written back into a copy of the .onnx at their offsets, so the
// result serves in onnxruntime-web exactly as the model did.
//
// What trains: every param the graph marks trainable but the manifest's
// `frozen` (BatchNorm's running statistics, which the forward pass reads as
// constants), on the painted voxels alone: no distillation, no good
// regions, no class balance or label smoothing, which are the head's.
// The epochs and the learning rate are the form's; the batch is in tiles
// (the form's Batch Size), the tile's side the form's Tile size, checked
// against the sides the graph takes (graph.ts).
//
// The pieces are a parameter (createRunFull) so a test can run it with its
// own TF.js loader and files.
import type { DatasetInfo, ModelEntry } from "../../contracts";
import { chooseRawLevel, geometryOf, inputShape, outputLevel, outputShape, ratio, sameVoxelSize } from "../../infer/geometry";
import { mulberry32, shuffle, SLICE_MS, yieldEvery } from "../head-trainer";
import { count, epochSummary, plural } from "../log-words";
import { refuseMode, sourceOf } from "../modes";
import { readGraph, readModel } from "../onnx-proto";
import { targetTransformFor } from "../targets";
import type { Finetuned, RunFinetune } from "../types";
import { buildNet, type InitializerOffsets, readOnnxWeights, trainStep, writeOnnxWeights } from "./cellmap-tfjs-train";
import { type FetchFullFiles, fetchFullFiles } from "./files";
import { defaultTile, describeSides, outputSide, tileChoices, tileSides } from "./graph";
import { type LoadedTf, loadTf } from "./tfjs";
import { paintDepthAxes, paintedVoxels, planTiles, PLANE_NAMES, readPaint, type TileSample, tileInput, tileLabels, tileTargets } from "./tiles";

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
    report.log(`The network: ${count(trainableValues)} weights in ${plural(trainable.size, "tensor")} train${frozen.size ? `, ${plural(frozen.size, "frozen statistic")} held` : ""}; ${plural(options.epochs, "epoch")} of ${plural(Math.min(options.batchPatches, tiles.length), "tile")} per step at learning rate ${options.learningRate}, masked BCE on the logits, Adam`);
    if (run.goodRegions?.length) report.log(`${plural(run.goodRegions.length, "good region")} marked: whole-network training does not use them (it trains on the paint alone; the last layer's training holds the model in them)`);

    const transform = deps.targetTransformFor(entry);
    const samples: TileSample[] = [];
    for (const tile of tiles) {
      if (report.cancelled()) throw new Error("cancelled before training");
      const labels = tileLabels(tile, output, chunks, volume);
      const t = tileTargets(labels, output, transform);
      if (!t.masked) continue;  // paint the loss cannot count here (an affinity needs a painted pair)
      const input = await tileInput(tile, side, output, run.infer, dataset, geo.rawLevel, volume.input_norm);
      samples.push({ tile, input, target: t.target, mask: t.mask, masked: t.masked });
    }
    if (!samples.length) throw new Error("the paint gives the loss nothing to count: an affinity is between two painted neighbours, so paint strokes, not single voxels");
    if (samples.length < tiles.length) report.log(`${plural(tiles.length - samples.length, "tile")} of lone painted voxels give the loss nothing to count (an affinity needs two painted neighbours) and ${tiles.length - samples.length === 1 ? "is" : "are"} left out`);
    const channels = samples[0].target.length / (output * output);

    const { tf, backend, fallback, release } = await deps.loadTf();
    report.log(backend === "webgpu"
      ? "TF.js on WebGPU"
      : `TF.js on its CPU backend: ${fallback ?? "no WebGPU"}. It trains, but a step takes minutes here where a GPU takes a fraction of a second; try Chrome or Edge on a machine with a GPU, or a smaller tile`);
    let net: ReturnType<typeof buildNet> | null = null, opt: ReturnType<typeof tf.train.adam> | null = null;
    try {
      net = buildNet(tf, graph, readOnnxWeights(start, offsets, names), { trainable });
      opt = tf.train.adam(options.learningRate);
      const batch = Math.max(1, Math.min(options.batchPatches, samples.length));
      const order = new Uint32Array(samples.length).map((_, i) => i);
      const random = mulberry32(options.seed);
      const breathe = yieldEvery(SLICE_MS);
      const losses: number[] = [];
      const x = new Float32Array(batch * side * side), t = new Float32Array(batch * output * output * channels), m = new Float32Array(t.length);
      let steps = 0, firstStepMs = 0;
      const stepsPerEpoch = Math.ceil(samples.length / batch);
      training: for (let epoch = 1; epoch <= options.epochs; epoch++) {
        if (report.cancelled()) break;
        report.log(`Starting epoch ${epoch} of ${options.epochs}`);
        shuffle(order, random);
        let sum = 0, weight = 0;
        for (let from = 0; from < samples.length; from += batch) {
          const rows = Math.min(batch, samples.length - from);
          let masked = 0;
          for (let r = 0; r < rows; r++) {
            const s = samples[order[from + r]];
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
          let value: number;
          try {
            const loss = trainStep(tf, net, opt, xs, ts, ms);
            value = (await loss.data())[0];
            loss.dispose();
          } finally {
            xs.dispose(); ts.dispose(); ms.dispose();
          }
          if (!Number.isFinite(value)) throw new Error(`the loss is ${value} at epoch ${epoch}: the learning rate ${options.learningRate} is too high for the whole network; lower it`);
          sum += value * masked;
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
        report.log(`Epoch ${epoch}/${options.epochs} - Loss: ${(sum / weight).toFixed(6)}`);
      }
      const bytes = writeOnnxWeights(start, offsets, net.exportWeights());
      const finalLoss = losses.length ? losses[losses.length - 1] : null;
      report.log(epochSummary(losses, options.epochs));
      return {
        bytes: bytes.buffer, epochs: losses.length, losses, finalLoss, supervisedLosses: losses.slice(),
        pixels: painted, chunks: chunks.size, distilled: 0, tiles: samples.length,
      };
    } finally {
      net?.dispose();
      opt?.dispose();
      release();
    }
  };
}

/** "12 s", "3.4 min". */
function seconds(s: number): string {
  return s < 90 ? `${s < 10 ? s.toFixed(1) : Math.round(s)} s` : `${(s / 60).toFixed(1)} min`;
}

export const runFull: RunFinetune = createRunFull();
