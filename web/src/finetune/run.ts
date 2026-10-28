// One finetune, start to finish: the model's head found in its bytes, its
// features on every painted chunk, the head trained on the painted voxels
// and written into a copy of the model (see types.ts for why the head).
// With distillation, the voxels the term holds are sampled from the same
// feature maps as they go by (distillation.ts): the unpainted voxels of the
// painted chunks, or all of them, as the form's scope says, or, when the
// volume has good regions, their unpainted voxels instead, for which the
// regions' blocks are run through the model too when nothing in them is
// painted. The pieces are taken as a parameter (createRunFinetune) so a
// test can run the whole with its own samples and targets over the real
// ONNX step; runHeadFinetune is the run with the real ones. runFinetune,
// what the backend's finetune routes call, sends a run to it or, for the
// whole network (the options' mode "full"), to full/run-full.ts.
//
// A run of one channel of several (the options' selectChannel) trains that
// channel's row of the head alone: the trainer is handed a head of the one
// row (headRow), the channel's targets and its distillation, and the other
// rows are written back as the run started them, bit for bit. The features
// do not change, so neither does those channels' prediction, and they need
// no term to hold them (the whole network does: full/loss.ts).
import { concatDistillation, DISTILLATION_VOXELS, type DistillationPick, quotas, sampleDistillation } from "./distillation";
import { createFeatureRunner } from "./features";
import { anchorBoxes, anchorChunks } from "./good-regions";
import { concatPixels, type DistillationSet, gatherPixels, headSteps, mulberry32, type PixelSet, trainHead } from "./head-trainer";
import { runFull } from "./full/run-full";
import { count, epochSummary, plural } from "./log-words";
import { scheduleWords } from "./schedule";
import { headOf, withHead } from "./onnx-head";
import { collectSamples } from "./samples";
import { targetTransformFor } from "./target-call";
import type { FinetuneOptions, FinetuneRun, Head, HeadFinetuned, RunFinetune } from "./types";
import { chunkKey, paintedChunkIndices } from "./volume";

export interface RunDeps {
  collectSamples: typeof collectSamples;
  targetTransformFor: typeof targetTransformFor;
  createFeatureRunner: typeof createFeatureRunner;
}

/** The seed of the distillation voxels' draw, a stream apart from the trainer's. */
const SAMPLE_STREAM = 0x85ebca6b;

/** The supervised loss in words, for the plan's line. */
function lossWords(options: FinetuneOptions): string {
  if (options.lossType === "margin") return `the margin loss at margin ${options.margin}`;
  return options.lossType === "mse" ? "MSE on probabilities" : "BCE-with-logits";
}

/** A RunFinetune over `deps`: the plan logged (the painted chunks, the
 * head, the loss, the distillation), the head found, the model run once
 * per painted chunk (and per good region's block, when the regions anchor
 * the distillation) for its features, the painted voxels gathered and the
 * distillation voxels sampled, the head trained from the run's initial
 * weights (the model's own by default) toward the model's own head as the
 * teacher, and written into a copy of the model's bytes. No painted voxel
 * is an error: there is nothing to train on. */
export function createRunFinetune(deps: RunDeps = { collectSamples, targetTransformFor, createFeatureRunner }): (run: FinetuneRun) => Promise<HeadFinetuned> {
  return async (run: FinetuneRun): Promise<HeadFinetuned> => {
    const { report, options, volume } = run;
    const paintedIndices = await paintedChunkIndices(run.paint, volume.id);
    const chunks = paintedIndices.length;
    const head = headOf(run.baseBytes);
    // The manifest's head, where it declares one, must be the one found:
    // the export names its weights, and the browser writes into them.
    const declared = run.entry.manifest.finetune?.head;
    if (declared && (declared.weight !== head.weightName || (declared.bias ?? null) !== head.biasName || declared.features !== head.featureTensor)) {
      throw new Error(`${run.entry.name}'s manifest declares its head as ${declared.weight} on ${declared.features}, and its graph ends in ${head.weightName} on ${head.featureTensor}: the two disagree, so nothing is written`);
    }
    // What trains: the head, or for one channel its row (headRow), from the
    // model's own, whose other rows are written back as they are.
    const channel = options.selectChannel;
    const trainee = channel === null ? head : headRow(head, channel);
    report.log(`Finetuning ${run.entry.name}'s head on volume ${volume.id}: ${count(chunks)} painted chunk${chunks === 1 ? "" : "s"} of ${volume.chunk_size.join("×")} voxels`);
    report.log(`The head: a 1×1 convolution, ${head.inChannels} → ${head.outChannels} channels (${head.weightName}${head.biasName ? `, ${head.biasName}` : ", no bias"}, ${head.dtype}); ${options.epochs} epochs of ${count(options.batchPixels)} voxels per step, ${lossWords(options)}${options.balanceClasses ? ", classes balanced" : ""}${options.labelSmoothing > 0 && options.lossType !== "margin" ? `, labels smoothed by ${options.labelSmoothing}` : ""}`);
    if (channel !== null && head.outChannels > 1) report.log(`Training channel ${channel} (${run.entry.manifest.channels[channel] ?? "unnamed"}) alone, its row of the head; the other ${plural(head.outChannels - 1, "row")} ${head.outChannels === 2 ? "is" : "are"} kept as the model has them, and with them those channels' prediction`);

    // Where the distillation term applies (lora_trainer.py's scope): the
    // good regions' unpainted voxels when the volume has any, whatever the
    // form's scope, as the server's anchors decide alone; else the form's.
    const lambda = options.distillationLambda;
    const { boxes, malformed, outside } = await anchorBoxes(run.infer, volume, run.goodRegions ?? []);
    for (const region of malformed) report.log(`Skipping a good region that is not one (offset_nm and shape_nm of three numbers, the shape positive): ${JSON.stringify(region)}`);
    for (const region of outside) report.log(`Good region ${region.label ?? "?"} (${region.id ?? "?"}) centres outside the volume; skipping it`);
    let pick: DistillationPick | null = null;
    let indices = paintedIndices;
    const paintedKeys = new Set(paintedIndices.map(chunkKey));
    const anchorQuota = new Map<string, { quota: number; inRegion: Uint8Array }>();
    if (lambda > 0 && boxes.length) {
      pick = "anchor";
      const regionChunks = anchorChunks(volume, boxes);
      const shares = quotas(regionChunks.map((c) => c.voxels));
      regionChunks.forEach((c, i) => anchorQuota.set(chunkKey(c.index), { quota: shares[i], inRegion: c.inRegion }));
      // The painted chunks in the store's order, as without regions, then
      // the regions' blocks nothing was painted in.
      indices = [...paintedIndices, ...regionChunks.filter((c) => !paintedKeys.has(chunkKey(c.index))).map((c) => c.index)];
      report.log(`Distillation, λ ${lambda}: held to the starting head in ${plural(boxes.length, "good region")} (${plural(regionChunks.length, "block")}), their unpainted voxels, at most ${count(DISTILLATION_VOXELS)}; the form's scope (${options.distillationScope}) does not apply while good regions are marked, as on the server`);
    } else if (lambda > 0) {
      pick = options.distillationScope;
      report.log(`Distillation, λ ${lambda}: held to the starting head on ${pick === "all" ? "every voxel" : "the unpainted voxels"} of the painted chunks, at most ${count(DISTILLATION_VOXELS)} (${count(Math.ceil(DISTILLATION_VOXELS / Math.max(chunks, 1)))} a chunk)`);
    } else if (boxes.length) {
      report.log(`${plural(boxes.length, "good region")} marked, but distillation is off (λ 0), so they hold nothing: give distillation a weight to use them`);
    }
    const equalQuota = Math.ceil(DISTILLATION_VOXELS / Math.max(chunks, 1));
    const random = mulberry32((options.seed ^ SAMPLE_STREAM) >>> 0);

    const targets = deps.targetTransformFor(run.entry, run.infer, channel);
    const runner = await deps.createFeatureRunner({ bytes: run.baseBytes, head, inputName: run.entry.manifest.io.input, outputName: run.entry.manifest.io.output, log: (line) => report.log(line) });
    try {
      const sets: PixelSet[] = [], heldSets: DistillationSet[] = [];
      let i = 0, paintedChunks = 0, painted = 0, held = 0;
      for await (const sample of deps.collectSamples(run, pick === "anchor" ? indices : undefined)) {
        i++;
        if (report.cancelled()) throw new Error("cancelled before training");
        const features = await runner.run(sample.input, sample.inputDims);
        const chunkTargets = await targets(sample.chunk);
        const set = gatherPixels(features, chunkTargets, trainee, sample.chunk.shape);
        if (paintedKeys.has(chunkKey(sample.chunk.index))) paintedChunks++;
        painted += set.count;
        if (set.count) sets.push(set);
        let words = `${count(set.count)} painted voxels`;
        if (pick) {
          const anchor = pick === "anchor" ? anchorQuota.get(chunkKey(sample.chunk.index)) : undefined;
          const quota = pick === "anchor" ? (anchor?.quota ?? 0) : equalQuota;
          if (quota > 0) {
            const heldSet = sampleDistillation(features, chunkTargets, sample.chunk, trainee, { pick, quota, inRegion: anchor?.inRegion, random });
            held += heldSet.count;
            if (heldSet.count) heldSets.push(heldSet);
            words += `, ${count(heldSet.count)} held to the starting head`;
          }
        }
        report.log(`chunk ${i}/${indices.length} [${sample.chunk.index.join(", ")}]: ${words}`);
      }
      if (!painted) throw new Error("no painted voxels: paint background (1) and objects (2 and up) on the annotation layer first, then train");
      // The run's steps, and so its schedule, are known only now: the epochs pass over every painted voxel.
      report.log(`${count(painted)} painted voxels in ${count(paintedChunks)} chunk${paintedChunks === 1 ? "" : "s"}; training the head at learning rate ${scheduleWords(options.learningRate, headSteps(painted, options), options.learningRateSchedule)}`);
      if (pick && !held) report.log(`Distillation has no voxel to hold (${pick === "anchor" ? "every voxel of the good regions is painted" : "every voxel of the painted chunks is painted"}); training on the paint alone`);
      const pixels = concatPixels(sets, trainee);
      const heldSet = held ? concatDistillation(heldSets, trainee) : null;
      // The teacher is the head in the base's bytes, the one it starts from.
      const trained = await trainHead(pixels, trainee, options, report, heldSet);
      const { weight, bias } = channel === null ? trained : withRow(head, head.inChannels, channel, trained);
      const bytes = withHead(run.baseBytes, head, weight, bias);
      const finalLoss = trained.losses.length ? trained.losses[trained.losses.length - 1] : null;
      const finalSupervised = trained.supervised.length ? trained.supervised[trained.supervised.length - 1] : null;
      const supervisedWords = heldSet && finalSupervised !== null ? ` (supervised ${finalSupervised.toFixed(6)})` : "";
      report.log(epochSummary(trained.losses, options.epochs, supervisedWords));
      return {
        bytes, head, weight, bias,
        epochs: trained.losses.length, losses: trained.losses, finalLoss, supervisedLosses: trained.supervised,
        pixels: painted, chunks: paintedChunks, distilled: heldSet?.count ?? 0,
      };
    } finally {
      await runner.dispose();
    }
  };
}

/** Weights of a head ([outChannels × inChannels] and a bias per output channel). */
type HeadWeights = { weight: Float32Array; bias: Float32Array };

/** Row `channel` of `weights`, a head of `inChannels` inputs: that output channel's weights and bias. */
function rowOf(weights: HeadWeights, inChannels: number, channel: number): HeadWeights {
  return { weight: weights.weight.slice(channel * inChannels, (channel + 1) * inChannels), bias: weights.bias.slice(channel, channel + 1) };
}

/** `head` cut down to its row `channel`, a head of one output channel: what
 * a run of that channel trains, the base's row its teacher. Its names are
 * the whole convolution's, so it is never written on its own (withHead
 * refuses a head of the wrong size): its row goes back through withRow. */
function headRow(head: Head, channel: number): Head {
  if (!(Number.isInteger(channel) && channel >= 0 && channel < head.outChannels)) {
    throw new Error(`channel ${channel} is not one of the ${head.outChannels} output channels the head writes`);
  }
  return { ...head, outChannels: 1, ...rowOf(head, head.inChannels, channel) };
}

/** A copy of `weights` with row `channel` replaced by `row`'s, every other row as it was, bit for bit. */
function withRow(weights: HeadWeights, inChannels: number, channel: number, row: HeadWeights): HeadWeights {
  const weight = weights.weight.slice(), bias = weights.bias.slice();
  weight.set(row.weight, channel * inChannels);
  bias.set(row.bias, channel);
  return { weight, bias };
}

/** The head's run (createRunFinetune with the real pieces). */
export const runHeadFinetune: RunFinetune = createRunFinetune();

/** A run of the mode its options say: the head's, or the whole network's,
 * which loads TensorFlow.js when it starts (full/tfjs.ts). */
export const runFinetune: RunFinetune = (run) => (run.options.mode === "full" ? runFull(run) : runHeadFinetune(run));
