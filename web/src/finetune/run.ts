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
import { concatDistillation, DISTILLATION_VOXELS, type DistillationPick, quotas, sampleDistillation } from "./distillation";
import { createFeatureRunner } from "./features";
import { anchorBoxes, anchorChunks } from "./good-regions";
import { concatPixels, type DistillationSet, gatherPixels, mulberry32, type PixelSet, trainHead } from "./head-trainer";
import { runFull } from "./full/run-full";
import { count, epochSummary, plural } from "./log-words";
import { headOf, withHead } from "./onnx-head";
import { collectSamples } from "./samples";
import { targetTransformFor } from "./targets";
import type { FinetuneOptions, FinetuneRun, HeadFinetuned, RunFinetune } from "./types";
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
    report.log(`Finetuning ${run.entry.name}'s head on volume ${volume.id}: ${count(chunks)} painted chunk${chunks === 1 ? "" : "s"} of ${volume.chunk_size.join("×")} voxels`);
    report.log(`The head: a 1×1 convolution, ${head.inChannels} → ${head.outChannels} channels (${head.weightName}${head.biasName ? `, ${head.biasName}` : ", no bias"}, ${head.dtype}); ${options.epochs} epochs of ${count(options.batchPixels)} voxels per step at learning rate ${options.learningRate}, ${lossWords(options)}${options.balanceClasses ? ", classes balanced" : ""}${options.labelSmoothing > 0 && options.lossType !== "margin" ? `, labels smoothed by ${options.labelSmoothing}` : ""}`);

    // Where the distillation term applies (lora_trainer.py's scope): the
    // good regions' unpainted voxels when the volume has any, whatever the
    // form's scope, as the server's anchors decide alone; else the form's.
    const lambda = options.distillationLambda;
    const { boxes, malformed, outside } = anchorBoxes(volume, run.goodRegions ?? []);
    for (const region of malformed) report.log(`Skipping a good region that is not one (an id, a label, offset_nm and shape_nm of three numbers): ${JSON.stringify(region)}`);
    for (const region of outside) report.log(`Good region ${region.label} (${region.id}) centres outside the volume; skipping it`);
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

    const targets = deps.targetTransformFor(run.entry);
    const runner = await deps.createFeatureRunner({ bytes: run.baseBytes, head, inputName: run.entry.manifest.io.input, outputName: run.entry.manifest.io.output, log: (line) => report.log(line) });
    try {
      const sets: PixelSet[] = [], heldSets: DistillationSet[] = [];
      let i = 0, paintedChunks = 0, painted = 0, held = 0;
      for await (const sample of deps.collectSamples(run, pick === "anchor" ? indices : undefined)) {
        i++;
        if (report.cancelled()) throw new Error("cancelled before training");
        const features = await runner.run(sample.input, sample.inputDims);
        const chunkTargets = targets(sample.chunk);
        const set = gatherPixels(features, chunkTargets, head, sample.chunk.shape);
        if (paintedKeys.has(chunkKey(sample.chunk.index))) paintedChunks++;
        painted += set.count;
        if (set.count) sets.push(set);
        let words = `${count(set.count)} painted voxels`;
        if (pick) {
          const anchor = pick === "anchor" ? anchorQuota.get(chunkKey(sample.chunk.index)) : undefined;
          const quota = pick === "anchor" ? (anchor?.quota ?? 0) : equalQuota;
          if (quota > 0) {
            const heldSet = sampleDistillation(features, chunkTargets, sample.chunk, head, { pick, quota, inRegion: anchor?.inRegion, random });
            held += heldSet.count;
            if (heldSet.count) heldSets.push(heldSet);
            words += `, ${count(heldSet.count)} held to the starting head`;
          }
        }
        report.log(`chunk ${i}/${indices.length} [${sample.chunk.index.join(", ")}]: ${words}`);
      }
      if (!painted) throw new Error("no painted voxels: paint background (1) and objects (2 and up) on the annotation layer first, then train");
      report.log(`${count(painted)} painted voxels in ${count(paintedChunks)} chunk${paintedChunks === 1 ? "" : "s"}; training the head`);
      if (pick && !held) report.log(`Distillation has no voxel to hold (${pick === "anchor" ? "every voxel of the good regions is painted" : "every voxel of the painted chunks is painted"}); training on the paint alone`);
      const pixels = concatPixels(sets, head);
      const heldSet = held ? concatDistillation(heldSets, head) : null;
      // The teacher is the head in the base's bytes, the weights the job
      // started from, also when a restart starts from the last run's head.
      const trained = await trainHead(pixels, head, options, report, run.initialHead, heldSet);
      const bytes = withHead(run.baseBytes, head, trained.weight, trained.bias);
      const finalLoss = trained.losses.length ? trained.losses[trained.losses.length - 1] : null;
      const finalSupervised = trained.supervised.length ? trained.supervised[trained.supervised.length - 1] : null;
      const supervisedWords = heldSet && finalSupervised !== null ? ` (supervised ${finalSupervised.toFixed(6)})` : "";
      report.log(epochSummary(trained.losses, options.epochs, supervisedWords));
      return {
        bytes, head, weight: trained.weight, bias: trained.bias,
        epochs: trained.losses.length, losses: trained.losses, finalLoss, supervisedLosses: trained.supervised,
        pixels: painted, chunks: paintedChunks, distilled: heldSet?.count ?? 0,
      };
    } finally {
      await runner.dispose();
    }
  };
}

/** The head's run (createRunFinetune with the real pieces). */
export const runHeadFinetune: RunFinetune = createRunFinetune();

/** A run of the mode its options say: the head's, or the whole network's,
 * which loads TensorFlow.js when it starts (full/tfjs.ts). */
export const runFinetune: RunFinetune = (run) => (run.options.mode === "full" ? runFull(run) : runHeadFinetune(run));
