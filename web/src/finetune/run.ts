// One finetune, start to finish: the model's head found in its bytes, its
// features on every painted chunk, the head trained on the painted voxels
// and written into a copy of the model (see types.ts for why the head).
// The pieces are taken as a parameter (createRunFinetune) so a test can run
// the whole with its own samples and targets over the real ONNX step;
// runFinetune is the run with the real ones, what the backend's finetune
// routes call.
import { createFeatureRunner } from "./features";
import { concatPixels, gatherPixels, type PixelSet, trainHead } from "./head-trainer";
import { headOf, withHead } from "./onnx-head";
import { collectSamples } from "./samples";
import { targetTransformFor } from "./targets";
import type { Finetuned, FinetuneRun, RunFinetune } from "./types";
import { paintedChunkIndices } from "./volume";

export interface RunDeps {
  collectSamples: typeof collectSamples;
  targetTransformFor: typeof targetTransformFor;
  createFeatureRunner: typeof createFeatureRunner;
}

const count = (n: number) => n.toLocaleString("en-US");

/** A RunFinetune over `deps`: the plan logged (the painted chunks, the
 * head), the head found, the model run once per painted chunk for its
 * features, the painted voxels gathered, the head trained from the run's
 * initial weights (the model's own by default) and written into a copy of
 * the model's bytes. No painted voxel is an error: there is nothing to
 * train on. */
export function createRunFinetune(deps: RunDeps = { collectSamples, targetTransformFor, createFeatureRunner }): RunFinetune {
  return async (run: FinetuneRun): Promise<Finetuned> => {
    const { report, options } = run;
    const chunks = (await paintedChunkIndices(run.paint, run.volume.id)).length;
    const head = headOf(run.baseBytes);
    report.log(`Finetuning ${run.entry.name}'s head on volume ${run.volume.id}: ${count(chunks)} painted chunk${chunks === 1 ? "" : "s"} of ${run.volume.chunk_size.join("×")} voxels`);
    report.log(`The head: a 1×1 convolution, ${head.inChannels} → ${head.outChannels} channels (${head.weightName}${head.biasName ? `, ${head.biasName}` : ", no bias"}, ${head.dtype}); ${options.epochs} epochs of ${count(options.batchPixels)} voxels per step at learning rate ${options.learningRate}${options.balanceClasses ? ", classes balanced" : ""}${options.labelSmoothing > 0 ? `, labels smoothed by ${options.labelSmoothing}` : ""}`);
    const targets = deps.targetTransformFor(run.entry);
    const runner = await deps.createFeatureRunner({ bytes: run.baseBytes, head, inputName: run.entry.manifest.io.input, outputName: run.entry.manifest.io.output, log: (line) => report.log(line) });
    try {
      const sets: PixelSet[] = [];
      let i = 0, painted = 0;
      for await (const sample of deps.collectSamples(run)) {
        i++;
        if (report.cancelled()) throw new Error("cancelled before training");
        const features = await runner.run(sample.input, sample.inputDims);
        const set = gatherPixels(features, targets(sample.chunk), head, sample.chunk.shape);
        painted += set.count;
        report.log(`chunk ${i}/${chunks} [${sample.chunk.index.join(", ")}]: ${count(set.count)} painted voxels`);
        if (set.count) sets.push(set);
      }
      if (!painted) throw new Error("no painted voxels: paint background (1) and objects (2 and up) on the annotation layer first, then train");
      report.log(`${count(painted)} painted voxels in ${count(i)} chunk${i === 1 ? "" : "s"}; training the head`);
      const pixels = concatPixels(sets, head);
      const trained = await trainHead(pixels, head, options, report, run.initialHead);
      const bytes = withHead(run.baseBytes, head, trained.weight, trained.bias);
      const finalLoss = trained.losses.length ? trained.losses[trained.losses.length - 1] : null;
      report.log(trained.losses.length === options.epochs ? `Trained ${trained.losses.length} epochs; final loss ${finalLoss!.toFixed(6)}` : `Stopped after ${trained.losses.length} of ${options.epochs} epochs${finalLoss === null ? "" : `; loss ${finalLoss.toFixed(6)}`}`);
      return { bytes, head, weight: trained.weight, bias: trained.bias, epochs: trained.losses.length, losses: trained.losses, finalLoss, pixels: painted, chunks: i };
    } finally {
      await runner.dispose();
    }
  };
}

export const runFinetune: RunFinetune = createRunFinetune();
