// A served model's view as stages of chunkmirage's engine (Engine.serveStages,
// web/chunkmirage/web/src/stages.ts): the browser's side of the Python
// server's pipeline (serving/chunkmirage_ops.py), two stages as there are two
// ops there:
//
//   1. the model stage: the layer's input normalizers on the block the engine
//      read, one forward pass, the model's output block [C, z, y, x] in
//      float32 (InferenceOp, which also holds the layer's input_norm);
//   2. the postprocess stage: the layer's postprocessors on that block, in the
//      chain's output dtype (PostprocessOp, one per step there; one stage for
//      the chain here, since the engine's per-stage cache is what a stage
//      buys and the model's output is the one worth keeping).
//
// The engine does the rest: it reads the raw level at the model's input
// voxel size over the chunk grown by the model's halo, pads past the volume
// with its outermost voxels (as the server's Pipeline(padding="edge")), keeps each
// stage's output under the stage's key, places the last stage's block in a
// whole chunk and encodes it. So a postprocess change finds the model's
// output made (its key is the model stage's plus the steps), a normalizer
// change recomputes everything (the model stage's key names the
// normalizers), and nothing here reads, pads or encodes.
//
// Why normalization is in the model stage rather than a stage of its own.
// stages.ts lets only the first stage read on another grid than the view's
// (its `inputVoxelSize`, the `scaled` case in Engine.stage): a model that
// reads 8 nm voxels and writes 16 nm ones must be the first stage, so a
// normalizer before it would have to change the grid itself, which a
// pointwise step cannot. And a normalized block kept under its own key would
// buy nothing: 148³ float32 is 13 MB a chunk, pushing fifteen model outputs
// out of the 512 MB staged cache each, for a reuse that never comes (a
// normalizer change changes the key, and the stage after wants the model's
// output, not its input). The Python op folds them the same way.
//
// The forward pass is a function the engine wires (chunkmirage's Onnx.run),
// so everything here runs in node and the tests feed the stages fake blocks
// and a fake model.
import type { Output } from "../../chunkmirage/web/src/onnx";
import { inputBox } from "../../chunkmirage/web/src/stages";
import type { Block, Stage, StageAt } from "../../chunkmirage/web/src/stages";
import type { Numbers } from "../../chunkmirage/web/src/asread";
import type { Chain, ModelEntry, Step } from "../contracts";
import { canonicalJson } from "../shared/args";
import { asDtype } from "./dtype";
import { type BlockGeometry, clipBox, halo as haloOf, inputShape, outputShape, ratio } from "./geometry";
import { applyNormalizers, missingChannel, selectedChannel } from "./norms";
import { type NdArray, outputChannels, outputDtype, type Postprocessor, runPostprocessors } from "./post";

/** What a forward pass returns: the output tensor's values, dims and type,
 * and the pass's time alone (chunkmirage's Onnx.run). */
export type ModelOutput = Output;

/** The model on a float32 input of `dims` ([1, 1, z, y, x]). */
export type Forward = (data: Float32Array, dims: number[]) => Promise<ModelOutput>;

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/** The model stage's key: the model file and the normalizers it is fed
 * through, so two layers of one model with the same normalizers share its
 * output whatever their postprocessing, and another normalizer is another
 * key. The steps as the dashboard sent them (strings), canonically ordered;
 * the Python server builds the steps first so "0.0" and 0 agree, which the
 * blob does for a layer's URL (shared/args.ts) and would do here too.
 *
 * The voxel size the model is run at (ModelEntry.runVoxelSize) is not in
 * it, and need not be: the engine keys a stage's output by the level's
 * voxel size and the box as well, and what the model computes depends on
 * nothing else of it. Its voxel counts and its halo in voxels are its own at
 * every voxel size, and so is the ratio of its output to its input voxel
 * size, so two run voxel sizes reading the same raw level compute the same
 * blocks on the same grid, and two reading different levels serve levels
 * of different voxel sizes. The view's name carries it (shared/args.ts
 * viewName), for the caches that go by URL. */
export function modelStageKey(entry: ModelEntry, chain: Chain): string {
  return `${entry.id}|${entry.variant.file}|norm:${canonicalJson(chain.input_norm)}|model`;
}

/** The postprocess stage's key: the model stage's and the steps. */
export function postprocessStageKey(modelKey: string, chain: Chain): string {
  return `${modelKey}|post:${canonicalJson(chain.postprocess)}`;
}

// ---------------------------------------------------------------------------
// The model stage
// ---------------------------------------------------------------------------

export interface ModelStageSpec {
  key: string;
  geometry: BlockGeometry;
  /** The raw level's voxels: what lies inside the data. Only those are
   * normalized; the rest repeats the data's outermost normalized voxels
   * (edgePadded), as the Python server pads (BlockInput.to_ndarray_ts:
   * padding is never normalized). */
  rawShape: number[];
  /** The raw level's voxel size as the engine's source declares it (the
   * level `chooseRawLevel` picked, in the source's units): the engine reads
   * that level for this stage. */
  inputVoxelSize: number[];
  normalizers: Step[];
  /** The model's output channels, as its manifest says. */
  channels: number;
  /** Whether the served array has a channel axis (model_info's has_channel). */
  hasChannel: boolean;
  run: Forward;
}

/** The model as a stage: the first of the view's, read at `inputVoxelSize`
 * with the model's context as its halo; its block is the model's output
 * over the chunk, `[C, z, y, x]` float32 (or `[z, y, x]` for a view without
 * a channel axis).
 *
 * A chunk at the volume's edge is clipped to the level by the engine (as the
 * Python pipeline clips chunk_box), so its input block is smaller than the
 * model's fixed input: the stage runs the model on the whole block all the
 * same, the data's edge repeated past what was read, and crops the output to the chunk, as
 * InferenceOp.apply_at does ("the whole chunk, as the model always computed
 * it; one the data ends in is cropped after"). */
export function modelStage(spec: ModelStageSpec): Stage {
  const g = spec.geometry;
  const halo = haloOf(g), r = ratio(g), inShape = inputShape(g), outShape = outputShape(g);
  const added = spec.hasChannel ? [spec.channels] : [];
  return {
    key: spec.key,
    halo,
    inputVoxelSize: spec.inputVoxelSize,
    dtype: "float32",
    added,
    async produce(input: Block, at: StageAt): Promise<Block> {
      const box = at.hi.map((h, a) => h - at.lo[a]);
      const block = modelInput(spec, input, at, halo, r, inShape);
      const out = await spec.run(block, [1, 1, ...inShape]);
      const nd = outputBlock(out, spec.channels, outShape);
      const cropped = box.every((n, a) => n === outShape[a]) ? nd : leading(nd, [nd.shape[0], ...box]);
      if (!spec.hasChannel) {
        if (cropped.shape[0] !== 1) throw new Error(`a view without a channel axis cannot serve ${cropped.shape[0]} channels`);
        return { data: cropped.data, shape: cropped.shape.slice(1) };
      }
      return cropped;
    },
  };
}

/** The model's input block for the chunk `at` describes, float32 and
 * normalized: the engine's `input` (the chunk's box grown by the halo, on
 * the raw level's grid; of a volume whose reader reads its channels
 * together, all of them first, of which the one the normalizers' channel
 * pick names is taken, as the server reads one: norms.ts selectedChannel)
 * converted, the voxels inside the volume normalized
 * and placed in a block of the model's input shape (the engine's block is
 * that, or its leading part for a chunk the volume ends in), every voxel past
 * the volume the nearest one inside it (edgePadded). A flat fill there is a
 * wall the model never saw in training, and it misses what the volume's edge
 * cuts through: on jrc_jurkat-1, setup_56 found 28% of the nuclei the edge
 * cuts with zeros there and 60% with the edge repeated. */
export function modelInput(spec: Pick<ModelStageSpec, "normalizers" | "rawShape">, input: Block, at: StageAt, halo: number[], r: number[], inShape: number[]): Float32Array {
  const channels = prod(input.shape.slice(0, -3)), channel = channels > 1 ? selectedChannel(spec.normalizers) : 0;
  const missing = missingChannel(channel, channels);
  if (missing) throw new Error(missing);
  const read = input.shape.slice(-3);
  // Where the engine's block lies on the raw level: the box it read for this
  // stage, as stages.ts computes it (Engine.input).
  const { lo: inLo, hi: inHi } = inputBox(at.lo, at.hi, r, halo);
  if (!read.every((n, a) => n === inHi[a] - inLo[a])) {
    throw new Error(`the engine gave a block of ${read.join("×")} for an input box of ${inHi.map((h, a) => h - inLo[a]).join("×")}`);
  }
  if (read.some((n, a) => n > inShape[a])) throw new Error(`a block of ${read.join("×")} is larger than the model's input of ${inShape.join("×")}: the view's chunks are not the model's output block`);
  // The data in the model's block: the whole block's box clipped to the
  // volume. The engine read the chunk's box (clipped to the level) grown by
  // the halo, which holds all of it unless the halo is smaller than the
  // ratio, where the Python op raises too.
  const inside = clipBox(inLo, inShape, spec.rawShape);
  if (inside && inside.hi.some((h, a) => h > inHi[a])) {
    throw new Error(`the model reads voxels ${inside.lo.join(",")} to ${inside.hi.join(",")} of the raw level, outside the block read for it (to ${inHi.join(",")})`);
  }
  if (!inside) return new Float32Array(prod(inShape));  // nothing of the volume to repeat: zeros
  const n = prod(read), picked = channels > 1 ? input.data.subarray(channel * n, (channel + 1) * n) : input.data;
  const data = Float32Array.from(picked as ArrayLike<number>);  // the engine's block, in the model's dtype
  const span = inside.hi.map((h, a) => h - inside.lo[a]);
  if (span.every((n, a) => n === inShape[a])) return applyNormalizers(data, spec.normalizers);  // the whole block is data
  // Only the data is normalized: it is cut out, normalized and its edge
  // repeated out to the model's input shape.
  const values = new Float32Array(prod(span));
  copyBox(values, span, [0, 0, 0], data, read, inside.offset, span);
  applyNormalizers(values, spec.normalizers);
  return edgePadded(values, span, inside.offset, inShape);
}

/** `values` (a box of `span` voxels, C order) placed at `offset` in a block
 * of `shape`, every voxel of the block past the box the nearest voxel in it:
 * the box's outermost voxels repeated outward, as numpy's pad(mode="edge"). */
export function edgePadded(values: Float32Array, span: readonly number[], offset: readonly number[], shape: readonly number[]): Float32Array {
  // The box's voxel nearest each of the block's along each axis.
  const near = shape.map((n, a) => Int32Array.from({ length: n }, (_, i) => Math.min(Math.max(i - offset[a], 0), span[a] - 1)));
  const block = new Float32Array(shape.reduce((p, n) => p * n, 1));
  for (let z = 0, o = 0; z < shape[0]; z++) for (let y = 0; y < shape[1]; y++) {
    const row = (near[0][z] * span[1] + near[1][y]) * span[2];
    for (let x = 0; x < shape[2]; x++) block[o++] = values[row + near[2][x]];
  }
  return block;
}

/** The output tensor as the model's block `[C, z, y, x]`: a 5-d
 * `[1, C, z, y, x]` loses its batch axis, a 4-d `[1, z, y, x]` is one
 * channel; its channels and voxels must be what the manifest says. */
export function outputBlock(out: ModelOutput, channels: number, outShape: number[]): { data: Float32Array; shape: number[] } {
  const dims = Array.from(out.dims);
  if (out.type !== "float32") throw new Error(`the model's output is ${out.type}; these models output float32`);
  const data = out.data instanceof Float32Array ? out.data : Float32Array.from(out.data as ArrayLike<number>);
  let shape: number[];
  if (dims.length === 5) {
    if (dims[0] !== 1) throw new Error(`the model's output has a batch of ${dims[0]}`);
    shape = dims.slice(1);
  } else if (dims.length === 4) shape = [1, ...dims.slice(1)];
  else throw new Error(`unexpected model output rank ${dims.length} (dims ${dims.join("×")})`);
  if (!outShape.every((n, a) => n === shape[a + 1])) {
    throw new Error(`the model wrote ${shape.slice(1).join("×")} voxels, not the ${outShape.join("×")} its manifest says`);
  }
  if (shape[0] !== channels) throw new Error(`the model wrote ${shape[0]} channel(s), not the ${channels} its manifest says`);
  if (data.length !== prod(shape)) throw new Error(`the model wrote ${data.length} values for a block of ${shape.join("×")}`);
  return { data, shape };
}

/** The leading `shape` of `nd` (its first voxels along each axis, every
 * channel): the chunk's part of a whole block computed for a chunk the
 * volume ends in. */
export function leading<T extends NdArray["data"]>(nd: { data: T; shape: number[] }, shape: number[]): { data: T; shape: number[] } {
  const [c, z, y, x] = nd.shape, [, oz, oy, ox] = shape;
  if (shape[0] !== c || oz > z || oy > y || ox > x) throw new Error(`a block of ${nd.shape.join("×")} has no leading ${shape.join("×")}`);
  const out = new (nd.data.constructor as { new (n: number): T })(prod(shape));
  for (let ic = 0; ic < c; ic++) for (let iz = 0; iz < oz; iz++) for (let iy = 0; iy < oy; iy++) {
    const from = ((ic * z + iz) * y + iy) * x, to = ((ic * oz + iz) * oy + iy) * ox;
    out.set(nd.data.subarray(from, from + ox) as never, to);
  }
  return { data: out, shape };
}

/** `shape` voxels of `src` (of `srcShape`) from `srcAt` into `dst` (of
 * `dstShape`) at `dstAt`; three axes, C order, row by row. */
export function copyBox(dst: Float32Array, dstShape: number[], dstAt: number[], src: Float32Array, srcShape: number[], srcAt: number[], shape: number[]): void {
  for (let a = 0; a < 3; a++) {
    if (dstAt[a] < 0 || dstAt[a] + shape[a] > dstShape[a] || srcAt[a] < 0 || srcAt[a] + shape[a] > srcShape[a]) {
      throw new Error(`a box of ${shape.join("×")} at ${srcAt.join(",")} in ${srcShape.join("×")} does not fit at ${dstAt.join(",")} in ${dstShape.join("×")}`);
    }
  }
  for (let z = 0; z < shape[0]; z++) for (let y = 0; y < shape[1]; y++) {
    const from = ((z + srcAt[0]) * srcShape[1] + y + srcAt[1]) * srcShape[2] + srcAt[2];
    const to = ((z + dstAt[0]) * dstShape[1] + y + dstAt[1]) * dstShape[2] + dstAt[2];
    dst.set(src.subarray(from, from + shape[2]), to);
  }
}

// ---------------------------------------------------------------------------
// The postprocess stage
// ---------------------------------------------------------------------------

export interface PostprocessStageSpec {
  key: string;
  geometry: BlockGeometry;
  steps: Step[];
  /** The steps built once for this layer (buildPostprocessors): stateful ones
   * (the merger's equivalences) keep their state across chunks. */
  postprocessors: Postprocessor[];
  /** The model's output channels, before the steps. */
  modelChannels: number;
  hasChannel: boolean;
}

/** The chain's postprocessors as the view's last stage: the model's block in,
 * the chunk's values out, in the chain's output dtype with its channels. The
 * steps that key their ids on the chunk get its index on the served grid
 * and the voxels of one whole output block, as PostprocessOp.apply_at
 * passes them (the block's, not a clipped chunk's). */
export function postprocessStage(spec: PostprocessStageSpec): Stage {
  const dtype = outputDtype(spec.steps, "float32");
  const channels = outputChannels(spec.steps, spec.modelChannels);
  const chunkNumVoxels = prod(outputShape(spec.geometry));
  return {
    key: spec.key,
    dtype,
    added: spec.hasChannel ? [channels] : [],
    async produce(input: Block, at: StageAt): Promise<Block> {
      const nd: NdArray = input.shape.length === 3 ? { data: input.data as NdArray["data"], shape: [1, ...input.shape] } : { data: input.data as NdArray["data"], shape: input.shape };
      const out = runPostprocessors(spec.postprocessors, nd, { chunkIndex: at.index, chunkNumVoxels });
      if (out.shape[0] !== channels) throw new Error(`the postprocessors left ${out.shape[0]} channel(s), not the ${channels} the chain declares`);
      // In the dtype the stage declares, as the engine checks: a
      // BigUint64Array for the segmentation steps' uint64 ids.
      const data = asDtype(out.data, dtype) as unknown as Numbers;
      if (!spec.hasChannel) {
        if (channels !== 1) throw new Error(`a view without a channel axis cannot serve ${channels} channels`);
        return { data, shape: out.shape.slice(1) };
      }
      return { data, shape: out.shape };
    },
  };
}
