// One output chunk of a served model: the body of chunkmirage's produce()
// for a model view, from Marwan Zouinkhi's web/legacy/compute.js
// (handleChunk) and zarr-reader.js (readRoi).
//
//   read the input block (the output chunk grown by the model's context,
//   clipped to the raw level and zero-padded past it)
//   → normalizers → one forward pass → postprocessors → the chunk's bytes.
//
// The reading and the forward pass are injected (`read` is Engine.read on the
// raw view's level, `run` the ORT session behind its lock), so everything
// here runs in node too and the tests check the whole pipeline with a fake
// model. Nothing in it is on a Worker: Marwan ran it in a module Worker
// because the service worker could not run ORT, but here the Engine already
// runs on the page and ORT's WebGPU work is asynchronous, so the main thread
// only spends the normalizer and postprocessor loops (a few ms per block). A
// Worker could take those later without changing this file's interface.
import type { Chain } from "../contracts";
import { chunkLayout, encodeChunk, placeBlock, prod, toFloat32 } from "./encode";
import { type BlockGeometry, clipBox, inputBox, inputShape, outputShape } from "./geometry";
import { applyNormalizers } from "./norms";
import { applyPostprocessors, outputChannels, outputDtype, type NdArray } from "./post";

/** What a forward pass returns: the output tensor's values, dims and type. */
export interface ModelOutput {
  data: ArrayLike<number>;
  dims: readonly number[];
  type: string;
}

export interface ProduceContext {
  geometry: BlockGeometry;
  /** The raw level read: its voxels and dtype. */
  rawShape: number[];
  rawDtype: string;
  chain: Chain;
  /** Whether the served array has a channel axis (model_info's has_channel). */
  hasChannel: boolean;
  /** Voxels [lo, hi) of the raw level, C order, in its dtype (Engine.read). */
  read(lo: number[], hi: number[]): Promise<ArrayBuffer>;
  /** The model on an input of `dims` ([1, 1, z, y, x]). */
  run(input: Float32Array, dims: number[]): Promise<ModelOutput>;
}

/** The model's input block for output chunk `index`, as float32: what lies
 * inside the raw level, zeros outside it. */
export async function readInputBlock(ctx: ProduceContext, index: number[]): Promise<Float32Array> {
  const { begin, shape } = inputBox(ctx.geometry, index);
  const clipped = clipBox(begin, shape, ctx.rawShape);
  const out = new Float32Array(prod(shape));
  if (!clipped) return out;  // wholly outside the data
  const blockShape = clipped.hi.map((h, a) => h - clipped.lo[a]);
  const block = toFloat32(await ctx.read(clipped.lo, clipped.hi), ctx.rawDtype);
  if (block.length !== prod(blockShape)) {
    throw new Error(`read ${block.length} voxels for a block of ${blockShape.join("×")} (${prod(blockShape)})`);
  }
  if (blockShape.every((n, a) => n === shape[a])) return block;  // the whole block is data
  placeBlock(out, shape, block, blockShape, clipped.offset);
  return out;
}

/** The output tensor as a channel-first ndarray [C, z, y, x]: a 5-d
 * [1, C, z, y, x] loses its batch axis, a 4-d [1, z, y, x] is one channel. */
export function outputToNdArray(out: ModelOutput): NdArray {
  const dims = Array.from(out.dims);
  if (out.type !== "float32") throw new Error(`the model's output is ${out.type}; these models output float32`);
  const data = out.data instanceof Float32Array ? out.data : Float32Array.from(out.data);
  if (dims.length === 5) {
    if (dims[0] !== 1) throw new Error(`the model's output has a batch of ${dims[0]}`);
    return { data, shape: dims.slice(1) };
  }
  if (dims.length === 4) return { data, shape: [1, ...dims.slice(1)] };
  throw new Error(`unexpected model output rank ${dims.length} (dims ${dims.join("×")})`);
}

/** The bytes of output chunk `index`: one whole chunk, channel first (see
 * encode.ts), in the chain's output dtype. */
export async function produceChunk(ctx: ProduceContext, index: number[]): Promise<ArrayBuffer> {
  const input = await readInputBlock(ctx, index);
  applyNormalizers(input, ctx.chain.input_norm);
  const dims = [1, 1, ...inputShape(ctx.geometry)];
  let nd = outputToNdArray(await ctx.run(input, dims));
  const block = outputShape(ctx.geometry);
  if (!block.every((n, a) => n === nd.shape[a + 1])) {
    throw new Error(`the model wrote ${nd.shape.slice(1).join("×")} voxels, not the ${block.join("×")} its manifest says`);
  }
  nd = applyPostprocessors(nd, ctx.chain.postprocess);
  const channels = outputChannels(ctx.chain.postprocess, nd.shape[0]);
  const laid = chunkLayout(nd, ctx.hasChannel);
  const chunkShape = ctx.hasChannel ? [channels, ...block] : block;
  return encodeChunk(laid, outputDtype(ctx.chain.postprocess, "float32"), chunkShape);
}
