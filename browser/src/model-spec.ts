// Browser-side analogue of cellmap-flow's Python model spec. Mirrors the
// declarative globals every example/*.py defines (input_voxel_size,
// output_voxel_size, read_shape, write_shape, output_channels, block_shape)
// plus normalize + postprocess steps that we can express declaratively.
//
// Spec authors translate their Python `model_spec.py` to a JSON file matching
// this interface. The pipeline here is the JS equivalent of
// `Inferencer.process_chunk_basic` + `apply_postprocess`.

export type Triple = [number, number, number]; // (z, y, x)

export type DType = "uint8" | "uint16" | "int16" | "float32";

// How the model expects its input tensor to be shaped from a (Cin, Dz, Dy, Dx)
// volume:
//   "NCDHW"        -> (1, Cin, Dz, Dy, Dx) — standard 3D conv input
//   "NDHWC"        -> (1, Dz, Dy, Dx, Cin) — channels-last
//   "BatchZ_NCHW"  -> (Dz, Cin, Dy, Dx)    — Z-as-batch, for 2D models
export type TensorLayout = "NCDHW" | "NDHWC" | "BatchZ_NCHW";

// cellmap-flow class-name-based pipeline. Mirrors cellmap_flow's
// `to_dict()` output so a spec exported from the Python pipeline builder
// works as-is here. See browser/src/cellmap-pipeline.ts for ports.
export type { NormSpec, PostSpec, PostContext } from "./cellmap-pipeline";
import {
  type NormSpec,
  type PostSpec,
  type PostContext,
  makeNormalizer,
  makePostprocessor,
} from "./cellmap-pipeline";

export interface ModelSpec {
  // Geometry — all shapes are in WORLD UNITS (e.g. nanometers).
  inputVoxelSize: Triple;
  outputVoxelSize: Triple;
  readShape: Triple;        // input ROI fed to model (world units)
  writeShape: Triple;       // output ROI returned by model (world units)

  // I/O dtype + channels
  inputDtype: DType;        // dtype of values fed into the ONNX tensor
  outputDtype: DType;       // dtype of chunks served to NG
  outputChannels: number;
  blockShape: Triple;       // chunk shape in OUTPUT VOXELS (z, y, x)

  // Tensor layout
  tensorLayout: TensorLayout;

  // Pipeline ops, declared by cellmap-flow class name + kwargs.
  // `normalize` runs in order on the float input volume before the model.
  // `postprocess` runs in order on the float (C, Dz, Dy, Dx) output.
  normalize?: NormSpec[];
  postprocess?: PostSpec[];
}

// Defaults that match the existing demo (`scripts/export_demo_onnx.py`).
// Used when the user provides only an ONNX URL without a spec URL.
export const DEMO_SPEC: ModelSpec = {
  inputVoxelSize: [1, 1, 1],
  outputVoxelSize: [1, 1, 1],
  readShape: [16, 128, 128],
  writeShape: [16, 128, 128],
  inputDtype: "float32",
  outputDtype: "uint8",
  outputChannels: 1,
  blockShape: [16, 128, 128],
  tensorLayout: "BatchZ_NCHW",
  // Demo model bakes /255 into its own forward; output sigmoid is in [0,1].
  // Map to uint8 with bias=0, multiplier=255 (no clip needed).
  normalize: [],
  postprocess: [
    { name: "DefaultPostprocessor", clip_min: 0, clip_max: 1, bias: 0, multiplier: 255 },
  ],
};

export async function loadSpec(url: string): Promise<ModelSpec> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`failed to fetch spec ${url}: ${r.status}`);
  const raw = (await r.json()) as Partial<ModelSpec>;
  return validateSpec(raw);
}

function validateSpec(raw: Partial<ModelSpec>): ModelSpec {
  const required: (keyof ModelSpec)[] = [
    "inputVoxelSize",
    "outputVoxelSize",
    "readShape",
    "writeShape",
    "inputDtype",
    "outputDtype",
    "outputChannels",
    "blockShape",
    "tensorLayout",
  ];
  for (const k of required) {
    if (raw[k] === undefined) throw new Error(`spec missing required field: ${k}`);
  }
  // Sanity: read >= write, both divisible by voxel size, blockShape > 0.
  const ws = raw.writeShape!;
  const rs = raw.readShape!;
  for (let i = 0; i < 3; i++) {
    if (rs[i] < ws[i]) {
      throw new Error(`readShape[${i}] (${rs[i]}) < writeShape[${i}] (${ws[i]})`);
    }
    if ((rs[i] - ws[i]) % 2 !== 0) {
      throw new Error(`(read-write)/2 not integer at axis ${i}: ${rs[i]} - ${ws[i]}`);
    }
  }
  return {
    inputVoxelSize: raw.inputVoxelSize!,
    outputVoxelSize: raw.outputVoxelSize!,
    readShape: rs,
    writeShape: ws,
    inputDtype: raw.inputDtype!,
    outputDtype: raw.outputDtype!,
    outputChannels: raw.outputChannels!,
    blockShape: raw.blockShape!,
    tensorLayout: raw.tensorLayout!,
    normalize: raw.normalize ?? [],
    postprocess: raw.postprocess ?? [],
  };
}

// world-units / voxel-size  -> integer voxel coords (round)
export function context(spec: ModelSpec): Triple {
  return [
    (spec.readShape[0] - spec.writeShape[0]) / 2,
    (spec.readShape[1] - spec.writeShape[1]) / 2,
    (spec.readShape[2] - spec.writeShape[2]) / 2,
  ];
}

// Apply normalize ops in order. Each instance comes from cellmap-pipeline's
// registry by class name (cellmap-flow's InputNormalizer subclasses).
export function applyNormalize(buf: Float32Array, ops: NormSpec[] | undefined): Float32Array {
  if (!ops || ops.length === 0) return buf;
  let cur = buf;
  for (const spec of ops) {
    cur = makeNormalizer(spec).process(cur);
  }
  return cur;
}

// Apply postprocess ops in order. `data` is (C, Dz, Dy, Dx) flattened C-major.
// Each instance comes from cellmap-pipeline's PostProcessor registry by name.
export function applyPostprocess(
  data: Float32Array,
  channels: number,
  spatial: number,
  ops: PostSpec[],
  ctx: PostContext,
): { data: Float32Array; channels: number } {
  let cur = data;
  let curC = channels;
  for (const spec of ops) {
    const r = makePostprocessor(spec).process(cur, curC, spatial, ctx);
    cur = r.data;
    curC = r.channels;
  }
  return { data: cur, channels: curC };
}

// Float32 (C,Dz,Dy,Dx) -> packed bytes for a Zarr v2 chunk in `dtype`.
// Output order in the chunk is (Dz, Dy, Dx, C) so NG sees a multichannel volume.
export function encodeChunk(
  data: Float32Array,
  channels: number,
  block: Triple,
  dtype: DType,
): ArrayBuffer {
  const [dz, dy, dx] = block;
  const spatial = dz * dy * dx;
  const total = spatial * channels;
  if (data.length !== total) {
    throw new Error(`postproc length ${data.length} != ${total} = ${dz}x${dy}x${dx}x${channels}`);
  }

  // Re-layout (C, S) -> (S, C)
  const interleaved =
    channels === 1 ? data : interleaveChannels(data, channels, spatial);

  switch (dtype) {
    case "uint8": {
      const out = new Uint8Array(total);
      for (let i = 0; i < total; i++) {
        const v = interleaved[i];
        out[i] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
      return out.buffer;
    }
    case "uint16": {
      const out = new Uint16Array(total);
      for (let i = 0; i < total; i++) {
        const v = interleaved[i];
        out[i] = v < 0 ? 0 : v > 65535 ? 65535 : v;
      }
      return out.buffer;
    }
    case "int16": {
      const out = new Int16Array(total);
      for (let i = 0; i < total; i++) {
        const v = interleaved[i];
        out[i] = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
      }
      return out.buffer;
    }
    case "float32":
      return interleaved.slice().buffer;
  }
}

function interleaveChannels(data: Float32Array, channels: number, spatial: number): Float32Array {
  const out = new Float32Array(channels * spatial);
  for (let s = 0; s < spatial; s++) {
    for (let c = 0; c < channels; c++) {
      out[s * channels + c] = data[c * spatial + s];
    }
  }
  return out;
}

// Zarr v2 dtype string for this output dtype.
export function zarrDtype(dtype: DType): string {
  switch (dtype) {
    case "uint8": return "|u1";
    case "uint16": return "<u2";
    case "int16": return "<i2";
    case "float32": return "<f4";
  }
}

// Compute the dims to feed runModel given a (Cin, Dz, Dy, Dx) input volume.
export function tensorDims(
  layout: TensorLayout,
  cin: number,
  dz: number,
  dy: number,
  dx: number,
): number[] {
  switch (layout) {
    case "NCDHW": return [1, cin, dz, dy, dx];
    case "NDHWC": return [1, dz, dy, dx, cin];
    case "BatchZ_NCHW": return [dz, cin, dy, dx];
  }
}

// Reshape ORT output (whatever the layout) to (Cout, Dz, Dy, Dx) Float32 contiguous.
export function reshapeOutputToCDHW(
  outData: Float32Array,
  outDims: readonly number[],
  layout: TensorLayout,
  expectChannels: number,
  expectDz: number,
  expectDy: number,
  expectDx: number,
): Float32Array {
  if (layout === "NCDHW") {
    // (1, Cout, Dz, Dy, Dx) -> already in (Cout, Dz, Dy, Dx) order; just slice.
    void outDims;
    return outData.subarray(0, expectChannels * expectDz * expectDy * expectDx).slice();
  }
  if (layout === "BatchZ_NCHW") {
    // (Dz, Cout, Dy, Dx) -> (Cout, Dz, Dy, Dx)
    const spatialZ = expectDy * expectDx;
    const out = new Float32Array(expectChannels * expectDz * spatialZ);
    for (let z = 0; z < expectDz; z++) {
      for (let c = 0; c < expectChannels; c++) {
        const src = (z * expectChannels + c) * spatialZ;
        const dst = (c * expectDz + z) * spatialZ;
        out.set(outData.subarray(src, src + spatialZ), dst);
      }
    }
    return out;
  }
  // NDHWC: (1, Dz, Dy, Dx, Cout) -> (Cout, Dz, Dy, Dx)
  const spatial = expectDz * expectDy * expectDx;
  const out = new Float32Array(expectChannels * spatial);
  for (let s = 0; s < spatial; s++) {
    for (let c = 0; c < expectChannels; c++) {
      out[c * spatial + s] = outData[s * expectChannels + c];
    }
  }
  return out;
}
