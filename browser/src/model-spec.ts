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

export type NormSpec =
  | { type: "identity" }
  | { type: "scale_offset"; scale: number; offset: number }   // (x*scale)+offset
  | { type: "mean_std"; mean: number; std: number }            // (x - mean) / std
  | { type: "minmax"; min: number; max: number };              // (x - min) / (max - min)

export type PostSpec =
  | { type: "clip"; min: number; max: number }
  | { type: "scale"; factor: number }                  // y *= factor
  | { type: "offset"; value: number }                  // y += value
  | { type: "channel"; index: number }                 // pick one channel
  | { type: "threshold"; value: number; below?: number; above?: number };

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

  // Pipeline ops (applied in order to the input -> tensor -> output stream)
  normalize?: NormSpec;
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
  normalize: { type: "identity" },             // demo model does /255 itself
  postprocess: [{ type: "scale", factor: 255 }],
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
    normalize: raw.normalize ?? { type: "identity" },
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

// Apply the normalize op in-place on a Float32Array.
export function applyNormalize(buf: Float32Array, n: NormSpec | undefined): Float32Array {
  if (!n || n.type === "identity") return buf;
  if (n.type === "scale_offset") {
    for (let i = 0; i < buf.length; i++) buf[i] = buf[i] * n.scale + n.offset;
    return buf;
  }
  if (n.type === "mean_std") {
    const inv = 1 / (n.std || 1);
    for (let i = 0; i < buf.length; i++) buf[i] = (buf[i] - n.mean) * inv;
    return buf;
  }
  if (n.type === "minmax") {
    const range = n.max - n.min || 1;
    for (let i = 0; i < buf.length; i++) buf[i] = (buf[i] - n.min) / range;
    return buf;
  }
  return buf;
}

// Apply postprocess ops in order. `data` is (C, Dz, Dy, Dx) flattened C-major.
// Returns possibly a new buffer + new channel count (channel-select reduces C).
export function applyPostprocess(
  data: Float32Array,
  channels: number,
  spatial: number,
  ops: PostSpec[],
): { data: Float32Array; channels: number } {
  let cur = data;
  let curC = channels;
  for (const op of ops) {
    if (op.type === "clip") {
      for (let i = 0; i < cur.length; i++) {
        const v = cur[i];
        cur[i] = v < op.min ? op.min : v > op.max ? op.max : v;
      }
    } else if (op.type === "scale") {
      for (let i = 0; i < cur.length; i++) cur[i] *= op.factor;
    } else if (op.type === "offset") {
      for (let i = 0; i < cur.length; i++) cur[i] += op.value;
    } else if (op.type === "threshold") {
      const below = op.below ?? 0;
      const above = op.above ?? 1;
      for (let i = 0; i < cur.length; i++) cur[i] = cur[i] > op.value ? above : below;
    } else if (op.type === "channel") {
      if (op.index < 0 || op.index >= curC) {
        throw new Error(`channel ${op.index} out of range [0, ${curC})`);
      }
      const out = new Float32Array(spatial);
      out.set(cur.subarray(op.index * spatial, (op.index + 1) * spatial));
      cur = out;
      curC = 1;
    }
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
