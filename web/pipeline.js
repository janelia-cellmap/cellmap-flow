// pipeline.js — dependency-free JS port of the CellMap Flow inference pipeline.
//
// Mirrors the Python "virtual Zarr" server so the service worker can emit
// byte-compatible responses to Neuroglancer:
//   - cellmap_flow/server.py            (.zattrs / .zarray / chunk layout)
//   - cellmap_flow/inferencer.py        (read ROI -> normalize -> model -> postprocess)
//   - cellmap_flow/norm/input_normalize.py
//   - cellmap_flow/post/postprocessors.py
//
// All ROIs are expressed in physical nanometres, exactly like funlib.geometry.Roi
// in the Python code. Axis order throughout is (z, y, x) [+ channel].
//
// v1 scope: fp32, uncompressed chunks, simple normalizer/postprocessor chain.

// ---------------------------------------------------------------------------
// ndarray helper: a flat C-order typed array plus its shape.
// ---------------------------------------------------------------------------
export function prod(arr) {
  return arr.reduce((a, b) => a * b, 1);
}

// ---------------------------------------------------------------------------
// Geometry — port of server.py _chunk_impl / inferencer context growing.
// meta carries the model geometry (all lengths-of-3 are [z, y, x]):
//   input_size, output_size            (voxels)
//   input_voxel_size, output_voxel_size (nm)
//   output_channels, has_channel
// ---------------------------------------------------------------------------
export function context(meta) {
  // context (nm) = (read_shape - write_shape) / 2   [inferencer.py:64]
  const read = meta.input_size.map((s, i) => s * meta.input_voxel_size[i]);
  const write = meta.output_size.map((s, i) => s * meta.output_voxel_size[i]);
  return read.map((r, i) => (r - write[i]) / 2);
}

// Returns the input ROI (nm) to read for a given output chunk index.
// server.py:244-246  +  inferencer.py:110 (output_roi.grow(context, context))
export function chunkInputRoi(meta, cz, cy, cx) {
  const block = meta.output_size; // block_shape[:3] in voxels
  const ovs = meta.output_voxel_size;
  const cidx = [cz, cy, cx];
  const ctx = context(meta);

  // output ROI (nm)
  const outBegin = block.map((b, i) => b * cidx[i] * ovs[i]);
  const outSize = block.map((b, i) => b * ovs[i]);

  // grow by context on both sides -> input ROI (nm)
  const begin = outBegin.map((v, i) => v - ctx[i]);
  const size = outSize.map((v, i) => v + 2 * ctx[i]);
  return { begin, size };
}

// ---------------------------------------------------------------------------
// Zarr metadata — port of server.py _top_level_attributes_impl / _attributes_impl.
// v1 difference: compressor is null (uncompressed) so we ship no Blosc codec.
// ---------------------------------------------------------------------------
const DTYPE_MAP = {
  uint8: "|u1",
  uint16: "<u2",
  uint32: "<u4",
  uint64: "<u8",
  int8: "<i1",
  int16: "<i2",
  int32: "<i4",
  int64: "<i8",
  float32: "<f4",
  float64: "<f8",
};

// Full output volume shape (voxels): input_array_shape * ivs / ovs  [server.py:77-86]
export function volumeShape(meta, inputArrayShape) {
  const spatial = inputArrayShape.map((s, i) =>
    Math.floor((s * meta.input_voxel_size[i]) / meta.output_voxel_size[i])
  );
  return meta.has_channel ? [...spatial, outputChannels(meta)] : spatial;
}

export function blockShape(meta) {
  return meta.has_channel
    ? [...meta.output_size, outputChannels(meta)]
    : [...meta.output_size];
}

// Channel count can be narrowed by a ChannelSelection postprocessor.
export function outputChannels(meta) {
  for (const p of meta.postprocess || []) {
    if (p.name === "ChannelSelection") {
      return String(p.channels ?? "0").split(",").length;
    }
  }
  return meta.output_channels;
}

export function buildZattrs(meta, dataset) {
  const ovs = meta.output_voxel_size;
  const axes = ["z", "y", "x"].map((name) => ({
    name,
    type: "space",
    unit: "nanometer",
  }));
  const scaleValues = [...ovs];
  const translationValues = [0.0, 0.0, 0.0];
  if (meta.has_channel) {
    axes.push({ name: "c", type: "channel" });
    scaleValues.push(1.0);
    translationValues.push(0.0);
  }
  const topScale = meta.has_channel ? [1.0, 1.0, 1.0, 1.0] : [1.0, 1.0, 1.0];
  return {
    multiscales: [
      {
        version: "0.4",
        name: dataset,
        axes,
        datasets: [
          {
            coordinateTransformations: [
              { type: "scale", scale: scaleValues },
              { type: "translation", translation: translationValues },
            ],
            path: "s0",
          },
        ],
        coordinateTransformations: [{ type: "scale", scale: topScale }],
      },
    ],
  };
}

export function buildZarray(meta, inputArrayShape) {
  const dtype = outputDtype(meta);
  return {
    chunks: blockShape(meta),
    compressor: null,
    dtype: DTYPE_MAP[dtype] || dtype,
    fill_value: 0,
    filters: null,
    order: "C",
    shape: volumeShape(meta, inputArrayShape),
    zarr_format: 2,
  };
}

// ---------------------------------------------------------------------------
// Normalizers — port of cellmap_flow/norm/input_normalize.py
// Each takes a Float32Array (in place) and returns it.
// ---------------------------------------------------------------------------
function makeNormalizer(cfg) {
  switch (cfg.name) {
    case "MinMaxNormalizer": {
      const min = Number(cfg.min_value ?? 0.0);
      const max = Number(cfg.max_value ?? 255.0);
      const invert = cfg.invert === true || cfg.invert === "true";
      const range = max - min;
      return (data) => {
        for (let i = 0; i < data.length; i++) {
          let v = Math.min(Math.max(data[i], min), max);
          v = (v - min) / range;
          data[i] = invert ? 1 - v : v;
        }
        return data;
      };
    }
    case "ZScoreNormalizer": {
      const mean = Number(cfg.mean ?? 0.0);
      const std = Number(cfg.std ?? 1.0);
      return (data) => {
        for (let i = 0; i < data.length; i++) data[i] = (data[i] - mean) / std;
        return data;
      };
    }
    case "LambdaNormalizer": {
      // Scalar arithmetic only (e.g. "x*2-1"). numpy-style ops are unsupported.
      const fn = new Function("x", `return (${cfg.expression});`);
      return (data) => {
        for (let i = 0; i < data.length; i++) data[i] = fn(data[i]);
        return data;
      };
    }
    default:
      console.warn(`[pipeline] unsupported normalizer ${cfg.name}, skipping`);
      return (data) => data;
  }
}

export function applyNormalizers(float32, normCfgs) {
  let data = float32;
  for (const cfg of normCfgs || []) data = makeNormalizer(cfg)(data);
  return data;
}

// ---------------------------------------------------------------------------
// Postprocessors — port of cellmap_flow/post/postprocessors.py
// Operate on a channel-first ndarray {data, shape:[C, z, y, x]} (model output).
// ---------------------------------------------------------------------------
function makePostprocessor(cfg) {
  switch (cfg.name) {
    case "DefaultPostprocessor": {
      const clipMin = Number(cfg.clip_min ?? -1.0);
      const clipMax = Number(cfg.clip_max ?? 1.0);
      const bias = Number(cfg.bias ?? 1.0);
      const mult = Number(cfg.multiplier ?? 127.5);
      return ({ data, shape }) => {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) {
          let v = Math.min(Math.max(data[i], clipMin), clipMax);
          out[i] = (v + bias) * mult; // truncates toward zero, like np.astype(uint8)
        }
        return { data: out, shape };
      };
    }
    case "ThresholdPostprocessor": {
      const thr = Number(cfg.threshold ?? 0.5);
      return ({ data, shape }) => {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = data[i] > thr ? 1 : 0;
        return { data: out, shape };
      };
    }
    case "ChannelSelection": {
      const channels = String(cfg.channels ?? "0")
        .split(",")
        .map((c) => parseInt(c, 10));
      return ({ data, shape }) => {
        const [, z, y, x] = [shape[0], shape[1], shape[2], shape[3]];
        const planeStride = shape[1] * shape[2] * shape[3]; // per-channel voxels
        const out = new data.constructor(channels.length * planeStride);
        channels.forEach((c, j) => {
          out.set(data.subarray(c * planeStride, (c + 1) * planeStride), j * planeStride);
        });
        return { data: out, shape: [channels.length, z, y, x] };
      };
    }
    case "LambdaPostprocessor": {
      const fn = new Function("x", `return (${cfg.expression});`);
      return ({ data, shape }) => {
        const out = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = fn(data[i]);
        return { data: out, shape };
      };
    }
    default:
      console.warn(`[pipeline] unsupported postprocessor ${cfg.name}, skipping`);
      return (nd) => nd;
  }
}

export function applyPostprocessors(nd, postCfgs) {
  let cur = nd;
  for (const cfg of postCfgs || []) cur = makePostprocessor(cfg)(cur);
  return cur;
}

const POST_DTYPE = {
  DefaultPostprocessor: "uint8",
  ThresholdPostprocessor: "uint8",
  LambdaPostprocessor: "float32",
};

export function outputDtype(meta) {
  // Last postprocessor that declares a dtype wins; otherwise model output (float32).
  let dtype = "float32";
  for (const p of meta.postprocess || []) {
    if (POST_DTYPE[p.name]) dtype = POST_DTYPE[p.name];
  }
  return dtype;
}

// ---------------------------------------------------------------------------
// Axis reorder + encode — port of server.py _reorder_to_zarr_axes / encode.
// Model output is channel-first [C, z, y, x]; Zarr expects [z, y, x, c].
// ---------------------------------------------------------------------------
export function reorderToZarr(nd) {
  const [c, z, y, x] = nd.shape;
  if (c === 1) return { data: nd.data, shape: [z, y, x, 1] }; // byte layout identical
  const out = new nd.data.constructor(nd.data.length);
  const src = nd.data;
  let o = 0;
  for (let iz = 0; iz < z; iz++)
    for (let iy = 0; iy < y; iy++)
      for (let ix = 0; ix < x; ix++)
        for (let ic = 0; ic < c; ic++)
          out[o++] = src[((ic * z + iz) * y + iy) * x + ix];
  return { data: out, shape: [z, y, x, c] };
}

const TYPED = {
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  float32: Float32Array,
  float64: Float64Array,
};

// Cast an ndarray's data to the target dtype and return a raw little-endian buffer.
export function encodeChunk(nd, dtype) {
  const Ctor = TYPED[dtype] || Float32Array;
  let typed;
  if (nd.data instanceof Ctor) {
    typed = nd.data;
  } else {
    typed = new Ctor(nd.data.length);
    for (let i = 0; i < nd.data.length; i++) typed[i] = nd.data[i];
  }
  // TypedArrays are already little-endian on all supported (x86/ARM) platforms.
  return typed.buffer.slice(typed.byteOffset, typed.byteOffset + typed.byteLength);
}
