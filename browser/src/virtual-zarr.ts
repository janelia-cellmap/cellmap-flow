import { openArray, readSubvolume3D, toFloat32 } from "./zarr-client";
import type { ZarrArray } from "./zarr-client";
import { createSession, runModel } from "./onnx-session";
import type * as ort from "onnxruntime-web/webgpu";
import {
  type ModelSpec,
  type Triple,
  applyNormalize,
  applyPostprocess,
  context,
  encodeChunk,
  reshapeOutputToCDHW,
  tensorDims,
  zarrDtype,
} from "./model-spec";

export interface VzConfig {
  zarrUrl: string;
  modelUrl: string;
  spec: ModelSpec;
}

export interface VzResponse {
  status: number;
  headers?: Record<string, string>;
  body?: ArrayBuffer | string | null;
}

interface ActiveState {
  cfg: VzConfig;
  raw: ZarrArray;
  session: ort.InferenceSession;
  // Volume shape in OUTPUT VOXELS (z, y, x).
  outShape: Triple;
}

let state: ActiveState | null = null;

export async function activate(cfg: VzConfig): Promise<ActiveState> {
  const raw = await openArray(cfg.zarrUrl);
  if (raw.shape.length < 3) throw new Error(`need ndim >= 3, got ${raw.shape.length}`);

  // Volume in INPUT voxels = last 3 dims of source zarr.
  const zA = raw.shape.length - 3;
  const inVoxels: Triple = [raw.shape[zA], raw.shape[zA + 1], raw.shape[zA + 2]];

  // Convert to OUTPUT voxels via voxel-size ratio.
  const ivs = cfg.spec.inputVoxelSize;
  const ovs = cfg.spec.outputVoxelSize;
  const outShape: Triple = [
    Math.floor((inVoxels[0] * ivs[0]) / ovs[0]),
    Math.floor((inVoxels[1] * ivs[1]) / ovs[1]),
    Math.floor((inVoxels[2] * ivs[2]) / ovs[2]),
  ];

  const { session } = await createSession(cfg.modelUrl);
  state = { cfg, raw, session, outShape };
  return state;
}

export async function handleRequest(path: string): Promise<VzResponse> {
  if (!state) {
    return text(503, "virtual zarr not activated");
  }
  const { cfg: { spec }, outShape } = state;

  // OME-Zarr v0.4 layout: /vz/ is the group, /vz/0/ is the array.
  if (path === "" || path === ".zgroup") return jsonResp({ zarr_format: 2 });
  if (path === ".zattrs") {
    return jsonResp({
      multiscales: [
        {
          version: "0.4",
          name: "inference",
          axes: [
            { name: "z", type: "space", unit: "nanometer" },
            { name: "y", type: "space", unit: "nanometer" },
            { name: "x", type: "space", unit: "nanometer" },
          ],
          datasets: [
            {
              path: "0",
              coordinateTransformations: [
                { type: "scale", scale: [spec.outputVoxelSize[0], spec.outputVoxelSize[1], spec.outputVoxelSize[2]] },
              ],
            },
          ],
        },
      ],
    });
  }
  if (path === "0/.zgroup") return text(404, "not a group");
  if (path === "0/.zattrs") return jsonResp({});
  if (path === "0/.zarray") {
    const shape = spec.outputChannels > 1
      ? [...outShape, spec.outputChannels]
      : [...outShape];
    const chunks = spec.outputChannels > 1
      ? [...spec.blockShape, spec.outputChannels]
      : [...spec.blockShape];
    return jsonResp({
      zarr_format: 2,
      shape,
      chunks,
      dtype: zarrDtype(spec.outputDtype),
      compressor: null,
      fill_value: 0,
      filters: null,
      order: "C",
      dimension_separator: ".",
    });
  }

  let chunkPath = path.startsWith("0/") ? path.slice(2) : path;
  const coords = parseChunkPath(chunkPath, spec.outputChannels > 1 ? 4 : 3);
  if (!coords) return text(404, `not found: ${path}`);

  // First three coords are spatial; if 4D path, last must be 0 (single chunk in C).
  if (coords.length === 4 && coords[3] !== 0) return text(404, "channel chunk index must be 0");

  const bytes = await computeChunk([coords[0], coords[1], coords[2]]);
  return {
    status: 200,
    headers: {
      "content-type": "application/octet-stream",
      "cache-control": "no-store",
    },
    body: bytes,
  };
}

function jsonResp(obj: unknown): VzResponse {
  return { status: 200, headers: { "content-type": "application/json" }, body: JSON.stringify(obj) };
}
function text(status: number, body: string): VzResponse {
  return { status, headers: { "content-type": "text/plain" }, body };
}

function parseChunkPath(path: string, expected: number): number[] | null {
  const parts = path.split(".");
  if (parts.length !== expected) return null;
  const nums = parts.map((s) => Number(s));
  if (nums.some((n) => !Number.isInteger(n) || n < 0)) return null;
  return nums;
}

async function computeChunk(coords: Triple): Promise<ArrayBuffer> {
  if (!state) throw new Error("not activated");
  const { cfg: { spec }, raw, session, outShape } = state;
  const [cz, cy, cx] = coords;
  const [bz, by, bx] = spec.blockShape;
  const ovs = spec.outputVoxelSize;
  const ivs = spec.inputVoxelSize;

  // 1) Write ROI in OUTPUT VOXELS, clipped to volume.
  const writeOutZ: [number, number] = [cz * bz, Math.min((cz + 1) * bz, outShape[0])];
  const writeOutY: [number, number] = [cy * by, Math.min((cy + 1) * by, outShape[1])];
  const writeOutX: [number, number] = [cx * bx, Math.min((cx + 1) * bx, outShape[2])];
  if (writeOutZ[0] >= outShape[0] || writeOutY[0] >= outShape[1] || writeOutX[0] >= outShape[2]) {
    return zeroChunk(spec);
  }

  // 2) Convert to WORLD UNITS.
  const writeWorldZ: [number, number] = [writeOutZ[0] * ovs[0], writeOutZ[1] * ovs[0]];
  const writeWorldY: [number, number] = [writeOutY[0] * ovs[1], writeOutY[1] * ovs[1]];
  const writeWorldX: [number, number] = [writeOutX[0] * ovs[2], writeOutX[1] * ovs[2]];

  // 3) Expand by context to get READ ROI in WORLD UNITS.
  const ctx = context(spec);
  const readWorldZ: [number, number] = [writeWorldZ[0] - ctx[0], writeWorldZ[1] + ctx[0]];
  const readWorldY: [number, number] = [writeWorldY[0] - ctx[1], writeWorldY[1] + ctx[1]];
  const readWorldX: [number, number] = [writeWorldX[0] - ctx[2], writeWorldX[1] + ctx[2]];

  // 4) Convert read ROI to INPUT VOXELS (assume integer divisibility).
  const readInZ: [number, number] = [readWorldZ[0] / ivs[0], readWorldZ[1] / ivs[0]];
  const readInY: [number, number] = [readWorldY[0] / ivs[1], readWorldY[1] / ivs[1]];
  const readInX: [number, number] = [readWorldX[0] / ivs[2], readWorldX[1] / ivs[2]];

  // 5) Read raw with zero-padding for parts that fall outside the source.
  const rawShape = raw.shape;
  const inAxis = rawShape.length - 3;
  const inExtent: Triple = [rawShape[inAxis], rawShape[inAxis + 1], rawShape[inAxis + 2]];
  const inputBuf = await readWithZeroPad(raw, [readInZ, readInY, readInX], inExtent);
  const expectedReadVox: Triple = [
    Math.round((readWorldZ[1] - readWorldZ[0]) / ivs[0]),
    Math.round((readWorldY[1] - readWorldY[0]) / ivs[1]),
    Math.round((readWorldX[1] - readWorldX[0]) / ivs[2]),
  ];

  // 6) Normalize.
  applyNormalize(inputBuf, spec.normalize);

  // 7) Run the model. Cin is 1 for now (multi-channel input is phase 2).
  const dims = tensorDims(spec.tensorLayout, 1, expectedReadVox[0], expectedReadVox[1], expectedReadVox[2]);
  const out = await runModel(session, inputBuf, dims);

  // 8) Reshape ORT output to (Cout, Dz_out, Dy_out, Dx_out).
  const expectedWriteVox: Triple = [
    Math.round((writeWorldZ[1] - writeWorldZ[0]) / ovs[0]),
    Math.round((writeWorldY[1] - writeWorldY[0]) / ovs[1]),
    Math.round((writeWorldX[1] - writeWorldX[0]) / ovs[2]),
  ];
  const cdhw = reshapeOutputToCDHW(
    out.data,
    out.dims,
    spec.tensorLayout,
    spec.outputChannels,
    expectedWriteVox[0],
    expectedWriteVox[1],
    expectedWriteVox[2],
  );

  // 9) Postprocess (ops mutate cdhw in place except channel-select).
  const spatial = expectedWriteVox[0] * expectedWriteVox[1] * expectedWriteVox[2];
  const post = applyPostprocess(cdhw, spec.outputChannels, spatial, spec.postprocess ?? []);

  // 10) Pad to full block shape (when at volume edge, output is smaller).
  const padded = padToBlock(post.data, post.channels, expectedWriteVox, spec.blockShape);

  // 11) Encode bytes for the chunk dtype.
  return encodeChunk(padded, post.channels, spec.blockShape, spec.outputDtype);
}

function zeroChunk(spec: ModelSpec): ArrayBuffer {
  const bs = spec.blockShape[0] * spec.blockShape[1] * spec.blockShape[2] * spec.outputChannels;
  switch (spec.outputDtype) {
    case "uint8": return new Uint8Array(bs).buffer;
    case "uint16": return new Uint16Array(bs).buffer;
    case "int16": return new Int16Array(bs).buffer;
    case "float32": return new Float32Array(bs).buffer;
  }
}

async function readWithZeroPad(
  arr: ZarrArray,
  rois: [[number, number], [number, number], [number, number]],
  extent: Triple,
): Promise<Float32Array> {
  const [z0, z1] = rois[0];
  const [y0, y1] = rois[1];
  const [x0, x1] = rois[2];
  const Dz = z1 - z0;
  const Dy = y1 - y0;
  const Dx = x1 - x0;
  const out = new Float32Array(Dz * Dy * Dx);

  const cz0 = Math.max(0, z0);
  const cy0 = Math.max(0, y0);
  const cx0 = Math.max(0, x0);
  const cz1 = Math.min(extent[0], z1);
  const cy1 = Math.min(extent[1], y1);
  const cx1 = Math.min(extent[2], x1);

  if (cz0 >= cz1 || cy0 >= cy1 || cx0 >= cx1) return out; // entire ROI outside

  const inner = await readSubvolume3D(arr, [cz0, cz1], [cy0, cy1], [cx0, cx1]);
  const innerF = toFloat32(inner.data);

  const offZ = cz0 - z0;
  const offY = cy0 - y0;
  const offX = cx0 - x0;
  const innerDy = inner.height;
  const innerDx = inner.width;
  for (let z = 0; z < inner.depth; z++) {
    for (let y = 0; y < innerDy; y++) {
      const src = (z * innerDy + y) * innerDx;
      const dst = ((z + offZ) * Dy + (y + offY)) * Dx + offX;
      out.set(innerF.subarray(src, src + innerDx), dst);
    }
  }
  return out;
}

function padToBlock(
  data: Float32Array,
  channels: number,
  actual: Triple,
  block: Triple,
): Float32Array {
  const [adz, ady, adx] = actual;
  const [bdz, bdy, bdx] = block;
  if (adz === bdz && ady === bdy && adx === bdx) return data;
  const out = new Float32Array(channels * bdz * bdy * bdx);
  for (let c = 0; c < channels; c++) {
    for (let z = 0; z < adz; z++) {
      for (let y = 0; y < ady; y++) {
        const src = ((c * adz + z) * ady + y) * adx;
        const dst = ((c * bdz + z) * bdy + y) * bdx;
        out.set(data.subarray(src, src + adx), dst);
      }
    }
  }
  return out;
}
