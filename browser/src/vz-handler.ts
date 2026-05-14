// Virtual-zarr handler. The service worker forwards same-origin /vz/<path>
// requests here; we synthesize zarr v2 metadata + serve inference chunks on
// demand. One active config per tab.
//
// Simplifications vs the old virtual-zarr.ts (commit b68225f):
//   - Output is single-channel float32 (channel 0 of the 2-channel BMZ
//     sigmoid), matched to source spatial shape, same voxel size as input.
//   - Chunks are model-sized (e.g. 512×512 for hiding-blowfish), no halo —
//     boundary artifacts accepted as known limitation in v1.
//   - 2D model only; treats source's trailing two dims as (Y, X) and runs
//     inference per Z-slice for higher-rank source volumes.

import * as zarr from "zarrita";

import { loadBmzModel, minMax, runOnTensor } from "./bmz-inference";
import type { LoadedBmz } from "./bmz-inference";
import {
  makeNormalizer,
  makePostprocessor,
  type NormSpec,
  type PostSpec,
  type PostContext,
} from "./cellmap-pipeline";

export interface VzResponse {
  status: number;
  headers?: Record<string, string>;
  body?: ArrayBuffer | string | null;
}

export interface VzConfig {
  modelId: string;
  zarrUrl: string;
  // Optional dashboard-supplied pipeline. Normalizers run on raw input
  // BEFORE the model's RDF-declared scale_range/etc. Postprocessors run
  // on the model output (multichannel) and can reduce channels.
  normalizers?: NormSpec[];
  postprocessors?: PostSpec[];
}

interface SourceSpatial {
  // Per-axis voxel size in source's native unit. Length = 3 (z, y, x).
  scale: [number, number, number];
  // Length = 3, e.g. "nanometer". Default "nanometer" if source didn't declare.
  unit: string;
}

interface ActiveState {
  cfg: VzConfig;
  bmz: LoadedBmz;
  raw: zarr.Array<zarr.DataType, zarr.FetchStore>;
  // Source spatial shape (Z, Y, X) — last 3 dims of raw, padded with 1 for 2D.
  spatialShape: [number, number, number];
  // Model input shape (Hm, Wm) — chunk size in the virtual array.
  modelHW: [number, number];
  // Inherited from source OME-NGFF (or defaulted) so the vz layer overlays
  // pixel-perfect on the source layer in Neuroglancer.
  spatial: SourceSpatial;
  // Compiled instances of the dashboard-checked pipeline ops.
  normalizers: Array<ReturnType<typeof makeNormalizer>>;
  postprocessors: Array<ReturnType<typeof makePostprocessor>>;
  // Channels in the output zarr after postprocessors run. Defaults to 1
  // (we pick channel 0 of the model output); ChannelSelection may change it.
  outputChannels: number;
}

let state: ActiveState | null = null;
const chunkCache = new Map<string, ArrayBuffer>();
const MAX_CACHED_CHUNKS = 256;

// Cap concurrent model forwards. WebGPU/WASM both serialize internally,
// but unbounded queueing of Promise callbacks freezes the main thread when
// NG fans out hundreds of tile requests at once (esp. orthogonal views of
// a 2D-per-slice model). 2 lets the GPU pipeline; higher just queues up.
const MAX_CONCURRENT_INFERENCES = 2;
// Also cap the number of pending-but-not-yet-started chunks. When NG zooms
// out the screen-wide tile count balloons (this synthesized vz only has
// one mip level); beyond the cap we reject with 503 and let NG retry when
// existing work clears. Without this the wait queue grows unbounded and
// the main thread starves.
const MAX_PENDING_INFERENCES = 16;
let inflight = 0;
const waiters: Array<() => void> = [];

class QueueFullError extends Error {
  constructor() { super("vz: inference queue full"); }
}

async function acquireSlot(): Promise<() => void> {
  if (inflight >= MAX_CONCURRENT_INFERENCES) {
    if (waiters.length >= MAX_PENDING_INFERENCES) {
      throw new QueueFullError();
    }
    // Wait. When our resolver fires, the slot has been pre-credited to us
    // by the releaser, so we don't increment again here.
    await new Promise<void>((resolve) => waiters.push(resolve));
  } else {
    inflight++;
  }
  return () => {
    const next = waiters.shift();
    if (next) {
      // Hand the slot to the next waiter; inflight stays the same.
      next();
    } else {
      inflight--;
    }
  };
}

export interface VzActivation {
  origin: string;
  vzUrl: string;
  // URL the user should hand NG for the *source* layer. If the input was
  // (or had) an OME-NGFF group, this points at the group so NG picks up
  // proper z/y/x dim labels and the source's own multiscale mip selection.
  // Otherwise just the array URL.
  sourceLayerUrl: string;
  // Scale level chosen for inference (e.g. "s2") and its voxel size, both
  // for display in the status box.
  chosenLevel: string | null;
  chosenVoxelSizeNm: number | null;
  // World-space center + half-extent (in source units) for fitting NG to data.
  centerWorld: [number, number, number];
  halfExtentWorld: [number, number, number];
  unit: string;
}

interface MultiscaleResolution {
  groupBase: string; // absolute URL of the group (no trailing slash)
  arrayBase: string; // absolute URL of the chosen array level
  levelPath: string; // e.g. "s2"
  scale: [number, number, number];
  unit: string;
  // All available levels, for telemetry.
  levels: Array<{ path: string; scale: [number, number, number] }>;
}

export async function activateVz(cfg: VzConfig): Promise<VzActivation> {
  const bmz = await loadBmzModel(cfg.modelId);
  if (bmz.manifest.shape_in.length !== 4) {
    throw new Error(
      `vz: model ${cfg.modelId} has shape_in=${bmz.manifest.shape_in}; ` +
        `only 4D (B, C, Y, X) models are supported in v1`,
    );
  }
  const Hm = bmz.manifest.shape_in[2];
  const Wm = bmz.manifest.shape_in[3];

  const u = new URL(cfg.zarrUrl);
  const userBase = `${u.origin}${u.pathname.replace(/\/$/, "")}`;

  // Resolve the user input into (group_url, array_url) — handles three cases:
  //   1) user pasted a group URL with multiscales → pick the level whose
  //      voxel size matches the model's preferred_voxel_size_nm.
  //   2) user pasted an array URL whose parent is a multiscale group →
  //      use that array, expose the parent group to NG so dim labels show.
  //   3) user pasted a plain array URL → use it directly, no group.
  const preferredNm = bmz.manifest.preferred_voxel_size_nm ?? 8;
  const resolved = await resolveMultiscale(userBase, preferredNm);

  const arrayBase = resolved ? resolved.arrayBase : userBase;
  const groupBase = resolved ? resolved.groupBase : null;
  const store = new zarr.FetchStore(arrayBase);
  const raw = (await zarr.open(store, { kind: "array" })) as ActiveState["raw"];

  if (raw.shape.length < 2) {
    throw new Error(`vz: source array has ndim=${raw.shape.length}, expected >= 2`);
  }
  const Y = raw.shape[raw.shape.length - 2];
  const X = raw.shape[raw.shape.length - 1];
  const Z = raw.shape.length >= 3 ? raw.shape[raw.shape.length - 3] : 1;
  const spatialShape: [number, number, number] = [Z, Y, X];

  const spatial: SourceSpatial = resolved
    ? { scale: resolved.scale, unit: resolved.unit }
    : await probeSourceSpatial(arrayBase);

  // Compile dashboard pipeline ops. Channel selection (if present) reduces
  // the number of output channels we stream; otherwise we stream channel 0.
  const normalizers = (cfg.normalizers ?? []).map(makeNormalizer);
  const postprocessors = (cfg.postprocessors ?? []).map(makePostprocessor);
  // The vz output is single-channel float32 unless a ChannelSelection in
  // the pipeline picks more — derive from the postprocessor chain.
  let outputChannels = 1;
  const modelOutChannels = bmz.manifest.shape_out[1];
  let runningChannels = modelOutChannels;
  for (const p of postprocessors) {
    runningChannels = p.numChannels(runningChannels);
  }
  if (postprocessors.length > 0) outputChannels = runningChannels;

  state = {
    cfg, bmz, raw, spatialShape, modelHW: [Hm, Wm], spatial,
    normalizers, postprocessors, outputChannels,
  };
  chunkCache.clear();
  console.log(
    `[vz] activated: model=${cfg.modelId} (${bmz.provider}) src=${arrayBase} ` +
      `shape=${raw.shape} spatial=${spatialShape.join("×")} chunk=${Hm}×${Wm} ` +
      `scale=${spatial.scale.join(",")}${spatial.unit}` +
      (resolved ? ` (picked ${resolved.levelPath} from ${resolved.levels.length}-level group)` : "") +
      (normalizers.length ? ` +norm[${normalizers.length}]` : "") +
      (postprocessors.length ? ` +post[${postprocessors.length}] outC=${outputChannels}` : ""),
  );

  const center: [number, number, number] = [
    (spatialShape[0] / 2) * spatial.scale[0],
    (spatialShape[1] / 2) * spatial.scale[1],
    (spatialShape[2] / 2) * spatial.scale[2],
  ];
  const halfExtent: [number, number, number] = [
    (spatialShape[0] / 2) * spatial.scale[0],
    (spatialShape[1] / 2) * spatial.scale[1],
    (spatialShape[2] / 2) * spatial.scale[2],
  ];

  return {
    origin: location.origin,
    vzUrl: `zarr://${location.origin}/vz/`,
    sourceLayerUrl: `zarr://${groupBase ?? arrayBase}/`,
    chosenLevel: resolved ? resolved.levelPath : null,
    chosenVoxelSizeNm: resolved
      ? resolved.scale[resolved.scale.length - 1]
      : null,
    centerWorld: center,
    halfExtentWorld: halfExtent,
    unit: spatial.unit,
  };
}

// Resolve a user-pasted URL to a (group, array) pair, picking the level
// whose trailing (XY) voxel size is closest to the model's preferred.
// Returns null if neither URL nor parent is a multiscale group; caller
// then treats userBase as a plain array.
async function resolveMultiscale(
  userBase: string,
  preferredVoxelNm: number,
): Promise<MultiscaleResolution | null> {
  const asGroup = await readMultiscaleAttrs(userBase);
  if (asGroup) {
    return pickLevel(userBase, asGroup, preferredVoxelNm);
  }
  // Try parent in case user pasted a specific array level.
  const slash = userBase.lastIndexOf("/");
  if (slash < 0) return null;
  const parent = userBase.slice(0, slash);
  const arrayName = userBase.slice(slash + 1);
  const parentAttrs = await readMultiscaleAttrs(parent);
  if (!parentAttrs) return null;
  // Use the level the user actually pointed at, even though we know about
  // the parent group now. That way we respect the user's choice.
  const found = parentAttrs.datasets.find((d) => d.path === arrayName);
  if (!found) {
    // User pointed at something inside the group but not a known dataset;
    // fall back to picking by voxel size.
    return pickLevel(parent, parentAttrs, preferredVoxelNm);
  }
  return {
    groupBase: parent,
    arrayBase: userBase,
    levelPath: arrayName,
    scale: trailingScale(found.scale, parentAttrs.axes),
    unit: firstSpatialUnit(parentAttrs.axes),
    levels: parentAttrs.datasets.map((d) => ({
      path: d.path,
      scale: trailingScale(d.scale, parentAttrs.axes),
    })),
  };
}

interface MultiscaleAttrs {
  axes: Array<{ name?: string; type?: string; unit?: string }>;
  datasets: Array<{ path: string; scale: number[] }>;
}

async function readMultiscaleAttrs(base: string): Promise<MultiscaleAttrs | null> {
  try {
    const r = await fetch(`${base}/.zattrs`);
    if (!r.ok) return null;
    const attrs = (await r.json()) as {
      multiscales?: Array<{
        axes?: Array<{ name?: string; type?: string; unit?: string }>;
        datasets?: Array<{
          path?: string;
          coordinateTransformations?: Array<{ type: string; scale?: number[] }>;
        }>;
      }>;
    };
    const ms = attrs.multiscales?.[0];
    if (!ms || !ms.datasets || ms.datasets.length === 0) return null;
    const axes = ms.axes ?? [];
    const datasets: Array<{ path: string; scale: number[] }> = [];
    for (const d of ms.datasets) {
      if (!d.path) continue;
      const tx = d.coordinateTransformations?.find((t) => t.type === "scale" && t.scale);
      if (!tx || !tx.scale) continue;
      datasets.push({ path: d.path, scale: tx.scale });
    }
    if (datasets.length === 0) return null;
    return { axes, datasets };
  } catch {
    return null;
  }
}

function pickLevel(
  groupBase: string,
  attrs: MultiscaleAttrs,
  preferredVoxelNm: number,
): MultiscaleResolution {
  // Compare each level by its trailing (XY) voxel size — normalize to nm.
  const unit = firstSpatialUnit(attrs.axes);
  const toNm = unitToNm(unit);
  let best = attrs.datasets[0];
  let bestDist = Infinity;
  for (const d of attrs.datasets) {
    const trailing = d.scale[d.scale.length - 1];
    const dist = Math.abs(trailing * toNm - preferredVoxelNm);
    if (dist < bestDist) {
      best = d;
      bestDist = dist;
    }
  }
  return {
    groupBase,
    arrayBase: `${groupBase}/${best.path}`,
    levelPath: best.path,
    scale: trailingScale(best.scale, attrs.axes),
    unit,
    levels: attrs.datasets.map((d) => ({
      path: d.path,
      scale: trailingScale(d.scale, attrs.axes),
    })),
  };
}

function trailingScale(
  scale: number[],
  axes: Array<{ type?: string }>,
): [number, number, number] {
  // Prefer the trailing 3 space axes; fall back to plain trailing 3.
  if (axes.length === scale.length) {
    const spaceIdx: number[] = [];
    axes.forEach((a, i) => { if (a.type === "space") spaceIdx.push(i); });
    if (spaceIdx.length >= 3) {
      const last3 = spaceIdx.slice(-3);
      return [scale[last3[0]], scale[last3[1]], scale[last3[2]]];
    }
  }
  const last3 = scale.slice(-3);
  while (last3.length < 3) last3.unshift(1);
  return [last3[0], last3[1], last3[2]];
}

function firstSpatialUnit(axes: Array<{ type?: string; unit?: string }>): string {
  const u = axes.find((a) => a.type === "space" && a.unit)?.unit;
  return u && u.length > 0 ? u : "nanometer";
}

function unitToNm(unit: string): number {
  switch (unit) {
    case "nanometer": return 1;
    case "micrometer": return 1000;
    case "millimeter": return 1e6;
    case "meter": return 1e9;
    case "angstrom": return 0.1;
    case "picometer": return 0.001;
    default: return 1;
  }
}

// Try to read OME-NGFF v0.4 multiscales metadata from the source's parent
// group (e.g. .../fibsem-uint8/.zattrs when user opened .../fibsem-uint8/s1).
// Returns the matched dataset's scale + axis units. Falls back to scale=1,
// unit="nanometer" so the vz layer at least uses a real unit.
async function probeSourceSpatial(arrayBase: string): Promise<SourceSpatial> {
  const fallback: SourceSpatial = { scale: [1, 1, 1], unit: "nanometer" };
  const slash = arrayBase.lastIndexOf("/");
  if (slash < 0) return fallback;
  const parent = arrayBase.slice(0, slash);
  const arrayName = arrayBase.slice(slash + 1);
  try {
    const r = await fetch(`${parent}/.zattrs`);
    if (!r.ok) return fallback;
    const attrs = (await r.json()) as {
      multiscales?: Array<{
        axes?: Array<{ name?: string; type?: string; unit?: string }>;
        datasets?: Array<{
          path?: string;
          coordinateTransformations?: Array<{ type: string; scale?: number[] }>;
        }>;
      }>;
    };
    const ms = attrs.multiscales?.[0];
    if (!ms || !ms.datasets) return fallback;
    const ds = ms.datasets.find((d) => d.path === arrayName);
    if (!ds) return fallback;
    const tx = ds.coordinateTransformations?.find((t) => t.type === "scale" && t.scale);
    if (!tx || !tx.scale) return fallback;
    const fullScale = tx.scale;
    const axes = ms.axes ?? [];
    // OME-NGFF includes non-spatial axes too (t, c). Pull the trailing 3 that
    // are type=space; fall back to last 3 unconditionally.
    let scale3 = fullScale.slice(-3);
    let unit3: (string | undefined)[] = axes.slice(-3).map((a) => a.unit);
    if (axes.length === fullScale.length) {
      const spaceIdx: number[] = [];
      axes.forEach((a, i) => { if (a.type === "space") spaceIdx.push(i); });
      if (spaceIdx.length >= 3) {
        const last3 = spaceIdx.slice(-3);
        scale3 = last3.map((i) => fullScale[i]);
        unit3 = last3.map((i) => axes[i].unit);
      }
    }
    while (scale3.length < 3) scale3.unshift(1);
    return {
      scale: scale3 as [number, number, number],
      unit: (unit3.find(Boolean) ?? "nanometer") || "nanometer",
    };
  } catch (e) {
    console.warn("[vz] could not probe source OME-NGFF metadata, using fallback:", e);
    return fallback;
  }
}

export async function handleVzRequest(path: string): Promise<VzResponse> {
  if (!state) return text(503, "vz not activated yet");
  const { spatialShape, modelHW } = state;
  const [Hm, Wm] = modelHW;
  const [Z, Y, X] = spatialShape;

  const outC = state.outputChannels;
  const isMulti = outC > 1;
  if (path === "" || path === ".zgroup") return json({ zarr_format: 2 });
  if (path === ".zattrs") {
    const unit = state.spatial.unit;
    const axes = isMulti
      ? [
          { name: "c", type: "channel" },
          { name: "z", type: "space", unit },
          { name: "y", type: "space", unit },
          { name: "x", type: "space", unit },
        ]
      : [
          { name: "z", type: "space", unit },
          { name: "y", type: "space", unit },
          { name: "x", type: "space", unit },
        ];
    const scale = isMulti
      ? [1, state.spatial.scale[0], state.spatial.scale[1], state.spatial.scale[2]]
      : state.spatial.scale;
    return json({
      multiscales: [
        {
          version: "0.4",
          name: state.cfg.modelId,
          axes,
          datasets: [{ path: "0", coordinateTransformations: [{ type: "scale", scale }] }],
        },
      ],
    });
  }
  if (path === "0/.zgroup") return text(404, "not a group");
  if (path === "0/.zattrs") return json({});
  if (path === "0/.zarray") {
    const shape = isMulti ? [outC, Z, Y, X] : [Z, Y, X];
    const chunks = isMulti ? [outC, 1, Hm, Wm] : [1, Hm, Wm];
    return json({
      zarr_format: 2,
      shape,
      chunks,
      dtype: "<f4",
      compressor: null,
      fill_value: 0,
      filters: null,
      order: "C",
      dimension_separator: ".",
    });
  }

  const chunkPath = path.startsWith("0/") ? path.slice(2) : path;
  const expected = isMulti ? 4 : 3;
  const coords = parseChunkPath(chunkPath, expected);
  if (!coords) return text(404, `not found: ${path}`);

  let cz: number, cy: number, cx: number;
  if (isMulti) {
    const [cc, _cz, _cy, _cx] = coords;
    if (cc !== 0) return text(404, "channel chunk index must be 0");
    cz = _cz; cy = _cy; cx = _cx;
  } else {
    [cz, cy, cx] = coords;
  }
  if (cz < 0 || cz >= Z) return chunkOk(zeros(outC * Hm * Wm));

  const key = `${cz}.${cy}.${cx}`;
  const cached = chunkCache.get(key);
  if (cached) return chunkOk(cached);

  let buf: ArrayBuffer;
  try {
    buf = await computeChunkBytes(cz, cy, cx);
  } catch (e) {
    if (e instanceof QueueFullError) {
      // NG retries 503s naturally as in-flight requests complete.
      return text(503, "inference queue full; backing off");
    }
    throw e;
  }
  if (chunkCache.size >= MAX_CACHED_CHUNKS) {
    const first = chunkCache.keys().next().value;
    if (first !== undefined) chunkCache.delete(first);
  }
  chunkCache.set(key, buf);
  return chunkOk(buf);
}

async function computeChunkBytes(cz: number, cy: number, cx: number): Promise<ArrayBuffer> {
  if (!state) throw new Error("vz: not activated");
  const { bmz, raw, spatialShape, modelHW, normalizers, postprocessors, outputChannels } = state;
  const [Hm, Wm] = modelHW;
  const [, Y, X] = spatialShape;
  const planeSize = Hm * Wm;

  const y0 = cy * Hm;
  const x0 = cx * Wm;
  const y1 = Math.min(y0 + Hm, Y);
  const x1 = Math.min(x0 + Wm, X);
  if (y0 >= Y || x0 >= X || y1 <= y0 || x1 <= x0) {
    return zeros(outputChannels * planeSize).buffer;
  }

  // Build a zarrita selection for the trailing 2D slice; zero-pad leading dims.
  const ndim = raw.shape.length;
  const sel: (number | zarr.Slice)[] = [];
  if (ndim >= 3) sel.push(Math.max(0, Math.min(cz, raw.shape[ndim - 3] - 1)));
  sel.push(zarr.slice(y0, y1));
  sel.push(zarr.slice(x0, x1));
  while (sel.length < ndim) sel.unshift(0);

  const inner = await zarr.get(raw, sel);
  const innerF = toFloat32(inner.data as TypedArray);

  // Build a Hm × Wm input, zero-padded on right/bottom edges.
  const Hi = inner.shape[inner.shape.length - 2];
  const Wi = inner.shape[inner.shape.length - 1];
  let rawInput: Float32Array;
  if (Hi === Hm && Wi === Wm) {
    rawInput = innerF;
  } else {
    rawInput = new Float32Array(Hm * Wm);
    for (let r = 0; r < Hi; r++) {
      rawInput.set(innerF.subarray(r * Wi, r * Wi + Wi), r * Wm);
    }
  }

  // 1) Run user-checked normalizers (in dashboard order) BEFORE the model's
  //    own RDF-declared scale_range. Most BMZ models expect [0,1] input;
  //    passing through user normalizers first lets them adjust contrast.
  let modelInput: Float32Array = rawInput;
  for (const n of normalizers) {
    modelInput = n.process(modelInput);
  }
  // RDF-declared preprocessing — for hiding-blowfish this is min-max per-sample.
  modelInput = minMax(modelInput);

  // 2) Inference.
  const slot = await acquireSlot();
  let out: Float32Array;
  try {
    out = await runOnTensor(bmz, modelInput, bmz.manifest.shape_in);
  } finally {
    slot();
  }

  // 3) Run user-checked postprocessors. out is (B=1, C_model, Hm, Wm) flat
  //    = C_model planes of size planeSize. Postprocessors may reduce C.
  let chunkData: Float32Array = out;
  let runningChannels = bmz.manifest.shape_out[1];
  if (postprocessors.length === 0) {
    // No postprocessors: default to channel 0.
    chunkData = out.subarray(0, planeSize) as Float32Array;
    runningChannels = 1;
  } else {
    const ctx: PostContext = { chunkCorner: [cz, cy, cx], chunkNumVoxels: planeSize };
    for (const p of postprocessors) {
      const r = p.process(chunkData, runningChannels, planeSize, ctx);
      chunkData = r.data;
      runningChannels = r.channels;
    }
  }
  // Sanity: must match what we declared in .zarray.
  if (runningChannels !== outputChannels) {
    console.warn(
      `[vz] postprocess produced ${runningChannels} channels, .zarray declared ${outputChannels}; ` +
        "the layer will not render correctly. Reactivate to re-derive output channel count.",
    );
  }

  // 4) Crop back to (y1-y0, x1-x0) inside a Hm×Wm output buffer per channel.
  const Hc = y1 - y0;
  const Wc = x1 - x0;
  if (Hc === Hm && Wc === Wm) {
    // Ensure it's a Float32Array we own (subarray views are fine; postprocessors
    // sometimes return owned arrays, sometimes views — copy to be safe).
    return chunkData.slice(0, runningChannels * planeSize).buffer;
  }
  const padded = new Float32Array(runningChannels * planeSize);
  for (let c = 0; c < runningChannels; c++) {
    const srcBase = c * planeSize;
    const dstBase = c * planeSize;
    for (let r = 0; r < Hc; r++) {
      padded.set(
        chunkData.subarray(srcBase + r * Wm, srcBase + r * Wm + Wc),
        dstBase + r * Wm,
      );
    }
  }
  return padded.buffer;
}

type TypedArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array;

function toFloat32(arr: TypedArray): Float32Array {
  return arr instanceof Float32Array ? arr : Float32Array.from(arr);
}

function zeros(n: number): Float32Array {
  return new Float32Array(n);
}

function json(obj: unknown): VzResponse {
  return { status: 200, headers: { "content-type": "application/json" }, body: JSON.stringify(obj) };
}
function text(status: number, body: string): VzResponse {
  return { status, headers: { "content-type": "text/plain" }, body };
}
function chunkOk(body: ArrayBuffer | Float32Array): VzResponse {
  const buf = body instanceof Float32Array ? body.buffer : body;
  return {
    status: 200,
    headers: { "content-type": "application/octet-stream", "cache-control": "no-store" },
    body: buf,
  };
}

function parseChunkPath(p: string, expected: number): number[] | null {
  const parts = p.split(".");
  if (parts.length !== expected) return null;
  const nums = parts.map(Number);
  return nums.every((n) => Number.isInteger(n) && n >= 0) ? nums : null;
}
