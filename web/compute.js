// compute.js — the CellMap Flow inference pipeline, run in a module Worker.
//
// This is the work that used to live in the service worker. It moved here
// because service workers forbid dynamic import() (zarrita lazily import()s its
// decompression codecs, and onnxruntime-web may too) — module Workers allow it,
// and this also keeps heavy compute off the main thread. The service worker now
// just intercepts Neuroglancer's requests and delegates each one here.
//
// handleCf(segment, tail) reproduces server.py's routes:
//   tail ".zattrs"      -> OME multiscales JSON
//   tail "s0/.zattrs"   -> {} (optional per-array attrs; keeps Neuroglancer quiet)
//   tail "s0/.zarray"   -> array metadata JSON
//   tail "s0/z.y.x[.c]" -> inference chunk (raw bytes)

// Nightly build: testing whether the WebGPU EP runs Conv3D (the stable 1.27
// operator matrix lists conv3d as unsupported). Pinned to a specific dev build.
const ORT_VER = "1.27.0-dev.20260506-673c3320fc";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0-dev.20260506-673c3320fc/dist/ort.webgpu.bundle.min.mjs";
// Propagate this module's cache-busting query (?v=…) to the relative imports so
// the whole worker graph refreshes together on reload (GitHub Pages caches JS
// for 10 min otherwise). CDN imports above are version-pinned already.
const _v = new URL(import.meta.url).search;
const { getModel, getHandle, getBlob } = await import("./local-store.js" + _v);
const { openArray, readRoi } = await import("./zarr-reader.js" + _v);
const P = await import("./pipeline.js" + _v);

const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
ort.env.wasm.wasmPaths = ORT_DIST;
// Use all cores for WASM (only takes effect when crossOriginIsolated, i.e. when
// the SW has injected COOP/COEP and SharedArrayBuffer is available).
if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency;
}
// Flip on to debug op placement (WebGPU vs WASM fallback) in the worker console.
// ort.env.logLevel = "verbose";
// ort.env.debug = true;
console.log("[compute] WebGPU available:", typeof navigator !== "undefined" && !!navigator.gpu);

// Limit concurrent inferences. Neuroglancer may request many chunks at once
// (its "Concurrent chunk requests"), but each is a full volumetric forward pass;
// running them all at once exhausts GPU/CPU memory. Configurable per model
// (meta.max_concurrent, the page's "Max concurrent inferences" field); default 1:
// an onnxruntime-web InferenceSession is not re-entrant, and two overlapping
// session.run() calls on the WebGPU EP never resolve (0 blocks done, page hangs).
const DEFAULT_MAX_CONCURRENT = 1;
let maxConcurrent = DEFAULT_MAX_CONCURRENT;
let inflight = 0;
let announcedFirst = false;
let blocksDone = 0;
const waiters = [];
function setMaxConcurrent(n) {
  maxConcurrent = Number.isInteger(n) && n >= 1 ? n : DEFAULT_MAX_CONCURRENT;
  while (inflight < maxConcurrent && waiters.length) {
    inflight++;
    waiters.shift()();
  }
}
function acquire() {
  if (inflight < maxConcurrent) {
    inflight++;
    return Promise.resolve();
  }
  return new Promise((resolve) => waiters.push(resolve));
}
function release() {
  inflight--;
  if (inflight < maxConcurrent && waiters.length) {
    inflight++;
    waiters.shift()();
  }
}

const ARGS_KEY = "__CFLOW_ARGS__";

const sessionCache = new Map(); // session key -> Promise<InferenceSession>
const arrayCache = new Map(); // model name -> Promise<{ arr, shape }>

// Build (and cache) a session from a remote URL (meta.onnxUrl) or bytes stored
// in IndexedDB by the page (meta.onnxKey, set for a local .onnx file).
// Returns { session, provider } where provider is "webgpu" (GPU) or "wasm" (CPU).
// The provider is fixed per session, so we detect it once: try WebGPU alone, and
// fall back to WASM if WebGPU is unavailable or fails to initialize.
function getSession(meta) {
  // Include backend so switching auto/webgpu/wasm creates a fresh session
  // instead of reusing the cached one.
  const key = `${meta.onnxKey || meta.onnxUrl}::${meta.backend || "auto"}`;
  if (!sessionCache.has(key)) {
    sessionCache.set(
      key,
      (async () => {
        let src = meta.onnxUrl;
        if (meta.onnxKey) {
          src = await getBlob(meta.onnxKey);
          if (!src) throw new Error(`no stored ONNX bytes for ${meta.onnxKey}`);
        }
        const pref = meta.backend || "auto"; // "auto" | "webgpu" | "wasm"
        // Forcing WebGPU is a debug mode: log op-by-op so the last op before a
        // stall is visible (e.g. an unsupported Conv3d falling back).
        if (pref === "webgpu") {
          ort.env.logLevel = "verbose";
          ort.env.debug = true;
        }
        const status = (text) => self.postMessage({ type: "status", text });
        status(`initializing model session (${meta.onnxKey ? "local file" : "remote URL"}) [${pref}]…`);
        if (pref !== "wasm" && navigator.gpu) {
          try {
            const session = await ort.InferenceSession.create(src, {
              executionProviders: ["webgpu"],
            });
            self.postMessage({ type: "status", text: "session ready · WebGPU (GPU)", provider: "webgpu" });
            return { session, provider: "webgpu" };
          } catch (e) {
            console.warn("[compute] WebGPU init failed:", e.message);
            if (pref === "webgpu") throw e;
            status("WebGPU failed, retrying on WASM (CPU)…");
          }
        } else if (pref !== "wasm") {
          status("navigator.gpu unavailable — using WASM (CPU)");
        }
        const session = await ort.InferenceSession.create(src, { executionProviders: ["wasm"] });
        self.postMessage({ type: "status", text: "session ready · WASM (CPU)", provider: "wasm" });
        return { session, provider: "wasm" };
      })()
    );
  }
  return sessionCache.get(key);
}

async function resolveSource(meta) {
  if (meta.source.type === "local") {
    const dirHandle = await getHandle(meta.source.handleKey);
    if (!dirHandle) throw new Error(`no stored handle ${meta.source.handleKey}`);
    return { ...meta.source, dirHandle };
  }
  return meta.source;
}

async function getInputArray(name, meta) {
  if (!arrayCache.has(name)) {
    arrayCache.set(
      name,
      (async () => {
        const source = await resolveSource(meta);
        const arr = await openArray(source, meta);
        return { arr, shape: arr.shape };
      })()
    );
  }
  return arrayCache.get(name);
}

function parseModelSegment(segment) {
  const i = segment.indexOf(ARGS_KEY);
  if (i === -1) return { name: decodeURIComponent(segment), args: null };
  const name = decodeURIComponent(segment.slice(0, i));
  let rest = segment.slice(i + ARGS_KEY.length);
  if (rest.endsWith(ARGS_KEY)) rest = rest.slice(0, -ARGS_KEY.length);
  let args = null;
  try {
    let b64 = rest.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    args = JSON.parse(atob(b64));
  } catch (e) {
    console.warn("[compute] failed to decode CFLOW args", e);
  }
  return { name, args };
}

async function handleChunk(meta, cz, cy, cx) {
  const { arr } = await getInputArray(meta.__name, meta);
  const roi = P.chunkInputRoi(meta, cz, cy, cx);
  const input = await readRoi(arr, meta, roi); // { data: Float32Array, shape: input_size }

  P.applyNormalizers(input.data, meta.input_norm);

  const { session, provider } = await getSession(meta);
  const dims = [1, 1, ...meta.input_size];

  // Bound concurrent forward passes (see DEFAULT_MAX_CONCURRENT above).
  setMaxConcurrent(Number(meta.max_concurrent));
  if (!announcedFirst) {
    announcedFirst = true;
    self.postMessage({
      type: "status",
      text: "first inference — compiling GPU kernels (first block can take a few seconds)…",
    });
  }
  await acquire();
  let out;
  const t0 = performance.now();
  try {
    const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input.data, dims) };
    const results = await session.run(feeds);
    out = results[session.outputNames[0]];
  } finally {
    release();
  }
  const ms = performance.now() - t0;
  blocksDone++;
  if (blocksDone === 1 || blocksDone % 10 === 0) {
    self.postMessage({ type: "status", text: `running · ${blocksDone} block(s) done · last ${Math.round(ms)} ms` });
  }
  const outDims = Array.from(out.dims);

  let nd;
  if (outDims.length === 5) nd = { data: out.data, shape: outDims.slice(1) };
  else if (outDims.length === 4) nd = { data: out.data, shape: [1, ...outDims.slice(1)] };
  else throw new Error(`unexpected model output rank ${outDims.length}`);

  nd = P.applyPostprocessors(nd, meta.postprocess);

  let zarrNd;
  if (meta.has_channel) zarrNd = P.reorderToZarr(nd); // [C,z,y,x] -> [z,y,x,c]
  else zarrNd = { data: nd.data, shape: nd.shape.slice(1) }; // squeeze channel

  return { buf: P.encodeChunk(zarrNd, P.outputDtype(meta)), provider, ms };
}

// Returns { status?, contentType, body } where body is a JSON string or ArrayBuffer.
export async function handleCf(segment, tail) {
  const { name, args } = parseModelSegment(segment);
  let meta;
  if (args && args.input_size) {
    // Self-contained meta embedded in the URL (shareable link).
    meta = { ...args, __name: name };
  } else {
    // Legacy: geometry from IndexedDB, args may override norms/post.
    const stored = await getModel(name);
    if (!stored) {
      return { status: 404, contentType: "text/plain", body: `unknown model ${name}` };
    }
    meta = { ...stored, ...(args || {}), __name: name };
  }

  if (tail === ".zattrs") {
    return { contentType: "application/json", body: JSON.stringify(P.buildZattrs(meta, name)) };
  }
  if (tail === "s0/.zattrs") {
    return { contentType: "application/json", body: "{}" };
  }
  if (tail === "s0/.zarray") {
    const { shape } = await getInputArray(name, meta);
    return { contentType: "application/json", body: JSON.stringify(P.buildZarray(meta, shape)) };
  }
  const m = tail.match(/^s0\/(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?$/);
  if (m) {
    const { buf, provider, ms } = await handleChunk(meta, +m[1], +m[2], +m[3]);
    return { contentType: "application/octet-stream", body: buf, provider, ms };
  }
  return { status: 404, contentType: "text/plain", body: "not found" };
}
