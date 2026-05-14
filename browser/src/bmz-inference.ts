// Run a BioImage Model Zoo model end-to-end in the browser via ONNX
// Runtime Web (WebGPU EP, WASM fallback). Models and test tensors are
// exported by ../scripts/export-bmz-onnx.py and served from /bmz/<id>/.

import type * as ortNS from "onnxruntime-web/webgpu";

export interface BmzManifest {
  id: string;
  name: string;
  shape_in: number[];
  shape_out: number[];
  dtype: "float32";
  normalization: { kind: "min_max" };
  opset: number;
  preferred_voxel_size_nm?: number;
  verification: {
    torch_vs_reference_max_abs_err: number;
    onnx_cpu_vs_reference_max_abs_err: number;
  };
}

export interface InferenceResult {
  manifest: BmzManifest;
  input: Float32Array;
  output: Float32Array;
  reference?: Float32Array;
  provider: "webgpu" | "wasm";
  loadMs: number;
  runMs: number;
}

let ortPromise: Promise<typeof ortNS> | null = null;

function loadOrt(): Promise<typeof ortNS> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/webgpu").then((m) => {
      m.env.wasm.wasmPaths = "/ort/";
      m.env.wasm.numThreads = 1;
      return m;
    });
  }
  return ortPromise;
}

async function fetchBin(url: string): Promise<Float32Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const buf = await r.arrayBuffer();
  return new Float32Array(buf);
}

function minMax(x: Float32Array): Float32Array {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const range = hi - lo || 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - lo) / range;
  return out;
}

export interface LoadedBmz {
  manifest: BmzManifest;
  session: ortNS.InferenceSession;
  provider: "webgpu" | "wasm";
}

const sessionCache = new Map<string, Promise<LoadedBmz>>();

export function loadBmzModel(modelId: string): Promise<LoadedBmz> {
  const cached = sessionCache.get(modelId);
  if (cached) return cached;
  const p = (async () => {
    const base = `/bmz/${modelId}/`;
    const manifest = await fetch(base + "manifest.json").then((r) => {
      if (!r.ok) throw new Error(`manifest: ${r.status} (did 'npm run export-bmz' run?)`);
      return r.json() as Promise<BmzManifest>;
    });
    const ort = await loadOrt();
    const modelBytes = new Uint8Array(await (await fetch(base + "model.onnx")).arrayBuffer());
    let session: ortNS.InferenceSession;
    let provider: "webgpu" | "wasm" = "webgpu";
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu"],
      });
    } catch (e) {
      console.warn("[bmz] webgpu unavailable, falling back to wasm:", e);
      provider = "wasm";
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
      });
    }
    return { manifest, session, provider };
  })();
  sessionCache.set(modelId, p);
  // Don't cache failures.
  p.catch(() => sessionCache.delete(modelId));
  return p;
}

export async function runOnTensor(
  loaded: LoadedBmz,
  input: Float32Array,
  shape: number[],
): Promise<Float32Array> {
  const ort = await loadOrt();
  const inputName = loaded.session.inputNames[0];
  const outputName = loaded.session.outputNames[0];
  const tensor = new ort.Tensor("float32", input, shape);
  const outputs = await loaded.session.run({ [inputName]: tensor });
  return outputs[outputName].data as Float32Array;
}

export async function runBmzOnTestInput(
  modelId: string,
): Promise<InferenceResult> {
  const base = `/bmz/${modelId}/`;
  const [input, reference] = await Promise.all([
    fetchBin(base + "test_input.bin"),
    fetchBin(base + "test_output.bin").catch(() => undefined),
  ]);

  const t0 = performance.now();
  const loaded = await loadBmzModel(modelId);
  const t1 = performance.now();
  const { manifest, provider } = loaded;

  const normalized =
    manifest.normalization.kind === "min_max" ? minMax(input) : input;

  let output: Float32Array;
  try {
    output = await runOnTensor(loaded, normalized, manifest.shape_in);
  } catch (e) {
    if (provider === "webgpu") {
      console.warn("[bmz] webgpu run failed, evicting and retrying on wasm:", e);
      sessionCache.delete(modelId);
      const wasmLoaded = await loadBmzModel(modelId);
      output = await runOnTensor(wasmLoaded, normalized, manifest.shape_in);
      return {
        manifest,
        input: normalized,
        output,
        reference,
        provider: wasmLoaded.provider,
        loadMs: t1 - t0,
        runMs: performance.now() - t1,
      };
    }
    throw e;
  }
  const t2 = performance.now();

  return {
    manifest,
    input: normalized,
    output,
    reference,
    provider,
    loadMs: t1 - t0,
    runMs: t2 - t1,
  };
}

export { minMax };

// Render a [H, W] slice of a Float32Array (values roughly in [0, 1]) into
// an offscreen canvas as 8-bit grayscale, returning the canvas so it can
// be appended to the DOM.
export function sliceToCanvas(
  data: Float32Array,
  shape: number[],
  channel: number,
  cmap: "gray" | "magma" = "gray",
): HTMLCanvasElement {
  // Expect either [1,C,H,W] (bcyx) or [H,W]
  let h: number, w: number, base: number, stride: number;
  if (shape.length === 4) {
    const [, C, H, W] = shape;
    h = H;
    w = W;
    stride = 1;
    base = (channel % C) * H * W;
  } else if (shape.length === 2) {
    [h, w] = shape;
    stride = 1;
    base = 0;
  } else {
    throw new Error(`sliceToCanvas: unsupported shape ${shape}`);
  }

  // Auto-range so single-sigmoid outputs and raw intensities both look
  // right. (BMZ test_input is already [0,1] but data crops from the wild
  // won't be.)
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < h * w; i++) {
    const v = data[base + i * stride];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const range = hi - lo || 1;

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    const v01 = (data[base + i * stride] - lo) / range;
    const [r, g, b] = cmap === "magma" ? magma(v01) : [v01 * 255, v01 * 255, v01 * 255];
    img.data[i * 4 + 0] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Cheap magma-ish ramp (good enough for a foreground probability map).
function magma(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.min(255, Math.max(0, Math.round(255 * Math.pow(x, 0.5))));
  const g = Math.min(255, Math.max(0, Math.round(255 * Math.pow(x, 2.0))));
  const b = Math.min(255, Math.max(0, Math.round(255 * (0.6 * x + 0.3 * Math.sin(3.14 * x)))));
  return [r, g, b];
}
