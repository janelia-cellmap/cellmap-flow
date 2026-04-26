import type * as ortNS from "onnxruntime-web/webgpu";

export type InferenceSession = ortNS.InferenceSession;

export interface SessionInfo {
  inputNames: string[];
  outputNames: string[];
  provider: string;
}

let ortPromise: Promise<typeof ortNS> | null = null;

function loadOrt(): Promise<typeof ortNS> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/webgpu").then((m) => {
      m.env.wasm.wasmPaths = "/ort/";
      m.env.wasm.numThreads = 1;
      // Verbose so the WebGPU EP prints node-by-node assignment decisions.
      // Surfaces "node X falls back to CPU because <reason>" — exactly what
      // we need to know when Conv3d gets refused.
      m.env.logLevel = "verbose";
      return m;
    });
  }
  return ortPromise;
}

export type ProgressCallback = (loaded: number, total: number | undefined) => void;

async function fetchWithProgress(url: string, onProgress?: ProgressCallback): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch model (${resp.status}) at ${url}`);
  const totalHeader = resp.headers.get("content-length");
  const total = totalHeader ? parseInt(totalHeader, 10) : undefined;
  if (!resp.body) {
    const buf = new Uint8Array(await resp.arrayBuffer());
    onProgress?.(buf.byteLength, total);
    return buf;
  }
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastReport = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    // Throttle progress callbacks to ~10/sec so we don't thrash the DOM.
    const now = performance.now();
    if (now - lastReport > 100) {
      onProgress?.(loaded, total);
      lastReport = now;
    }
  }
  onProgress?.(loaded, total);
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

export async function createSession(
  url: string,
  onProgress?: ProgressCallback,
): Promise<{ session: ortNS.InferenceSession; info: SessionInfo }> {
  const ort = await loadOrt();
  const bytes = await fetchWithProgress(url, onProgress);

  // WebGPU is required for 3D models (the WASM backend doesn't implement
  // Conv3d). We try WebGPU first; if it fails we surface the actual error
  // and do NOT silently fall back to WASM — silent fallback was hiding
  // the root cause and producing useless "Only Conv1d/Conv2d supported"
  // errors at runtime. Pass `?wasm` in the URL to force WASM (only useful
  // for 2D models).
  let session: ortNS.InferenceSession;
  let provider = "webgpu";
  const allowWasm = url.includes("?wasm") || url.includes("&wasm");

  if (typeof navigator !== "undefined" && !(navigator as Navigator & { gpu?: object }).gpu) {
    console.warn(
      "[ort] navigator.gpu is undefined — your browser does not expose WebGPU. " +
        "3D UNet models will not run. Use a recent Chrome (>= 113) on a desktop GPU.",
    );
  }

  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["webgpu"],
      logSeverityLevel: 0, // 0 = verbose
      logVerbosityLevel: 1,
    });
  } catch (e) {
    const ge = e as Error;
    console.error("[ort] WebGPU session.create failed:", ge.message);
    if (!allowWasm) {
      throw new Error(
        `WebGPU unavailable for this model. ` +
          `Underlying error: ${ge.message}. ` +
          `If you're sure the model is 2D-only, append "?wasm" to the model URL to force the WASM backend.`,
      );
    }
    console.warn("[ort] explicit ?wasm — falling back to WASM backend (no Conv3d support).");
    provider = "wasm";
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    });
  }

  return {
    session,
    info: {
      inputNames: [...session.inputNames],
      outputNames: [...session.outputNames],
      provider,
    },
  };
}

let runLock: Promise<unknown> = Promise.resolve();

export async function runModel(
  session: ortNS.InferenceSession,
  input: Float32Array,
  dims: number[],
): Promise<{ data: Float32Array; dims: readonly number[] }> {
  const next = runLock.then(async () => {
    const ort = await loadOrt();
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const tensor = new ort.Tensor("float32", input, dims);
    const outputs = await session.run({ [inputName]: tensor });
    const out = outputs[outputName];
    return { data: out.data as Float32Array, dims: out.dims };
  });
  runLock = next.catch(() => undefined);
  return next;
}
