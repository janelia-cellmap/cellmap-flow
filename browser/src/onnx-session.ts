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

  let session: ortNS.InferenceSession;
  let provider = "webgpu";
  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["webgpu"],
    });
  } catch (e) {
    console.warn("webgpu unavailable, falling back to wasm:", e);
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
