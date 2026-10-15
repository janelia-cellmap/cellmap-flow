// onnxruntime-web sessions: one per model file, built from the bytes the
// catalog gives (so its IndexedDB cache is used), on WebGPU when the browser
// has it and on WASM (CPU) otherwise, each behind a one-slot lock. From
// Marwan Zouinkhi's web/legacy/compute.js (getSession, the concurrency bound).
//
// `Sessions` is the browser mode's ONNX runner (`ModelRunner`), the shape of
// chunkmirage's `Onnx` class (web/chunkmirage/web/src/onnx.ts: run, retain,
// release, reset, limits, device, provider), so the engine (engine.ts) can
// take either: this file today, and chunkmirage's when it settles there (a
// one-line change in engine.ts). The interface is spelled out here rather
// than imported from chunkmirage so this file stands whether or not the
// pinned chunkmirage has an onnx.ts.
//
// "onnxruntime-web/webgpu" is the build with the native WebGPU execution
// provider (the package's default export carries the older JSEP provider,
// which is deprecated). Its one WebAssembly binary,
// ort-wasm-simd-threaded.asyncify.wasm, runs both the WASM provider and the
// WebGPU provider's host side; the bundle fetches it from env.wasm.wasmPaths,
// which points at ./ort/ beside the page, where build/vite-ort.ts serves it
// in development and copies it in the build (self-hosted: the page then
// works offline from its own origin, and no CDN version can drift from the
// npm one).
//
// Marwan pinned a nightly ORT to get Conv3D on WebGPU; the models on Hugging
// Face are since exported with 2-D convolutions and run on the release.
import * as ort from "onnxruntime-web/webgpu";

import type { GpuLimits, Provider } from "../contracts";
import { Semaphore } from "./semaphore";
import type { ModelOutput } from "./stages";

/** A model to run: its bytes when first needed (the catalog keeps them),
 * under a key that is its identity (the .onnx file's URL); the tensors to
 * feed and read, the session's first of each by default. (chunkmirage's
 * OnnxModel, which is assignable to it.) */
export interface RunnerModel {
  key: string;
  bytes: () => Promise<ArrayBuffer>;
  input?: string;
  output?: string;
}

/** What runs the models: sessions built on first use and shared by the
 * views that use them, one forward pass at a time per session. */
export interface ModelRunner {
  /** The provider of the first session built: WebGPU, or WASM (the CPU). */
  readonly provider: Provider | null;
  /** One forward pass of `model` on `data` of `dims` (float32), after the
   * session's other passes: the output tensor, and the pass's time alone. */
  run(model: RunnerModel, data: Float32Array, dims: number[]): Promise<ModelOutput>;
  /** A view starts using `key`'s session. */
  retain(key: string): void;
  /** A view is done with `key`'s session: the last one releases it. */
  release(key: string): Promise<void>;
  /** Every session released and forgotten; the next block builds its session again. */
  reset(): Promise<void>;
  /** The WebGPU adapter's limits the variant choice depends on; nulls without WebGPU. */
  limits(): Promise<GpuLimits>;
  /** The GPU as the adapter describes it, or null without WebGPU. */
  device(): Promise<string | null>;
}

interface OrtSession {
  session: ort.InferenceSession;
  provider: Provider;
  /** One forward pass at a time: the session is not re-entrant, and two
   * overlapping runs on WebGPU never resolved. */
  lock: Semaphore;
  inputName: string;
  outputName: string;
}

let configured = false;

/** ORT's global settings, once, before the first session. */
function configureOrt() {
  if (configured) return;
  configured = true;
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
  // Threads need SharedArrayBuffer, which needs the COOP/COEP headers GitHub
  // Pages cannot set; without cross-origin isolation ORT would warn and fall
  // back to one thread anyway.
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 1;
}

export class Sessions implements ModelRunner {
  private cache = new Map<string, Promise<OrtSession>>();
  private users = new Map<string, number>();  // served models per session key
  private adapter: Promise<GPUAdapter | null> | null = null;
  provider: Provider | null = null;

  /** `log` hears what the sessions do; `status` the step under way (loading
   * a model, building its session), for the dashboard's status line. */
  constructor(private log: (line: string) => void, private status: (line: string) => void) {}

  /** The session for `model`, built from its bytes on the first call: WebGPU
   * if `navigator.gpu` exists and the session can be created on it, else
   * WASM. A failed build is forgotten, so the next chunk tries again. */
  private session(model: RunnerModel): Promise<OrtSession> {
    let pending = this.cache.get(model.key);
    if (!pending) {
      pending = this.build(model).catch((e) => {
        this.cache.delete(model.key);
        throw e;
      });
      this.cache.set(model.key, pending);
    }
    return pending;
  }

  private async build(model: RunnerModel): Promise<OrtSession> {
    configureOrt();
    const name = model.key.split("/").pop() ?? model.key;
    this.status(`loading ${name}…`);
    const bytes = new Uint8Array(await model.bytes());
    let session: ort.InferenceSession | null = null;
    let provider: Provider = "wasm";
    if (navigator.gpu) {
      this.status(`${name}: creating the WebGPU session…`);
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
        provider = "webgpu";
      } catch (e) {
        this.log(`${name}: WebGPU session failed (${(e as Error).message ?? e}); retrying on WASM (CPU)`);
      }
    } else {
      this.log("navigator.gpu is unavailable: running on WASM (CPU), which is slow");
    }
    if (!session) {
      this.status(`${name}: creating the WASM session…`);
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    }
    const inputName = model.input ?? session.inputNames[0], outputName = model.output ?? session.outputNames[0];
    if (!session.inputNames.includes(inputName)) throw new Error(`${name} has no input ${inputName} (it has ${session.inputNames.join(", ")})`);
    if (!session.outputNames.includes(outputName)) throw new Error(`${name} has no output ${outputName} (it has ${session.outputNames.join(", ")})`);
    this.provider ??= provider;
    this.status(`${name}: session ready · ${provider === "webgpu" ? "WebGPU (GPU)" : "WASM (CPU)"}`);
    return { session, provider, lock: new Semaphore(1), inputName, outputName };
  }

  async run(model: RunnerModel, data: Float32Array, dims: number[]): Promise<ModelOutput> {
    const s = await this.session(model);
    return s.lock.run(async () => {
      const t0 = performance.now();
      const results = await s.session.run({ [s.inputName]: new ort.Tensor("float32", data, dims) });
      const out = results[s.outputName];
      return { data: out.data as ArrayLike<number>, dims: out.dims, type: out.type, ms: performance.now() - t0 };
    });
  }

  retain(key: string) { this.users.set(key, (this.users.get(key) ?? 0) + 1); }

  async release(key: string): Promise<void> {
    const left = (this.users.get(key) ?? 1) - 1;
    if (left > 0) { this.users.set(key, left); return; }
    this.users.delete(key);
    const pending = this.cache.get(key);
    this.cache.delete(key);
    if (pending) await releaseQuietly(pending);
  }

  /** Every session released and forgotten; the next chunk builds its session
   * again. The WebGPU device is left to ORT: it creates the one GPUDevice its
   * backend uses when the first session is built and does not make another
   * if that one is destroyed, so destroying it would leave every later
   * session failing. A forward pass that truly hangs the GPU queue needs a
   * reload; this recovers from a session in a bad state. */
  async reset(): Promise<void> {
    const pending = [...this.cache.values()];
    this.cache.clear();
    this.users.clear();
    await Promise.all(pending.map(releaseQuietly));
  }

  async limits(): Promise<GpuLimits> {
    const adapter = await this.getAdapter();
    if (!adapter) return { maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false };
    return {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      shaderF16: adapter.features.has("shader-f16"),
    };
  }

  async device(): Promise<string | null> {
    const adapter = await this.getAdapter();
    if (!adapter) return null;
    const info = adapter.info as Partial<GPUAdapterInfo> | undefined;
    const name = [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(" ").trim();
    return name || "WebGPU adapter";
  }

  private getAdapter(): Promise<GPUAdapter | null> {
    if (!this.adapter) {
      this.adapter = typeof navigator !== "undefined" && navigator.gpu
        ? navigator.gpu.requestAdapter().catch(() => null)
        : Promise.resolve(null);
    }
    return this.adapter;
  }
}

/** Release a session, not waiting forever on one whose run hangs (the reason
 * reset() is called), and not failing on one that never got built. */
async function releaseQuietly(pending: Promise<OrtSession>): Promise<void> {
  try {
    const s = await pending;
    await Promise.race([s.session.release(), new Promise<void>((r) => setTimeout(r, 3000))]);
  } catch {
    // a session that failed to build has nothing to release
  }
}
