// TensorFlow.js for whole-network finetuning, loaded the first time a run
// needs it: @tensorflow/tfjs-core with its WebGPU backend where the browser
// has a GPU, and its CPU backend otherwise (Node, where the tests train on
// it). The dynamic imports keep TF.js out of the page's bundle:
// Vite splits them into chunks of their own that only a whole-network run
// fetches: six chunks, about 0.68 MB of JavaScript, 0.19 MB gzipped, once
// Vite has shaken out the ops nothing calls (the WebGPU backend 0.28 MB of
// it); the head's training needs none of it.
//
// The CPU backend is plain JavaScript on the calling thread: it trains,
// but a step at 300² takes minutes where WebGPU takes a fraction of a
// second, and the page answers nothing meanwhile, so the page offers the
// whole network only where webgpuMissing finds a GPU. Version 4.22.0 is the one Marwan measured with (the
// manifests' spike); package.json pins it.
//
// A run holds TF.js from loadTf to its release(). When the last run lets
// go, the engine is reset: its backend's GPU buffers, which TF.js pools for
// reuse rather than freeing (over a gigabyte after a 556² run), go back to
// the GPU that serves the models; the next run sets the backend up again.
import type { Tf } from "./cellmap-tfjs-train";

export type TfBackend = "webgpu" | "cpu";

export interface LoadedTf {
  tf: Tf;
  backend: TfBackend;
  /** Why the run is not on WebGPU, when it is not. */
  fallback: string | null;
  /** The run is done with TF.js (once is enough; later calls do nothing). */
  release(): void;
}

let modules: Promise<Tf> | null = null;
let backend: Promise<{ backend: TfBackend; fallback: string | null }> | null = null;
let holders = 0;

/** TF.js with its backend set, for one run; a failed load is tried again on the next call. */
export async function loadTf(): Promise<LoadedTf> {
  holders++;
  try {
    const tf = await (modules ??= importTf().catch((e) => {
      modules = null;
      throw e;
    }));
    const set = await (backend ??= chooseBackend(tf).catch((e) => {
      backend = null;
      throw e;
    }));
    let released = false;
    return {
      tf,
      ...set,
      release: () => {
        if (released) return;
        released = true;
        if (--holders === 0) {
          backend = null;
          tf.engine().reset();
        }
      },
    };
  } catch (e) {
    holders--;
    throw e;
  }
}

async function importTf(): Promise<Tf> {
  const tf = await import("@tensorflow/tfjs-core");
  // tfjs-core's index registers no gradients (the @tensorflow/tfjs bundle
  // imports this for it): without them every step fails on its first op.
  await import("@tensorflow/tfjs-core/dist/register_all_gradients");
  await import("@tensorflow/tfjs-backend-cpu");
  if ((await webgpuMissing()) === null) {
    try {
      await import("@tensorflow/tfjs-backend-webgpu");
    } catch {
      // chooseBackend says so: the backend is then not registered.
    }
  }
  return tf;
}

/**
 * Why this browser cannot give TF.js a GPU, or null when it can, without
 * loading TF.js. navigator.gpu exists in Chrome even where requestAdapter()
 * then answers null (a headless Chromium without the GPU flags, a
 * blocklisted driver), and TF.js's WebGPU factory throws on that null
 * adapter rather than letting setBackend fall back, so the adapter is asked
 * for here first. The finetune routes refuse a whole-network run without
 * one: on the CPU backend a step takes minutes of the page's own thread,
 * which then answers nothing else (the viewer's chunks, the tab's status).
 */
export async function webgpuMissing(): Promise<string | null> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) return "this browser has no WebGPU";
  try {
    return (await navigator.gpu.requestAdapter()) ? null : "this browser's WebGPU found no GPU to use";
  } catch (e) {
    return `this browser's WebGPU failed to find a GPU (${String((e as Error)?.message ?? e)})`;
  }
}

async function chooseBackend(tf: Tf): Promise<{ backend: TfBackend; fallback: string | null }> {
  let fallback = await webgpuMissing();
  if (!fallback && !tf.findBackendFactory("webgpu")) fallback = "TF.js's WebGPU backend failed to load";
  if (!fallback) {
    if (await tf.setBackend("webgpu")) {
      await tf.ready();
      return { backend: "webgpu", fallback: null };
    }
    fallback = "TF.js could not start its WebGPU backend (the device was refused)";
  }
  if (!(await tf.setBackend("cpu"))) throw new Error("TF.js could start neither its WebGPU nor its CPU backend");
  await tf.ready();
  return { backend: "cpu", fallback };
}
