// The model run for its logits and its features at once: an onnxruntime-web
// session over the model with the head's input added as a second output
// (onnx-head.ts withFeatureOutput), fed the same float32 blocks the model
// stage feeds at inference. The session is built as chunkmirage's Onnx
// builds its (web/chunkmirage/web/src/onnx.ts): on WebGPU when the browser
// has it, else on WASM (the CPU); ORT's WebAssembly runtime from ort/ beside
// the page unless something set the path already (the page's inference
// shares this module instance of ORT, and with it the setting); in Node
// nothing is set and ORT finds its runtime in its own package. The session
// is the run's own: the base model's session serves the layer meanwhile,
// and this one is released when the run ends (dispose).
import * as ort from "onnxruntime-web/webgpu";

import { float16ArrayToFloat32, withFeatureOutput } from "./onnx-head";
import type { Head, ModelFeatures } from "./types";

export interface FeatureRunnerOptions {
  /** The base model's bytes; the session runs them with the feature output added. */
  bytes: ArrayBuffer | Uint8Array;
  head: Head;
  /** The tensor to feed, the session's first input by default. */
  inputName?: string;
  /** The logits tensor, the model's first output by default. */
  outputName?: string;
  log?: (line: string) => void;
}

export interface FeatureRunner {
  /** One forward pass on `input` of `dims` (float32, as the model stage
   * feeds it): the logits and the features, both float32 (a float16 model's
   * converted), with their dims. Passes run one at a time. */
  run(input: Float32Array, dims: number[]): Promise<ModelFeatures>;
  /** Release the session. */
  dispose(): Promise<void>;
  provider: "webgpu" | "wasm";
}

export async function createFeatureRunner(options: FeatureRunnerOptions): Promise<FeatureRunner> {
  const { head } = options;
  const log = options.log ?? (() => {});
  configure();
  const bytes = withFeatureOutput(options.bytes, head);
  let session: ort.InferenceSession | null = null, provider: FeatureRunner["provider"] = "wasm";
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      provider = "webgpu";
    } catch (e) {
      log(`no WebGPU session for the features (${(e as Error).message ?? e}); on WASM (the CPU) instead`);
    }
  } else {
    log("navigator.gpu is unavailable: the features run on WASM (the CPU), which is slow");
  }
  session ??= await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const input = options.inputName ?? session.inputNames[0], output = options.outputName ?? session.outputNames[0];
  for (const [what, name] of [["input", input], ["output", output], ["feature tensor", head.featureTensor]] as const) {
    const names = what === "input" ? session.inputNames : session.outputNames;
    if (!names.includes(name)) {
      await session.release();
      throw new Error(`the model has no ${what} ${name} (it has ${names.join(", ")})`);
    }
  }
  log(`features session ready on ${provider === "webgpu" ? "WebGPU" : "WASM (the CPU)"}: ${head.inChannels} channels of ${head.featureTensor}, ${head.outChannels} of ${output}`);

  // One pass at a time: an InferenceSession is not re-entrant, and two passes
  // overlapping on WebGPU never resolve (chunkmirage's Onnx has the same turn).
  let turn: Promise<unknown> = Promise.resolve();
  const s = session;
  return {
    provider,
    run(data, dims) {
      const pass = turn.then(async (): Promise<ModelFeatures> => {
        const results = await s.run({ [input]: new ort.Tensor("float32", data, dims) }, [output, head.featureTensor]);
        const logits = results[output], features = results[head.featureTensor];
        return { logits: asFloat32(logits, output), logitsDims: [...logits.dims], features: asFloat32(features, head.featureTensor), featuresDims: [...features.dims] };
      });
      turn = pass.catch(() => {});
      return pass;
    },
    async dispose() {
      await turn;
      await s.release();
    },
  };
}

/** ORT's WebAssembly runtime's location, ort/ beside the page, unless set
 * already (by chunkmirage's Onnx, which the page's inference configures
 * first); nothing in Node, where ORT reads it from its own package. */
function configure(): void {
  if (typeof document === "undefined" || ort.env.wasm.wasmPaths) return;
  ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
}

/** A tensor's values as float32: as they are, or converted from the float16 bits ORT hands back as a Uint16Array. */
function asFloat32(tensor: ort.Tensor, name: string): Float32Array {
  if (tensor.type === "float32") return tensor.data as Float32Array;
  if (tensor.type === "float16") return float16ArrayToFloat32(tensor.data as Uint16Array);
  throw new Error(`${name} is ${tensor.type}, not float32 or float16`);
}
