// Load a cellmap-flow-compatible model from a HuggingFace repo.
// Repos under https://huggingface.co/cellmap/<repo> follow a known layout:
//   metadata.json   — the cellmap metadata (input/output shape, voxel sizes, etc.)
//   model.onnx      — exported ONNX (only present on a few repos so far)
//   model.pt[.2|.ts] — PyTorch checkpoints (not browser-loadable)
//
// We fetch metadata.json via the HF resolve URL, then translate it to our
// browser ModelSpec format.

import type { ModelSpec, TensorLayout } from "./model-spec";

const HF_BASE = "https://huggingface.co";

export interface HfMetadata {
  model_type?: string;
  framework?: string;
  spatial_dims?: number;
  in_channels?: number;
  out_channels?: number;
  input_voxel_size?: [number, number, number];
  output_voxel_size?: [number, number, number];
  input_shape?: [number, number, number];          // training input voxels
  output_shape?: [number, number, number];          // training output voxels
  inference_input_shape?: [number, number, number]; // larger inference voxels
  inference_output_shape?: [number, number, number];
  channels_names?: string[];
}

export interface HfLoadResult {
  modelUrl: string;
  spec: ModelSpec;
  metadata: HfMetadata;
}

export async function loadHfModel(repoOrUrl: string): Promise<HfLoadResult> {
  const repo = parseRepo(repoOrUrl);
  const metaUrl = `${HF_BASE}/${repo}/resolve/main/metadata.json`;
  const modelUrl = `${HF_BASE}/${repo}/resolve/main/model.onnx`;

  const r = await fetch(metaUrl);
  if (!r.ok) throw new Error(`failed to fetch ${metaUrl}: ${r.status}`);
  const metadata = (await r.json()) as HfMetadata;

  // Probe model.onnx so we fail fast with a clear message if the repo only
  // ships PyTorch checkpoints (most cellmap repos do today).
  const head = await fetch(modelUrl, { method: "HEAD" });
  if (!head.ok) {
    throw new Error(
      `${repo} has no model.onnx (HEAD ${head.status}). Most cellmap HF repos ` +
        `currently ship PyTorch checkpoints only. Ask the author to also export ONNX.`,
    );
  }

  const spec = specFromHfMetadata(metadata);
  return { modelUrl, spec, metadata };
}

function parseRepo(input: string): string {
  let s = input.trim();
  // Allow the user to paste either "user/repo" or a full HF URL.
  s = s.replace(/^https?:\/\/huggingface\.co\//, "");
  s = s.replace(/\/(?:tree|blob|resolve)\/.+$/, "");
  s = s.replace(/\/$/, "");
  return s;
}

function specFromHfMetadata(m: HfMetadata): ModelSpec {
  const ivs = m.input_voxel_size ?? [1, 1, 1];
  const ovs = m.output_voxel_size ?? ivs;

  // ONNX exports use the inference tile (the model is shape-fixed unless
  // exported with dynamic axes), so prefer inference_* over training shapes.
  // Fall back to training shape only if inference_* isn't in the metadata.
  const inVox = m.inference_input_shape ?? m.input_shape ?? [128, 128, 128];
  const outVox = m.inference_output_shape ?? m.output_shape ?? [64, 64, 64];

  const tensorLayout: TensorLayout = (m.spatial_dims ?? 3) === 3 ? "NCDHW" : "BatchZ_NCHW";

  return {
    inputVoxelSize: ivs,
    outputVoxelSize: ovs,
    readShape: [inVox[0] * ivs[0], inVox[1] * ivs[1], inVox[2] * ivs[2]],
    writeShape: [outVox[0] * ovs[0], outVox[1] * ovs[1], outVox[2] * ovs[2]],
    inputDtype: "float32",
    outputDtype: "uint8",
    outputChannels: m.out_channels ?? 1,
    blockShape: outVox,
    tensorLayout,
    // Common cellmap convention: rescale raw uint8 [0,255] -> [0,1].
    normalize: [
      { name: "MinMaxNormalizer", min_value: 0, max_value: 255 },
    ],
    // Standard output mapping: clip [-1, 1], shift to [0, 2], multiply by 127.5
    // -> uint8 [0, 255]. (cellmap-flow's DefaultPostprocessor defaults.)
    postprocess: [
      { name: "DefaultPostprocessor" },
    ],
  };
}
