// Display-only loader for cellmap HF model metadata. We don't build runtime
// pipelines from HF metadata anymore — the inference server (HF Space or
// Colab) loads the model directly via cellmap-flow's HuggingFaceModelConfig.
// All this does is fetch metadata.json so the dashboard can show the user
// what they're pointed at (channels, voxel size, etc.).

const HF_BASE = "https://huggingface.co";

export interface HfMetadata {
  model_type?: string;
  framework?: string;
  spatial_dims?: number;
  in_channels?: number;
  out_channels?: number;
  input_voxel_size?: [number, number, number];
  output_voxel_size?: [number, number, number];
  input_shape?: [number, number, number];
  output_shape?: [number, number, number];
  inference_input_shape?: [number, number, number];
  inference_output_shape?: [number, number, number];
  channels_names?: string[];
  description?: string;
}

export interface HfLoadResult {
  metadata: HfMetadata;
  spec: {
    outputVoxelSize: [number, number, number];
    inputVoxelSize: [number, number, number];
    outputChannels: number;
  };
}

export async function loadHfModel(repoOrUrl: string): Promise<HfLoadResult> {
  const repo = parseRepo(repoOrUrl);
  const metaUrl = `${HF_BASE}/${repo}/resolve/main/metadata.json`;
  const r = await fetch(metaUrl);
  if (!r.ok) throw new Error(`failed to fetch ${metaUrl}: ${r.status}`);
  const metadata = (await r.json()) as HfMetadata;
  return {
    metadata,
    spec: {
      outputVoxelSize: metadata.output_voxel_size ?? metadata.input_voxel_size ?? [1, 1, 1],
      inputVoxelSize: metadata.input_voxel_size ?? [1, 1, 1],
      outputChannels: metadata.out_channels ?? 1,
    },
  };
}

function parseRepo(input: string): string {
  let s = input.trim();
  s = s.replace(/^https?:\/\/huggingface\.co\//, "");
  s = s.replace(/\/(?:tree|blob|resolve)\/.+$/, "");
  s = s.replace(/\/$/, "");
  return s;
}
