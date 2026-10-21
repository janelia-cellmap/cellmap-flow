// metadataFor: the shape the Models tab renders for a Hugging Face repo,
// compared with a real PyTorch repo's metadata.json fields.
import { describe, expect, it } from "vitest";

import { metadataFor } from "../../src/models/metadata";
import { entryFixture } from "./models-helpers";

// The keys of cellmap/mito-aff-unet-setup-16's metadata.json (read 2026-10-14).
const PYTORCH_METADATA_KEYS = [
  "model_name", "model_type", "framework", "spatial_dims", "in_channels", "out_channels", "iteration",
  "input_voxel_size", "output_voxel_size", "channels_names", "input_shape", "output_shape",
  "inference_input_shape", "inference_output_shape", "author", "description", "version", "format_version",
];

describe("metadataFor", () => {
  it("has every field a PyTorch repo's metadata.json has", () => {
    const metadata = metadataFor(entryFixture("setup_54"));
    for (const key of PYTORCH_METADATA_KEYS) expect(metadata, key).toHaveProperty(key);
  });

  it("fills them from the manifest and the chosen variant", () => {
    const entry = entryFixture("setup_54");
    const metadata = metadataFor(entry);
    expect(metadata).toMatchObject({
      model_name: "mito_aff_setup54_16nm",
      out_channels: 3,
      iteration: 260000,
      input_shape: [132, 132, 132],
      output_shape: [44, 44, 44],
      onnx_url: entry.onnxUrl,
    });
  });
});
