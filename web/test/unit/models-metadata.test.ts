// metadataFor: the shape the Models tab renders for a Hugging Face repo,
// compared with a real PyTorch repo's metadata.json fields; and listFacts,
// what every model list's row tags a model with.
import { describe, expect, it } from "vitest";

import { finetunedEntry } from "../../src/models/finetuned";
import { listFacts, metadataFor } from "../../src/models/metadata";
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

describe("listFacts", () => {
  it("gives a model's format, its ways to finetune, its size and each precision's file size, a finetuned one's its own", () => {
    const entry = entryFixture("setup_53_2d");
    expect(listFacts(entry)).toEqual({
      formats: ["ONNX"], finetune_modes: ["head", "full"], size_mb: 7.75, browser_copy: entry.id,
      variants: [{ variant: "fp32", file_mb: 7.75 }, { variant: "fp16", file_mb: 3.9 }], params: 1927907,
    });
    // A model finetuned in this browser is one file, the one it trained.
    const finetuned = finetunedEntry(entryFixture("setup_54"), "mito_aff_setup54_16nm-ft1", 42.9);
    expect(listFacts(finetuned)).toMatchObject({ size_mb: 42.9, browser_copy: finetuned.id, variants: [{ variant: "fp32", file_mb: 42.9 }] });
  });
});
