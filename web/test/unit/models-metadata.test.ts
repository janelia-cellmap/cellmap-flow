// metadataFor: the shape the Models tab renders for a Hugging Face repo,
// compared with a real PyTorch repo's metadata.json fields.
import { describe, expect, it } from "vitest";

import { checkpointIteration, describe as describeEntry, metadataFor, MODELS_AUTHOR } from "../../src/models/metadata";
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
      model_type: "ResUNet3DValid",
      framework: "onnx",
      spatial_dims: 3,
      in_channels: 1,
      out_channels: 3,
      iteration: 260000,
      input_voxel_size: [16, 16, 16],
      output_voxel_size: [16, 16, 16],
      channels_names: ["mito_aff_z", "mito_aff_y", "mito_aff_x"],
      input_shape: [132, 132, 132],
      output_shape: [44, 44, 44],
      inference_input_shape: [132, 132, 132],
      inference_output_shape: [44, 44, 44],
      author: MODELS_AUTHOR,
      version: "main",
      format_version: "1",
      // the export
      repo: "cellmap/onnx-web-models",
      folder: "setup_54",
      task: "mito_affinities",
      checkpoint: "model_checkpoint_260000",
      params: 10702083,
      variant: "fp32",
      file: "setup_54_mito_affinities_16nm_in132_fp32.onnx",
      file_mb: 42.91,
      onnx_url: entry.onnxUrl,
      output: "raw logits (apply sigmoid / threshold at 0)",
    });
    expect(metadata.input_norm).toEqual(entry.manifest.input_norm);
  });

  it("describes the model, the variant and its size in one line", () => {
    const entry = entryFixture("setup_54");
    expect(describeEntry(entry)).toBe(
      "mito affinities at 16 nm, ResUNet3DValid (10.7M parameters); ONNX fp32, block 132³ → 44³, 42.9 MB; raw logits (apply sigmoid / threshold at 0)",
    );
    expect(metadataFor(entry).description).toBe(describeEntry(entry));
    expect(describeEntry(entryFixture("setup_55"))).toBe(
      "nuc at 128 nm, Unet3dIso (5.6M parameters); ONNX fp32, block 140³ → 52³, 22.5 MB; raw logits (apply sigmoid / threshold at 0)",
    );
  });

  it("serialises to JSON, which the tab puts in each row's search string", () => {
    const text = JSON.stringify(metadataFor(entryFixture("setup_67")));
    expect(text).toContain("er_setup67_8nm");
    expect(text).toContain("fp32");
    expect(JSON.parse(text)).toEqual(metadataFor(entryFixture("setup_67")));
  });

  it("reads the iteration off the checkpoint name", () => {
    expect(checkpointIteration("model_checkpoint_260000")).toBe(260000);
    expect(checkpointIteration("iter 20000")).toBe(20000);
    expect(checkpointIteration("final")).toBeNull();
    expect(checkpointIteration("")).toBeNull();
  });
});
