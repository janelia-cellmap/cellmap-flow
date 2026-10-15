// modelInfo: the /__control__/model_info answer for a browser-served model.
import { describe, expect, it } from "vitest";

import type { GpuLimits } from "../../src/contracts";
import { UNKNOWN_LIMITS } from "../../src/models/variant";
import { modelInfo, outputClassOf } from "../../src/shared/model-info";
import { entryFixture } from "./models-helpers";

describe("modelInfo", () => {
  it("gives the geometry of the chosen variant, channels first", () => {
    const info = modelInfo(entryFixture("setup_54", UNKNOWN_LIMITS));
    expect(info).toEqual({
      output_channels: 3,
      channels: ["mito_aff_z", "mito_aff_y", "mito_aff_x"],
      write_shape: [704, 704, 704], // 44 voxels at 16 nm
      read_shape: [2112, 2112, 2112], // 132 voxels
      output_voxel_size: [16, 16, 16],
      input_voxel_size: [16, 16, 16],
      effective_output_voxel_size: [16, 16, 16],
      has_channel: true,
      output_axes: ["c", "z", "y", "x"],
      available: true,
      output_class: "unbounded",
      output_min: null,
      output_max: null,
      display_channel: null,
      engine: "browser",
    });
  });

  it("follows the variant: a bigger block on a bigger GPU", () => {
    const big: GpuLimits = { maxBufferSize: 4 * 1024 ** 3, maxStorageBufferBindingSize: null, shaderF16: true };
    const info = modelInfo(entryFixture("setup_53", big));
    expect(info.read_shape).toEqual([2752, 2752, 2752]); // 172 voxels at 16 nm
    expect(info.write_shape).toEqual([1344, 1344, 1344]); // 84 voxels
    expect(info.output_channels).toBe(3);
  });

  it("works for the one-channel models at their voxel sizes", () => {
    const nuc = modelInfo(entryFixture("setup_61"));
    expect(nuc.channels).toEqual(["nuc"]);
    expect(nuc.output_channels).toBe(1);
    expect(nuc.output_voxel_size).toEqual([256, 256, 256]);
    const er = modelInfo(entryFixture("setup_67"));
    expect(er.input_voxel_size).toEqual([8, 8, 8]);
  });

  it("copies the arrays rather than sharing the manifest's", () => {
    const entry = entryFixture("setup_54");
    const info = modelInfo(entry);
    info.channels.push("extra");
    info.write_shape[0] = 1;
    expect(entry.manifest.channels).toHaveLength(3);
    expect(entry.variant.output_size[0]).toBe(44);
  });

  it("is JSON as the dashboard reads it", () => {
    const info = modelInfo(entryFixture("setup_54"));
    expect(JSON.parse(JSON.stringify(info))).toEqual(info);
  });
});

describe("outputClassOf", () => {
  it("maps the manifest's output text to probe.py's classes", () => {
    expect(outputClassOf("raw logits (apply sigmoid / threshold at 0)")).toBe("unbounded");
    expect(outputClassOf("logits")).toBe("unbounded");
    expect(outputClassOf("signed distances")).toBe("unbounded");
    expect(outputClassOf("probabilities (sigmoid applied)")).toBe("unit");
    expect(outputClassOf("softmax over classes")).toBe("unit");
    expect(outputClassOf("tanh, in [-1, 1]")).toBe("signed_unit");
    expect(outputClassOf("")).toBeNull();
    expect(outputClassOf("something else")).toBeNull();
  });
});
