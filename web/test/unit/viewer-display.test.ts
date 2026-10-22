// The shaders and display ranges (src/viewer/display.ts) against what the
// Python they port prints. Every expected string and range below was printed
// with the pixi python from the worktree root:
//
//   from cellmap_flow.viewer.raw import prediction_shader, RAW_SHADER
//   from cellmap_flow.serving.probe import output_display_range
import { describe, expect, it } from "vitest";

import { dtypeRange, outputDisplayRange, predictionShader, rawShader } from "../../src/viewer/display";

describe("predictionShader (prediction_shader)", () => {
  const shader = (bounds: string, color: string) =>
    `#uicontrol invlerp normalized(${bounds});\n#uicontrol vec3 color color(default="${color}");\nvoid main(){emitRGB(color * normalized());}`;

  it("falls back to [0, 1], padded half a range each way, without a range", () => {
    expect(predictionShader("red")).toBe(shader("range=[0, 1], window=[-0.5, 1.5]", "red"));
    expect(predictionShader("red", null, null)).toBe(shader("range=[0, 1], window=[-0.5, 1.5]", "red"));
  });
  it("takes a float range as it is", () => {
    expect(predictionShader("green", [0, 1])).toBe(shader("range=[0, 1], window=[-0.5, 1.5]", "green"));
    expect(predictionShader("red", [-1, 1])).toBe(shader("range=[-1, 1], window=[-2, 2]", "red"));
    expect(predictionShader("red", [0, 1], "float32")).toBe(shader("range=[0, 1], window=[-0.5, 1.5]", "red"));
  });
  it("keeps a uint8 output's bounds within 0..255 and whole", () => {
    expect(predictionShader("blue", [0, 255], "uint8")).toBe(shader("range=[0, 255], window=[0, 255]", "blue"));
    // After a sigmoid then a DefaultPostprocessor: the output starts at 127.5.
    expect(predictionShader("red", [127.5, 255], "uint8")).toBe(shader("range=[127, 255], window=[63, 255]", "red"));
  });
});

describe("rawShader (RAW_SHADER over the dtype's range)", () => {
  const shader = (bounds: string) =>
    `#uicontrol invlerp normalized(${bounds});\n#uicontrol vec3 color color(default="white");\nvoid main(){emitRGB(color * normalized());}`;

  it("spans an integer dtype", () => {
    expect(rawShader("uint8")).toBe(shader("range=[0, 255], window=[0, 255]"));
    expect(rawShader("uint16")).toBe(shader("range=[0, 65535], window=[0, 65535]"));
    expect(dtypeRange("int8")).toEqual([-128, 127]);
  });
  it("spans [0, 1] for a float", () => {
    expect(rawShader("float32")).toBe(shader("range=[0, 1], window=[0, 1]"));
  });
});

describe("outputDisplayRange (probe.output_display_range)", () => {
  const sigmoid = { name: "SigmoidPostprocessor" };
  const threshold = { name: "ThresholdPostprocessor", threshold: 0.5 };
  const defaults = { name: "DefaultPostprocessor", clip_min: -1.0, clip_max: 1.0, bias: 1.0, multiplier: 127.5 };

  it("starts from the model's output class", () => {
    expect(outputDisplayRange([], "logits")).toBeNull();
    expect(outputDisplayRange([], "unbounded")).toBeNull();
    expect(outputDisplayRange([], null)).toBeNull();
    expect(outputDisplayRange([], undefined)).toBeNull();
    expect(outputDisplayRange([], "unit")).toEqual([0, 1]);
    expect(outputDisplayRange([], "signed_unit")).toEqual([-1, 1]);
  });
  it("is [0, 1] after a sigmoid or a threshold", () => {
    expect(outputDisplayRange([sigmoid], "logits")).toEqual([0, 1]);
    expect(outputDisplayRange([threshold], "logits")).toEqual([0, 1]);
    expect(outputDisplayRange([sigmoid, threshold], "signed_unit")).toEqual([0, 1]);
    expect(outputDisplayRange([sigmoid, { name: "ChannelSelection", channels: "0" }], "logits")).toEqual([0, 1]);
  });
  it("lands a DefaultPostprocessor in 0..255, clipped against what arrives", () => {
    expect(outputDisplayRange([defaults], "logits")).toEqual([0, 255]);
    expect(outputDisplayRange([{ name: "DefaultPostprocessor" }], "logits")).toEqual([0, 255]);
    // After a sigmoid, clip_min=-1 is never reached: the output starts at 127.5.
    expect(outputDisplayRange([sigmoid, defaults], "logits")).toEqual([127.5, 255]);
    expect(outputDisplayRange([{ name: "DefaultPostprocessor", clip_min: 0.0, clip_max: 1.0, bias: 0.0, multiplier: 255.0 }], "unit")).toEqual([0, 255]);
    // Swapped clips, or a multiplier past 255, still give 0..255.
    expect(outputDisplayRange([{ ...defaults, clip_min: 1.0, clip_max: -1.0 }], "logits")).toEqual([0, 255]);
    expect(outputDisplayRange([{ ...defaults, multiplier: 1000.0 }], "logits")).toEqual([0, 255]);
  });
});
