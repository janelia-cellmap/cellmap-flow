// The shaders and display ranges (src/viewer/display.ts) against what the
// Python they port prints. Every expected string and range below was printed
// with the pixi python from the worktree root:
//
//   from cellmap_flow.viewer.raw import prediction_shader, RAW_SHADER, _invlerp_bounds
//   from cellmap_flow.serving.probe import output_display_range
import { describe, expect, it } from "vitest";

import {
  dtypeRange, formatG6, invlerpBounds, outputDisplayRange, PREDICTION_COLORS, predictionShader, rawShader, shaderColor,
} from "../../src/viewer/display";

describe("formatG6 (Python's .6g)", () => {
  it("rounds to six significant digits and drops trailing zeros", () => {
    expect(formatG6(0.123456789)).toBe("0.123457");
    expect(formatG6(123456.789)).toBe("123457");
    expect(formatG6(255)).toBe("255");
    expect(formatG6(-0.3)).toBe("-0.3");
    expect(formatG6(1.5)).toBe("1.5");
    expect(formatG6(0)).toBe("0");
    expect(formatG6(0.0001)).toBe("0.0001");
  });
  it("uses Python's exponent form below 1e-4 and from 1e6", () => {
    expect(formatG6(1e-7)).toBe("1e-07");
    expect(formatG6(-1e-7)).toBe("-1e-07");
    expect(formatG6(2e6)).toBe("2e+06");
    expect(formatG6(1e-5)).toBe("1e-05");
  });
});

describe("invlerpBounds (_invlerp_bounds)", () => {
  it("passes floats through, six significant digits", () => {
    expect(invlerpBounds(0.123456789, 0.987654321, -0.3, 1.4)).toBe("range=[0.123457, 0.987654], window=[-0.3, 1.4]");
    expect(invlerpBounds(1e-7, 123456.789, -1e-7, 2e6)).toBe("range=[1e-07, 123457], window=[-1e-07, 2e+06]");
    expect(invlerpBounds(0, 0, -1, 1, null)).toBe("range=[0, 0], window=[-1, 1]");
    expect(invlerpBounds(100, 200, 50, 250, "float64")).toBe("range=[100, 200], window=[50, 250]");
  });
  it("widens an integer dtype's range to whole numbers and clips to what it holds", () => {
    expect(invlerpBounds(0.5, 2.5, -200, 300, "int8")).toBe("range=[0, 3], window=[-128, 127]");
    expect(invlerpBounds(127.5, 255, 63.75, 318.75, "uint8")).toBe("range=[127, 255], window=[63, 255]");
    expect(invlerpBounds(0, 255, 0, 255, "uint8")).toBe("range=[0, 255], window=[0, 255]");
  });
});

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
  it("cycles the prediction colours in Python's order", () => {
    expect(PREDICTION_COLORS).toEqual(["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]);
    const colours = Array.from({ length: 10 }, (_, i) => PREDICTION_COLORS[i % PREDICTION_COLORS.length]);
    expect(colours).toEqual(["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "red", "green"]);
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

describe("shaderColor (_COLOR_RE)", () => {
  it("reads the colour control's default", () => {
    expect(shaderColor(predictionShader("cyan", [0, 1]))).toBe("cyan");
    expect(shaderColor('#uicontrol vec3 color color(default="#ff8800");\nvoid main(){}')).toBe("#ff8800");
  });
  it("is null without one", () => {
    expect(shaderColor("void main(){emitGrayscale(normalized());}")).toBeNull();
    expect(shaderColor(null)).toBeNull();
    expect(shaderColor(undefined)).toBeNull();
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
  it("reads the form's strings as numbers, and a value that is none as the whole uint8 range", () => {
    expect(outputDisplayRange([{ name: "DefaultPostprocessor", clip_min: "-1", clip_max: "1", bias: "1", multiplier: "127.5" }], "logits")).toEqual([0, 255]);
    expect(outputDisplayRange([{ name: "DefaultPostprocessor", clip_min: "x" }], "logits")).toEqual([0, 255]);
    expect(outputDisplayRange([{ name: "DefaultPostprocessor", clip_min: "" }], "logits")).toEqual([0, 255]);
  });
  it("is undetermined for an empty or inverted range", () => {
    expect(outputDisplayRange([{ name: "DefaultPostprocessor", clip_min: 0.0, clip_max: 0.0, bias: 0.0, multiplier: 1.0 }], "logits")).toBeNull();
    expect(outputDisplayRange([{ ...defaults, multiplier: -127.5 }], "logits")).toBeNull();
  });
  it("is undetermined after a step whose arithmetic is unknown", () => {
    expect(outputDisplayRange([{ name: "LambdaPostprocessor", expression: "x*2" }], "unit")).toBeNull();
    expect(outputDisplayRange([sigmoid, { name: "LambdaPostprocessor", expression: "x*2" }], "logits")).toBeNull();
    expect(outputDisplayRange([threshold, { name: "LabelPostprocessor" }], "logits")).toBeNull();
  });
});
