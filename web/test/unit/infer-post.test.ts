// Postprocessors against values printed by the Python classes
// (cellmap_flow.post.postprocessors) and pipeline_spec's chain_output_dtype /
// chain_num_channels / chain_is_segmentation on the pixi environment:
//
//   y = np.array([-3, -1, -0.5, 0, 0.5, 1, 3, 10], np.float32).reshape(1, 1, 2, 4)
//   z = np.arange(24, np.float32).reshape(3, 2, 2, 2)
import { describe, expect, it } from "vitest";

import type { Step } from "../../src/contracts";
import {
  type NdArray, SUPPORTED_POSTPROCESSORS, applyPostprocessors, isSegmentation, outputChannels, outputDtype, parseChannels, validatePostprocessors,
} from "../../src/infer/post";

const Y = [-3, -1, -0.5, 0, 0.5, 1, 3, 10];
const y = (): NdArray => ({ data: new Float32Array(Y), shape: [1, 1, 2, 4] });
const z = (): NdArray => ({ data: Float32Array.from({ length: 24 }, (_, i) => i), shape: [3, 2, 2, 2] });
const run = (nd: NdArray, steps: Step[]) => applyPostprocessors(nd, steps);
const values = (nd: NdArray) => Array.from(nd.data);

describe("SigmoidPostprocessor", () => {
  it("turns logits into probabilities, float32", () => {
    const out = run(y(), [{ name: "SigmoidPostprocessor" }]);
    expect(out.data).toBeInstanceOf(Float32Array);
    expect(out.shape).toEqual([1, 1, 2, 4]);
    const want = [0.04742587357759476, 0.26894140243530273, 0.3775406777858734, 0.5, 0.622459352016449, 0.7310585975646973, 0.9525741338729858, 0.9999545812606812];
    values(out).forEach((v, i) => expect(v).toBeCloseTo(want[i], 6));
  });
});

describe("DefaultPostprocessor", () => {
  it("maps [-1, 1] onto uint8 by default", () => {
    const out = run(y(), [{ name: "DefaultPostprocessor" }]);
    expect(out.data).toBeInstanceOf(Uint8Array);
    expect(values(out)).toEqual([0, 0, 63, 127, 191, 255, 255, 255]);
  });

  it("takes the forms' strings: [0, 1] * 255 on probabilities", () => {
    const out = run(y(), [{ name: "SigmoidPostprocessor" }, { name: "DefaultPostprocessor", clip_min: "0", clip_max: "1", bias: "0", multiplier: "255" }]);
    expect(values(out)).toEqual([12, 68, 96, 127, 158, 186, 242, 254]);
  });

  it("truncates and wraps like astype(uint8)", () => {
    const w = (): NdArray => ({ data: new Float32Array([-1.0, -0.999, 0.996, 1.0, 1.5]), shape: [1, 1, 1, 5] });
    expect(values(run(w(), [{ name: "DefaultPostprocessor" }]))).toEqual([0, 0, 254, 255, 255]);
    expect(values(run(w(), [{ name: "DefaultPostprocessor", clip_min: -1, clip_max: 1, bias: 1.5, multiplier: 127.5 }]))).toEqual([63, 63, 62, 62, 62]);
  });
});

describe("ThresholdPostprocessor", () => {
  it("is strictly greater than the threshold, 0.5 by default, uint8", () => {
    const out = run(y(), [{ name: "ThresholdPostprocessor" }]);
    expect(out.data).toBeInstanceOf(Uint8Array);
    expect(values(out)).toEqual([0, 0, 0, 0, 0, 1, 1, 1]);
    expect(values(run(y(), [{ name: "ThresholdPostprocessor", threshold: "0" }]))).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  });

  it("thresholds uint8 input too", () => {
    expect(values(run(y(), [{ name: "DefaultPostprocessor" }, { name: "ThresholdPostprocessor", threshold: 127 }]))).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  });
});

describe("ChannelSelection", () => {
  it("picks the listed channels in the listed order, keeping the dtype", () => {
    const out = run(z(), [{ name: "ChannelSelection", channels: "2,0" }]);
    expect(out.shape).toEqual([2, 2, 2, 2]);
    expect(out.data).toBeInstanceOf(Float32Array);
    expect(values(out)).toEqual([16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it("takes one channel as a string or a number", () => {
    expect(values(run(z(), [{ name: "ChannelSelection", channels: "1" }]))).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
    expect(values(run(z(), [{ name: "ChannelSelection", channels: 2 }]))).toEqual([16, 17, 18, 19, 20, 21, 22, 23]);
    expect(run(z(), [{ name: "ChannelSelection" }]).shape).toEqual([1, 2, 2, 2]);
  });

  it("keeps a uint8 input uint8", () => {
    expect(run(z(), [{ name: "DefaultPostprocessor" }, { name: "ChannelSelection", channels: "0" }]).data).toBeInstanceOf(Uint8Array);
  });

  it("parses channel lists", () => {
    expect(parseChannels("0, 2")).toEqual([0, 2]);
    expect(parseChannels([1, "2"])).toEqual([1, 2]);
    expect(parseChannels(undefined)).toEqual([0]);
    expect(() => parseChannels("a")).toThrow(/not a list of channel indices/);
  });

  it("refuses a channel the output does not have", () => {
    expect(() => run(z(), [{ name: "ChannelSelection", channels: "3" }])).toThrow(/channel 3 of an output with 3/);
  });
});

describe("LambdaPostprocessor", () => {
  it("evaluates on float32 and returns float32", () => {
    const out = run(y(), [{ name: "DefaultPostprocessor" }, { name: "LambdaPostprocessor", expression: "x*2+1" }]);
    expect(out.data).toBeInstanceOf(Float32Array);
    expect(values(out)).toEqual([1, 1, 127, 255, 383, 511, 511, 511]);
    expect(values(run(y(), [{ name: "LambdaPostprocessor", expression: "np.clip(x, 0, 1)" }]))).toEqual([0, 0, 0, 0, 0.5, 1, 1, 1]);
  });
});

describe("applyPostprocessors", () => {
  it("returns the input untouched with no steps", () => {
    const nd = y();
    expect(run(nd, [])).toBe(nd);
  });

  it("names an unsupported postprocessor, at validation and when applied", () => {
    expect(() => validatePostprocessors([{ name: "LabelPostprocessor" }])).toThrow(/"LabelPostprocessor" is not available in the browser/);
    expect(() => run(y(), [{ name: "AffinityPostprocessor" }])).toThrow(/AffinityPostprocessor/);
    expect(SUPPORTED_POSTPROCESSORS).toContain("SigmoidPostprocessor");
  });
});

describe("what a chain produces (pipeline_spec.py's rule: the last step that declares wins)", () => {
  const sigmoid = { name: "SigmoidPostprocessor" };
  const dflt = { name: "DefaultPostprocessor" };
  const threshold = { name: "ThresholdPostprocessor" };
  const select = { name: "ChannelSelection", channels: "0,2" };
  const lambda = { name: "LambdaPostprocessor", expression: "x*2" };

  it("outputDtype", () => {
    expect(outputDtype([], "float32")).toBe("float32");
    expect(outputDtype([sigmoid])).toBe("float32");
    expect(outputDtype([dflt])).toBe("uint8");
    expect(outputDtype([threshold])).toBe("uint8");
    expect(outputDtype([select])).toBe("float32");
    expect(outputDtype([threshold, lambda])).toBe("float32");
    expect(outputDtype([dflt, { name: "ChannelSelection", channels: "1" }])).toBe("uint8");
    expect(outputDtype([select], "uint16")).toBe("uint16");
    expect(outputDtype([{ name: "SomethingServerOnly" }], "float32")).toBe("float32");  // declares nothing here
  });

  it("outputChannels", () => {
    expect(outputChannels([], 3)).toBe(3);
    expect(outputChannels([sigmoid], 3)).toBe(3);
    expect(outputChannels([select], 3)).toBe(2);
    expect(outputChannels([dflt, { name: "ChannelSelection", channels: "1" }], 3)).toBe(1);
  });

  it("isSegmentation", () => {
    expect(isSegmentation([])).toBe(false);
    expect(isSegmentation([sigmoid])).toBe(null);
    expect(isSegmentation([dflt])).toBe(false);
    expect(isSegmentation([threshold])).toBe(true);
    expect(isSegmentation([select])).toBe(null);
    expect(isSegmentation([threshold, lambda])).toBe(true);
    expect(isSegmentation([sigmoid, { name: "ThresholdPostprocessor", threshold: "0.5" }, { name: "ChannelSelection", channels: "1" }])).toBe(true);
  });
});
