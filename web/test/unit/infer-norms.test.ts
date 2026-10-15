// Normalizers against values printed by the Python classes
// (cellmap_flow.norm.input_normalize) on the pixi environment:
//
//   x = np.array([0, 1, 127, 128, 254, 255, 300, -5], dtype=np.float32)
//   MinMaxNormalizer()(x), MinMaxNormalizer("0", "255", "true")(x),
//   MinMaxNormalizer(10, 200)(x), ZScoreNormalizer("100", "50")(x),
//   LambdaNormalizer("2*x-1")(MinMaxNormalizer()(x)), LambdaNormalizer("x/255")(x)
import { describe, expect, it } from "vitest";

import { SUPPORTED_NORMALIZERS, applyNormalizers, asBool, asNumber, validateNormalizers } from "../../src/infer/norms";

const X = [0, 1, 127, 128, 254, 255, 300, -5];
const run = (steps: Parameters<typeof applyNormalizers>[1], x = X) => Array.from(applyNormalizers(new Float32Array(x), steps));

/** Equal as float32 values, within one ulp of double rounding differences. */
function expectClose(got: number[], want: number[]) {
  expect(got.length).toBe(want.length);
  got.forEach((g, i) => expect(g).toBeCloseTo(want[i], 6));
}

describe("MinMaxNormalizer", () => {
  it("clips to [0, 255] and scales to [0, 1] by default", () => {
    expectClose(run([{ name: "MinMaxNormalizer" }]), [0.0, 0.003921568859368563, 0.49803921580314636, 0.501960813999176, 0.9960784316062927, 1.0, 1.0, 0.0]);
  });

  it("parses the forms' strings and inverts", () => {
    expectClose(run([{ name: "MinMaxNormalizer", min_value: "0", max_value: "255", invert: "true" }]),
      [1.0, 0.9960784316062927, 0.5019607543945312, 0.498039186000824, 0.003921568393707275, 0.0, 0.0, 1.0]);
  });

  it("takes other bounds", () => {
    expectClose(run([{ name: "MinMaxNormalizer", min_value: 10, max_value: 200 }]), [0.0, 0.0, 0.6157894730567932, 0.621052622795105, 1.0, 1.0, 1.0, 0.0]);
  });

  it("reads invert as the Python constructor does", () => {
    // a string: lower-cased "true" and nothing else (bool("False") would be True)
    expect(asBool("False", false)).toBe(false);
    expect(asBool("True", false)).toBe(true);
    expect(asBool("TRUE", false)).toBe(true);
    expect(asBool("yes", false)).toBe(false);
    // a real bool: bool(value)
    expect(asBool(true, false)).toBe(true);
    expect(asBool(false, true)).toBe(false);
    // not given: the default
    expect(asBool(undefined, false)).toBe(false);
    expect(run([{ name: "MinMaxNormalizer", invert: "False" }], [255])).toEqual([1]);
    expect(run([{ name: "MinMaxNormalizer", invert: true }], [255])).toEqual([0]);
  });

  it("reads numbers as float() does", () => {
    expect(asNumber("255.0", 0, "p")).toBe(255);
    expect(asNumber("-1", 0, "p")).toBe(-1);
    expect(asNumber(undefined, 7, "p")).toBe(7);
    expect(asNumber("", 7, "p")).toBe(7);
    expect(() => asNumber("abc", 0, "MinMaxNormalizer.min_value")).toThrow(/MinMaxNormalizer.min_value must be a number/);
  });
});

describe("ZScoreNormalizer", () => {
  it("subtracts the mean and divides by the std, given as strings", () => {
    expectClose(run([{ name: "ZScoreNormalizer", mean: "100", std: "50" }]),
      [-2.0, -1.9800000190734863, 0.5400000214576721, 0.5600000023841858, 3.0799999237060547, 3.0999999046325684, 4.0, -2.0999999046325684]);
  });

  it("defaults to mean 0, std 1", () => {
    expect(run([{ name: "ZScoreNormalizer" }], [3, -2])).toEqual([3, -2]);
  });
});

describe("LambdaNormalizer", () => {
  it("evaluates the models' 2*x-1 after MinMax", () => {
    expectClose(run([{ name: "MinMaxNormalizer" }, { name: "LambdaNormalizer", expression: "2*x-1" }]),
      [-1.0, -0.9921568632125854, -0.003921568393707275, 0.003921627998352051, 0.9921568632125854, 1.0, 1.0, -1.0]);
  });

  it("evaluates x/255 without clipping", () => {
    expectClose(run([{ name: "LambdaNormalizer", expression: "x/255" }]),
      [0.0, 0.003921568859368563, 0.49803921580314636, 0.501960813999176, 0.9960784316062927, 1.0, 1.1764706373214722, -0.019607843831181526]);
  });

  it("evaluates x**2", () => {
    expect(run([{ name: "LambdaNormalizer", expression: "x**2" }], [0.5, -2])).toEqual([0.25, 4]);
  });

  it("rejects an expression outside the whitelist when the chain is validated", () => {
    expect(() => validateNormalizers([{ name: "LambdaNormalizer", expression: "__import__('os')" }])).toThrow(/not allowed/);
  });
});

describe("applyNormalizers", () => {
  it("works in place and returns the same array", () => {
    const data = new Float32Array([255]);
    expect(applyNormalizers(data, [{ name: "MinMaxNormalizer" }])).toBe(data);
    expect(data[0]).toBe(1);
  });

  it("leaves the data alone with no steps", () => {
    expect(run([], [1, 2])).toEqual([1, 2]);
  });

  it("names an unsupported normalizer", () => {
    expect(() => run([{ name: "EuclideanDistance" }])).toThrow(/"EuclideanDistance" is not available in the browser/);
    expect(SUPPORTED_NORMALIZERS).toEqual(["MinMaxNormalizer", "ZScoreNormalizer", "LambdaNormalizer"]);
  });
});
