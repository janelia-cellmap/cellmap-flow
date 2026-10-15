// The args blob codec and digest against values printed by the Python side
// (cellmap_flow.pipeline_spec.PipelineSpec, serving.protocol) with the pixi
// environment:
//
//   PipelineSpec(DEFAULT.input_norm, DEFAULT.postprocess).to_url_blob()
//   .to_url_blob(dashboard_url="http://localhost:5173/", digest=spec.digest())
//   .digest()  → 26e542790eb63be8
import { describe, expect, it } from "vitest";

import { ARGS_KEY, type Chain } from "../../src/contracts";
import { canonicalJson, chainDigest, decodeArgs, encodeArgs, normalizeSteps, parseViewName, viewName } from "../../src/shared/args";

/** The browser mode's default chain: the Hugging Face models' normalization, logits to probabilities. */
const DEFAULT: Chain = {
  input_norm: [
    { name: "MinMaxNormalizer", min_value: 0, max_value: 255 },
    { name: "LambdaNormalizer", expression: "2*x-1" },
  ],
  postprocess: [{ name: "SigmoidPostprocessor" }],
};
const DEFAULT_BLOB =
  "eyJpbnB1dF9ub3JtIjpbeyJuYW1lIjoiTWluTWF4Tm9ybWFsaXplciIsIm1pbl92YWx1ZSI6MCwibWF4X3ZhbHVlIjoyNTV9LHsibmFtZSI6IkxhbWJkYU5vcm1hbGl6ZXIiLCJleHByZXNzaW9uIjoiMip4LTEifV0sInBvc3Rwcm9jZXNzIjpbeyJuYW1lIjoiU2lnbW9pZFBvc3Rwcm9jZXNzb3IifV19";
const DEFAULT_WITH_EXTRAS_BLOB =
  "eyJpbnB1dF9ub3JtIjpbeyJuYW1lIjoiTWluTWF4Tm9ybWFsaXplciIsIm1pbl92YWx1ZSI6MCwibWF4X3ZhbHVlIjoyNTV9LHsibmFtZSI6IkxhbWJkYU5vcm1hbGl6ZXIiLCJleHByZXNzaW9uIjoiMip4LTEifV0sInBvc3Rwcm9jZXNzIjpbeyJuYW1lIjoiU2lnbW9pZFBvc3Rwcm9jZXNzb3IifV0sImRhc2hib2FyZF91cmwiOiJodHRwOi8vbG9jYWxob3N0OjUxNzMvIiwiZGlnZXN0IjoiMjZlNTQyNzkwZWI2M2JlOCJ9";

/** As the dashboard's forms send a chain: every value a string. */
const FORM_CHAIN: Chain = {
  input_norm: [{ name: "MinMaxNormalizer", min_value: "0", max_value: "255", invert: "true" }],
  postprocess: [
    { name: "DefaultPostprocessor", clip_min: "-1", clip_max: "1", bias: "1", multiplier: "127.5" },
    { name: "ChannelSelection", channels: "0,2" },
  ],
};
const FORM_BLOB =
  "eyJpbnB1dF9ub3JtIjpbeyJuYW1lIjoiTWluTWF4Tm9ybWFsaXplciIsIm1pbl92YWx1ZSI6IjAiLCJtYXhfdmFsdWUiOiIyNTUiLCJpbnZlcnQiOiJ0cnVlIn1dLCJwb3N0cHJvY2VzcyI6W3sibmFtZSI6IkRlZmF1bHRQb3N0cHJvY2Vzc29yIiwiY2xpcF9taW4iOiItMSIsImNsaXBfbWF4IjoiMSIsImJpYXMiOiIxIiwibXVsdGlwbGllciI6IjEyNy41In0seyJuYW1lIjoiQ2hhbm5lbFNlbGVjdGlvbiIsImNoYW5uZWxzIjoiMCwyIn1dfQ";

describe("encodeArgs", () => {
  it("writes the Python blob for the default chain", () => {
    expect(encodeArgs(DEFAULT)).toBe(DEFAULT_BLOB);
  });

  it("appends extras after the two chains, in the order given", () => {
    expect(encodeArgs(DEFAULT, { dashboard_url: "http://localhost:5173/", digest: "26e542790eb63be8" })).toBe(DEFAULT_WITH_EXTRAS_BLOB);
  });

  it("keeps string values as the forms send them", () => {
    expect(encodeArgs(FORM_CHAIN)).toBe(FORM_BLOB);
  });

  it("is URL safe without padding", () => {
    for (const blob of [encodeArgs(DEFAULT), encodeArgs(FORM_CHAIN), encodeArgs({ input_norm: [], postprocess: [] })]) {
      expect(blob).toMatch(/^[A-Za-z0-9_-]+$/);
    }
  });

  it("refuses extras that would replace a chain", () => {
    expect(() => encodeArgs(DEFAULT, { postprocess: [] })).toThrow(/may not replace a chain: postprocess/);
  });
});

describe("decodeArgs", () => {
  it("round-trips the chain and separates the extras", () => {
    const { chain, extras } = decodeArgs(DEFAULT_WITH_EXTRAS_BLOB);
    expect(chain).toEqual(DEFAULT);
    expect(extras).toEqual({ dashboard_url: "http://localhost:5173/", digest: "26e542790eb63be8" });
  });

  it("reads the legacy {Name: {params}} chain form, the key naming the class", () => {
    // encode_to_str of the dict form, as an older dashboard wrote it
    const legacy = btoa(JSON.stringify({
      input_norm: { MinMaxNormalizer: { min_value: 0, max_value: 255, name: "ignored" } },
      postprocess: { ThresholdPostprocessor: { threshold: 0.5 } },
      dashboard_url: "x",
    })).replace(/=+$/, "");
    const { chain, extras } = decodeArgs(legacy);
    expect(chain).toEqual({
      input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }],
      postprocess: [{ name: "ThresholdPostprocessor", threshold: 0.5 }],
    });
    expect(extras).toEqual({ dashboard_url: "x" });
  });

  it("treats a missing or null chain as empty", () => {
    const blob = btoa(JSON.stringify({ input_norm: null })).replace(/=+$/, "");
    expect(decodeArgs(blob).chain).toEqual({ input_norm: [], postprocess: [] });
  });

  it("accepts a blob with its padding", () => {
    expect(decodeArgs(`${DEFAULT_BLOB}=`).chain).toEqual(DEFAULT);
  });

  it("rejects a blob that is not an object", () => {
    expect(() => decodeArgs(btoa("[1,2]"))).toThrow(/not a JSON object/);
  });
});

describe("normalizeSteps", () => {
  it("copies list steps and drops elements that are not objects", () => {
    const steps = [{ name: "A", p: 1 }, "junk", null, 3];
    const out = normalizeSteps(steps);
    expect(out).toEqual([{ name: "A", p: 1 }]);
    expect(out[0]).not.toBe(steps[0]);
  });

  it("rejects a chain that is neither", () => {
    expect(() => normalizeSteps("MinMaxNormalizer")).toThrow(/list or an object/);
  });
});

describe("chainDigest", () => {
  it("matches PipelineSpec.digest() for the default chain", async () => {
    expect(await chainDigest(DEFAULT)).toBe("26e542790eb63be8");
  });

  it("matches for a form-valued chain", async () => {
    expect(await chainDigest(FORM_CHAIN)).toBe("2ed2fdb5b97206a8");
  });

  it("matches for an empty chain", async () => {
    expect(await chainDigest({ input_norm: [], postprocess: [] })).toBe("4d687d4743088632");
  });

  it("matches for numeric values that render alike in both languages", async () => {
    expect(await chainDigest({ input_norm: [{ name: "ZScoreNormalizer", mean: 0.5, std: 2 }], postprocess: [] })).toBe("97d0a9741afeb118");
  });

  it("ignores the order of a step's parameters but not of the steps", async () => {
    const a = await chainDigest({ input_norm: [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255 }], postprocess: [] });
    const b = await chainDigest({ input_norm: [{ max_value: 255, name: "MinMaxNormalizer", min_value: 0 }], postprocess: [] });
    const c = await chainDigest({ input_norm: [...DEFAULT.input_norm].reverse(), postprocess: DEFAULT.postprocess });
    expect(a).toBe(b);
    expect(c).not.toBe(await chainDigest(DEFAULT));
  });

  it("sorts keys at every depth, like json.dumps(sort_keys=True)", () => {
    expect(canonicalJson({ b: [{ z: 1, a: null }], a: "x" })).toBe('{"a":"x","b":[{"a":null,"z":1}]}');
  });
});

describe("view names", () => {
  it("wraps the blob in the markers", () => {
    expect(viewName("setup_54", "abc")).toBe(`setup_54${ARGS_KEY}abc${ARGS_KEY}`);
  });

  it("parses a name with and without a blob", () => {
    expect(parseViewName(viewName("setup_54", DEFAULT_BLOB))).toEqual({ name: "setup_54", blob: DEFAULT_BLOB });
    expect(parseViewName("setup_54")).toEqual({ name: "setup_54", blob: null });
  });

  it("percent-decodes the model name", () => {
    expect(parseViewName(`my%20model${ARGS_KEY}x${ARGS_KEY}`).name).toBe("my model");
  });

  it("rejects any number of markers other than two, as Python does", () => {
    expect(() => parseViewName(`a${ARGS_KEY}b`)).toThrow(/two occurrences/);
    expect(() => parseViewName(`a${ARGS_KEY}b${ARGS_KEY}c${ARGS_KEY}`)).toThrow(/two occurrences/);
  });
});
