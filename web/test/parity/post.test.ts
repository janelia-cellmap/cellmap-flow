// Parity of web/src/infer/post.ts with cellmap_flow/post/postprocessors.py
// and the chain summaries in cellmap_flow/pipeline_spec.py.
//
// The fixture (test/fixtures/post.json) was written by the Python classes
// themselves (test/fixtures/make_fixtures.py): one float32 block of logits,
// channel-first [C, z, y, x] as the server hands a model's output to the
// chain, and for each case the steps, the array the Python returned (with
// its dtype: the uint8 steps truncate toward zero and wrap), and what
// chain_output_dtype / chain_num_channels / chain_is_segmentation say of the
// chain. Regenerate the fixture when the Python changes;
// tests/web/test_fixtures_fresh.py fails when it is stale. What JSON cannot
// hold (a NaN) or the block is too small for (ties, past 255 channels) is
// held to what tests/utils/test_postprocessors.py says of the Python, on the
// same block.
import { describe, expect, it } from "vitest";

import type { Step } from "../../src/contracts";
import { SUPPORTED_POSTPROCESSORS, applyPostprocessors, isSegmentation, outputChannels, outputDtype } from "../../src/infer/post";
import { type ArrayRecord, dtypeOf, expectArrayMatches, loadFixture, supports } from "./helpers";

interface PostCase {
  id: string;
  steps: Step[];
  output: ArrayRecord;
  chain_output_dtype: string;
  chain_num_channels: number;
  /** true/false, or null when the chain has steps but none says. */
  chain_is_segmentation: boolean | null;
}

interface PostFixture {
  seed: number;
  model_dtype: string;
  model_channels: number;
  input: ArrayRecord;
  cases: PostCase[];
  /** Chains only declared, with the model's channels not known. */
  declared: { id: string; steps: Step[]; chain_output_dtype: string }[];
}

const fixture = loadFixture<PostFixture>("post.json");

function input() {
  return { data: new Float32Array(fixture.input.data), shape: [...fixture.input.shape] };
}

describe("applyPostprocessors matches the Python postprocessors", () => {
  it("has a channel-first float32 input block to work on", () => {
    expect(fixture.input.dtype).toBe("float32");
    expect(fixture.input.shape).toHaveLength(4);
    expect(fixture.input.shape[0]).toBe(fixture.model_channels);
  });

  for (const c of fixture.cases) {
    it(c.id, () => {
      const out = applyPostprocessors(input(), c.steps);
      expect(out.shape, `${c.id}: shape`).toEqual(c.output.shape);
      expect(dtypeOf(out.data), `${c.id}: dtype`).toBe(c.output.dtype);
      expectArrayMatches(out.data, c.output, c.id);
    });
  }
});

describe("the chain summaries match pipeline_spec.py", () => {
  for (const c of fixture.cases) {
    it(`${c.id}: dtype ${c.chain_output_dtype}, ${c.chain_num_channels} channels, segmentation ${c.chain_is_segmentation}`, () => {
      expect(outputDtype(c.steps, fixture.model_dtype, fixture.model_channels), "outputDtype").toBe(c.chain_output_dtype);
      expect(outputChannels(c.steps, fixture.model_channels), "outputChannels").toBe(c.chain_num_channels);
      // The Python says None when steps are present but none declares; the
      // port may say null or undefined for that.
      expect(isSegmentation(c.steps) ?? null, "isSegmentation").toBe(c.chain_is_segmentation);
    });
  }

  for (const c of fixture.declared) {
    it(`${c.id}, the model's channels not known: dtype ${c.chain_output_dtype}`, () => {
      expect(outputDtype(c.steps, fixture.model_dtype), "outputDtype").toBe(c.chain_output_dtype);
    });
  }

  it("lists every postprocessor the fixtures use as supported", () => {
    const names = new Set(fixture.cases.flatMap((c) => c.steps.map((s) => s.name)));
    for (const name of names) {
      expect(supports(SUPPORTED_POSTPROCESSORS, name), `${name} in SUPPORTED_POSTPROCESSORS`).toBe(true);
    }
  });
});

describe("ArgmaxPostprocessor gives the Python's ids where the fixture cannot show them", () => {
  // test_argmax_gives_each_voxel_its_best_channel_plus_one's block: one
  // voxel per row, its scores in channels 0, 1, 2.
  const voxels = [
    [0.1, 0.7, 0.2],  // channel 1 is best: id 2
    [0.6, 0.6, 0.1],  // a tie: the lower channel, id 1
    [0.2, 0.3, 0.35],  // best below the threshold
    [NaN, 0.1, 0.9],  // the NaN does not win
    [NaN, NaN, NaN],  // nothing to pick: background
  ];
  const scores = { data: Float32Array.from([0, 1, 2].flatMap((ch) => voxels.map((v) => v[ch]))), shape: [3, 1, 1, voxels.length] };
  const ids = (steps: Step[], nd = scores) => {
    const out = applyPostprocessors({ data: nd.data.slice(), shape: [...nd.shape] }, steps);
    return { dtype: dtypeOf(out.data), shape: out.shape, ids: Array.from(out.data, Number) };
  };

  it("ties to the lowest channel, never picks a NaN, and gives 0 below the threshold", () => {
    expect(ids([{ name: "ArgmaxPostprocessor" }])).toEqual({ dtype: "uint8", shape: [1, 1, 1, 5], ids: [2, 1, 3, 3, 0] });
    expect(ids([{ name: "ArgmaxPostprocessor", threshold: "None" }]).ids).toEqual([2, 1, 3, 3, 0]);
    expect(ids([{ name: "ArgmaxPostprocessor", threshold: "0.5" }]).ids).toEqual([2, 1, 0, 3, 0]);
    // Compared in float64, as the Python compares: float32's 0.7 is below 0.7.
    expect(ids([{ name: "ArgmaxPostprocessor", threshold: 0.7 }], { data: new Float32Array([0.7, 0.7]), shape: [2, 1, 1, 1] }).ids).toEqual([0]);
  });

  it("gives each channel the id it is given, 0 making it background, in the smallest dtype that holds them", () => {
    for (const given of ["50,0,37", "50 0 37", " 50, 0  37 "]) {
      expect(ids([{ name: "ArgmaxPostprocessor", ids: given }])).toEqual({ dtype: "uint8", shape: [1, 1, 1, 5], ids: [0, 50, 37, 37, 0] });
    }
    expect(ids([{ name: "ArgmaxPostprocessor", ids: "None" }]).ids).toEqual([2, 1, 3, 3, 0]);
    expect(ids([{ name: "ArgmaxPostprocessor", ids: "300,1,2" }]).dtype).toBe("uint16");
    expect(ids([{ name: "ArgmaxPostprocessor", ids: "70000,1,2" }]).dtype).toBe("uint32");
    expect(outputDtype([{ name: "ArgmaxPostprocessor", ids: "2,1,300" }], "float32")).toBe("uint16");
    expect(() => ids([{ name: "ArgmaxPostprocessor", ids: "1,2" }])).toThrow(/2\D+3/);  // says both counts
    for (const refused of ["1,-2", "1,2.5"]) expect(() => ids([{ name: "ArgmaxPostprocessor", ids: refused }])).toThrow();
  });

  it("gives uint16 ids past 255 channels, and declares them", () => {
    const many = { data: new Float32Array(300 * 2), shape: [300, 1, 1, 2] };
    many.data[299 * 2] = 1;
    expect(ids([{ name: "ArgmaxPostprocessor" }], many)).toEqual({ dtype: "uint16", shape: [1, 1, 1, 2], ids: [300, 1] });
    expect(outputDtype([{ name: "ArgmaxPostprocessor" }], "float32", 300)).toBe("uint16");
  });
});
