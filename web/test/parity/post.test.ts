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
// tests/web/test_fixtures_fresh.py fails when it is stale.
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
      expect(outputDtype(c.steps, fixture.model_dtype), "outputDtype").toBe(c.chain_output_dtype);
      expect(outputChannels(c.steps, fixture.model_channels), "outputChannels").toBe(c.chain_num_channels);
      // The Python says None when steps are present but none declares; the
      // port may say null or undefined for that.
      expect(isSegmentation(c.steps) ?? null, "isSegmentation").toBe(c.chain_is_segmentation);
    });
  }

  it("lists every postprocessor the fixtures use as supported", () => {
    const names = new Set(fixture.cases.flatMap((c) => c.steps.map((s) => s.name)));
    for (const name of names) {
      expect(supports(SUPPORTED_POSTPROCESSORS, name), `${name} in SUPPORTED_POSTPROCESSORS`).toBe(true);
    }
  });
});
