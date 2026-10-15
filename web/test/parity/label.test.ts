// Parity of web/src/infer/label.ts with LabelPostprocessor in
// cellmap_flow/post/postprocessors.py (segment.connected_components:
// scipy.ndimage.label, per slice when asked, then remove_small).
//
// The fixture (test/fixtures/label.json) was written by the Python class
// itself (test/fixtures/make_segment_fixtures.py): one uint8 block of masks,
// channel first [C, z, y, x], and for each case the step's parameters and
// the uint32 block the Python returned. Ids are compared exactly: scipy
// numbers objects by the raster order of their first voxel, and so must
// the port. tests/web/test_fixtures_fresh.py fails when the fixture is stale.
import { describe, expect, it } from "vitest";

import type { Step } from "../../src/contracts";
import { applyPostprocessors, isSegmentation, outputDtype } from "../../src/infer/post";
import { type ArrayRecord, dtypeOf, expectArrayMatches, loadFixture } from "./helpers";

interface LabelCase {
  id: string;
  step: Step;
  output: ArrayRecord;
  /** How many objects the labelled channel has (its largest id). */
  num_objects: number;
}

interface LabelFixture {
  seed: number;
  input: ArrayRecord;
  cases: LabelCase[];
}

const fixture = loadFixture<LabelFixture>("label.json");

function input() {
  return { data: new Uint8Array(fixture.input.data), shape: [...fixture.input.shape] };
}

function channelOf(step: Step): number {
  return Number(step.channel ?? 0);
}

describe("LabelPostprocessor matches scipy's labels", () => {
  it("has a uint8 (c, z, y, x) block of masks with two channels to work on", () => {
    expect(fixture.input.dtype).toBe("uint8");
    expect(fixture.input.shape).toHaveLength(4);
    expect(fixture.input.shape[0]).toBe(2);
  });

  for (const c of fixture.cases) {
    it(c.id, () => {
      const out = applyPostprocessors(input(), [c.step]);
      expect(out.shape, `${c.id}: shape`).toEqual(c.output.shape);
      expect(dtypeOf(out.data), `${c.id}: dtype`).toBe("uint32");
      expect(c.output.dtype).toBe("uint32");
      expectArrayMatches(out.data, c.output, c.id);
      // The ids run 1..n with none skipped.
      const [, z, y, x] = out.shape;
      const plane = z * y * x;
      const labels = out.data.subarray(channelOf(c.step) * plane, (channelOf(c.step) + 1) * plane) as Uint32Array;
      const ids = new Set<number>();
      for (const v of labels) if (v) ids.add(v);
      expect(ids.size, `${c.id}: objects`).toBe(c.num_objects);
      expect(Math.max(0, ...ids), `${c.id}: largest id`).toBe(c.num_objects);
    });
  }

  it("the fixture tells the parameters apart", () => {
    const byId = Object.fromEntries(fixture.cases.map((c) => [c.id, c]));
    expect(byId.conn_1.output.data).not.toEqual(byId.conn_2.output.data);
    expect(byId.conn_2.output.data).not.toEqual(byId.conn_3.output.data);
    expect(byId.min_size_4.num_objects).toBeLessThan(byId.conn_1.num_objects);
    expect(byId.per_slice.output.data).not.toEqual(byId.conn_1.output.data);
    // The forms' strings parse to the same step as the numbers.
    expect(byId.conn_2_string.output.data).toEqual(byId.conn_2.output.data);
    expect(byId.per_slice_false_string.output.data).toEqual(byId.defaults.output.data);
    // Connectivity 3 in a 2-D slice is connectivity 2.
    expect(byId.per_slice_conn_3_capped.output.data).toEqual(byId.per_slice_string_conn_2.output.data);
  });

  it("declares what the Python class declares", () => {
    const step: Step = { name: "LabelPostprocessor" };
    expect(outputDtype([step], "float32")).toBe("uint32");
    expect(isSegmentation([step])).toBe(true);
  });
});
