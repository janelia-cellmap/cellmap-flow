// Parity of web/src/infer/norms.ts with cellmap_flow/norm/input_normalize.py.
//
// The fixture (test/fixtures/norms.json) was written by the Python classes
// themselves (test/fixtures/make_fixtures.py): one float32 block with
// out-of-range values in it, and for each case the chain of steps as the
// dashboard would send them (numbers, or the strings its forms send) and the
// array the Python normalizers returned. Regenerate the fixture when the
// Python changes; tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import type { Step } from "../../src/contracts";
import { SUPPORTED_NORMALIZERS, applyNormalizers } from "../../src/infer/norms";
import { type ArrayRecord, expectArrayMatches, loadFixture, supports } from "./helpers";

interface NormsFixture {
  seed: number;
  input: ArrayRecord;
  cases: { id: string; steps: Step[]; output: ArrayRecord }[];
}

const fixture = loadFixture<NormsFixture>("norms.json");

describe("applyNormalizers matches the Python normalizers", () => {
  it("has a float32 input block to work on", () => {
    expect(fixture.input.dtype).toBe("float32");
    expect(fixture.input.data.length).toBe(fixture.input.shape.reduce((a, b) => a * b, 1));
  });

  for (const c of fixture.cases) {
    it(c.id, () => {
      // A fresh copy per case: applyNormalizers works in place.
      const data = new Float32Array(fixture.input.data);
      const out = applyNormalizers(data, c.steps);
      expect(out, "normalizers return float32").toBeInstanceOf(Float32Array);
      expect(c.output.dtype, "the Python returned float32").toBe("float32");
      expectArrayMatches(out, c.output, c.id);
    });
  }

  it("returns the array it was given (in place)", () => {
    const data = new Float32Array(fixture.input.data);
    const steps = fixture.cases.find((c) => c.id === "chain_default_browser")!.steps;
    expect(applyNormalizers(data, steps)).toBe(data);
  });

  it("lists every normalizer the fixtures use as supported", () => {
    const names = new Set(fixture.cases.flatMap((c) => c.steps.map((s) => s.name)));
    for (const name of names) {
      expect(supports(SUPPORTED_NORMALIZERS, name), `${name} in SUPPORTED_NORMALIZERS`).toBe(true);
    }
  });
});
