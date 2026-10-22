// The browser's postprocessors refuse a step they cannot run, at validation
// and when applied. What each step computes, and what a chain declares, is
// test/parity/post.test.ts's, against the Python classes.
import { describe, expect, it } from "vitest";

import { applyPostprocessors, validatePostprocessors } from "../../src/infer/post";

describe("applyPostprocessors", () => {
  it("names an unsupported postprocessor, at validation and when applied", () => {
    const y = { data: new Float32Array([-3, -1, -0.5, 0, 0.5, 1, 3, 10]), shape: [1, 1, 2, 4] };
    expect(() => validatePostprocessors([{ name: "CellposeMasksPostprocessor" }])).toThrow(/CellposeMasksPostprocessor/);
    expect(() => applyPostprocessors(y, [{ name: "CellposeMasksPostprocessor" }])).toThrow(/CellposeMasksPostprocessor/);
  });
});
