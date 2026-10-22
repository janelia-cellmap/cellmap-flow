// The blockwise merger's step (src/infer/merger.ts) and the layer's hook it
// reports to: the throttled reporting, and a listener that fails. The ids it
// joins, chunk by chunk, are test/parity/merger.test.ts's, against the
// Python class.
import { afterEach, describe, expect, it, vi } from "vitest";

import { blockwiseMerger, REPORT_EVERY_MS } from "../../src/infer/merger";
import type { NdArray, PostContext } from "../../src/infer/post";

/** A (1, z, y, x) block filled with one id. */
function filled(shape: [number, number, number], id: number): NdArray {
  const n = shape[0] * shape[1] * shape[2];
  return { data: new Uint32Array(n).fill(id), shape: [1, ...shape] };
}

function at(chunkIndex: number[]): PostContext {
  return { chunkIndex, chunkNumVoxels: 0 };
}

describe("blockwiseMerger, the step", () => {
  afterEach(() => vi.useRealTimers());

  it("reports the joined ids as sets, at once the first time and at most every 5 s after", () => {
    vi.useFakeTimers();
    const reports: string[][][] = [];
    const step = blockwiseMerger.make({ name: "SimpleBlockwiseMerger", channel: "0", face_erosion_iterations: "0" }, { onEquivalences: (p) => reports.push(p) });
    step(filled([1, 1, 1], 1), at([0, 0, 0]));
    expect(reports, "nothing joined yet").toEqual([]);
    step(filled([1, 1, 1], 2), at([0, 0, 1]));
    expect(reports).toEqual([[["1", "2"]]]);
    step(filled([1, 1, 1], 3), at([0, 0, 2]));
    expect(reports, "within the interval: held back").toHaveLength(1);
    vi.advanceTimersByTime(REPORT_EVERY_MS);
    expect(reports).toEqual([[["1", "2"]], [["1", "2", "3"]]]);
  });

  it("a failing listener does not fail the chunk", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const step = blockwiseMerger.make({ name: "SimpleBlockwiseMerger" }, { onEquivalences: () => { throw new Error("viewer gone"); } });
    step(filled([1, 1, 1], 1), at([0, 0, 0]));
    expect(() => step(filled([1, 1, 1], 2), at([0, 0, 1]))).not.toThrow();
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });
});
