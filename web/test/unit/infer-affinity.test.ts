// AffinityPostprocessor and the mutex watershed behind it: the time a real
// block takes. The labels themselves (the partition, the background rules,
// the numbering, the Morton offset) are test/parity/affinity.test.ts's,
// against what mwatershed.agglom gives on the same blocks.
import { describe, expect, it } from "vitest";

import { applyPostprocessors } from "../../src/infer/post";

// The step with a bias of 0: an edge above 0 joins. (The class's own default
// is 0.5, a sigmoid's midpoint; the parity fixtures cover it.)
const AFFINITY = { name: "AffinityPostprocessor", bias: 0 };

describe("AffinityPostprocessor", () => {
  it("segments a 60³ block of 3 and of 9 channels in about the time the Rust mwatershed takes", () => {
    // random logits as a model outputs them: ~650 k edges for 3 channels, 1.8 M for 9
    const seeded = (seed: number) => () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
    const timed = (channels: number, neighborhood?: string) => {
      const rnd = seeded(channels);
      const data = Float32Array.from({ length: channels * 60 ** 3 }, () => rnd() * 2 - 1);
      const step = neighborhood ? { ...AFFINITY, neighborhood } : AFFINITY;
      const start = performance.now();
      const out = applyPostprocessors({ data, shape: [channels, 60, 60, 60] }, [step], { chunkIndex: [1, 2, 3], chunkNumVoxels: 60 ** 3 });
      const ms = performance.now() - start;
      expect(out.shape).toEqual([1, 60, 60, 60]);
      expect((out.data as BigUint64Array).some((v) => v !== 0n)).toBe(true);
      return ms;
    };
    const three = timed(3, "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]");
    const nine = timed(9);
    console.log(`AffinityPostprocessor on 60³: 3 channels ${three.toFixed(0)} ms, 9 channels ${nine.toFixed(0)} ms`);
    // About 420 and 770 ms in node on the workstation this was written on,
    // where mwatershed.agglom (Rust, what the Python runs) took 350 and 620 ms
    // on the same blocks. The bounds are loose so that a slow CI machine does
    // not fail the test; they catch a pathological slowdown, not a regression.
    expect(three).toBeLessThan(2500);
    expect(nine).toBeLessThan(5000);
  });
});
