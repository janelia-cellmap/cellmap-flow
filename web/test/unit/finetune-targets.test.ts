// The training targets through the browser's call (src/finetune/
// target-call.ts): cellmap_flow/portable/targets.py in node's Pyodide
// (test/unit/pyodide.ts), as the engine's worker runs it, from a model
// entry as the trainers and the ranking ask for them. The maths itself is
// tested by pytest (tests/finetune/test_target_transforms.py). Without
// Pyodide (no network for numpy's wheel) these skip.
import { describe, expect, it, vi } from "vitest";

import { blockTargets } from "../../src/finetune/full/tiles";
import { targetTransformFor } from "../../src/finetune/target-call";
import { entry } from "./backend-fakes";
import { portablePython } from "./pyodide";

// Pyodide and numpy load once for the file, a few seconds; each call after is quick.
vi.setConfig({ testTimeout: 120_000 });
const python = await portablePython();

describe.skipIf(!python)("a chunk's targets, from the portable Python", () => {
  const infer = { python: python! };

  it("are an affinity model's along the offsets its channels name, and a binary model's on every channel", async () => {
    // One slice of 2 × 3, an instance volume's ids: 2 2 3 over 1 0 2.
    const chunk = { index: [0, 0, 0], shape: [1, 2, 3], labels: Uint16Array.from([2, 2, 3, 1, 0, 2]) };
    const aff = await targetTransformFor(entry("setup_54"), infer)(chunk);
    // z pairs nothing in one slice; y pairs the rows where both are painted,
    // 2 over 1 and 3 over 2, neither one object; x pairs 2|2 (one object) and 2|3.
    expect(aff.channels).toBe(3);
    expect(Array.from(aff.mask)).toEqual([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]);
    expect(Array.from(aff.target)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);

    const nuc = await targetTransformFor(entry("setup_56", "nuc", ["nuc"], 128), infer)({ ...chunk, labels: Uint8Array.from(chunk.labels) });
    expect(nuc.channels).toBe(1);
    expect(Array.from(nuc.target)).toEqual([1, 1, 1, 0, 0, 1]);
    expect(Array.from(nuc.mask)).toEqual([1, 1, 1, 1, 0, 1]);
  });

  it("pair a cube's z neighbours, channels last, for the whole network's blocks", async () => {
    // A 2³ object whose corner voxel is another object (3): the z affinity
    // pairs each voxel of the lower slice with the one above, 3 over 2 a
    // boundary, 2 over 2 the same object.
    const labels = new Uint16Array(8).map((_, i) => (i ? 2 : 3));
    const t = await blockTargets(labels, [2, 2, 2], targetTransformFor(entry("setup_54"), infer));
    const zPairs = [0, 1, 2, 3].map((p) => [t.mask[p * 3], t.target[p * 3]]);
    expect(zPairs).toEqual([[1, 0], [1, 1], [1, 1], [1, 1]]);
  });
});
