// Parity of web/src/finetune/targets.ts with cellmap_flow/finetune/
// target_transforms.py (BinaryTargetTransform, BroadcastBinaryTargetTransform,
// AffinityTargetTransform).
//
// The fixture (test/fixtures/targets.json) was written by the Python classes
// themselves (test/fixtures/make_fixtures.py): label blocks [z, y, x] with the
// paint layer's values (0 unpainted, 1 background, 2+ objects; a one-slice
// block, a fully painted one, one with uint16 ids, one all zero), and for
// each case the class with its constructor arguments and the target and
// mask it returned, float32 [C, z, y, x] with the batch axis dropped, which
// is the port's channels × N layout. Targets are 0 or 1, so the match is
// exact. Regenerate the fixture when the Python changes;
// tests/web/test_fixtures_fresh.py fails when it is stale.
import { describe, expect, it } from "vitest";

import { affinityTargets, binaryTargets } from "../../src/finetune/targets";
import type { PaintedChunk, TargetTransform } from "../../src/finetune/types";
import { type ArrayRecord, expectArrayMatches, loadFixture } from "./helpers";

interface TargetCase {
  id: string;
  transform: "BinaryTargetTransform" | "BroadcastBinaryTargetTransform" | "AffinityTargetTransform";
  params: { num_channels?: number; offsets?: number[][] };
  input: string;
  target: ArrayRecord;
  mask: ArrayRecord;
}

interface TargetsFixture {
  seed: number;
  inputs: Record<string, ArrayRecord>;
  cases: TargetCase[];
}

const fixture = loadFixture<TargetsFixture>("targets.json");

/** The port of the case's class, built as the Python was. */
function transformOf(c: TargetCase): TargetTransform {
  switch (c.transform) {
    case "BinaryTargetTransform":
      return binaryTargets(1);
    case "BroadcastBinaryTargetTransform":
      return binaryTargets(c.params.num_channels!);
    case "AffinityTargetTransform":
      return affinityTargets(c.params.offsets!, c.params.num_channels!);
  }
}

/** A fixture block as the painted chunk the port takes: its labels in the
 * volume's dtype (uint8, or uint16 for an instance volume's ids). */
function chunkOf(record: ArrayRecord): PaintedChunk {
  const labels = record.dtype === "uint16" ? Uint16Array.from(record.data) : Uint8Array.from(record.data);
  return { index: [0, 0, 0], shape: record.shape.slice(), labels };
}

const sum = (values: ArrayLike<number>, from = 0, to = values.length) => {
  let s = 0;
  for (let i = from; i < to; i++) s += values[i];
  return s;
};

describe("the target transforms match the Python classes", () => {
  it("has label blocks in the paint layer's dtypes", () => {
    for (const [name, block] of Object.entries(fixture.inputs)) {
      expect(["uint8", "uint16"], `${name}: dtype`).toContain(block.dtype);
      expect(block.shape.length, `${name}: axes`).toBe(3);
      expect(block.data.length, `${name}: voxels`).toBe(block.shape.reduce((a, b) => a * b, 1));
    }
  });

  for (const c of fixture.cases) {
    it(c.id, () => {
      const block = fixture.inputs[c.input];
      const out = transformOf(c)(chunkOf(block));
      const n = block.data.length;
      expect(c.target.dtype, "the Python returned float32").toBe("float32");
      expect(c.target.shape.slice(1), "the Python's target covers the block").toEqual(block.shape);
      expect(out.channels, "channels").toBe(c.target.shape[0]);
      expect(out.target.length, "target length").toBe(out.channels * n);
      expect(out.mask.length, "mask length").toBe(out.channels * n);
      expectArrayMatches(out.target, c.target, `${c.id}: target`);
      expectArrayMatches(out.mask, c.mask, `${c.id}: mask`);
    });
  }

  it("covers what the port must get right: an empty z mask on a one-slice chunk, a masked-out extra channel, both affinity values", () => {
    const byId = new Map(fixture.cases.map((c) => [c.id, c]));
    // A 2D student's chunk is one slice: the z offset pairs nothing, so its
    // channel has no mask (and its head keeps the model's weights); the y
    // and x channels do.
    const slice = byId.get("affinity_3_slice_1_6_7")!;
    const n = fixture.inputs[slice.input].data.length;
    expect(sum(slice.mask.data, 0, n)).toBe(0);
    expect(sum(slice.mask.data, n, 2 * n)).toBeGreaterThan(0);
    expect(sum(slice.mask.data, 2 * n, 3 * n)).toBeGreaterThan(0);
    // num_channels past the offsets: the fourth channel is masked out.
    const four = byId.get("affinity_4_block_4_5_6")!;
    const m = fixture.inputs[four.input].data.length;
    expect(four.target.shape[0]).toBe(4);
    expect(sum(four.mask.data, 3 * m, 4 * m)).toBe(0);
    expect(sum(four.target.data, 3 * m, 4 * m)).toBe(0);
    // Masked affinities of both values: same object (1) and different (0).
    const masked = four.mask.data.map((v, i) => (v ? four.target.data[i] : -1));
    expect(masked).toContain(1);
    expect(masked).toContain(0);
    // Nothing painted: nothing in any mask.
    for (const c of fixture.cases) if (c.input.startsWith("zeros")) expect(sum(c.mask.data), c.id).toBe(0);
    // The one-channel binary target is the broadcast's first channel.
    const binary = byId.get("binary_block_4_5_6")!, broadcast = byId.get("broadcast_3_block_4_5_6")!;
    expect(broadcast.target.data.slice(0, m)).toEqual(binary.target.data);
    expect(broadcast.target.data.slice(2 * m, 3 * m)).toEqual(binary.target.data);
  });
});
