// The distillation voxels of a chunk: "unlabeled" takes the voxels some
// channel's mask leaves out, weighted 1 on those channels only (the
// server's 1 - mask); "all" every voxel; "anchor" the unpainted voxels of a
// good region (a scribble inside one wins); each with its features by the
// layout rule. At most the quota is kept, drawn uniformly (every candidate
// as likely) and deterministically for a seed; the cap is shared out
// equally or by the region voxels; sets concatenate in order.
import { describe, expect, it } from "vitest";

import { concatDistillation, DISTILLATION_VOXELS, quotas, sampleDistillation } from "../../src/finetune/distillation";
import { mulberry32 } from "../../src/finetune/head-trainer";
import type { Head, ModelFeatures, PaintedChunk, Targets } from "../../src/finetune/types";

const head = (inChannels: number, outChannels: number): Head => ({
  featureTensor: "features", weightName: "w", biasName: "b", inChannels, outChannels, dtype: "float32",
  weight: new Float32Array(inChannels * outChannels), bias: new Float32Array(outChannels),
});

// A 1×2×3 chunk, 6 voxels, its features as [1, 2, 2, 3]: channel c of voxel v is 10 v + c.
const SHAPE = [1, 2, 3], N = 6, CIN = 2, COUT = 2;
const FEATURES: ModelFeatures = {
  features: Float32Array.from({ length: CIN * N }, (_, i) => 10 * (i % N) + Math.floor(i / N)),
  featuresDims: [1, CIN, 2, 3],
  logits: new Float32Array(COUT * N),
  logitsDims: [1, COUT, 2, 3],
};
// Voxels 0 and 1 painted (masked on both channels), voxel 2 masked on channel 0 only (an affinity's edge).
const TARGETS: Targets = {
  channels: COUT,
  target: new Float32Array(COUT * N),
  mask: Float32Array.from([1, 1, 1, 0, 0, 0, /* channel 1 */ 1, 1, 0, 0, 0, 0]),
};
const CHUNK: PaintedChunk = { index: [0, 0, 0], shape: SHAPE, labels: Uint8Array.from([1, 2, 1, 0, 0, 0]) };

const rows = (x: Float32Array) => Array.from({ length: x.length / CIN }, (_, r) => Array.from(x.subarray(r * CIN, (r + 1) * CIN)));

describe("sampleDistillation", () => {
  const all = { quota: 100, random: mulberry32(1) };

  it("unlabeled: the voxels some channel leaves out, weighted on those channels alone", () => {
    const set = sampleDistillation(FEATURES, TARGETS, CHUNK, head(CIN, COUT), { pick: "unlabeled", ...all });
    expect(set.count).toBe(4);  // voxels 2 (channel 1 only), 3, 4, 5
    expect(rows(set.x)).toEqual([[20, 21], [30, 31], [40, 41], [50, 51]]);
    expect(Array.from(set.w)).toEqual([0, 1, 1, 1, 1, 1, 1, 1]);
  });

  it("all: every voxel, every channel", () => {
    const set = sampleDistillation(FEATURES, TARGETS, CHUNK, head(CIN, COUT), { pick: "all", ...all });
    expect(set.count).toBe(6);
    expect(rows(set.x).map((r) => r[0])).toEqual([0, 10, 20, 30, 40, 50]);
    expect(Array.from(set.w).every((w) => w === 1)).toBe(true);
  });

  it("anchor: the region's unpainted voxels, every channel: paint inside a region wins", () => {
    const inRegion = Uint8Array.from([0, 1, 1, 1, 1, 0]);
    const set = sampleDistillation(FEATURES, TARGETS, CHUNK, head(CIN, COUT), { pick: "anchor", inRegion, ...all });
    expect(rows(set.x).map((r) => r[0])).toEqual([30, 40]);  // 1 and 2 are painted, 0 and 5 are outside
    expect(Array.from(set.w)).toEqual([1, 1, 1, 1]);
    expect(() => sampleDistillation(FEATURES, TARGETS, CHUNK, head(CIN, COUT), { pick: "anchor", ...all })).toThrow(/region mask/);
  });

  it("keeps at most the quota, in the chunk's order, each candidate as likely, the same draw for a seed", () => {
    // A 1×1×1000 chunk, every voxel a candidate, 100 kept.
    const n = 1000;
    const features: ModelFeatures = { features: Float32Array.from({ length: n }, (_, v) => v), featuresDims: [1, 1, 1, n], logits: new Float32Array(n), logitsDims: [1, 1, 1, n] };
    const targets: Targets = { channels: 1, target: new Float32Array(n), mask: new Float32Array(n) };
    const chunk: PaintedChunk = { index: [0, 0, 0], shape: [1, 1, n], labels: new Uint8Array(n) };
    const draw = (seed: number) => Array.from(sampleDistillation(features, targets, chunk, head(1, 1), { pick: "unlabeled", quota: 100, random: mulberry32(seed) }).x);
    const a = draw(3);
    expect(a).toHaveLength(100);
    expect(a).toEqual([...a].sort((p, q) => p - q));
    expect(new Set(a).size).toBe(100);
    expect(draw(3)).toEqual(a);
    expect(draw(4)).not.toEqual(a);
    // Uniform: over many draws each tenth of the chunk gets about a tenth of the voxels kept.
    const tenths = new Array(10).fill(0);
    for (let seed = 0; seed < 200; seed++) for (const v of draw(seed)) tenths[Math.floor(v / 100)]++;
    for (const t of tenths) expect(Math.abs(t / 20000 - 0.1)).toBeLessThan(0.01);
    // A quota of 0 keeps nothing; more than there are keeps them all.
    expect(sampleDistillation(features, targets, chunk, head(1, 1), { pick: "all", quota: 0, random: mulberry32(1) }).count).toBe(0);
    expect(sampleDistillation(features, targets, chunk, head(1, 1), { pick: "all", quota: 5000, random: mulberry32(1) }).count).toBe(n);
  });
});

describe("the cap", () => {
  it("is 32,768 voxels a run, shared out in proportion and rounded up", () => {
    expect(DISTILLATION_VOXELS).toBe(32768);
    expect(quotas([1, 1, 1, 1])).toEqual([8192, 8192, 8192, 8192]);
    expect(quotas([3, 1], 100)).toEqual([75, 25]);
    expect(quotas([2, 1], 100)).toEqual([67, 34]);
    expect(quotas([0, 0], 100)).toEqual([0, 0]);
  });

  it("concatenates sets in order and refuses one for another head", () => {
    const set = sampleDistillation(FEATURES, TARGETS, CHUNK, head(CIN, COUT), { pick: "all", quota: 2, random: mulberry32(1) });
    const both = concatDistillation([set, set], head(CIN, COUT));
    expect(both.count).toBe(4);
    expect(Array.from(both.x.subarray(4))).toEqual(Array.from(set.x));
    expect(() => concatDistillation([set], head(3, COUT))).toThrow(/2 → 2 channels for a 3 → 2 head/);
  });
});
