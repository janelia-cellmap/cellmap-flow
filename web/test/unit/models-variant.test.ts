// chooseVariant across GPU limits: the precision from shader-f16, the block
// from the memory estimate against maxBufferSize (the largest that fits for
// a 3D model, the one nearest PREFERRED_2D_SIDE for a one-slice model).
import { describe, expect, it } from "vitest";

import type { GpuLimits } from "../../src/contracts";
import { parseManifest } from "../../src/models/manifest";
import { chooseVariant, UNKNOWN_LIMITS } from "../../src/models/variant";
import { rawManifest } from "./models-helpers";

const MiB = 1024 * 1024;
const limits = (over: Partial<GpuLimits>): GpuLimits => ({ ...UNKNOWN_LIMITS, ...over });

// setup_54's blocks are 116 / 132 / 148, setup_53_2d's one-slice blocks
// 300² / 556² / 1068² (fp32 and fp16). A block's peak is 40 values per input
// voxel, 4 bytes each in fp32 and 2 in fp16, and must fit in half of
// maxBufferSize.
const CASES: [string, string, GpuLimits, string, number[]][] = [
  ["limits unknown (no WebGPU: the CPU wants small blocks): fp32, the smallest block", "setup_54", UNKNOWN_LIMITS, "fp32", [116, 116, 116]],
  ["shader-f16 without a maxBufferSize: fp16, the smallest block", "setup_54", limits({ shaderF16: true }), "fp16", [116, 116, 116]],
  ["4 GiB: every fp32 block fits (148³ needs 519 MB), the largest", "setup_54", limits({ maxBufferSize: 4096 * MiB }), "fp32", [148, 148, 148]],
  ["800 MiB: 132³ fp32 (368 MB) fits in half, 148³ (519 MB) does not", "setup_54", limits({ maxBufferSize: 800 * MiB }), "fp32", [132, 132, 132]],
  ["WebGPU's default 256 MiB: none fits, so the smallest, that a run can at least be tried", "setup_54", limits({ maxBufferSize: 256 * MiB }), "fp32", [116, 116, 116]],
  ["600 MiB with shader-f16: a bigger block, fp16 148³ (259 MB), its values half the size", "setup_54", limits({ maxBufferSize: 600 * MiB, shaderF16: true }), "fp16", [148, 148, 148]],
  ["a one-slice model on 4 GiB: the block nearest PREFERRED_2D_SIDE, 556 of 300/556/1068, not the largest", "setup_53_2d", limits({ maxBufferSize: 4096 * MiB }), "fp32", [1, 556, 556]],
  ["a one-slice model on 60 MiB: still bounded by memory, 300² (14 MB) where 556² (49 MB) does not fit", "setup_53_2d", limits({ maxBufferSize: 60 * MiB }), "fp32", [1, 300, 300]],
];

describe("chooseVariant", () => {
  it.each(CASES)("%s", (_what, folder, gpu, variant, inputSize) => {
    const v = chooseVariant(parseManifest(rawManifest(folder)), gpu);
    expect(v.variant).toBe(variant);
    expect(v.input_size).toEqual(inputSize);
  });
});
