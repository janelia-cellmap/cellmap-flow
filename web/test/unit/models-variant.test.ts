// chooseVariant across GPU limits: the precision from shader-f16, the block
// from the memory estimate against maxBufferSize (the largest that fits for
// a 3D model, the one nearest PREFERRED_2D_SIDE for a one-slice model).
import { describe, expect, it } from "vitest";

import type { GpuLimits } from "../../src/contracts";
import { parseManifest } from "../../src/models/manifest";
import { bytesPerValue, chooseVariant, estimatePeakBytes, isOneSlice, PEAK_VALUES_PER_VOXEL, PREFERRED_2D_SIDE, UNKNOWN_LIMITS } from "../../src/models/variant";
import { rawManifest } from "./models-helpers";

const MiB = 1024 * 1024;
const limits = (over: Partial<GpuLimits>): GpuLimits => ({ ...UNKNOWN_LIMITS, ...over });
const resunet = () => parseManifest(rawManifest("setup_54")); // blocks 116 / 132 / 148
const unet = () => parseManifest(rawManifest("setup_53")); // blocks 124 / 140 / 172
const twoD = () => parseManifest(rawManifest("setup_53_2d")); // one-slice blocks 300² / 556² / 1068², fp32 and fp16 (saved 2026-10-15)

describe("estimatePeakBytes", () => {
  it("is 40 values per input voxel, 4 bytes in fp32 and 2 in fp16 (the README's 519 / 259 MB at N=148)", () => {
    expect(PEAK_VALUES_PER_VOXEL).toBe(40);
    expect(estimatePeakBytes([148, 148, 148], "fp32") / 1e6).toBeCloseTo(518.6, 0);
    expect(estimatePeakBytes([148, 148, 148], "fp16") / 1e6).toBeCloseTo(259.3, 0);
    expect(bytesPerValue("int8w")).toBe(4);
    expect(bytesPerValue("int8w_fp16")).toBe(2);
  });
});

describe("chooseVariant: precision", () => {
  it("is fp32 and the smallest block while the limits are unknown (no WebGPU: the CPU wants small blocks)", () => {
    const v = chooseVariant(resunet(), UNKNOWN_LIMITS);
    expect(v.variant).toBe("fp32");
    expect(v.input_size).toEqual([116, 116, 116]);
    expect(v.output_size).toEqual([28, 28, 28]);
    const u = chooseVariant(unet(), UNKNOWN_LIMITS);
    expect(u.variant).toBe("fp32");
    expect(u.input_size).toEqual([124, 124, 124]);
  });

  it("is fp16 when the device has shader-f16", () => {
    const v = chooseVariant(resunet(), limits({ shaderF16: true }));
    expect(v.variant).toBe("fp16");
    expect(v.input_size).toEqual([116, 116, 116]); // no maxBufferSize: the smallest block
    expect(v.file).toBe("setup_54_mito_affinities_16nm_in116_fp16.onnx");
  });

  it("stays fp32 with shader-f16 when the manifest has no fp16 file", () => {
    const m = resunet();
    m.variants = m.variants.filter((v) => v.variant !== "fp16");
    expect(chooseVariant(m, limits({ shaderF16: true })).variant).toBe("fp32");
  });

  it("takes the int8-weight files when asked to keep the download small", () => {
    expect(chooseVariant(resunet(), UNKNOWN_LIMITS, { preferSmallDownload: true }).variant).toBe("int8w");
    expect(chooseVariant(resunet(), limits({ shaderF16: true }), { preferSmallDownload: true }).variant).toBe("int8w_fp16");
    const m = resunet();
    m.variants = m.variants.filter((v) => !v.variant.startsWith("int8w"));
    expect(chooseVariant(m, UNKNOWN_LIMITS, { preferSmallDownload: true }).variant).toBe("fp32");
  });

  it("falls back to whatever precision the manifest offers", () => {
    const m = resunet();
    m.variants = m.variants.filter((v) => v.variant === "int8w_fp16");
    expect(chooseVariant(m, UNKNOWN_LIMITS).variant).toBe("int8w_fp16");
  });

  it("rejects a manifest with no variants", () => {
    const m = resunet();
    m.variants = [];
    expect(() => chooseVariant(m, UNKNOWN_LIMITS)).toThrow(/lists no variants/);
  });
});

describe("chooseVariant: block size", () => {
  it("is the largest whose peak fits in half of maxBufferSize", () => {
    // 4 GiB: every fp32 block fits (148³ needs 519 MB, half the budget is 2 GiB).
    expect(chooseVariant(resunet(), limits({ maxBufferSize: 4096 * MiB })).input_size).toEqual([148, 148, 148]);
    // 800 MiB: half is 400 MiB; 132³ fp32 is 368 MB, 148³ is 519 MB.
    expect(chooseVariant(resunet(), limits({ maxBufferSize: 800 * MiB })).input_size).toEqual([132, 132, 132]);
    // 600 MiB: half is 300 MiB; only 116³ fp32 (250 MB) fits.
    expect(chooseVariant(resunet(), limits({ maxBufferSize: 600 * MiB })).input_size).toEqual([116, 116, 116]);
  });

  it("is the smallest block when none fits, so a run can at least be tried", () => {
    // WebGPU's default maxBufferSize is 256 MiB: half is 128 MiB, under even 116³ fp32.
    const v = chooseVariant(resunet(), limits({ maxBufferSize: 256 * MiB }));
    expect(v.input_size).toEqual([116, 116, 116]);
    expect(v.variant).toBe("fp32");
  });

  it("fits a bigger block in fp16, since the values are half the size", () => {
    // 600 MiB: half is 300 MiB; fp16 148³ is 259 MB.
    const v = chooseVariant(resunet(), limits({ maxBufferSize: 600 * MiB, shaderF16: true }));
    expect(v.variant).toBe("fp16");
    expect(v.input_size).toEqual([148, 148, 148]);
    // 256 MiB default: half is 128 MiB; fp16 116³ is 125 MB.
    expect(chooseVariant(resunet(), limits({ maxBufferSize: 256 * MiB, shaderF16: true })).input_size).toEqual([116, 116, 116]);
  });

  it("applies to the Unet3dIso sizes too", () => {
    // 172³ fp32 is 814 MB, 140³ is 439 MB, 124³ is 305 MB.
    expect(chooseVariant(unet(), limits({ maxBufferSize: 1000 * MiB })).input_size).toEqual([140, 140, 140]);
    expect(chooseVariant(unet(), limits({ maxBufferSize: 2000 * MiB })).input_size).toEqual([172, 172, 172]);
  });

  it("ignores maxStorageBufferBindingSize for now", () => {
    const only = limits({ maxBufferSize: 4096 * MiB, maxStorageBufferBindingSize: 16 * MiB });
    expect(chooseVariant(resunet(), only).input_size).toEqual([148, 148, 148]);
  });
});

describe("chooseVariant: one-slice (2D) models", () => {
  it("knows a one-slice block by its first axis", () => {
    expect(isOneSlice([1, 556, 556])).toBe(true);
    expect(isOneSlice([116, 116, 116])).toBe(false);
    expect(isOneSlice([556, 556])).toBe(false);
    expect(twoD().variants.every((v) => isOneSlice(v.input_size))).toBe(true);
    expect(resunet().variants.some((v) => isOneSlice(v.input_size))).toBe(false);
  });

  it("is fp32 and the smallest block while the limits are unknown, as for a 3D model", () => {
    const v = chooseVariant(twoD(), UNKNOWN_LIMITS);
    expect(v.variant).toBe("fp32");
    expect(v.input_size).toEqual([1, 300, 300]);
    expect(v.output_size).toEqual([1, 212, 212]);
  });

  it("is the fitting block nearest PREFERRED_2D_SIDE on a GPU, not the largest: 556 of 300/556/1068", () => {
    expect(PREFERRED_2D_SIDE).toBe(512);
    // 4 GiB: every block fits (1068² fp32 is 182 MB); the 3D rule would take 1068.
    const v = chooseVariant(twoD(), limits({ maxBufferSize: 4096 * MiB }));
    expect(v.input_size).toEqual([1, 556, 556]);
    expect(v.output_size).toEqual([1, 468, 468]);
    expect(v.file).toBe("setup_53_mito_affinities_16nm_2d_in556_fp32.onnx");
    const h = chooseVariant(twoD(), limits({ maxBufferSize: 4096 * MiB, shaderF16: true }));
    expect(h.variant).toBe("fp16");
    expect(h.file).toBe("setup_53_mito_affinities_16nm_2d_in556_fp16.onnx");
  });

  it("is still bounded by memory: 300 when 556 does not fit, the smallest when nothing does", () => {
    // 60 MiB: half is 30 MiB; 300² fp32 is 14 MB, 556² is 49 MB.
    expect(chooseVariant(twoD(), limits({ maxBufferSize: 60 * MiB })).input_size).toEqual([1, 300, 300]);
    // 16 MiB: half is 8 MiB, under 300² fp32's 14 MB: the smallest, to be tried.
    expect(chooseVariant(twoD(), limits({ maxBufferSize: 16 * MiB })).input_size).toEqual([1, 300, 300]);
    // WebGPU's 256 MiB default: half is 128 MiB; 1068² fp32 (182 MB) is out, 556² in.
    expect(chooseVariant(twoD(), limits({ maxBufferSize: 256 * MiB })).input_size).toEqual([1, 556, 556]);
  });

  it("is the nearest, not the largest or the smallest: 300 of 300/1068, the smaller of two equally near", () => {
    const m = twoD();
    m.variants = m.variants.filter((v) => v.input_size[1] !== 556);
    // |300 − 512| = 212, |1068 − 512| = 556.
    expect(chooseVariant(m, limits({ maxBufferSize: 4096 * MiB })).input_size).toEqual([1, 300, 300]);
    const tie = twoD();
    const sides = [400, 624, 1000]; // 400 and 624 are both 112 from 512
    tie.variants = tie.variants
      .filter((v) => v.variant === "fp32")
      .map((v, i) => ({ ...v, input_size: [1, sides[i]!, sides[i]!], output_size: [1, sides[i]! - 88, sides[i]! - 88] }));
    expect(chooseVariant(tie, limits({ maxBufferSize: 4096 * MiB })).input_size).toEqual([1, 400, 400]);
  });
});
