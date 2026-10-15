// Which of a model's files to run on this machine: the precision, then the
// block size.
//
// Every model folder holds the same network exported twelve times: three
// input block sizes (116/132/148 or 124/140/172 voxels, valid convolutions,
// so the output is 88 voxels smaller) times four precisions. The file is
// fixed per block size, so the choice is made before the download.
//
// The numbers below come from the repo's README (its WebGPU benchmark on an
// Apple GPU, Chrome 154) and are a first guess: tune them by hand once the
// browser mode runs on a few GPUs.
import type { GpuLimits, ModelManifest, ModelVariant, VariantName } from "../contracts";

/** What a run needs at its peak, in values per input voxel: the README
 * measures the largest single activation tensor at about 40·N³ values for
 * the ResUNet3DValid exports (32·(N−2)³ for Unet3dIso, which this rounds up
 * to the same), e.g. 519 MB in fp32 or 259 MB in fp16 at N=148. */
export const PEAK_VALUES_PER_VOXEL = 40;

/** The share of maxBufferSize the peak tensor may take. The rest is headroom
 * for the other live activations and the weights, which the estimate above
 * does not count. */
export const PEAK_BUDGET_FRACTION = 0.5;

/** Prefer the int8-weight files (int8w, int8w_fp16)? They only shrink the
 * download, to a quarter of fp32: the weights are dequantized at load, so a
 * run takes the time and memory of the fp32 or fp16 file, and the error
 * against the fp32 model grows from a max |Δlogit| of ~0.01 to ~0.14. Off,
 * because the bytes are cached in IndexedDB after the first download anyway;
 * a flag rather than a hard choice so a slow-network deployment can flip it. */
export const PREFER_SMALL_DOWNLOAD: boolean = false;

/** The limits of a GPU nobody has asked yet: chooseVariant answers fp32 and
 * the middle block, the choice `Catalog.list()` makes for its default entries. */
export const UNKNOWN_LIMITS: GpuLimits = { maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false };

export interface VariantPreferences {
  preferSmallDownload: boolean;
}

/** Bytes per value a variant computes in: the fp16 files run their activations in half precision. */
export function bytesPerValue(variant: VariantName): 2 | 4 {
  return variant === "fp16" || variant === "int8w_fp16" ? 2 : 4;
}

/** The estimated peak memory of one run on a block of `inputSize` voxels, in bytes. */
export function estimatePeakBytes(inputSize: number[], variant: VariantName): number {
  const voxels = inputSize.reduce((a, b) => a * b, 1);
  return PEAK_VALUES_PER_VOXEL * voxels * bytesPerValue(variant);
}

/** The variant to run on a GPU with `limits`.
 *
 * Precision: fp16 when the device has shader-f16 and the manifest offers it
 * (1.3–2.2x faster, half the memory, flips under 0.05% of the logit > 0
 * decisions), otherwise fp32, which is exact and the only one that also runs
 * on the WASM fallback. With preferSmallDownload the int8-weight files of
 * the same precision are taken instead, when offered.
 *
 * Block size: the largest whose estimated peak (estimatePeakBytes) stays
 * within PEAK_BUDGET_FRACTION of maxBufferSize, the smallest when none does
 * (a run may still fail there; the inference side reports that), and the
 * middle one while the limits are unknown. Bigger blocks waste less on the
 * 44-voxel context each side and run more voxels per second (0.04 to 0.18
 * Mvox/s from 116³ to 148³ on the README's GPU), so the largest that fits is
 * the one to want. maxStorageBufferBindingSize caps one bound tensor too and
 * is often smaller than maxBufferSize; it is not used yet, and is the first
 * thing to look at when a block that "fits" fails to run. */
export function chooseVariant(
  manifest: ModelManifest,
  limits: GpuLimits,
  prefs: VariantPreferences = { preferSmallDownload: PREFER_SMALL_DOWNLOAD },
): ModelVariant {
  if (manifest.variants.length === 0) throw new Error(`${manifest.model} lists no variants`);
  const precision = choosePrecision(manifest.variants, limits, prefs);
  const candidates = manifest.variants
    .filter((v) => v.variant === precision)
    .sort((a, b) => volume(a.input_size) - volume(b.input_size));
  // No maxBufferSize: either the limits were not probed yet (the catalog's
  // default entries, retuned before serving) or there is no WebGPU and the
  // model runs on the CPU, where a smaller block answers sooner. The
  // smallest block serves both.
  if (limits.maxBufferSize === null) return candidates[0]!;
  const budget = limits.maxBufferSize * PEAK_BUDGET_FRACTION;
  const fitting = candidates.filter((v) => estimatePeakBytes(v.input_size, v.variant) <= budget);
  return fitting.length > 0 ? fitting[fitting.length - 1]! : candidates[0]!;
}

function choosePrecision(variants: ModelVariant[], limits: GpuLimits, prefs: VariantPreferences): VariantName {
  const offers = (name: VariantName) => variants.some((v) => v.variant === name);
  if (prefs.preferSmallDownload) {
    if (limits.shaderF16 && offers("int8w_fp16")) return "int8w_fp16";
    if (offers("int8w")) return "int8w";
  }
  if (limits.shaderF16 && offers("fp16")) return "fp16";
  if (offers("fp32")) return "fp32";
  // Neither plain precision: whatever the manifest offers (sorted, so the first is fixed).
  return variants[0]!.variant;
}

function volume(size: number[]): number {
  return size.reduce((a, b) => a * b, 1);
}
