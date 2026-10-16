// Which of a model's files to run on this machine: the precision, then the
// block size.
//
// A 3D model's folder holds the same network exported twelve times: three
// input block sizes (116/132/148 or 124/140/172 voxels, valid convolutions,
// so the output is 88 voxels smaller) times four precisions. A 2D model's
// folder (setup_53_2d and setup_54_2d, since 2026-10-15: one slice in, one
// slice out, input_size [1, N, N]) holds six: three sides (300/556/1068
// pixels, or 1060; the output is 88 pixels smaller) times fp32 and fp16.
// The file is fixed per block size, so the choice is made before the
// download.
//
// The numbers below come from the repo's README (its WebGPU benchmark on an
// Apple GPU, Chrome 154) and from the first day with the 2D models, and are
// a first guess: tune them by hand once the browser mode runs on a few GPUs.
import type { GpuLimits, ModelManifest, ModelVariant, VariantName } from "../contracts";

/** What a run needs at its peak, in values per input voxel: the README
 * measures the largest single activation tensor at about 40·N³ values for
 * the ResUNet3DValid exports (32·(N−2)³ for Unet3dIso, which this rounds up
 * to the same), e.g. 519 MB in fp32 or 259 MB in fp16 at N=148. Applied to
 * a one-slice block it is 40·N² values (182 MB for 1068² in fp32, 49 MB for
 * 556², 14 MB for 300²), which nobody has measured: it only matters at
 * WebGPU's 256 MiB default maxBufferSize, where it rules out 1068² in fp32. */
export const PEAK_VALUES_PER_VOXEL = 40;

/** The share of maxBufferSize the peak tensor may take. The rest is headroom
 * for the other live activations and the weights, which the estimate above
 * does not count. */
export const PEAK_BUDGET_FRACTION = 0.5;

/** The in-plane side, in pixels, to want for a one-slice (2D) model when
 * more than one block fits the GPU: of the fitting blocks, the one whose
 * side is nearest this is run (556 of 300/556/1068 today).
 *
 * The 3D rule, the largest block that fits, is wrong for a 2D model. Each
 * viewer chunk is one forward pass, and Neuroglancer shows nothing of a
 * chunk until the whole of it is done, so the block's side sets how long
 * the user waits for every tile: a 1068² block has 3.7x the pixels of a
 * 556² one and takes about that much longer per chunk. On a GPU the old
 * rule picked 1068², which the first user saw as "slower but huge"
 * (2026-10-15); Marwan, who exported the models, found 556 "very fast"
 * (Slack, the same day). Against latency stands the context each block
 * spends on its edges, 88 pixels in all: 50% of a 300² input is output,
 * 71% of 556², 84% of 1068². 512 is a round number beside the 556 that
 * was fast, far enough from 300 (half its pixels wasted) and from 1068
 * (seconds per chunk) that either would have to be chosen on purpose.
 *
 * A 3D block keeps the largest-that-fits rule: its latency is bounded by
 * memory before it is by patience (148³ is the largest exported), and its
 * context, 44 voxels each side in three dimensions, is the bigger waste
 * (28³ of 116³ is output: 1.4%; 60³ of 148³: 6.7%), so the README's
 * throughput grows with the block (0.04 to 0.18 Mvox/s from 116³ to 148³). */
export const PREFERRED_2D_SIDE = 512;

/** Prefer the int8-weight files (int8w, int8w_fp16)? They only shrink the
 * download, to a quarter of fp32: the weights are dequantized at load, so a
 * run takes the time and memory of the fp32 or fp16 file, and the error
 * against the fp32 model grows from a max |Δlogit| of ~0.01 to ~0.14. Off,
 * because the bytes are cached in IndexedDB after the first download anyway;
 * a flag rather than a hard choice so a slow-network deployment can flip it. */
export const PREFER_SMALL_DOWNLOAD: boolean = false;

/** The limits of a GPU nobody has asked yet: chooseVariant answers fp32 and
 * the smallest block, the choice `Catalog.list()` makes for its default entries. */
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

/** Is `inputSize` one slice, a 2D model's block ([1, y, x])? */
export function isOneSlice(inputSize: number[]): boolean {
  return inputSize.length === 3 && inputSize[0] === 1;
}

/** The variant to run on a GPU with `limits`.
 *
 * Precision: fp16 when the device has shader-f16 and the manifest offers it
 * (1.3–2.2x faster, half the memory, flips under 0.05% of the logit > 0
 * decisions), otherwise fp32, which is exact and the only one that also runs
 * on the WASM fallback. With preferSmallDownload the int8-weight files of
 * the same precision are taken instead, when offered.
 *
 * Block size: among the blocks whose estimated peak (estimatePeakBytes)
 * stays within PEAK_BUDGET_FRACTION of maxBufferSize, the largest for a 3D
 * model and the one whose side is nearest PREFERRED_2D_SIDE for a one-slice
 * model (why: at that constant); the smallest when none fits (a run may
 * still fail there; the inference side reports that), and the smallest
 * while the limits are unknown. maxStorageBufferBindingSize caps one bound
 * tensor too and is often smaller than maxBufferSize; it is not used yet,
 * and is the first thing to look at when a block that "fits" fails to run. */
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
  if (fitting.length === 0) return candidates[0]!;
  if (fitting.every((v) => isOneSlice(v.input_size))) return nearestSide(fitting, PREFERRED_2D_SIDE);
  return fitting[fitting.length - 1]!;
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

// The one of `fitting` (sorted smallest first) whose in-plane side is
// nearest `side`; the smaller block on a tie.
function nearestSide(fitting: ModelVariant[], side: number): ModelVariant {
  let best = fitting[0]!;
  for (const v of fitting) {
    if (Math.abs(planeSide(v.input_size) - side) < Math.abs(planeSide(best.input_size) - side)) best = v;
  }
  return best;
}

// A one-slice block's side: of a square with as many pixels, should an
// export ever be oblong.
function planeSide(inputSize: number[]): number {
  return Math.sqrt(inputSize[1]! * inputSize[2]!);
}

function volume(size: number[]): number {
  return size.reduce((a, b) => a * b, 1);
}
