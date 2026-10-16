// From an EM plane to the grey image a model is sent, and from its reply to
// a mask (src/ai-annotate/mask.ts) against cellmap_flow/ai_annotate/
// mask_extraction.py and staging.py's overlay. The parity fixture checks
// extractMask and normalizeToUint8 against the Python's own numbers; here
// the rules are spelled out.
import { describe, expect, it } from "vitest";

import { extractMask, greyToRgba, maskFraction, normalizeToUint8, overlay, OVERLAY_ALPHA } from "../../src/ai-annotate/mask";
import type { RgbaImage } from "../../src/ai-annotate/types";

/** An RGBA image of the given pixels (rows first), alpha 255. */
function image(width: number, height: number, pixels: [number, number, number][]): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  pixels.forEach((rgb, i) => {
    data.set(rgb, i * 4);
    data[i * 4 + 3] = 255;
  });
  return { width, height, data };
}

describe("normalizeToUint8", () => {
  it("passes uint8 values through unchanged", () => {
    const plane = Uint8Array.from([0, 7, 128, 255]);
    const out = normalizeToUint8(plane, true);
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out)).toEqual([0, 7, 128, 255]);
    expect(out).not.toBe(plane);
  });

  it("stretches the 1st to 99th percentile to 0-255 and clips the outliers", () => {
    // A ramp 0..999 with five saturated pixels at each end: numpy's linear
    // percentiles fall at ranks 9.99 and 989.01, inside the ramp, so the
    // outliers and the ramp's ends clip to 0 and 255 and the rest stretches.
    const values = Array.from({ length: 1000 }, (_, i) => i);
    for (let i = 0; i < 5; i++) {
      values[i] = -1e6;
      values[999 - i] = 1e6;
    }
    const out = normalizeToUint8(Float32Array.from(values), false);
    for (let i = 0; i < 5; i++) {
      expect(out[i]).toBe(0);
      expect(out[999 - i]).toBe(255);
    }
    expect(out[5]).toBe(0);
    expect(out[10]).toBeLessThanOrEqual(1);
    expect(out[989]).toBeGreaterThanOrEqual(253);
    expect(out[994]).toBe(255);
    expect(out[500]).toBeGreaterThan(120);
    expect(out[500]).toBeLessThan(135);
    // Monotone in between.
    for (let i = 6; i < 995; i++) expect(out[i]).toBeGreaterThanOrEqual(out[i - 1]);
  });

  it("a flat plane is all zeros (the range is widened to 1 rather than divided by 0)", () => {
    expect(Array.from(normalizeToUint8(Uint16Array.from([700, 700, 700]), false))).toEqual([0, 0, 0]);
    expect(normalizeToUint8(new Float32Array(0), false).length).toBe(0);
  });

  it("uint16 values stretch like any other", () => {
    const out = normalizeToUint8(Uint16Array.from([1000, 2000, 3000, 4000, 5000]), false);
    expect(out[0]).toBe(0);
    expect(out[4]).toBe(255);
    expect(out[2]).toBeGreaterThan(120);
    expect(out[2]).toBeLessThan(135);
  });
});

describe("greyToRgba", () => {
  it("puts the grey in every channel with alpha 255, columns across", () => {
    const rgba = greyToRgba(Uint8Array.from([1, 2, 3, 4, 5, 6]), 2, 3);
    expect([rgba.width, rgba.height]).toEqual([3, 2]);
    expect(Array.from(rgba.data.slice(0, 8))).toEqual([1, 1, 1, 255, 2, 2, 2, 255]);
    expect(Array.from(rgba.data.slice(20, 24))).toEqual([6, 6, 6, 255]);
    expect(() => greyToRgba(new Uint8Array(5), 2, 3)).toThrow(/2×3/);
  });
});

describe("extractMask", () => {
  const pixels: [number, number, number][] = [
    [255, 0, 0], [230, 20, 10], [200, 0, 0], [255, 128, 0], [255, 100, 0], [128, 0, 255],
    [120, 10, 250], [255, 255, 255], [0, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 128],
  ];
  const img = image(6, 2, pixels);

  it("finds red that dominates: R − max(G, B) at least the threshold", () => {
    expect(Array.from(extractMask(img, [255, 0, 0]))).toEqual([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    // A lower threshold admits orange too: 255 − 128 = 127 is red that dominates by 100.
    expect(Array.from(extractMask(img, [255, 0, 0], 100))).toEqual([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("finds a part-strength colour (orange, purple) by its distance from the half channel", () => {
    expect(Array.from(extractMask(img, [255, 128, 0]))).toEqual([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(extractMask(img, [128, 0, 255]))).toEqual([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
    expect(Array.from(extractMask(img, [0, 255, 128]))).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
  });

  it("white is not any colour, black is nothing, yellow is not red or green", () => {
    for (const target of [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]] as const) {
      const mask = extractMask(img, target);
      expect(mask[7], `white vs ${target}`).toBe(target[0] === 255 && target[1] === 255 ? 0 : 0);
      expect(mask[8], `black vs ${target}`).toBe(0);
    }
    expect(extractMask(img, [255, 255, 0])[9]).toBe(1);
    expect(extractMask(img, [255, 0, 0])[9]).toBe(0);
    expect(extractMask(img, [0, 255, 0])[9]).toBe(0);
  });

  it("a target with every channel full (white) scores by the weakest channel alone", () => {
    expect(Array.from(extractMask(img, [255, 255, 255]))).toEqual([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
  });
});

describe("overlay and maskFraction", () => {
  it("tints the masked pixels towards the colour by the alpha and leaves the rest", () => {
    const grey = greyToRgba(Uint8Array.from([100, 100, 100, 100]), 2, 2);
    const out = overlay(grey, Uint8Array.from([1, 0, 0, 1]), [255, 0, 0]);
    expect(out).not.toBe(grey);
    expect(Array.from(out.data.slice(0, 4))).toEqual([Math.round(100 * (1 - OVERLAY_ALPHA) + 255 * OVERLAY_ALPHA), Math.round(100 * (1 - OVERLAY_ALPHA)), Math.round(100 * (1 - OVERLAY_ALPHA)), 255]);
    expect(Array.from(out.data.slice(4, 8))).toEqual([100, 100, 100, 255]);
    expect(Array.from(grey.data.slice(0, 3))).toEqual([100, 100, 100]);  // the input untouched
    expect(() => overlay(grey, new Uint8Array(3), [255, 0, 0])).toThrow(/3 pixels/);
  });

  it("maskFraction is the share of pixels set", () => {
    expect(maskFraction(Uint8Array.from([1, 0, 0, 1]))).toBe(0.5);
    expect(maskFraction(new Uint8Array(0))).toBe(0);
    expect(maskFraction(Uint8Array.from([0, 0, 0]))).toBe(0);
  });
});
