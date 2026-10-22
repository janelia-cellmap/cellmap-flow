// From a model's reply to a mask (src/ai-annotate/mask.ts extractMask)
// against cellmap_flow/ai_annotate/mask_extraction.py. The parity fixture
// checks extractMask and normalizeToUint8 against the Python's own numbers;
// here the colour rule is spelled out.
import { describe, expect, it } from "vitest";

import { extractMask } from "../../src/ai-annotate/mask";
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
});
