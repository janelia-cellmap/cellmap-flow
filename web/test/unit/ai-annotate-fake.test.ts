// fake.ts: the provider that calls nothing. Its rule is fake.py's, and the
// cases are tests/ai_annotate/test_backends.py's for the FakeBackend.
import { describe, expect, it } from "vitest";

import { createFakeProvider, FAKE_DESTINATION, fakeSegment } from "../../src/ai-annotate/fake";
import { FAKE_MODEL, type RgbaImage, type SegmentRequest } from "../../src/ai-annotate/types";

const RED: [number, number, number] = [255, 0, 0];

/** A grey plane, bright (200) with a dark square (30) in rows 10-19, cols 20-34. */
function plane(height = 40, width = 60): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const v = r >= 10 && r < 20 && c >= 20 && c < 35 ? 30 : 200;
      data.set([v, v, v, 255], (r * width + c) * 4);
    }
  }
  return { width, height, data };
}

function request(click: [number, number] | null, image = plane()): SegmentRequest {
  return { image, prompt: "paint it", target_rgb: RED, click_px: click };
}

function pixel(img: RgbaImage, row: number, col: number): number[] {
  return Array.from(img.data.subarray((row * img.width + col) * 4, (row * img.width + col) * 4 + 4));
}

describe("createFakeProvider", () => {
  it("describes itself as the config route shows it", () => {
    const { info } = createFakeProvider();
    expect(info).toEqual({ id: "fake", type: "fake", models: [FAKE_MODEL], destination: FAKE_DESTINATION, credentials: [] });
    expect(info.destination).toBe("nowhere (fake provider, runs in this browser)");
  });

  it("masks the dark pixels near the click, deterministically, and paints them in the model image", async () => {
    const provider = createFakeProvider();
    const first = await provider.segment(request([15, 27]), FAKE_MODEL, {});
    const second = await provider.segment(request([15, 27]), FAKE_MODEL, {});
    expect(first.mask).toEqual(second.mask);
    expect(first.mask.length).toBe(40 * 60);
    expect(first.mask[15 * 60 + 27]).toBe(1);
    expect(first.mask.reduce((n, v) => n + v, 0)).toBe(150);  // the whole dark square is within reach
    expect(first.mask[0]).toBe(0);

    const painted = first.model_image!;
    expect([painted.width, painted.height]).toEqual([60, 40]);
    expect(pixel(painted, 15, 27)).toEqual([255, 0, 0, 255]);
    expect(pixel(painted, 0, 0)).toEqual([200, 200, 200, 255]);  // untouched outside the mask
    expect(first.model).toBe(FAKE_MODEL);
    expect(first.usage).toEqual({});
  });

  it("limits the mask to a disk of min(H, W) / 4 around the click", () => {
    const far = fakeSegment(request([39, 0]), FAKE_MODEL);
    expect(far.mask.every((v) => v === 0)).toBe(true);
    // Radius 10 around (15, 27): the dark square's corner (10, 20) is at distance √74 < 10, (19, 34) at √65 < 10; but a
    // click at (5, 27) reaches rows 10-14 only, and not all of their columns.
    const edge = fakeSegment(request([5, 27]), FAKE_MODEL);
    let expected = 0;
    for (let r = 10; r < 20; r++) for (let c = 20; c < 35; c++) if ((r - 5) ** 2 + (c - 27) ** 2 <= 100) expected++;
    expect(expected).toBeGreaterThan(0);
    expect(edge.mask.reduce((n, v) => n + v, 0)).toBe(expected);
  });

  it("uses the plane's centre when there is no click", () => {
    const centred = fakeSegment(request(null), FAKE_MODEL);
    let expected = 0;
    for (let r = 10; r < 20; r++) for (let c = 20; c < 35; c++) if ((r - 20) ** 2 + (c - 30) ** 2 <= 100) expected++;
    expect(centred.mask.reduce((n, v) => n + v, 0)).toBe(expected);
    expect(centred.mask[15 * 60 + 27]).toBe(1);
  });

  it("takes dark as below the median grey, so a plane of one value masks nothing", () => {
    const flat = plane();
    flat.data.fill(200);
    for (let j = 3; j < flat.data.length; j += 4) flat.data[j] = 255;
    const nothing = fakeSegment(request([20, 30], flat), FAKE_MODEL);
    expect(nothing.mask.every((v) => v === 0)).toBe(true);
    expect(Array.from(nothing.model_image!.data)).toEqual(Array.from(flat.data));
  });
});
