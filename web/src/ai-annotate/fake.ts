// A provider that calls nothing and runs in this browser: the port of
// cellmap_flow/ai_annotate/backends/fake.py. It stands in for a hosted model
// wherever the page's own behaviour is what is being checked or tried: the
// panel, the review, Accept and Reject, Undo, with no key and nothing sent.
// Its mask is deterministic and plausible on EM: the pixels darker than the
// plane's median within a disk around the click, which on most EM picks out
// membranes and dense organelles near where the user pointed.
import { medianOfBytes, toGrey } from "./images";
import { FAKE_MODEL, type Provider, type RgbaImage, type SegmentRequest, type SegmentResult } from "./types";

export const FAKE_DESTINATION = "nowhere (fake provider, runs in this browser)";

export function createFakeProvider(): Provider {
  return {
    info: { id: "fake", type: "fake", models: [FAKE_MODEL], destination: FAKE_DESTINATION, credentials: [] },
    async segment(request, model) {
      return fakeSegment(request, model);
    },
  };
}

/** fake.py's rule: the pixels below the plane's median grey (PIL's "L" of
 * the image, numpy's median) within min(H, W) / 4 of the click, or of the
 * centre when there is none; and the input with those pixels painted
 * `target_rgb` as the "model's" image. */
export function fakeSegment(request: SegmentRequest, model: string): SegmentResult {
  const { image, target_rgb, click_px } = request;
  const { width, height } = image;
  const grey = toGrey(image);
  const median = medianOfBytes(grey);
  const [row, col] = click_px ?? [height / 2, width / 2];
  const radiusSquared = (Math.min(height, width) / 4) ** 2;

  const mask = new Uint8Array(width * height);
  const painted = new Uint8ClampedArray(image.data);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const i = r * width + c;
      if ((r - row) ** 2 + (c - col) ** 2 <= radiusSquared && grey[i] < median) {
        mask[i] = 1;
        painted[i * 4] = target_rgb[0];
        painted[i * 4 + 1] = target_rgb[1];
        painted[i * 4 + 2] = target_rgb[2];
        painted[i * 4 + 3] = 255;
      }
    }
  }
  const model_image: RgbaImage = { width, height, data: painted };
  return { mask, model_image, model, usage: {} };
}
