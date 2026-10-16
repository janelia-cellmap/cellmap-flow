// images.ts: the checks a model's image passes before it is decoded (the
// magic bytes, the PNG header's size), the jsCodec stand-in's round trip and
// resamplers, and the two helpers that line a model's answer up with the
// plane (padToSquare with its median-grey rule, cropTo). The canvas codec
// needs a browser and is exercised by the e2e test.
import { describe, expect, it } from "vitest";

import {
  checkBeforeDecode, cropTo, jsCodec, medianOfBytes, padToSquare, PNG_SIGNATURE, pngDimensions, RAW_CONTAINER_TAG,
  resizeBilinear, resizeNearest, sniffImageType, toGrey,
} from "../../src/ai-annotate/images";
import { AIAnnotateError, MAX_IMAGE_PIXELS, type RgbaImage } from "../../src/ai-annotate/types";

/** A width × height image whose pixel (row, col) is `rgb(row, col)`, alpha 255. */
function image(width: number, height: number, rgb: (row: number, col: number) => [number, number, number]): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const [red, green, blue] = rgb(r, c);
      data.set([red, green, blue, 255], (r * width + c) * 4);
    }
  }
  return { width, height, data };
}

/** The grey plane the provider tests use too: 200 with a dark square (30) at rows 10-19, cols 20-34. */
function plane(height = 40, width = 60): RgbaImage {
  return image(width, height, (r, c) => (r >= 10 && r < 20 && c >= 20 && c < 35 ? [30, 30, 30] : [200, 200, 200]));
}

function pixel(img: RgbaImage, row: number, col: number): number[] {
  return Array.from(img.data.subarray((row * img.width + col) * 4, (row * img.width + col) * 4 + 4));
}

/** A PNG's first 24 bytes: the signature and an IHDR chunk declaring width × height. */
function pngHeader(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(24);
  bytes.set(PNG_SIGNATURE, 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(8, 13);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12);  // "IHDR"
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes;
}

const GIF = Uint8Array.from("GIF89a".split(""), (ch) => ch.charCodeAt(0));
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0x10, 0x4a, 0x46]);
const WEBP = Uint8Array.from("RIFF\0\0\0\0WEBPVP8 ".split(""), (ch) => ch.charCodeAt(0));

async function failure(promise: Promise<unknown>): Promise<AIAnnotateError> {
  try {
    await promise;
  } catch (e) {
    expect(e).toBeInstanceOf(AIAnnotateError);
    return e as AIAnnotateError;
  }
  throw new Error("expected the promise to reject");
}

describe("sniffImageType and pngDimensions (what the bytes say, whatever was declared)", () => {
  it("knows the PNG, JPEG and WebP signatures and nothing else", () => {
    expect(sniffImageType(pngHeader(1, 1))).toBe("image/png");
    expect(sniffImageType(JPEG)).toBe("image/jpeg");
    expect(sniffImageType(WEBP)).toBe("image/webp");
    expect(sniffImageType(GIF)).toBeNull();
    expect(sniffImageType(new Uint8Array())).toBeNull();
    expect(sniffImageType(new Uint8Array([0x89, 0x50]))).toBeNull();
  });

  it("reads a PNG's size from its IHDR without decoding, and nothing from other bytes", () => {
    expect(pngDimensions(pngHeader(640, 480))).toEqual({ width: 640, height: 480 });
    expect(pngDimensions(JPEG)).toBeNull();
    expect(pngDimensions(new Uint8Array(PNG_SIGNATURE))).toBeNull();  // too short for an IHDR
  });
});

describe("checkBeforeDecode (images.py decode_model_image's checks, with its messages)", () => {
  it("refuses an empty image", async () => {
    expect(() => checkBeforeDecode(new Uint8Array(), "image/png")).toThrow("The model returned an empty image.");
  });

  it("refuses bytes that are neither a declared nor a sniffed PNG, JPEG or WebP", () => {
    expect(() => checkBeforeDecode(GIF, "image/gif")).toThrow("The model returned an image that is not a PNG, JPEG or WebP.");
    expect(() => checkBeforeDecode(GIF, "")).toThrow(/not a PNG, JPEG or WebP/);
    let caught: unknown;
    try { checkBeforeDecode(GIF, "application/octet-stream"); } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(AIAnnotateError);
    expect((caught as AIAnnotateError).category).toBe("bad_response");
  });

  it("trusts the magic bytes over a wrong or missing declared type, and the declared type when the bytes say nothing", () => {
    expect(checkBeforeDecode(pngHeader(2, 2), "image/jpeg")).toBe("image/png");
    expect(checkBeforeDecode(pngHeader(2, 2), "")).toBe("image/png");
    expect(checkBeforeDecode(WEBP, "application/octet-stream")).toBe("image/webp");
    expect(checkBeforeDecode(GIF, "IMAGE/PNG; charset=binary")).toBe("image/png");  // the decoder decides from here
  });

  it("refuses a PNG whose header declares too many pixels, or none, before anything is decoded", () => {
    expect(() => checkBeforeDecode(pngHeader(5000, 5000), "image/png")).toThrow("The model returned a 5000x5000 image, larger than this dashboard accepts.");
    expect(() => checkBeforeDecode(pngHeader(0, 10), "image/png")).toThrow(/larger than this dashboard accepts/);
    expect(() => checkBeforeDecode(pngHeader(4096, 4096), "image/png")).not.toThrow();
    expect(4096 * 4096).toBe(MAX_IMAGE_PIXELS);
  });
});

describe("jsCodec (the stand-in: a raw container behind the PNG signature)", () => {
  const codec = jsCodec();

  it("round-trips an image through bytes that the checks take for a PNG", async () => {
    const original = plane();
    const bytes = await codec.encodePng(original);
    expect(Array.from(bytes.subarray(0, 8))).toEqual(PNG_SIGNATURE);
    expect(String.fromCharCode(...bytes.subarray(8, 12))).toBe(RAW_CONTAINER_TAG);
    expect(sniffImageType(bytes)).toBe("image/png");
    expect(pngDimensions(bytes)).toBeNull();  // no IHDR: the container is not a PNG

    const back = await codec.decode(bytes, "image/png");
    expect([back.width, back.height]).toEqual([60, 40]);
    expect(Array.from(back.data)).toEqual(Array.from(original.data));
    expect(back.data).not.toBe(original.data);
    // The bytes decide: a wrong or missing declared type is fine.
    expect((await codec.decode(bytes, "image/jpeg")).width).toBe(60);
    expect((await codec.decode(bytes, "")).width).toBe(60);
  });

  it("refuses what the shared checks refuse, with their messages", async () => {
    expect((await failure(codec.decode(new Uint8Array(), "image/png"))).userMessage).toBe("The model returned an empty image.");
    expect((await failure(codec.decode(GIF, "image/gif"))).userMessage).toMatch(/not a PNG, JPEG or WebP/);
  });

  it("refuses a container declaring too many pixels before reading them, and one it cannot read", async () => {
    const huge = new Uint8Array(20);
    huge.set(PNG_SIGNATURE, 0);
    huge.set([0x52, 0x47, 0x42, 0x41], 8);  // "RGBA"
    new DataView(huge.buffer).setUint32(12, 4097);
    new DataView(huge.buffer).setUint32(16, 4096);
    const refused = await failure(codec.decode(huge, "image/png"));
    expect(refused.category).toBe("bad_response");
    expect(refused.userMessage).toBe("The model returned a 4097x4096 image, larger than this dashboard accepts.");

    const truncated = (await codec.encodePng(plane())).subarray(0, 100);
    expect((await failure(codec.decode(truncated, "image/png"))).userMessage).toBe("The model returned an image that could not be decoded.");
    // A real PNG header is a PNG to the checks, but not something this stand-in can decode.
    expect((await failure(codec.decode(pngHeader(2, 2), "image/png"))).userMessage).toMatch(/could not be decoded/);
  });

  it("resizes by nearest neighbour or bilinearly, and returns the image itself at its own size", async () => {
    const img = plane();
    expect(await codec.resize(img, 60, 40, true)).toBe(img);
    const nearest = await codec.resize(image(2, 2, (r, c) => ((r + c) % 2 ? [255, 255, 255] : [0, 0, 0])), 4, 4, false);
    expect(pixel(nearest, 0, 0)).toEqual([0, 0, 0, 255]);
    expect(pixel(nearest, 1, 1)).toEqual([0, 0, 0, 255]);
    expect(pixel(nearest, 0, 2)).toEqual([255, 255, 255, 255]);
    expect(pixel(nearest, 3, 2)).toEqual([0, 0, 0, 255]);
    const ramp = await codec.resize(image(2, 1, (_r, c) => (c ? [255, 255, 255] : [0, 0, 0])), 4, 1, true);
    expect([0, 1, 2, 3].map((c) => pixel(ramp, 0, c)[0])).toEqual([0, 64, 191, 255]);
  });
});

describe("resizeNearest and resizeBilinear", () => {
  it("keep a flat image flat and a block's interior intact when shrinking", () => {
    const flat = image(7, 5, () => [90, 90, 90]);
    for (const out of [resizeNearest(flat, 3, 2), resizeBilinear(flat, 3, 2), resizeBilinear(flat, 13, 9)]) {
      expect(new Set(out.data)).toEqual(new Set([90, 255]));
    }
    // A 120×120 reply with a red block at rows 20-39, cols 40-69 shrunk to 60×60: the block is at rows 10-19, cols 20-34, its interior pure red.
    const big = image(120, 120, (r, c) => (r >= 20 && r < 40 && c >= 40 && c < 70 ? [255, 0, 0] : [0, 0, 0]));
    const small = resizeBilinear(big, 60, 60);
    expect(pixel(small, 15, 27)).toEqual([255, 0, 0, 255]);
    expect(pixel(small, 11, 21)).toEqual([255, 0, 0, 255]);
    expect(pixel(small, 5, 5)).toEqual([0, 0, 0, 255]);
    expect(resizeNearest(big, 60, 60).data).toEqual(small.data);  // exact 2:1, so both agree
  });
});

describe("toGrey and medianOfBytes (PIL's L and numpy's median)", () => {
  it("converts colours as PIL does and leaves grey alone", () => {
    const img = image(4, 1, (_r, c) => [[255, 0, 0], [0, 255, 0], [0, 0, 255], [77, 77, 77]][c] as [number, number, number]);
    expect(Array.from(toGrey(img))).toEqual([76, 150, 29, 77]);
  });

  it("is the middle value, or the mean of the two middle ones", () => {
    expect(medianOfBytes(new Uint8Array([250, 10, 200]))).toBe(200);
    expect(medianOfBytes(new Uint8Array([10, 20, 200, 250]))).toBe(110);
    expect(medianOfBytes(new Uint8Array([100, 101]))).toBe(100.5);
    expect(medianOfBytes(new Uint8Array([7]))).toBe(7);
    expect(medianOfBytes(new Uint8Array())).toBe(0);
    const values = Uint8Array.from({ length: 1001 }, (_v, i) => (i * 7919) % 256);
    const sorted = Array.from(values).sort((a, b) => a - b);
    expect(medianOfBytes(values)).toBe(sorted[500]);
  });
});

describe("padToSquare (vertex_gemini.py _square)", () => {
  it("pads a wide plane below with its median grey and leaves the plane's pixels where they were", () => {
    const { image: padded, padded: added } = padToSquare(plane());
    expect(added).toBe(true);
    expect([padded.width, padded.height]).toEqual([60, 60]);
    const original = plane();
    expect(Array.from(padded.data.subarray(0, 60 * 40 * 4))).toEqual(Array.from(original.data));
    for (let r = 40; r < 60; r++) for (const c of [0, 30, 59]) expect(pixel(padded, r, c)).toEqual([200, 200, 200, 255]);
  });

  it("pads a tall plane on the right", () => {
    const tall = plane(60, 40);
    const { image: padded } = padToSquare(tall);
    expect([padded.width, padded.height]).toEqual([60, 60]);
    expect(pixel(padded, 15, 27)).toEqual([30, 30, 30, 255]);  // the dark square, where it was
    expect(pixel(padded, 15, 45)).toEqual([200, 200, 200, 255]);  // the padding
    expect(pixel(padded, 59, 59)).toEqual([200, 200, 200, 255]);
  });

  it("uses the integer part of numpy's median of PIL's grey", () => {
    const even = padToSquare(image(4, 1, (_r, c) => [[10, 10, 10], [20, 20, 20], [200, 200, 200], [250, 250, 250]][c] as [number, number, number]));
    expect(pixel(even.image, 3, 0)).toEqual([110, 110, 110, 255]);
    const halves = padToSquare(image(2, 1, (_r, c) => (c ? [101, 101, 101] : [100, 100, 100])));
    expect(pixel(halves.image, 1, 1)).toEqual([100, 100, 100, 255]);  // int(100.5)
    const colour = padToSquare(image(3, 1, (_r, c) => [[255, 0, 0], [0, 255, 0], [0, 0, 255]][c] as [number, number, number]));
    expect(pixel(colour.image, 2, 2)).toEqual([76, 76, 76, 255]);  // L of red is the middle of 29, 76, 150
  });

  it("returns a square image as it is", () => {
    const square = plane(60, 60);
    const result = padToSquare(square);
    expect(result.padded).toBe(false);
    expect(result.image).toBe(square);
  });
});

describe("cropTo", () => {
  it("keeps the top-left box, returns the image itself when nothing is cropped, and refuses a bigger box", () => {
    const padded = padToSquare(plane()).image;
    const cropped = cropTo(padded, 60, 40);
    expect([cropped.width, cropped.height]).toEqual([60, 40]);
    expect(Array.from(cropped.data)).toEqual(Array.from(plane().data));
    const narrower = cropTo(padded, 10, 60);
    expect(pixel(narrower, 15, 9)).toEqual([200, 200, 200, 255]);
    expect(narrower.data.length).toBe(10 * 60 * 4);
    expect(cropTo(padded, 60, 60)).toBe(padded);
    expect(() => cropTo(padded, 61, 60)).toThrow(/cannot crop a 60×60 image to 61×60/);
  });
});
