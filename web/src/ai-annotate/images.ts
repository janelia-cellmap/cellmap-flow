// Images for AI-assisted annotation: encoding the plane that is sent, checking
// and decoding the image a model sends back, and the padding the Gemini
// provider sends the plane with (lining its answer up with the plane again
// is the Python's: mask.ts modelMask). The port of
// cellmap_flow/ai_annotate/images.py (decode_model_image: the checks a
// model's image must pass before it is decoded) and of vertex_gemini.py's
// _square, for the browser, where the canvas is the only image codec there
// is. See PLAN_browser_finetune.md, "AI-assisted annotation in the browser".
//
// A model's image is untrusted input: it could be huge, malformed or in a
// format we never asked for. images.py reads the header first (PIL opens
// lazily), checks the format and the pixel count, and decodes only then. The
// browser's decoder (createImageBitmap) decodes in one go, so the checks
// here work on the bytes themselves: the magic bytes say what the format is
// (a model may declare the wrong MIME type, so the declared type alone is
// not trusted), a PNG's IHDR gives its size before anything is decoded, and
// for a JPEG or WebP the size is checked as soon as the bitmap exists. The
// messages are images.py's, so the panel says the same thing in both modes.
//
// Two codecs implement ImageCodec (types.ts):
//   - canvasCodec(): the page's, over OffscreenCanvas (or a <canvas> where
//     there is none) and createImageBitmap. It needs a browser.
//   - jsCodec(): a plain-JavaScript stand-in for the unit tests and Node,
//     where there is no canvas. It writes no real PNG: its bytes are a tiny
//     container of the raw RGBA pixels (see RAW_CONTAINER_TAG), prefixed with
//     the PNG signature so that the checks above treat them as a PNG and the
//     Gemini provider's whole flow (encode, send, decode) runs unchanged in a
//     test. It cannot decode a real PNG.
import { AIAnnotateError, ALLOWED_IMAGE_TYPES, type ImageCodec, MAX_IMAGE_PIXELS, type RgbaImage } from "./types";

// ---------------------------------------------------------------------------
// What a model's bytes are: the magic bytes decide, not the declared type
// ---------------------------------------------------------------------------

/** The eight bytes every PNG starts with. */
export const PNG_SIGNATURE: readonly number[] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** The image format `bytes` begin with, as a MIME type, or null when they
 * begin like none of the formats a model's image may be in (a PNG's
 * signature, a JPEG's SOI marker, a RIFF container tagged WEBP). */
export function sniffImageType(bytes: Uint8Array): "image/png" | "image/jpeg" | "image/webp" | null {
  if (bytes.length >= 8 && PNG_SIGNATURE.every((b, i) => bytes[i] === b)) return "image/png";
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "image/jpeg";
  if (bytes.length >= 12 && ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "WEBP") return "image/webp";
  return null;
}

/** The width and height a PNG's IHDR chunk declares, read without decoding
 * anything, or null when `bytes` are not a PNG with an IHDR where one belongs. */
export function pngDimensions(bytes: Uint8Array): { width: number; height: number } | null {
  if (sniffImageType(bytes) !== "image/png" || bytes.length < 24 || ascii(bytes, 12, 16) !== "IHDR") return null;
  return { width: u32be(bytes, 16), height: u32be(bytes, 20) };
}

/** `mimeType` without parameters or case, so "image/PNG; charset=binary" is "image/png". */
function normalizeType(mimeType: string): string {
  return (mimeType ?? "").split(";")[0].trim().toLowerCase();
}

/** The checks a model's image passes before any pixel is decoded
 * (images.py decode_model_image's first half): not empty; a PNG, JPEG or
 * WebP by its declared type or, since a model may declare the wrong one, by
 * its magic bytes; and, for a PNG, a size within MAX_IMAGE_PIXELS by its
 * header. Returns the type to decode the bytes as: what the bytes say when
 * they say something, else what was declared. Throws AIAnnotateError
 * ("bad_response") with images.py's messages otherwise. */
export function checkBeforeDecode(bytes: Uint8Array, mimeType: string): string {
  if (!(bytes instanceof Uint8Array) || bytes.length === 0) {
    throw new AIAnnotateError("bad_response", "The model returned an empty image.");
  }
  const sniffed = sniffImageType(bytes);
  const declared = normalizeType(mimeType);
  if (sniffed === null && !ALLOWED_IMAGE_TYPES.includes(declared)) {
    throw new AIAnnotateError("bad_response", "The model returned an image that is not a PNG, JPEG or WebP.");
  }
  const header = pngDimensions(bytes);
  if (header) checkPixelCount(header.width, header.height);
  return sniffed ?? declared;
}

/** Refuses a `width` × `height` image that is empty or has more than
 * MAX_IMAGE_PIXELS pixels, with images.py's message. */
export function checkPixelCount(width: number, height: number): void {
  if (!(width >= 1) || !(height >= 1) || width * height > MAX_IMAGE_PIXELS) {
    throw new AIAnnotateError("bad_response", `The model returned a ${width}x${height} image, larger than this dashboard accepts.`);
  }
}

const COULD_NOT_DECODE = "The model returned an image that could not be decoded.";

// ---------------------------------------------------------------------------
// Grey levels: PIL's "L" conversion and numpy's median, which two rules use
// ---------------------------------------------------------------------------

/** `image` as grey levels, one byte per pixel, as PIL's convert("L")
 * computes them: L = R × 299/1000 + G × 587/1000 + B × 114/1000, rounded to
 * the nearest integer through the same fixed-point arithmetic as PIL's C
 * code (so a grey pixel stays exactly itself). */
export function toGrey(image: RgbaImage): Uint8Array {
  const { width, height, data } = image;
  const grey = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < grey.length; i++, j += 4) {
    grey[i] = (data[j] * 19595 + data[j + 1] * 38470 + data[j + 2] * 7471 + 0x8000) >>> 16;
  }
  return grey;
}

/** The median of byte values as numpy computes it: the middle value, or the
 * mean of the two middle values when there is an even number (so it can end
 * in .5). Counted through a histogram, as there are only 256 values. */
export function medianOfBytes(values: Uint8Array): number {
  if (values.length === 0) return 0;
  const counts = new Uint32Array(256);
  for (let i = 0; i < values.length; i++) counts[values[i]]++;
  // The sorted positions (0-based) of the two middle values; equal when odd.
  const lower = Math.floor((values.length - 1) / 2), upper = Math.ceil((values.length - 1) / 2);
  let seen = 0, lowerValue = -1, upperValue = -1;
  for (let v = 0; v < 256 && upperValue < 0; v++) {
    seen += counts[v];
    if (lowerValue < 0 && seen > lower) lowerValue = v;
    if (seen > upper) upperValue = v;
  }
  return (lowerValue + upperValue) / 2;
}

// ---------------------------------------------------------------------------
// The plane padded square for the model (vertex_gemini.py _square)
// ---------------------------------------------------------------------------

/** `image` padded on the right or bottom to a square with its median grey
 * (vertex_gemini.py _square), and whether anything was added.
 *
 * Gemini's image models answer in a few fixed aspect ratios, and a square
 * one for a square input; a plane clipped at the volume's edge is not
 * square, and resizing the answer back to it would stretch the mask. The
 * padding is a flat grey rather than black, which the prompt asks the model
 * to paint everything else, so it reads as nothing rather than as a
 * structure's dark edge; the provider crops it off the answer (mask.ts modelMask). The
 * grey is the integer part of the median (int(np.median(...))). A square
 * image comes back as it is. */
export function padToSquare(image: RgbaImage): { image: RgbaImage; padded: boolean } {
  const { width, height } = image;
  if (width === height) return { image, padded: false };
  const side = Math.max(width, height);
  const grey = Math.trunc(medianOfBytes(toGrey(image)));
  const data = new Uint8ClampedArray(side * side * 4);
  for (let j = 0; j < data.length; j += 4) {
    data[j] = data[j + 1] = data[j + 2] = grey;
    data[j + 3] = 255;
  }
  for (let row = 0; row < height; row++) {
    data.set(image.data.subarray(row * width * 4, (row + 1) * width * 4), row * side * 4);
  }
  return { image: { width: side, height: side, data }, padded: true };
}

// ---------------------------------------------------------------------------
// The page's codec: the canvas
// ---------------------------------------------------------------------------

type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/** A canvas of `width` × `height` with its 2D context: an OffscreenCanvas
 * where the page (or a worker) has one, else a <canvas> element. */
function makeCanvas(width: number, height: number): { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } {
  let canvas: OffscreenCanvas | HTMLCanvasElement;
  if (typeof OffscreenCanvas !== "undefined") {
    canvas = new OffscreenCanvas(width, height);
  } else if (typeof document !== "undefined") {
    canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
  } else {
    throw new Error("no canvas here: images need a browser (tests use jsCodec)");
  }
  const ctx = canvas.getContext("2d") as Canvas2D | null;
  if (!ctx) throw new Error("the canvas gave no 2D context");
  return { canvas, ctx };
}

/** `image` drawn on a fresh canvas, for encoding. */
function canvasOf(image: RgbaImage): { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } {
  const made = makeCanvas(image.width, image.height);
  // ImageData wants a Uint8ClampedArray over an ArrayBuffer of its own; the copy also keeps the caller's pixels untouched.
  made.ctx.putImageData(new ImageData(new Uint8ClampedArray(image.data), image.width, image.height), 0, 0);
  return made;
}

/** The pixels of `ctx`'s whole canvas as an RgbaImage, alpha forced to 255:
 * a model's image may carry transparency, and PIL's convert("RGB") drops the
 * alpha channel rather than compositing it, so the canvas's colour values
 * are kept and the alpha set opaque. */
function readPixels(ctx: Canvas2D, width: number, height: number): RgbaImage {
  const data = new Uint8ClampedArray(ctx.getImageData(0, 0, width, height).data);
  for (let j = 3; j < data.length; j += 4) data[j] = 255;
  return { width, height, data };
}

/** The ImageCodec the page uses: PNG encoding through the canvas's own
 * encoder, decoding through createImageBitmap (PNG, JPEG and WebP, as every
 * browser decodes them). */
export function canvasCodec(): ImageCodec {
  return {
    async encodePng(image) {
      const { canvas } = canvasOf(image);
      const blob = canvas instanceof OffscreenCanvas
        ? await canvas.convertToBlob({ type: "image/png" })
        : await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("the canvas could not encode a PNG");
      return new Uint8Array(await blob.arrayBuffer());
    },

    async decode(bytes, mimeType) {
      const type = checkBeforeDecode(bytes, mimeType);
      let bitmap: ImageBitmap;
      try {
        // A copy over a plain ArrayBuffer: Blob takes no view over a SharedArrayBuffer.
        bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type }));
      } catch {
        throw new AIAnnotateError("bad_response", COULD_NOT_DECODE);
      }
      try {
        // A JPEG or WebP is sized only now; a PNG was checked from its header already.
        checkPixelCount(bitmap.width, bitmap.height);
        const { ctx } = makeCanvas(bitmap.width, bitmap.height);
        ctx.drawImage(bitmap, 0, 0);
        return readPixels(ctx, bitmap.width, bitmap.height);
      } finally {
        bitmap.close();
      }
    },
  };
}

// ---------------------------------------------------------------------------
// The stand-in codec for tests and Node
// ---------------------------------------------------------------------------

/** The four bytes after the PNG signature that mark jsCodec's container.
 * Layout: PNG_SIGNATURE (8 bytes), "RGBA", width u32 big-endian, height u32
 * big-endian, then width × height × 4 bytes of RGBA. */
export const RAW_CONTAINER_TAG = "RGBA";
const RAW_HEADER_BYTES = 8 + 4 + 4 + 4;

/** A pure-JavaScript ImageCodec for the unit tests and Node: `encodePng`
 * writes the raw container described at RAW_CONTAINER_TAG (not a PNG, though
 * it begins with one's signature so that checkBeforeDecode treats it as
 * one), and `decode` reads it back after the same checks the canvas codec
 * makes. It cannot decode a real PNG, JPEG or WebP: those fail as "could
 * not be decoded". */
export function jsCodec(): ImageCodec {
  return {
    async encodePng(image) {
      const bytes = new Uint8Array(RAW_HEADER_BYTES + image.data.length);
      bytes.set(PNG_SIGNATURE, 0);
      for (let i = 0; i < 4; i++) bytes[8 + i] = RAW_CONTAINER_TAG.charCodeAt(i);
      const view = new DataView(bytes.buffer);
      view.setUint32(12, image.width);
      view.setUint32(16, image.height);
      bytes.set(image.data, RAW_HEADER_BYTES);
      return bytes;
    },

    async decode(bytes, mimeType) {
      checkBeforeDecode(bytes, mimeType);
      if (bytes.length < RAW_HEADER_BYTES || ascii(bytes, 8, 12) !== RAW_CONTAINER_TAG) {
        throw new AIAnnotateError("bad_response", COULD_NOT_DECODE);
      }
      const width = u32be(bytes, 12), height = u32be(bytes, 16);
      checkPixelCount(width, height);
      if (bytes.length !== RAW_HEADER_BYTES + width * height * 4) {
        throw new AIAnnotateError("bad_response", COULD_NOT_DECODE);
      }
      return { width, height, data: new Uint8ClampedArray(bytes.subarray(RAW_HEADER_BYTES)) };
    },

  };
}

// ---------------------------------------------------------------------------
// Bytes
// ---------------------------------------------------------------------------

function ascii(bytes: Uint8Array, from: number, to: number): string {
  let s = "";
  for (let i = from; i < to; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

function u32be(bytes: Uint8Array, at: number): number {
  return ((bytes[at] << 24) >>> 0) + (bytes[at + 1] << 16) + (bytes[at + 2] << 8) + bytes[at + 3];
}
