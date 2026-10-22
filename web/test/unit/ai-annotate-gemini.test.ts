// gemini.ts against a scripted fetch: the request's URL, headers and JSON for
// both endpoints, the answer parsed (thought parts skipped, the last image
// taken, the padding cropped, another size resized), the retries and their
// waits, the failures' categories and statuses, and that no message ever
// carries the key, the token or the service's body. The Vertex cases follow
// tests/ai_annotate/test_backends.py's; the Gemini API has its own
// (listModels, the key header, API_KEY_INVALID).
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  base64ToBytes, bytesToBase64, createGeminiProvider, type FetchLike, GEMINI_API_BASE, type GeminiType,
  looksLikeAccessToken, MAX_ATTEMPTS, RETRY_WAITS_MS, TOKEN_AS_KEY_MESSAGE, VERTEX_BASE,
} from "../../src/ai-annotate/gemini";
import { jsCodec, PNG_SIGNATURE } from "../../src/ai-annotate/images";
import { AIAnnotateError, CALL_TIMEOUT_MS, type Provider, type RgbaImage, type SegmentRequest } from "../../src/ai-annotate/types";

const codec = jsCodec();
const RED: [number, number, number] = [255, 0, 0];
const KEY = "AIzaSECRET-KEY-0123456789";
const TOKEN = "ya29.SECRET-TOKEN-abcdef";
const PROJECT = "my-gcp-project";
const BODY_DETAIL = `INTERNAL-DETAIL projects/secret-proj/locations/global quota_metric=xyz key=${KEY} token=${TOKEN}`;
const SECRETS = [KEY, TOKEN, "INTERNAL-DETAIL", "secret-proj", "quota_metric"];

// --- images -----------------------------------------------------------------

/** A width × height image whose pixel (row, col) is `rgb(row, col)`. */
function image(width: number, height: number, rgb: (row: number, col: number) => [number, number, number]): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) data.set([...rgb(r, c), 255], (r * width + c) * 4);
  }
  return { width, height, data };
}

/** A grey plane, bright (200) with a dark square (30) in rows 10-19, cols 20-34. */
function plane(height = 40, width = 60): RgbaImage {
  return image(width, height, (r, c) => (r >= 10 && r < 20 && c >= 20 && c < 35 ? [30, 30, 30] : [200, 200, 200]));
}

/** A good answer to the 60×40 plane, which is sent padded to 60×60: the dark
 * square red on black, scaled by `side / 60` as a model answering bigger would. */
function paintedSquare(side = 60): RgbaImage {
  const k = side / 60;
  return image(side, side, (r, c) => (r >= 10 * k && r < 20 * k && c >= 20 * k && c < 35 * k ? RED : [0, 0, 0]));
}

function request(img = plane(), click: [number, number] | null = null): SegmentRequest {
  return { image: img, prompt: "paint it", target_rgb: RED, click_px: click };
}

function pixel(img: RgbaImage, row: number, col: number): number[] {
  return Array.from(img.data.subarray((row * img.width + col) * 4, (row * img.width + col) * 4 + 4));
}

function maskCount(mask: Uint8Array): number {
  return mask.reduce((n, v) => n + v, 0);
}

// --- answers ----------------------------------------------------------------

type Part = Record<string, unknown>;

async function imagePart(img: RgbaImage, mimeType = "image/png", extra: Part = {}): Promise<Part> {
  return { inlineData: { mimeType, data: bytesToBase64(await codec.encodePng(img)) }, ...extra };
}

function answer(parts: Part[], options: { finishReason?: string } = {}): Record<string, unknown> {
  return {
    usageMetadata: { promptTokenCount: 600, candidatesTokenCount: 1300, totalTokenCount: 1900 },
    candidates: [{ content: { role: "model", parts }, finishReason: options.finishReason ?? "STOP" }],
  };
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** An error answer shaped like Google's, whose message echoes the request and the credentials. */
function apiError(code: number, status: string, reasons: string[] = []): Response {
  const details = reasons.map((reason) => ({ "@type": "type.googleapis.com/google.rpc.ErrorInfo", reason, domain: "googleapis.com" }));
  return json(code, { error: { code, message: `${status}. ${BODY_DETAIL}`, status, details } });
}

// --- the scripted fetch -----------------------------------------------------

interface Call { url: string; method: string; headers: Record<string, string>; body: unknown }
type Scripted = (() => Promise<Response>) | Error;

/** Answers each call from the script in order, recording what was asked. */
class FakeFetch {
  calls: Call[] = [];
  constructor(private script: Scripted[]) {}
  fetch: FetchLike = async (url, init = {}) => {
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries((init.headers ?? {}) as Record<string, string>)) headers[k.toLowerCase()] = v;
    this.calls.push({ url, method: init.method ?? "GET", headers, body: typeof init.body === "string" ? JSON.parse(init.body) : null });
    const next = this.script.shift();
    if (!next) throw new Error("the fake fetch's script is exhausted");
    if (next instanceof Error) throw next;
    return await next();
  };
}

function provider(type: GeminiType, fake: FakeFetch, sleeps: number[] = []): Provider {
  return createGeminiProvider(type, { fetch: fake.fetch, codec, sleep: async (ms) => { sleeps.push(ms); } });
}

const GEMINI_CREDENTIALS = { api_key: KEY };
const VERTEX_CREDENTIALS = { project: PROJECT, access_token: TOKEN };

async function failure(promise: Promise<unknown>): Promise<AIAnnotateError> {
  try {
    await promise;
  } catch (e) {
    expect(e).toBeInstanceOf(AIAnnotateError);
    return e as AIAnnotateError;
  }
  throw new Error("expected the call to fail");
}

function expectNoSecrets(error: AIAnnotateError): void {
  for (const text of [error.message, error.userMessage, String(error)]) {
    for (const secret of SECRETS) expect(text).not.toContain(secret);
  }
}

afterEach(() => {
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------

describe("the request", () => {
  it("sends the plane padded square as PNG with the prompt to the Gemini API with the key header, and reads the mask", async () => {
    const draft = image(60, 60, () => RED);  // a thought image: all red, which would mask everything if taken
    const fake = new FakeFetch([async () => json(200, answer([
      await imagePart(draft, "image/png", { thought: true }),
      { text: "Here you go." },
      await imagePart(paintedSquare()),
    ]))]);
    const result = await provider("gemini_api", fake).segment(request(), "gemini-3-pro-image", GEMINI_CREDENTIALS);

    expect(fake.calls).toHaveLength(1);
    const [call] = fake.calls;
    expect(call.url).toBe(`${GEMINI_API_BASE}/models/gemini-3-pro-image:generateContent`);
    expect(call.method).toBe("POST");
    expect(call.headers["x-goog-api-key"]).toBe(KEY);
    expect(call.headers["content-type"]).toBe("application/json");
    expect(call.headers.authorization).toBeUndefined();

    const body = call.body as { contents: { role: string; parts: Part[] }[]; generationConfig: unknown };
    expect(body.generationConfig).toEqual({ responseModalities: ["IMAGE", "TEXT"] });
    expect(body.contents).toHaveLength(1);
    expect(body.contents[0].role).toBe("user");
    const [imageSent, textSent] = body.contents[0].parts;
    expect(textSent).toEqual({ text: "paint it" });
    const inline = imageSent.inlineData as { mimeType: string; data: string };
    expect(inline.mimeType).toBe("image/png");
    const sent = await codec.decode(base64ToBytes(inline.data), inline.mimeType);
    // Padded to a square below the plane, with flat grey, and not upsampled.
    expect([sent.width, sent.height]).toEqual([60, 60]);
    expect(Array.from(sent.data.subarray(0, 60 * 40 * 4))).toEqual(Array.from(plane().data));
    for (let r = 40; r < 60; r++) expect(pixel(sent, r, r)).toEqual([200, 200, 200, 255]);

    expect(result.mask.length).toBe(40 * 60);
    expect(maskCount(result.mask)).toBe(150);
    for (let r = 10; r < 20; r++) for (let c = 20; c < 35; c++) expect(result.mask[r * 60 + c]).toBe(1);
    expect([result.model_image!.width, result.model_image!.height]).toEqual([60, 40]);
    expect(pixel(result.model_image!, 15, 27)).toEqual([255, 0, 0, 255]);
    expect(result.model).toBe("gemini-3-pro-image");
    expect(result.usage).toEqual({ prompt_tokens: 600, output_tokens: 1300, total_tokens: 1900 });
  });

  it("sends to Vertex AI with the project, the location (global unless given) and the bearer token", async () => {
    const fake = new FakeFetch([async () => json(200, answer([await imagePart(paintedSquare())])), async () => json(200, answer([await imagePart(paintedSquare())]))]);
    const vertex = provider("vertex_gemini", fake);
    await vertex.segment(request(), "gemini-3-pro-image", VERTEX_CREDENTIALS);
    await vertex.segment(request(), "gemini-3-pro-image", { ...VERTEX_CREDENTIALS, location: "us-central1" });

    expect(fake.calls[0].url).toBe(`${VERTEX_BASE}/projects/${PROJECT}/locations/global/publishers/google/models/gemini-3-pro-image:generateContent`);
    expect(fake.calls[1].url).toContain("/locations/us-central1/");
    expect(fake.calls[0].headers.authorization).toBe(`Bearer ${TOKEN}`);
    expect(fake.calls[0].headers["x-goog-api-key"]).toBeUndefined();
    expect((fake.calls[0].body as { generationConfig: unknown }).generationConfig).toEqual({ responseModalities: ["IMAGE", "TEXT"] });
  });

  it("refuses to run without the provider's credentials, before any call", async () => {
    const fake = new FakeFetch([]);
    for (const credentials of [{}, { api_key: "   " }, { project: PROJECT, access_token: TOKEN }]) {
      const error = await failure(provider("gemini_api", fake).segment(request(), "m", credentials));
      expect([error.category, error.httpStatus]).toEqual(["config", 400]);
    }
    for (const credentials of [{}, { project: PROJECT }, { access_token: TOKEN }, { api_key: KEY }]) {
      const error = await failure(provider("vertex_gemini", fake).segment(request(), "m", credentials));
      expect([error.category, error.httpStatus]).toEqual(["config", 400]);
    }
    expect(fake.calls).toEqual([]);
  });
});

describe("the answer", () => {
  it("is a bad response without an image, quoting at most 200 plain characters of the model's text and the finish reason", async () => {
    const reply = "I can't do that. <script>alert(1)</script>\x00\x1b" + "x".repeat(500);
    const fake = new FakeFetch([async () => json(200, answer([{ text: reply }], { finishReason: "IMAGE_SAFETY" }))]);
    const error = await failure(provider("gemini_api", fake).segment(request(), "m", GEMINI_CREDENTIALS));
    expect(error.category).toBe("bad_response");
    expect(error.userMessage).toContain("IMAGE_SAFETY");
    expect(error.userMessage).toContain("I can't do that.");
    expect(error.userMessage).not.toMatch(/[\x00\x1b]/);
    expect(error.userMessage).not.toContain("x".repeat(201));
  });

  it("resizes an image of another size back to the sent size, then crops the padding off", async () => {
    const fake = new FakeFetch([
      async () => json(200, answer([await imagePart(paintedSquare(120))])),
      async () => json(200, answer([await imagePart(image(120, 80, (r, c) => (r >= 20 && r < 40 && c >= 40 && c < 70 ? RED : [0, 0, 0])))])),
    ]);
    const gemini = provider("gemini_api", fake);
    const bigger = await gemini.segment(request(), "m", GEMINI_CREDENTIALS);
    expect([bigger.model_image!.width, bigger.model_image!.height]).toEqual([60, 40]);
    expect(bigger.mask.length).toBe(40 * 60);
    for (let r = 12; r < 18; r++) for (let c = 22; c < 33; c++) expect(bigger.mask[r * 60 + c]).toBe(1);
    expect(bigger.mask[0]).toBe(0);
    // Another aspect ratio is stretched to fit (what the preview then shows), never an error.
    const stretched = await gemini.segment(request(), "m", GEMINI_CREDENTIALS);
    expect([stretched.model_image!.width, stretched.model_image!.height]).toEqual([60, 40]);
    expect(maskCount(stretched.mask)).toBeGreaterThan(0);
  });

  it("decodes by the bytes when the model declares the wrong type, and refuses what is not an image", async () => {
    const gifBytes = Uint8Array.from("GIF89a-not-an-image".split(""), (ch) => ch.charCodeAt(0));
    const huge = new Uint8Array(20);
    huge.set(PNG_SIGNATURE, 0);
    huge.set([0x52, 0x47, 0x42, 0x41], 8);
    new DataView(huge.buffer).setUint32(12, 5000);
    new DataView(huge.buffer).setUint32(16, 5000);
    const fake = new FakeFetch([
      async () => json(200, answer([await imagePart(paintedSquare(), "image/jpeg")])),
      async () => json(200, answer([await imagePart(paintedSquare(), "")])),
      async () => json(200, answer([{ inlineData: { mimeType: "image/gif", data: bytesToBase64(gifBytes) } }])),
      async () => json(200, answer([{ inlineData: { mimeType: "image/png", data: "!!!not base64!!!" } }])),
      async () => json(200, answer([{ inlineData: { mimeType: "image/png", data: bytesToBase64(huge) } }])),
    ]);
    const gemini = provider("gemini_api", fake);
    expect(maskCount((await gemini.segment(request(), "m", GEMINI_CREDENTIALS)).mask)).toBe(150);
    expect(maskCount((await gemini.segment(request(), "m", GEMINI_CREDENTIALS)).mask)).toBe(150);
    const gif = await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    expect(gif.category).toBe("bad_response");
    expect(gif.userMessage).toMatch(/not a PNG/);
    const garbage = await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    expect(garbage.userMessage).toMatch(/could not be decoded/);
    // Refused by the header's size, before the 25-megapixel image is decoded.
    const tooBig = await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    expect(tooBig.category).toBe("bad_response");
    expect(tooBig.userMessage).toMatch(/5000x5000/);
  });
});

describe("retries", () => {
  it("retries a 429 and a 503 with the bounded waits, then succeeds", async () => {
    const sleeps: number[] = [];
    const fake = new FakeFetch([
      async () => apiError(429, "RESOURCE_EXHAUSTED"),
      async () => apiError(503, "UNAVAILABLE"),
      async () => json(200, answer([await imagePart(paintedSquare())])),
    ]);
    const result = await provider("gemini_api", fake, sleeps).segment(request(), "m", GEMINI_CREDENTIALS);
    expect(fake.calls).toHaveLength(3);
    expect(sleeps).toEqual([...RETRY_WAITS_MS]);
    expect(maskCount(result.mask)).toBe(150);
  });

  it("stops after three attempts with a quota error", async () => {
    const sleeps: number[] = [];
    const fake = new FakeFetch(Array.from({ length: 10 }, () => async () => apiError(429, "RESOURCE_EXHAUSTED")));
    const error = await failure(provider("gemini_api", fake, sleeps).segment(request(), "m", GEMINI_CREDENTIALS));
    expect(fake.calls).toHaveLength(MAX_ATTEMPTS);
    expect(sleeps).toEqual([...RETRY_WAITS_MS]);
    expect([error.category, error.httpStatus]).toEqual(["quota", 429]);
    expectNoSecrets(error);
  });

  it("stops after three 503s with an unavailable error", async () => {
    const sleeps: number[] = [];
    const fake = new FakeFetch(Array.from({ length: 3 }, () => async () => apiError(503, "UNAVAILABLE")));
    const error = await failure(provider("vertex_gemini", fake, sleeps).segment(request(), "m", VERTEX_CREDENTIALS));
    expect(fake.calls).toHaveLength(3);
    expect([error.category, error.httpStatus]).toEqual(["unavailable", 503]);
    expectNoSecrets(error);
  });
});

describe("an access token pasted as the Gemini API key", () => {
  it("is refused before any call, saying it belongs to the vertex provider, when segmenting and when listing", async () => {
    const fake = new FakeFetch([]);
    const p = provider("gemini_api", fake);
    for (const call of [() => p.segment(request(), "m", { api_key: ` ${TOKEN} ` }), () => p.listModels!({ api_key: TOKEN })]) {
      const error = await failure(call());
      expect([error.category, error.httpStatus, error.userMessage]).toEqual(["config", 400, TOKEN_AS_KEY_MESSAGE]);
      expectNoSecrets(error);
    }
    expect(fake.calls).toHaveLength(0);
    expect(looksLikeAccessToken("ya29.a0Ae")).toBe(true);
    expect(looksLikeAccessToken(KEY)).toBe(false);
  });
});

describe("failures that are not retried", () => {
  const cases: [string, GeminiType, () => Promise<Response>, string, number][] = [
    ["400", "gemini_api", async () => apiError(400, "INVALID_ARGUMENT"), "bad_response", 502],
    ["400 API_KEY_INVALID", "gemini_api", async () => apiError(400, "INVALID_ARGUMENT", ["API_KEY_INVALID"]), "auth", 502],
    ["404", "gemini_api", async () => apiError(404, "NOT_FOUND"), "config", 500],
    ["500", "gemini_api", async () => apiError(500, "INTERNAL"), "unavailable", 503],
    ["502 HTML", "gemini_api", async () => new Response(`<html>${BODY_DETAIL}</html>`, { status: 502 }), "unavailable", 503],
    ["401 on Vertex", "vertex_gemini", async () => apiError(401, "UNAUTHENTICATED"), "auth", 502],
  ];

  it.each(cases)("%s fails at once as %s, never showing the body", async (_name, type, response, category, httpStatus) => {
    const sleeps: number[] = [];
    const fake = new FakeFetch([response, async () => json(200, answer([await imagePart(paintedSquare())]))]);
    const credentials = type === "gemini_api" ? GEMINI_CREDENTIALS : VERTEX_CREDENTIALS;
    const error = await failure(provider(type, fake, sleeps).segment(request(), "gemini-3-pro-image", credentials));
    expect(fake.calls).toHaveLength(1);
    expect(sleeps).toEqual([]);
    expect(error.category).toBe(category);
    expect(error.httpStatus).toBe(httpStatus);
    expectNoSecrets(error);
  });

  it("says the service could not be reached when fetch fails, and names only the class of anything else", async () => {
    const network = await failure(provider("gemini_api", new FakeFetch([new TypeError(`fetch failed ${BODY_DETAIL}`)])).segment(request(), "m", GEMINI_CREDENTIALS));
    expect(network.category).toBe("unavailable");
    const vertex = await failure(provider("vertex_gemini", new FakeFetch([new TypeError("fetch failed")])).segment(request(), "m", VERTEX_CREDENTIALS));
    expect(vertex.category).toBe("unavailable");
    const odd = await failure(provider("gemini_api", new FakeFetch([new RangeError(`weird ${BODY_DETAIL}`)])).segment(request(), "m", GEMINI_CREDENTIALS));
    expect(odd.category).toBe("bad_response");
    expect(odd.userMessage).toContain("RangeError");
    for (const error of [network, vertex, odd]) expectNoSecrets(error);
  });
});

describe("the timeout", () => {
  /** A fetch that answers only by rejecting when its signal aborts. */
  const hanging: FetchLike = (_url, init) => new Promise((_resolve, reject) => {
    const signal = init!.signal!;
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener("abort", () => reject(signal.reason));
  });

  it("gives up after CALL_TIMEOUT_MS with a timeout error, not retried", async () => {
    vi.useFakeTimers();
    const sleeps: number[] = [];
    const gemini = createGeminiProvider("gemini_api", { fetch: hanging, codec, sleep: async (ms) => { sleeps.push(ms); } });
    const pending = failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    await vi.advanceTimersByTimeAsync(CALL_TIMEOUT_MS - 1);
    await vi.advanceTimersByTimeAsync(2);
    const error = await pending;
    expect([error.category, error.httpStatus]).toEqual(["timeout", 504]);
    expect(sleeps).toEqual([]);  // a timeout is not retried
  });
});

describe("listModels", () => {
  it("fails as the call would, without retrying, and needs the key", async () => {
    const sleeps: number[] = [];
    const bad = await failure(provider("gemini_api", new FakeFetch([async () => apiError(400, "INVALID_ARGUMENT", ["API_KEY_INVALID"])]), sleeps).listModels!(GEMINI_CREDENTIALS));
    expect(bad.category).toBe("auth");
    expectNoSecrets(bad);
    const busy = await failure(provider("gemini_api", new FakeFetch([async () => apiError(429, "RESOURCE_EXHAUSTED")]), sleeps).listModels!(GEMINI_CREDENTIALS));
    expect(busy.category).toBe("quota");
    expect(sleeps).toEqual([]);
    const fake = new FakeFetch([]);
    const noKey = await failure(provider("gemini_api", fake).listModels!({}));
    expect(noKey.category).toBe("config");
    expect(fake.calls).toEqual([]);
  });
});
