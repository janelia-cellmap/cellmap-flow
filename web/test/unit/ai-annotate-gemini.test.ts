// gemini.ts against a scripted fetch: the request's URL, headers and JSON for
// both endpoints, the answer parsed (thought parts skipped, the last image
// taken, the padding cropped, another size resized), the retries and their
// waits, every failure's category and message, and that no message ever
// carries the key, the token or the service's body. The cases are
// tests/ai_annotate/test_backends.py's for the Vertex backend, plus the
// Gemini API's own (listModels, the key header, API_KEY_INVALID).
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  base64ToBytes, bytesToBase64, createGeminiProvider, endpoint, type FetchLike, GEMINI_API_BASE, GEMINI_DESTINATION, type GeminiType,
  MAX_ATTEMPTS, MISSING_KEY_MESSAGE, MISSING_VERTEX_MESSAGE, plain, RETRY_WAITS_MS, VERTEX_BASE, VERTEX_DESTINATION,
} from "../../src/ai-annotate/gemini";
import { jsCodec, PNG_SIGNATURE } from "../../src/ai-annotate/images";
import { createProviders, DEFAULT_PROVIDER } from "../../src/ai-annotate/providers";
import { AIAnnotateError, CALL_TIMEOUT_MS, DEFAULT_GEMINI_MODELS, type Provider, type RgbaImage, type SegmentRequest } from "../../src/ai-annotate/types";

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

function answer(parts: Part[], options: { finishReason?: string; blockReason?: string; candidates?: boolean } = {}): Record<string, unknown> {
  const body: Record<string, unknown> = {
    usageMetadata: { promptTokenCount: 600, candidatesTokenCount: 1300, totalTokenCount: 1900 },
  };
  if (options.candidates !== false) body.candidates = [{ content: { role: "model", parts }, finishReason: options.finishReason ?? "STOP" }];
  if (options.blockReason) body.promptFeedback = { blockReason: options.blockReason };
  return body;
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

describe("info and createProviders", () => {
  it("describes both Gemini providers as the plan says, and lists gemini, vertex, fake in that order", () => {
    const gemini = createGeminiProvider("gemini_api", { codec }).info;
    expect(gemini).toEqual({ id: "gemini", type: "gemini_api", models: [...DEFAULT_GEMINI_MODELS], destination: GEMINI_DESTINATION, credentials: ["api_key"] });
    expect(GEMINI_DESTINATION).toBe(
      "Google's Gemini API (generativelanguage.googleapis.com), with your API key. On the paid tier Google does not use what you send to train its models; on the free tier it may.",
    );
    const vertex = createGeminiProvider("vertex_gemini", { codec }).info;
    expect(vertex).toEqual({ id: "vertex", type: "vertex_gemini", models: [...DEFAULT_GEMINI_MODELS], destination: VERTEX_DESTINATION, credentials: ["project", "access_token"] });
    expect(VERTEX_DESTINATION).toBe("Google Cloud Vertex AI, location global (Google may process the data in any region)");

    const providers = createProviders({ codec });
    expect([...providers.keys()]).toEqual(["gemini", "vertex", "fake"]);
    expect(providers.get("fake")!.info.type).toBe("fake");
    expect(DEFAULT_PROVIDER).toBe("gemini");
    expect(providers.has(DEFAULT_PROVIDER)).toBe(true);
  });
});

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

  it("path-encodes the model id and drops a pasted models/ prefix", () => {
    expect(endpoint("gemini_api", "models/gemini-x", GEMINI_CREDENTIALS).url).toBe(`${GEMINI_API_BASE}/models/gemini-x:generateContent`);
    expect(endpoint("gemini_api", "odd/model id", GEMINI_CREDENTIALS).url).toBe(`${GEMINI_API_BASE}/models/odd%2Fmodel%20id:generateContent`);
    expect(endpoint("vertex_gemini", "m", { project: "a b", access_token: "t" }).url).toContain("/projects/a%20b/locations/global/");
  });

  it("refuses to run without the provider's credentials, before any call", async () => {
    const fake = new FakeFetch([]);
    for (const credentials of [{}, { api_key: "   " }, { project: PROJECT, access_token: TOKEN }]) {
      const error = await failure(provider("gemini_api", fake).segment(request(), "m", credentials));
      expect([error.category, error.httpStatus, error.userMessage]).toEqual(["config", 400, MISSING_KEY_MESSAGE]);
    }
    for (const credentials of [{}, { project: PROJECT }, { access_token: TOKEN }, { api_key: KEY }]) {
      const error = await failure(provider("vertex_gemini", fake).segment(request(), "m", credentials));
      expect([error.category, error.httpStatus, error.userMessage]).toEqual(["config", 400, MISSING_VERTEX_MESSAGE]);
    }
    expect(MISSING_KEY_MESSAGE).toBe("Enter your Gemini API key in the panel first.");
    expect(MISSING_VERTEX_MESSAGE).toBe("Enter the Google Cloud project and an access token (gcloud auth print-access-token) first.");
    expect(fake.calls).toEqual([]);
  });
});

describe("the answer", () => {
  it("is a bad response without an image, quoting at most 200 plain characters of the model's text and the finish reason", async () => {
    const reply = "I can't do that. <script>alert(1)</script>\x00\x1b" + "x".repeat(500);
    const fake = new FakeFetch([async () => json(200, answer([{ text: reply }], { finishReason: "IMAGE_SAFETY" }))]);
    const error = await failure(provider("gemini_api", fake).segment(request(), "m", GEMINI_CREDENTIALS));
    expect(error.category).toBe("bad_response");
    expect(error.userMessage).toContain("(stopped: IMAGE_SAFETY)");
    expect(error.userMessage).toContain('It said: "I can\'t do that.');
    expect(error.userMessage).not.toMatch(/[\x00\x1b]/);
    expect(error.userMessage).not.toContain("x".repeat(201));
    expect(error.userMessage.length).toBeLessThan(320);
    expect(error.userMessage.endsWith('..."' + ".")).toBe(true);
  });

  it("names the block reason when there is no candidate, and says so plainly when there is nothing at all", async () => {
    const blocked = { candidates: [], promptFeedback: { blockReason: "PROHIBITED_CONTENT" } };
    const fake = new FakeFetch([async () => json(200, blocked), async () => json(200, {}), async () => json(200, answer([{ text: "" }]))]);
    const gemini = provider("gemini_api", fake);
    expect((await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS))).userMessage).toBe("The model answered without an image (blocked: PROHIBITED_CONTENT).");
    expect((await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS))).userMessage).toBe("The model answered without an image.");
    expect((await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS))).userMessage).toBe("The model answered without an image.");
    expect(plain("  a\x07b\tc\n ", 200)).toBe("a b\tc");
    expect(plain("y".repeat(10), 8)).toBe("yyyyy...");
  });

  it("is a bad response when the body is not JSON", async () => {
    const fake = new FakeFetch([async () => new Response("<html>oops</html>", { status: 200 })]);
    const error = await failure(provider("gemini_api", fake).segment(request(), "m", GEMINI_CREDENTIALS));
    expect([error.category, error.userMessage]).toEqual(["bad_response", "The Gemini API answered with something other than JSON."]);
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

  it("sends a square plane as it is", async () => {
    const square = plane(60, 60);
    const fake = new FakeFetch([async () => json(200, answer([await imagePart(paintedSquare())]))]);
    const result = await provider("gemini_api", fake).segment(request(square), "m", GEMINI_CREDENTIALS);
    const inline = ((fake.calls[0].body as { contents: { parts: Part[] }[] }).contents[0].parts[0].inlineData) as { data: string };
    const sent = await codec.decode(base64ToBytes(inline.data), "image/png");
    expect(Array.from(sent.data)).toEqual(Array.from(square.data));
    expect(maskCount(result.mask)).toBe(150);
    expect([result.model_image!.width, result.model_image!.height]).toEqual([60, 60]);
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
    expect([gif.category, gif.userMessage]).toEqual(["bad_response", "The model returned an image that is not a PNG, JPEG or WebP."]);
    const garbage = await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    expect(garbage.userMessage).toBe("The model returned an image that could not be decoded.");
    const tooBig = await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    expect(tooBig.userMessage).toBe("The model returned a 5000x5000 image, larger than this dashboard accepts.");
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
    expect(sleeps).toEqual([10_000, 30_000]);
    expect(sleeps).toEqual([...RETRY_WAITS_MS]);
    expect(maskCount(result.mask)).toBe(150);
  });

  it("retries on the body's status UNAVAILABLE or RESOURCE_EXHAUSTED whatever the HTTP code", async () => {
    const sleeps: number[] = [];
    const fake = new FakeFetch([async () => apiError(500, "UNAVAILABLE"), async () => json(200, answer([await imagePart(paintedSquare())]))]);
    await provider("gemini_api", fake, sleeps).segment(request(), "m", GEMINI_CREDENTIALS);
    expect(fake.calls).toHaveLength(2);
    expect(sleeps).toEqual([10_000]);
  });

  it("stops after three attempts with a quota error", async () => {
    const sleeps: number[] = [];
    const fake = new FakeFetch(Array.from({ length: 10 }, () => async () => apiError(429, "RESOURCE_EXHAUSTED")));
    const error = await failure(provider("gemini_api", fake, sleeps).segment(request(), "m", GEMINI_CREDENTIALS));
    expect(fake.calls).toHaveLength(MAX_ATTEMPTS);
    expect(MAX_ATTEMPTS).toBe(3);
    expect(sleeps).toEqual([10_000, 30_000]);
    expect([error.category, error.httpStatus]).toEqual(["quota", 429]);
    expect(error.userMessage).toBe("The Gemini API's quota is used up (still refused after 3 attempts); wait a minute and try again.");
    expectNoSecrets(error);
  });

  it("stops after three 503s with an unavailable error", async () => {
    const sleeps: number[] = [];
    const fake = new FakeFetch(Array.from({ length: 3 }, () => async () => apiError(503, "UNAVAILABLE")));
    const error = await failure(provider("vertex_gemini", fake, sleeps).segment(request(), "m", VERTEX_CREDENTIALS));
    expect(fake.calls).toHaveLength(3);
    expect([error.category, error.httpStatus, error.userMessage]).toEqual(["unavailable", 503, "Vertex AI is unavailable (HTTP 503); try again later."]);
    expectNoSecrets(error);
  });
});

describe("failures that are not retried", () => {
  const cases: [string, GeminiType, () => Promise<Response>, string, number, string | RegExp][] = [
    ["400", "gemini_api", async () => apiError(400, "INVALID_ARGUMENT"), "bad_response", 502, "The Gemini API rejected the request (HTTP 400)."],
    ["400 API_KEY_INVALID", "gemini_api", async () => apiError(400, "INVALID_ARGUMENT", ["API_KEY_INVALID"]), "auth", 502, "The Gemini API refused the credentials: check the API key."],
    ["401", "gemini_api", async () => apiError(401, "UNAUTHENTICATED"), "auth", 502, "The Gemini API refused the credentials: check the API key."],
    ["403", "gemini_api", async () => apiError(403, "PERMISSION_DENIED"), "auth", 502, /refused the credentials: check the API key/],
    ["404", "gemini_api", async () => apiError(404, "NOT_FOUND"), "config", 500, "The Gemini API has no model gemini-3-pro-image."],
    ["500", "gemini_api", async () => apiError(500, "INTERNAL"), "unavailable", 503, "The Gemini API is unavailable (HTTP 500); try again later."],
    ["502 HTML", "gemini_api", async () => new Response(`<html>${BODY_DETAIL}</html>`, { status: 502 }), "unavailable", 503, "The Gemini API is unavailable (HTTP 502); try again later."],
    ["418", "gemini_api", async () => new Response("", { status: 418 }), "bad_response", 502, "The Gemini API rejected the request (HTTP 418)."],
    ["401 on Vertex", "vertex_gemini", async () => apiError(401, "UNAUTHENTICATED"), "auth", 502,
      "Vertex AI refused the credentials: check the access token (it expires after about an hour: run gcloud auth print-access-token again), the project, that the Vertex AI API is enabled and that you have the Vertex AI User role."],
    ["403 on Vertex", "vertex_gemini", async () => apiError(403, "PERMISSION_DENIED"), "auth", 502, /gcloud auth print-access-token/],
    ["404 on Vertex", "vertex_gemini", async () => apiError(404, "NOT_FOUND"), "config", 500, "Vertex AI has no model gemini-3-pro-image in location global for this project."],
    ["400 on Vertex", "vertex_gemini", async () => apiError(400, "INVALID_ARGUMENT"), "bad_response", 502, "Vertex AI rejected the request (HTTP 400)."],
  ];

  it.each(cases)("%s fails at once as %s, never showing the body", async (_name, type, response, category, httpStatus, message) => {
    const sleeps: number[] = [];
    const fake = new FakeFetch([response, async () => json(200, answer([await imagePart(paintedSquare())]))]);
    const credentials = type === "gemini_api" ? GEMINI_CREDENTIALS : VERTEX_CREDENTIALS;
    const error = await failure(provider(type, fake, sleeps).segment(request(), "gemini-3-pro-image", credentials));
    expect(fake.calls).toHaveLength(1);
    expect(sleeps).toEqual([]);
    expect(error.category).toBe(category);
    expect(error.httpStatus).toBe(httpStatus);
    if (typeof message === "string") expect(error.userMessage).toBe(message);
    else expect(error.userMessage).toMatch(message);
    expectNoSecrets(error);
  });

  it("says the service could not be reached when fetch fails, and names only the class of anything else", async () => {
    const network = await failure(provider("gemini_api", new FakeFetch([new TypeError(`fetch failed ${BODY_DETAIL}`)])).segment(request(), "m", GEMINI_CREDENTIALS));
    expect([network.category, network.userMessage]).toEqual(["unavailable", "Could not reach the Gemini API from this browser."]);
    const vertex = await failure(provider("vertex_gemini", new FakeFetch([new TypeError("fetch failed")])).segment(request(), "m", VERTEX_CREDENTIALS));
    expect(vertex.userMessage).toBe("Could not reach Vertex AI from this browser.");
    const odd = await failure(provider("gemini_api", new FakeFetch([new RangeError(`weird ${BODY_DETAIL}`)])).segment(request(), "m", GEMINI_CREDENTIALS));
    expect([odd.category, odd.userMessage]).toEqual(["bad_response", "The call to the Gemini API failed (RangeError)."]);
    for (const error of [network, vertex, odd]) expectNoSecrets(error);
  });
});

describe("the timeout and the caller's signal", () => {
  /** A fetch that answers only by rejecting when its signal aborts. */
  const hanging: FetchLike = (_url, init) => new Promise((_resolve, reject) => {
    const signal = init!.signal!;
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener("abort", () => reject(signal.reason));
  });

  it("gives up after CALL_TIMEOUT_MS with a timeout error", async () => {
    vi.useFakeTimers();
    const sleeps: number[] = [];
    const gemini = createGeminiProvider("gemini_api", { fetch: hanging, codec, sleep: async (ms) => { sleeps.push(ms); } });
    const pending = failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS));
    await vi.advanceTimersByTimeAsync(CALL_TIMEOUT_MS - 1);
    await vi.advanceTimersByTimeAsync(2);
    const error = await pending;
    expect([error.category, error.httpStatus, error.userMessage]).toEqual(["timeout", 504, "The Gemini API did not answer within 120 s."]);
    expect(sleeps).toEqual([]);  // a timeout is not retried
  });

  it("stops when the caller's signal aborts", async () => {
    const controller = new AbortController();
    const gemini = createGeminiProvider("gemini_api", { fetch: hanging, codec });
    const pending = failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS, { signal: controller.signal }));
    controller.abort();
    const error = await pending;
    expect([error.category, error.userMessage]).toEqual(["unavailable", "The call to the Gemini API was cancelled."]);
  });

  it("reads the caller's own deadline (AbortSignal.timeout, as the job's) as a timeout", async () => {
    const gemini = createGeminiProvider("vertex_gemini", { fetch: hanging, codec });
    const error = await failure(gemini.segment(request(), "m", VERTEX_CREDENTIALS, { signal: AbortSignal.timeout(5) }));
    expect([error.category, error.httpStatus, error.userMessage]).toEqual(["timeout", 504, "Vertex AI did not answer in time."]);
  });

  it("stops retrying when the caller aborts during a wait", async () => {
    const controller = new AbortController();
    const sleeps: number[] = [];
    const fake = new FakeFetch([async () => apiError(429, "RESOURCE_EXHAUSTED"), async () => json(200, answer([await imagePart(paintedSquare())]))]);
    const gemini = createGeminiProvider("gemini_api", { fetch: fake.fetch, codec, sleep: async (ms) => { sleeps.push(ms); controller.abort(); } });
    const error = await failure(gemini.segment(request(), "m", GEMINI_CREDENTIALS, { signal: controller.signal }));
    expect(fake.calls).toHaveLength(1);
    expect(sleeps).toEqual([10_000]);
    expect([error.category, error.userMessage]).toEqual(["unavailable", "The call to the Gemini API was cancelled."]);
  });
});

describe("listModels", () => {
  it("follows the pages, keeps the image models that generate content, sorted, and sends the key", async () => {
    const fake = new FakeFetch([
      async () => json(200, {
        models: [
          { name: "models/gemini-3-pro-image", supportedGenerationMethods: ["generateContent"] },
          { name: "models/gemini-2.5-flash", supportedGenerationMethods: ["generateContent"] },
          { name: "models/imagen-4.0-generate-001", supportedGenerationMethods: ["predict"] },
          { name: "models/gemini-3.1-flash-image", supportedGenerationMethods: ["generateContent", "countTokens"] },
        ],
        nextPageToken: "page/2",
      }),
      async () => json(200, { models: [{ name: "models/gemini-2.5-flash-image", supportedGenerationMethods: ["generateContent"] }, { name: "models/embedding-001" }] }),
    ]);
    const models = await provider("gemini_api", fake).listModels!(GEMINI_CREDENTIALS);
    expect(models).toEqual(["gemini-2.5-flash-image", "gemini-3-pro-image", "gemini-3.1-flash-image"]);
    expect(fake.calls.map((c) => c.url)).toEqual([`${GEMINI_API_BASE}/models?pageSize=200`, `${GEMINI_API_BASE}/models?pageSize=200&pageToken=page%2F2`]);
    expect(fake.calls.every((c) => c.method === "GET" && c.headers["x-goog-api-key"] === KEY && c.body === null)).toBe(true);
  });

  it("fails as the call would, without retrying, and needs the key", async () => {
    const sleeps: number[] = [];
    const bad = await failure(provider("gemini_api", new FakeFetch([async () => apiError(400, "INVALID_ARGUMENT", ["API_KEY_INVALID"])]), sleeps).listModels!(GEMINI_CREDENTIALS));
    expect([bad.category, bad.userMessage]).toEqual(["auth", "The Gemini API refused the credentials: check the API key."]);
    expectNoSecrets(bad);
    const busy = await failure(provider("gemini_api", new FakeFetch([async () => apiError(429, "RESOURCE_EXHAUSTED")]), sleeps).listModels!(GEMINI_CREDENTIALS));
    expect(busy.category).toBe("quota");
    expect(busy.userMessage).toContain("after 1 attempt)");
    const missing = await failure(provider("gemini_api", new FakeFetch([async () => apiError(404, "NOT_FOUND")]), sleeps).listModels!(GEMINI_CREDENTIALS));
    expect(missing.userMessage).toBe("The Gemini API rejected the request (HTTP 404).");  // no model to name
    expect(sleeps).toEqual([]);
    const fake = new FakeFetch([]);
    const noKey = await failure(provider("gemini_api", fake).listModels!({}));
    expect([noKey.category, noKey.userMessage]).toEqual(["config", MISSING_KEY_MESSAGE]);
    expect(fake.calls).toEqual([]);
  });

  it("is null for Vertex AI, which is not asked", async () => {
    const fake = new FakeFetch([]);
    expect(await provider("vertex_gemini", fake).listModels!(VERTEX_CREDENTIALS)).toBeNull();
    expect(fake.calls).toEqual([]);
  });
});

describe("base64", () => {
  it("round-trips bytes, in chunks bigger than one", () => {
    const bytes = Uint8Array.from({ length: 70_000 }, (_v, i) => (i * 31) % 256);
    expect(Array.from(base64ToBytes(bytesToBase64(bytes)))).toEqual(Array.from(bytes));
    expect(bytesToBase64(new Uint8Array([0x89, 0x50, 0x4e, 0x47]))).toBe("iVBORw==");
    expect(() => base64ToBytes("!!!")).toThrow();
  });
});
