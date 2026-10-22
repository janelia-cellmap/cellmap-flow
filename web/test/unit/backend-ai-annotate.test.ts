// The AI-annotate routes against fake context members, the fake provider
// (and a test provider that wants a key and records what it is given), the
// plain-JavaScript codec and a memory store: the JSON each answers, checked
// against what the tab's ai-annotate.js reads and against the Python
// routes' refusals, their order and their statuses; and what each does to the
// paint store (the mask written, the undo, the audit log), the viewer (the
// key bound, the layer refreshed) and the stores (credentials, the day's
// count). A run plans, reads, masks and accepts its plane through the
// portable Python (here in node's Pyodide, test/unit/pyodide.ts), so the
// cases that run one skip without it.
import { describe, expect, it, vi } from "vitest";

import { readLog } from "../../src/ai-annotate/audit";
import { CREDENTIALS_KEY, createCredentialStore } from "../../src/ai-annotate/credentials";
import { createFakeProvider, fakeSegment } from "../../src/ai-annotate/fake";
import { readBox, writeBox } from "../../src/ai-annotate/fill";
import { jsCodec } from "../../src/ai-annotate/images";
import { ORGANELLES } from "../../src/ai-annotate/organelles";
import { editablePrompt } from "../../src/ai-annotate/prompts";
import type { CredentialField, Credentials, Provider, ProviderInfo, ProviderType, SegmentRequest, SegmentResult } from "../../src/ai-annotate/types";
import { ACTION, AIAnnotateError, DAILY_CALL_LIMIT, DEFAULT_GEMINI_MODELS, FAKE_MODEL, KEY, KEYBINDING } from "../../src/ai-annotate/types";
import { createUsage, USAGE_KEY } from "../../src/ai-annotate/usage";
import { createBackend } from "../../src/backend";
import { type AiAnnotateDeps, type AiAnnotateState, aiAnnotateRoutes } from "../../src/backend/routes/ai-annotate";
import { finetuneRoutes } from "../../src/backend/routes/finetune";
import { Router } from "../../src/backend/router";
import { viewLabelsRoutes } from "../../src/backend/routes/view-labels";
import type { ApiRequest, ApiResponse, ModelEntry, PaintVolume } from "../../src/contracts";
import type { RunFinetune } from "../../src/finetune/types";
import { readVolume } from "../../src/finetune/volume";
import { type KeyValueStore, MemoryKeyValueStore } from "../../src/models/listing-cache";
import { paintLayerName } from "../../src/viewer/layers";
import { entry, type FakeContext, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";
import { portablePython } from "./pyodide";

// Pyodide, numpy and scipy load once for the file, a few seconds.
vi.setConfig({ testTimeout: 120_000 });
const python = await portablePython();

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/** Wait until `condition` holds (a job set by the Shift+G handler, which runs on its own). */
async function until(condition: () => boolean): Promise<void> {
  for (let i = 0; i < 200 && !condition(); i++) await tick();
  if (!condition()) throw new Error("the condition never held");
}

const BASE = "/api/finetune/ai-annotate";
const NOW = new Date(2026, 9, 15, 12, 0, 0);
/** A point inside the fakes' dataset (1000 voxels of 8 nm a side): voxel (100, 250, 250). */
const POINT = [800, 2000, 2000];
const SECRET = "AIza-SECRET-KEY-12345";

/** A provider like the Gemini one: wants an API key, lists models, records
 * what it is given, can be told to fail or to wait. Its mask is the fake's. */
class TestProvider implements Provider {
  readonly info: ProviderInfo;
  calls: { model: string; credentials: Credentials; request: SegmentRequest }[] = [];
  listed: Credentials[] = [];
  models: string[] | null | Error = ["gemini-3-pro-image", "listed-image-model"];
  fail: AIAnnotateError | null = null;
  /** When set, segment waits here until the test resolves it. */
  gate: Promise<void> | null = null;

  constructor(id = "gemini", type: ProviderType = "gemini_api", credentials: CredentialField[] = ["api_key"], destination = "Test destination (a service, with your key)") {
    this.info = { id, type, models: [...DEFAULT_GEMINI_MODELS], destination, credentials };
  }

  async segment(request: SegmentRequest, model: string, credentials: Credentials): Promise<SegmentResult> {
    this.calls.push({ model, credentials, request });
    if (this.gate) await this.gate;
    if (this.fail) throw this.fail;
    return fakeSegment(request, model);
  }

  async listModels(credentials: Credentials): Promise<string[] | null> {
    this.listed.push(credentials);
    if (this.models instanceof Error) throw this.models;
    return this.models;
  }
}

function providers(gemini = new TestProvider()): Map<string, Provider> {
  const vertex = new TestProvider("vertex", "vertex_gemini", ["project", "access_token"], "Google Cloud Vertex AI, location global (Google may process the data in any region)");
  vertex.models = null; // Vertex cannot list its models
  return new Map<string, Provider>([
    [gemini.info.id, gemini],
    ["vertex", vertex],
    ["fake", createFakeProvider()],
  ]);
}

/** The AI-annotate routes beside the finetune routes (a volume comes to
 * exist through create-volume, with the real plan) and the Label it routes
 * (whose Undo takes an accept back, from the undo stacks the two share, as
 * createRouter wires them), on a fake provider, the JavaScript codec, a
 * memory store and a fixed clock; the plane 64 voxels a side, so the images
 * are small. */
function harness(ctx: FakeContext, deps: Partial<AiAnnotateDeps> = {}) {
  const router = new Router();
  const run: RunFinetune = async () => {
    throw new Error("no finetune run in these tests");
  };
  finetuneRoutes(router, { run, now: () => NOW });
  const labels = viewLabelsRoutes(router);
  const storage = deps.storage !== undefined ? deps.storage : new MemoryKeyValueStore();
  const ai = aiAnnotateRoutes(router, { providers: providers(), codec: jsCodec(), storage, now: () => NOW, cropSizePx: 64, undo: labels.undo, ...deps });
  return { handle: (req: ApiRequest) => router.handle(req, ctx), ai, storage };
}

async function serve(ctx: FakeContext, ...entries: ModelEntry[]): Promise<void> {
  for (const e of entries) await ctx.infer.serve(e, ctx.session.state.chain, "blob", async () => new ArrayBuffer(8));
  await ctx.session.update({ models: [...ctx.session.state.models, ...entries] });
}

/** The settings body the page sends for the fake provider on mitochondria, and the acknowledgement of its destination. */
const FAKE_SETTINGS = { provider: "fake", model: FAKE_MODEL, label_key: "mito", prompt: null };
const FAKE_DESTINATION = createFakeProvider().info.destination;

/** A context with the raw a gradient, a model served, a volume created for
 * it (uint8 for setup_56, uint16 for setup_54) and its layer in the viewer,
 * the fake provider set up and acknowledged. */
async function ready(options: { model?: string; deps?: Partial<AiAnnotateDeps>; settings?: Record<string, unknown> } = {}) {
  const ctx = fakeContext();
  ctx.infer.rawPattern = "gradient";
  await serve(ctx, entry("setup_54"), entry("setup_56", "nuc", ["nuc"], 128));
  const h = harness(ctx, options.deps);
  const created = body(await h.handle(request("POST", "/api/finetune/create-volume", { model_name: options.model ?? NAME_56 })));
  expect(created.success).toBe(true);
  const id = created.volume_id as string;
  await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: id, minio_url: created.neuroglancer_url }));
  const settings = body(await h.handle(request("POST", `${BASE}/settings`, { ...FAKE_SETTINGS, acknowledge: true, destination: FAKE_DESTINATION, ...options.settings })));
  expect(settings.success).toBe(true);
  const volume = (await readVolume(ctx.paint, id)) as PaintVolume;
  return { ctx, h, id, volume };
}

/** Run at POINT in the XY plane and wait for the job to settle; the run's answer. */
async function runToEnd(h: ReturnType<typeof harness>, point: number[] = POINT, depthAxis = 0) {
  const started = body(await h.handle(request("POST", `${BASE}/run`, { point_nm: point, depth_axis: depthAxis })));
  expect(started).toMatchObject({ success: true, status: "running" });
  expect(started.annotate_id).toMatch(/^[0-9a-f]{32}$/);
  await h.ai.job!.settled();
  return started.annotate_id as string;
}

const status = (h: ReturnType<typeof harness>, preview = false) => h.handle(request("GET", `${BASE}/status${preview ? "?preview=1" : ""}`)).then(body);

/** The voxel (z, y, x) of index `i` of a write mask over the XY plane of `writeBox`. */
function voxelOf(writeBox: { lo: number[]; hi: number[] }, i: number): number[] {
  const cols = writeBox.hi[2] - writeBox.lo[2];
  return [writeBox.lo[0], writeBox.lo[1] + Math.floor(i / cols), writeBox.lo[2] + (i % cols)];
}

/** Paint one voxel `label` in the store, as a brush stroke would (the chunk written whole). */
async function paintVoxel(ctx: FakeContext, volume: PaintVolume, voxel: number[], label: number): Promise<void> {
  const labels = volume.dtype === "uint16" ? new Uint16Array([label]) : new Uint8Array([label]);
  await writeBox(ctx.paint, volume, voxel, voxel.map((v) => v + 1), labels);
}

// ---------------------------------------------------------------------------

describe("GET /api/finetune/ai-annotate/config", () => {
  it("is enabled, with the three providers (configured only where no credentials are needed), the catalog and the browser's usage note", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    const d = body(await h.handle(request("GET", `${BASE}/config`)));
    expect(d.success).toBe(true);
    expect(d.enabled).toBe(true);
    expect(d.providers.map((p: any) => [p.id, p.type, p.configured])).toEqual([
      ["gemini", "gemini_api", false],
      ["vertex", "vertex_gemini", false],
      ["fake", "fake", true],
    ]);
    expect(d.providers[0].models).toEqual(DEFAULT_GEMINI_MODELS);
    expect(d.providers[0].credentials).toEqual(["api_key"]);
    expect(d.providers[1].credentials).toEqual(["project", "access_token"]);
    expect(d.providers[2]).toMatchObject({ models: [FAKE_MODEL], destination: FAKE_DESTINATION, credentials: [] });
    expect(d.default_provider).toBe("gemini");
    expect(d.daily_call_limit).toBe(DAILY_CALL_LIMIT);
    expect(d.calls_today).toBe(0);
    expect(d.crop_size_px).toBe(64);
    expect(d.organelles.find((o: any) => o.key === "mito")).toEqual({ key: "mito", name: "mitochondria", prompt: editablePrompt(ORGANELLES.mito) });
    expect(d.organelles.length).toBe(Object.keys(ORGANELLES).length);
    expect(d.keybinding).toBe(KEYBINDING);
    expect(d.settings).toBeNull();
    expect(d.acknowledged).toEqual([]);
    // The words the page puts in the brackets after "0 of 200 calls used today", where the server names its config file.
    expect(d.usage_note).toContain(String(DAILY_CALL_LIMIT));
  });
});

describe("POST /api/finetune/ai-annotate/credentials", () => {
  it("keeps the key in the page's memory only, lists the models it reaches, and never shows, stores or logs it", async () => {
    const ctx = fakeContext();
    const gemini = new TestProvider();
    const h = harness(ctx, { providers: providers(gemini) });
    const res = await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", api_key: ` ${SECRET} ` }));
    expect(res.status).toBe(200);
    expect(body(res)).toEqual({ success: true, configured: true, models: ["gemini-3-pro-image", "listed-image-model"] });
    expect(gemini.listed).toEqual([{ api_key: SECRET }]); // trimmed
    expect(h.storage!.getItem(CREDENTIALS_KEY)).toBeNull();
    expect(h.ai.credentials.get("gemini")).toEqual({ api_key: SECRET });
    // Nothing of it in storage, under any key, even when the page asks to remember it.
    await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", api_key: SECRET, remember: true }));
    const stored = h.storage as MemoryKeyValueStore;
    for (const key of ["cellmap-flow:ai-annotate:usage", CREDENTIALS_KEY]) expect(stored.getItem(key) ?? "").not.toContain(SECRET);
    expect(JSON.stringify(body(res))).not.toContain(SECRET);
    expect(ctx.logged.join("\n")).not.toContain(SECRET);
    // The config now says configured, and lists the defaults plus what was listed.
    const config = body(await h.handle(request("GET", `${BASE}/config`)));
    expect(config.providers[0]).toMatchObject({ id: "gemini", configured: true, models: [...DEFAULT_GEMINI_MODELS, "listed-image-model"] });
    expect(JSON.stringify(config)).not.toContain(SECRET);
  });

  it("takes the key out with forget, and gives the picker the provider's default models back", async () => {
    const h = harness(fakeContext());
    expect(body(await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", api_key: SECRET }))).configured).toBe(true);
    const forgotten = body(await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", forget: true })));
    expect(forgotten).toEqual({ success: true, configured: false, models: DEFAULT_GEMINI_MODELS });
    expect(h.ai.credentials.get("gemini")).toBeNull();
    expect(body(await h.handle(request("GET", `${BASE}/config`))).providers[0].models).toEqual(DEFAULT_GEMINI_MODELS); // the live list went with the key
  });

});

describe("the credential store", () => {
  it("keeps credentials in memory only, dropping what is not a credential", () => {
    const store = createCredentialStore();
    store.set("gemini", { api_key: " k ", bogus: "x" } as never);
    expect(store.get("gemini")).toEqual({ api_key: "k" });
    expect(store.configured("gemini", ["api_key"])).toBe(true);
    store.set("vertex", { project: "p" });
    expect(store.configured("vertex", ["project", "access_token"])).toBe(false);
    expect(store.configured("fake", [])).toBe(true);
    expect(store.configured("fake")).toBe(false);
    store.forget("gemini");
    expect(store.get("gemini")).toBeNull();
  });

  it("deletes what an earlier build remembered in storage, and reads nothing from it", () => {
    const storage = new MemoryKeyValueStore();
    storage.setItem(CREDENTIALS_KEY, JSON.stringify({ v: 1, providers: { gemini: { api_key: "old-key" } } }));
    const store = createCredentialStore(storage);
    expect(storage.getItem(CREDENTIALS_KEY)).toBeNull();
    expect(store.get("gemini")).toBeNull();
    // A storage that refuses is left alone; the store still works.
    const broken: KeyValueStore = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => {
        throw new Error("blocked");
      },
    };
    const inMemory = createCredentialStore(broken);
    inMemory.set("gemini", { api_key: "k" });
    expect(inMemory.get("gemini")).toEqual({ api_key: "k" });
  });
});

describe("POST /api/finetune/ai-annotate/settings", () => {
  it("keeps the settings, with the catalog's own prompt as null and an edited one as the override, and binds Shift+G once", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    const d = body(await h.handle(request("POST", `${BASE}/settings`, { ...FAKE_SETTINGS, prompt: `  ${editablePrompt(ORGANELLES.mito)}  ` })));
    expect(d).toEqual({ success: true, settings: { provider: "fake", model: FAKE_MODEL, label_key: "mito", label_name: null, prompt: null }, acknowledged: [] });
    expect(h.ai.settings).toEqual(d.settings);
    const edited = body(await h.handle(request("POST", `${BASE}/settings`, { provider: "fake", model: FAKE_MODEL, label_name: " Golgi apparatus ", prompt: " Paint the Golgi. " })));
    expect(edited.settings).toEqual({ provider: "fake", model: FAKE_MODEL, label_key: null, label_name: " Golgi apparatus ", prompt: "Paint the Golgi." });
    expect(ctx.viewer.bindings.map((b) => [b.key, b.action])).toEqual([[KEY, ACTION]]);
    expect(h.ai.unbind).not.toBeNull();
    expect(body(await h.handle(request("GET", `${BASE}/config`))).settings).toEqual(edited.settings);
  });

  it("records the acknowledgement of the shown destination per provider and dataset, refuses a changed one (409), and takes it back on false", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    const moved = await h.handle(request("POST", `${BASE}/settings`, { ...FAKE_SETTINGS, acknowledge: true, destination: "somewhere else" }));
    expect(moved.status).toBe(409);
    expect(body(moved)).toEqual({
      success: false,
      error: "Where this provider sends images has changed; check the new destination and tick again.",
      needs_acknowledgement: true,
      provider: "fake",
      destination: FAKE_DESTINATION,
    });
    expect(h.ai.settings).toBeNull(); // refused before anything was kept
    const agreed = body(await h.handle(request("POST", `${BASE}/settings`, { ...FAKE_SETTINGS, acknowledge: true, destination: FAKE_DESTINATION })));
    expect(agreed.acknowledged).toEqual(["fake"]);
    expect(h.ai.acknowledged.get("https://example.org/data.zarr")?.get("fake")).toBe(FAKE_DESTINATION);
    expect(ctx.logged.some((line) => line.includes("may be sent to fake"))).toBe(true);
    // Left out: kept. False: withdrawn.
    expect(body(await h.handle(request("POST", `${BASE}/settings`, FAKE_SETTINGS))).acknowledged).toEqual(["fake"]);
    expect(body(await h.handle(request("POST", `${BASE}/settings`, { ...FAKE_SETTINGS, acknowledge: false }))).acknowledged).toEqual([]);
    expect(body(await h.handle(request("GET", `${BASE}/config`))).acknowledged).toEqual([]);
  });

  it("refuses an acknowledgement with no dataset open", async () => {
    const h = harness(fakeContext({ dataset: null }));
    const res = await h.handle(request("POST", `${BASE}/settings`, { ...FAKE_SETTINGS, acknowledge: true, destination: FAKE_DESTINATION }));
    expect(res.status).toBe(409);
    expect(body(res).error).toBe("Open a dataset before acknowledging where its images are sent.");
  });
});

describe.skipIf(!python)("POST /api/finetune/ai-annotate/run: the refusals, in _begin_run's order", () => {
  it("wants settings first, then a volume (needs_volume), then the acknowledgement, then the credentials", async () => {
    const ctx = fakeContext();
    ctx.viewer.positionNm = POINT;
    await serve(ctx, entry("setup_56", "nuc", ["nuc"], 128));
    const h = harness(ctx);
    const run = () => h.handle(request("POST", `${BASE}/run`, {}));

    const noSettings = await run();
    expect(noSettings.status).toBe(400);
    expect(body(noSettings).needs_volume).toBeUndefined();

    await h.handle(request("POST", `${BASE}/settings`, { provider: "gemini", model: "gemini-3-pro-image", label_key: "mito" }));
    const noVolume = await run();
    expect(noVolume.status).toBe(409);
    expect(body(noVolume)).toMatchObject({ success: false, needs_volume: true });

    await h.handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_56 }));
    const notAgreed = await run();
    expect(notAgreed.status).toBe(409);
    expect(body(notAgreed)).toMatchObject({
      success: false,
      needs_acknowledgement: true,
      provider: "gemini",
      destination: "Test destination (a service, with your key)",
    });

    await h.handle(request("POST", `${BASE}/settings`, { provider: "gemini", model: "gemini-3-pro-image", label_key: "mito", acknowledge: true, destination: "Test destination (a service, with your key)" }));
    const noKey = await run();
    expect(noKey.status).toBe(400);
    expect(body(noKey).error).toMatch(/API key/);
    expect(h.ai.job).toBeNull();
  });

  it("takes the point and plane from the viewer when the body has none (the panel's Run posts {}), and refuses a viewer without a position", async () => {
    const { ctx, h } = await ready();
    ctx.viewer.positionNm = null;
    const noPosition = await h.handle(request("POST", `${BASE}/run`, {}));
    expect(noPosition.status).toBe(400);
    expect(body(noPosition).success).toBe(false);
    ctx.viewer.positionNm = POINT;
    ctx.viewer.depthAxis = 1;
    const started = body(await h.handle(request("POST", `${BASE}/run`, {})));
    expect(started.success).toBe(true);
    expect(await status(h)).toMatchObject({ point_nm: POINT, depth_axis: 1, plane: "XZ" });
    await h.ai.job!.settled();
    expect((await status(h)).status).toBe("ready");
  });

  it("refuses a run while one runs, waits for review or is being written, and once the day's calls are used up", async () => {
    const gemini = new TestProvider();
    let release!: () => void;
    gemini.gate = new Promise<void>((resolve) => (release = resolve));
    const { h, ctx } = await ready({
      deps: { providers: providers(gemini) },
      settings: { provider: "gemini", model: "gemini-3-pro-image", destination: gemini.info.destination },
    });
    await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", api_key: SECRET }));
    const run = () => h.handle(request("POST", `${BASE}/run`, { point_nm: POINT, depth_axis: 0 }));
    expect(body(await run()).success).toBe(true);
    await until(() => gemini.calls.length === 1);
    expect(gemini.calls[0].credentials).toEqual({ api_key: SECRET });
    expect((await status(h))).toMatchObject({ status: "running", stage: "sending" });
    expect((await run()).status).toBe(409); // still running
    release();
    await h.ai.job!.settled();
    expect((await run()).status).toBe(409); // waiting for review
    h.ai.job!.status = "accepting";
    expect((await run()).status).toBe(409); // being written
    h.ai.job = null;
    // The day's calls used up: 429 before anything is read.
    h.storage!.setItem(USAGE_KEY, JSON.stringify({ date: "2026-10-15", calls: DAILY_CALL_LIMIT }));
    const limited = await run();
    expect(limited.status).toBe(429);
    expect(body(limited)).toMatchObject({ success: false, category: "limit" });
    expect(ctx.logged.join("\n")).not.toContain(SECRET);
  });
});

describe.skipIf(!python)("a run on another dataset than the volume's", () => {
  it("is refused before anything is sent or counted: the plane would be read from one dataset and written on another's grid", async () => {
    const { ctx, h, id } = await ready();
    ctx.viewer.positionNm = POINT;
    const volumeDataset = (await readVolume(ctx.paint, id))!.dataset_path;
    await ctx.infer.openDataset("https://example.org/other.zarr");
    const res = await h.handle(request("POST", `${BASE}/run`, {}));
    expect(res.status).toBe(409);
    expect(body(res).error).toBe(`The annotation volume was painted on ${volumeDataset}, but the viewer shows https://example.org/other.zarr: open that dataset again, or make a new volume on this one.`);
    expect(h.ai.usage.callsToday()).toBe(0);
    expect(await readLog(ctx.paint, id)).toEqual([]);
  });
});

describe.skipIf(!python)("a run to ready", () => {
  it("reads the plane around the point, asks the model, and stages the mask with its preview; the status's fields are set as it goes", async () => {
    const { ctx, h, id } = await ready();
    expect((await status(h))).toMatchObject({ success: true, annotate_id: null, status: "idle", error: null, preview: null, needs_volume: false });
    const started = body(await h.handle(request("POST", `${BASE}/run`, { point_nm: POINT, depth_axis: 0 })));
    expect(started).toEqual({ success: true, annotate_id: expect.stringMatching(/^[0-9a-f]{32}$/), status: "running" });
    const early = await status(h);
    expect(early).toMatchObject({ annotate_id: started.annotate_id, status: "running", stage: "fetching_crop", provider: "fake", model: FAKE_MODEL, label_name: "mitochondria", depth_axis: 0, plane: "XY", point_nm: POINT, preview: null });
    expect(early.prompt).toBe(editablePrompt(ORGANELLES.mito));
    await h.ai.job!.settled();

    const done = await status(h);
    expect(done.status).toBe("ready");
    expect(done.stage).toBeNull();
    expect(done.error).toBeNull();
    // The plane 64 voxels a side around voxel (100, 250, 250), one voxel thick in z.
    expect(done.write_box).toEqual({ lo: [100, 218, 218], hi: [101, 282, 282] });
    expect(done.mask_fraction).toBeGreaterThan(0);
    expect(done.mask_fraction).toBeLessThan(0.5);
    expect(done.preview).toBeNull(); // only with ?preview=1
    expect(ctx.infer.reads).toEqual([{ level: 0, lo: [100, 218, 218], hi: [101, 282, 282] }]);

    const withPreview = await status(h, true);
    const codec = jsCodec();
    const decode = (b64: string) => codec.decode(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)), "image/png");
    for (const key of ["input_png", "model_png", "overlay_png"]) {
      expect(withPreview.preview[key]).toMatch(/^[A-Za-z0-9+/]+={0,2}$/); // what the page's pngUrl accepts
      const image = await decode(withPreview.preview[key]);
      expect([image.width, image.height]).toEqual([64, 64]);
    }
    expect(withPreview.preview.overlay_png).not.toBe(withPreview.preview.input_png);
    // The input is the gradient as the fake raw gives it, grey; the overlay tints the mask red.
    const input = await decode(withPreview.preview.input_png);
    const overlayImage = await decode(withPreview.preview.overlay_png);
    expect(input.data[0]).toBe(input.data[1]);
    let tinted = 0;
    for (let i = 0; i < overlayImage.data.length; i += 4) if (overlayImage.data[i] !== input.data[i]) tinted++;
    expect(tinted / (64 * 64)).toBeCloseTo(done.mask_fraction, 2);

    expect(body(await h.handle(request("GET", `${BASE}/config`))).calls_today).toBe(1);
    const log = await readLog(ctx.paint, id);
    expect(log.map((e) => e.event)).toEqual(["requested", "staged"]);
    expect(log[0]).toMatchObject({ user: "browser", annotate_id: started.annotate_id, volume_id: id, dataset: "https://example.org/data.zarr", provider: "fake", model: FAKE_MODEL, label_name: "mitochondria", point_nm: POINT, depth_axis: 0, prompt_override: null });
    expect(log[0].time).toMatch(/^2026-10-15T12:00:00[+-]\d\d:\d\d$/);
    expect(log[1]).toMatchObject({ event: "staged", mask_fraction: done.mask_fraction, write_box: done.write_box, usage: {} });
  });

  it("fails, with the provider's message and category in the log, and refunds a call the model never saw", async () => {
    const gemini = new TestProvider();
    gemini.fail = new AIAnnotateError("auth", "The API key was not accepted.");
    const { ctx, h, id } = await ready({ deps: { providers: providers(gemini) }, settings: { provider: "gemini", model: "gemini-3-pro-image", destination: gemini.info.destination } });
    await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", api_key: SECRET }));
    await runToEnd(h);
    expect(await status(h)).toMatchObject({ status: "failed", stage: null, error: "The API key was not accepted.", preview: null, needs_volume: false });
    expect(h.ai.usage.callsToday()).toBe(0); // refunded: auth
    expect((await readLog(ctx.paint, id)).map((e) => [e.event, e.category, e.error])).toEqual([
      ["requested", undefined, undefined],
      ["failed", "auth", "The API key was not accepted."],
    ]);
    // A failure after the call was made stays counted.
    gemini.fail = new AIAnnotateError("bad_response", "The model returned no image.");
    await h.handle(request("POST", `${BASE}/reject`, { annotate_id: h.ai.job!.annotateId }));
    await runToEnd(h);
    expect((await status(h)).error).toBe("The model returned no image.");
    expect(h.ai.usage.callsToday()).toBe(1);
    // Anything else is the generic failure, with the detail logged, not shown.
    gemini.fail = new RangeError("boom") as unknown as AIAnnotateError;
    await h.handle(request("POST", `${BASE}/reject`, { annotate_id: h.ai.job!.annotateId }));
    await runToEnd(h);
    expect((await status(h)).status).toBe("failed");
    expect((await status(h)).error).not.toContain("boom");
    expect(ctx.logged.some((line) => line.includes("boom"))).toBe(true);
    expect(ctx.logged.join("\n")).not.toContain(SECRET);
  });

});

describe.skipIf(!python)("Shift+G", () => {
  it("without a volume says so with needs_volume, for the page to offer one", async () => {
    const ctx = fakeContext();
    const h = harness(ctx);
    await h.handle(request("POST", `${BASE}/settings`, FAKE_SETTINGS));
    ctx.viewer.pointer = POINT;
    ctx.viewer.bindings[0].handler();
    await until(() => h.ai.job !== null);
    expect(await status(h)).toMatchObject({ status: "failed", needs_volume: true });
  });

  it("with the pointer over the data starts a run in the viewer's plane, and a later refusal does not replace a result waiting for review", async () => {
    const { ctx, h } = await ready();
    ctx.viewer.pointer = POINT;
    ctx.viewer.depthAxis = 2;
    ctx.viewer.bindings[0].handler();
    await until(() => h.ai.job !== null);
    // The fake pipeline is quick: the job may be ready by the first poll.
    const begun = await status(h);
    expect(begun).toMatchObject({ plane: "YZ", point_nm: POINT, provider: "fake" });
    expect(["running", "ready"]).toContain(begun.status);
    await h.ai.job!.settled();
    const job = h.ai.job;
    expect((await status(h)).status).toBe("ready");
    ctx.viewer.pointer = null;
    ctx.viewer.bindings[0].handler();
    await tick();
    await tick();
    expect(h.ai.job).toBe(job);
    expect((await status(h)).status).toBe("ready");
    // Not dropped silently: the panel keeps the result, the log says why the key did nothing.
    expect(ctx.logged.some((line) => line.includes("Shift+G refused"))).toBe(true);
  });
});

describe.skipIf(!python)("POST /api/finetune/ai-annotate/resend", () => {
  it("sends the same plane again with the edited prompt under the same id, and replaces the preview", async () => {
    const gemini = new TestProvider();
    const { ctx, h, id } = await ready({ deps: { providers: providers(gemini) }, settings: { provider: "gemini", model: "gemini-3-pro-image", destination: gemini.info.destination } });
    await h.handle(request("POST", `${BASE}/credentials`, { provider: "gemini", api_key: SECRET }));
    const annotateId = await runToEnd(h);
    const firstImage = gemini.calls[0].request.image;
    const res = body(await h.handle(request("POST", `${BASE}/resend`, { annotate_id: annotateId, prompt: "Paint only the big ones." })));
    expect(res).toEqual({ success: true, annotate_id: annotateId, status: "running" });
    // The previous answer's preview and share of the plane are gone while the new call runs.
    expect(await status(h)).toMatchObject({ status: "running", stage: "sending", prompt: "Paint only the big ones.", preview: null, mask_fraction: null });
    await h.ai.job!.settled();
    expect(await status(h)).toMatchObject({ status: "ready", annotate_id: annotateId, prompt: "Paint only the big ones." });
    expect(gemini.calls.length).toBe(2);
    expect(gemini.calls[1].request.image).toBe(firstImage); // not read again
    expect(gemini.calls[1].request.prompt).toContain("Paint only the big ones.");
    expect(gemini.calls[1].credentials).toEqual({ api_key: SECRET });
    expect(ctx.infer.reads.length).toBe(1);
    expect(h.ai.usage.callsToday()).toBe(2);
    expect((await readLog(ctx.paint, id)).map((e) => e.event)).toEqual(["requested", "staged", "resent", "staged"]);
    expect((await readLog(ctx.paint, id))[2]).toMatchObject({ prompt_override: "Paint only the big ones.", provider: "gemini" });
    // The catalog's text sent back is no override.
    await h.handle(request("POST", `${BASE}/resend`, { annotate_id: annotateId, prompt: editablePrompt(ORGANELLES.mito) }));
    await h.ai.job!.settled();
    expect((await status(h)).prompt).toBe(editablePrompt(ORGANELLES.mito));
    expect((await readLog(ctx.paint, id))[4]).toMatchObject({ event: "resent", prompt_override: null });
  });

});

describe.skipIf(!python)("POST /api/finetune/ai-annotate/accept", () => {
  it("writes the mask's objects with ids of their own and background elsewhere in the plane, refreshes the layer, and leaves nothing to review", async () => {
    const { ctx, h, id, volume } = await ready();
    const annotateId = await runToEnd(h);
    const { write_box: box } = await status(h);
    const res = body(await h.handle(request("POST", `${BASE}/accept`, { annotate_id: annotateId, overwrite: false })));
    expect(res.success).toBe(true);
    expect(res.filled_foreground).toBeGreaterThan(0);
    expect(res.filled_foreground + res.filled_background).toBe(64 * 64);
    expect(res).toMatchObject({ overwritten: 0, reload_viewer: true, layer_refreshed: true, can_undo: true });
    expect(ctx.viewer.refreshed).toEqual([paintLayerName(id)]);

    const labels = await readBox(ctx.paint, volume, box.lo, box.hi);
    expect(labels).toBeInstanceOf(Uint8Array);
    let foreground = 0, background = 0;
    for (const v of labels) {
      if (v >= 2) foreground++;
      else if (v === 1) background++;
    }
    expect([foreground, background]).toEqual([res.filled_foreground, res.filled_background]);
    // The planes beside are untouched.
    const below = await readBox(ctx.paint, volume, [99, 218, 218], [100, 282, 282]);
    expect(below.every((v) => v === 0)).toBe(true);
    expect((await status(h)).status).toBe("idle");
    expect(h.ai.job).toBeNull();
    const log = await readLog(ctx.paint, id);
    expect(log[2]).toMatchObject({ event: "accepted", annotate_id: annotateId, volume_id: id, overwrite: false, filled_foreground: res.filled_foreground, filled_background: res.filled_background, overwritten: 0, write_box: box });
    // Accepted: nothing to accept or reject any more.
    expect((await h.handle(request("POST", `${BASE}/accept`, { annotate_id: annotateId, overwrite: false }))).status).toBe(409);
  });

  it("keeps what the user painted unless overwrite, and even then its background never erases paint", async () => {
    const { ctx, h, volume } = await ready();
    const annotateId = await runToEnd(h);
    const mask = h.ai.job!.maskWrite!;
    const box = h.ai.job!.writeBox!;
    const masked = voxelOf(box, mask.indexOf(1));
    const unmasked = voxelOf(box, mask.indexOf(0));
    // Painted since the plane was staged: a stroke of 5 inside the mask, 7 outside it.
    await paintVoxel(ctx, volume, masked, 5);
    await paintVoxel(ctx, volume, unmasked, 7);
    const kept = body(await h.handle(request("POST", `${BASE}/accept`, { annotate_id: annotateId, overwrite: false })));
    expect(kept.success).toBe(true);
    expect(kept.overwritten).toBe(0);
    let labels = await readBox(ctx.paint, volume, box.lo, box.hi);
    const at = (voxel: number[]) => labels[(voxel[1] - box.lo[1]) * 64 + (voxel[2] - box.lo[2])];
    expect(at(masked)).toBe(5);
    expect(at(unmasked)).toBe(7);
    // The object the stroke is in takes the stroke's id (fill.seed_labels).
    let fives = 0;
    for (const v of labels) if (v === 5) fives++;
    expect(fives).toBeGreaterThan(1);

    // Undo, paint a masked voxel background (1), and accept with overwrite:
    // the object replaces the 1 (overwritten), the 7 outside the mask stays.
    expect(body(await h.handle(request("POST", "/api/finetune/view-labels/undo", {}))).success).toBe(true);
    await runToEnd(h);
    const second = h.ai.job!.annotateId!;
    await paintVoxel(ctx, volume, masked, 1);
    const over = body(await h.handle(request("POST", `${BASE}/accept`, { annotate_id: second, overwrite: true })));
    expect(over.success).toBe(true);
    expect(over.overwritten).toBe(1);
    labels = await readBox(ctx.paint, volume, box.lo, box.hi);
    expect(at(masked)).toBeGreaterThanOrEqual(2);
    expect(at(unmasked)).toBe(7);
  });

  it("counts an instance target's ids up on a uint16 volume, past every id in the box", async () => {
    const { ctx, h, volume } = await ready({ model: NAME_54 });
    expect(volume.dtype).toBe("uint16");
    const annotateId = await runToEnd(h);
    const box = h.ai.job!.writeBox!;
    await paintVoxel(ctx, volume, voxelOf(box, h.ai.job!.maskWrite!.indexOf(0)), 9);
    const res = body(await h.handle(request("POST", `${BASE}/accept`, { annotate_id: annotateId, overwrite: false })));
    expect(res.success).toBe(true);
    const labels = await readBox(ctx.paint, volume, box.lo, box.hi);
    expect(labels).toBeInstanceOf(Uint16Array);
    const ids = new Set<number>();
    for (const v of labels) if (v >= 2 && v !== 9) ids.add(v);
    expect(ids.size).toBeGreaterThan(0);
    expect(Math.min(...ids)).toBe(10);
  });

});

describe.skipIf(!python)("POST /api/finetune/view-labels/undo", () => {
  it("restores the box as it was, keeping a stroke painted since, and says when nothing is left", async () => {
    const { ctx, h, volume } = await ready();
    const annotateId = await runToEnd(h);
    const accepted = body(await h.handle(request("POST", `${BASE}/accept`, { annotate_id: annotateId })));
    const box = { lo: [100, 218, 218], hi: [101, 282, 282] };
    // A stroke after the accept, on a voxel the accept wrote: kept by the undo.
    const labels = await readBox(ctx.paint, volume, box.lo, box.hi);
    const i = labels.findIndex((v) => v === 1);
    const stroke = voxelOf(box, i);
    await paintVoxel(ctx, volume, stroke, 3);
    const res = body(await h.handle(request("POST", "/api/finetune/view-labels/undo", {})));
    expect(res).toEqual({ success: true, restored: accepted.filled_foreground + accepted.filled_background - 1, reload_viewer: true, layer_refreshed: true, can_undo: false });
    const after = await readBox(ctx.paint, volume, box.lo, box.hi);
    expect(after[i]).toBe(3);
    let nonZero = 0;
    for (const v of after) if (v !== 0) nonZero++;
    expect(nonZero).toBe(1);
    expect(ctx.viewer.refreshed.length).toBe(2);
    expect((await h.handle(request("POST", "/api/finetune/view-labels/undo", {}))).status).toBe(409); // nothing left
  });

});

describe.skipIf(!python)("POST /api/finetune/ai-annotate/reject", () => {
  it("throws the staged mask away, writing nothing, and takes a failed job too", async () => {
    const { ctx, h, id } = await ready();
    const annotateId = await runToEnd(h);
    expect((await h.handle(request("POST", `${BASE}/reject`, { annotate_id: "0".repeat(32) }))).status).toBe(409);
    expect(body(await h.handle(request("POST", `${BASE}/reject`, { annotate_id: annotateId })))).toEqual({ success: true });
    expect((await status(h)).status).toBe("idle");
    expect(await ctx.paint.list(id, "annotation/s0/")).toEqual(["annotation/s0/.zarray"]);
    expect((await readLog(ctx.paint, id)).map((e) => e.event)).toEqual(["requested", "staged", "rejected"]);
    expect((await h.handle(request("POST", `${BASE}/reject`, { annotate_id: annotateId }))).status).toBe(409);
    expect((await h.handle(request("POST", `${BASE}/reject`, {}))).status).toBe(400);
  });
});

describe("the daily limit", () => {
  it("counts calls per local day in the store, refuses past the limit, refunds, and starts afresh the next day", () => {
    const storage = new MemoryKeyValueStore();
    let now = new Date(2026, 9, 15, 23, 59, 0);
    const usage = createUsage(storage, () => now);
    expect(usage.callsToday()).toBe(0);
    expect(usage.checkAndCount(2)).toBe(1);
    expect(usage.checkAndCount(2)).toBe(2);
    expect(() => usage.checkAndCount(2)).toThrow(AIAnnotateError);
    expect(JSON.parse(storage.getItem(USAGE_KEY)!)).toEqual({ date: "2026-10-15", calls: 2 });
    usage.refund();
    expect(usage.callsToday()).toBe(1);
    now = new Date(2026, 9, 16, 0, 1, 0);
    expect(usage.callsToday()).toBe(0);
    usage.refund(); // never below zero
    expect(usage.callsToday()).toBe(0);
    // Unreadable or another day's record: today starts from 0. No storage: the page's own count.
    storage.setItem(USAGE_KEY, "not json");
    expect(usage.callsToday()).toBe(0);
    const inMemory = createUsage(null, () => now);
    inMemory.checkAndCount(5);
    expect(inMemory.callsToday()).toBe(1);
  });

  it.skipIf(!python)("is applied to a run (429) and shown by the config", async () => {
    const { h } = await ready({ deps: { dailyCallLimit: 1 } });
    const annotateId = await runToEnd(h);
    expect(body(await h.handle(request("GET", `${BASE}/config`)))).toMatchObject({ daily_call_limit: 1, calls_today: 1 });
    const resend = await h.handle(request("POST", `${BASE}/resend`, { annotate_id: annotateId }));
    expect(resend.status).toBe(429);
    await h.handle(request("POST", `${BASE}/accept`, { annotate_id: annotateId }));
    const run = await h.handle(request("POST", `${BASE}/run`, { point_nm: POINT, depth_axis: 0 }));
    expect(run.status).toBe(429);
    expect(body(run).category).toBe("limit");
  });
});

describe("in the backend", () => {
  it("answers the AI-annotate routes and the undo", async () => {
    const ctx = fakeContext();
    const backend = createBackend(ctx, { aiAnnotate: { providers: providers(), codec: jsCodec(), storage: new MemoryKeyValueStore() } });
    const config = await backend.handle(request("GET", `${BASE}/config`));
    expect(config.status).toBe(200);
    expect(body(config).enabled).toBe(true);
    expect((await backend.handle(request("GET", `${BASE}/status`))).status).toBe(200);
    expect((await backend.handle(request("POST", "/api/finetune/view-labels/undo", {}))).status).toBe(409);
  });

  it("builds its codec and providers only when first used, so a page without a canvas can still register the routes", () => {
    const ai: AiAnnotateState = aiAnnotateRoutes(new Router());
    expect(ai.job).toBeNull();
    expect(ai.settings).toBeNull();
  });
});
