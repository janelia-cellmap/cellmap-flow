// The browser mode, end to end: the page boots, the dashboard appears, a
// public dataset opens in the viewer, a Hugging Face model is submitted, and
// Neuroglancer gets one computed chunk back from the service worker. Nothing
// about the output's values is checked: that is the parity suite's job
// (test/parity). This test says whether the parts are wired together.
//
// Opt-in with E2E=1 (see playwright.config.ts). On the WASM provider (no
// WebGPU in headless Chromium) a 16 nm mito model takes a minute or more per
// block, hence the long waits.
import { expect, type Locator, type Page, test } from "@playwright/test";

import { isChunkUrl } from "./helpers";

// A public OME-Zarr multiscale on S3 whose bucket answers with
// Access-Control-Allow-Origin: * (checked 2026-10-14). jrc_mus-liver's EM is
// 8 nm isotropic at s0, so s1 is exactly the 16 nm the mito models
// (setup_53, setup_54) read: 128^3 zstd chunks, uint8. An alternative is
// jrc_hela-2/.../fibsem-uint8 (s0 5.24 x 4 x 4 nm; s2 is 20.96 x 16 x 16 nm,
// the closest level to 16 nm, anisotropic in z).
const DATASET =
  process.env.E2E_DATASET ??
  "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
// The smallest model in cellmap/onnx-web-models: setup_53 (mito affinities, 5.6 M
// parameters, 5.8 MB as int8w). Matched by substring against the row's checkbox
// value/id and its label text, then by "mito", then the first row.
const MODEL = process.env.E2E_MODEL ?? "setup_53";
// On WASM a single block of the smallest model takes minutes on one CPU
// thread (GitHub Pages cannot set the COOP/COEP headers that would allow
// threads); E2E_CHUNK_TIMEOUT_MS overrides the wait.
const CHUNK_TIMEOUT_MS = Number(process.env.E2E_CHUNK_TIMEOUT_MS ?? 8 * 60_000);

/** The Hugging Face row's checkbox for `name`, as forgiving as the catalog's labels require. */
async function modelCheckbox(page: Page, name: string): Promise<Locator> {
  const rows = page.locator("#hfModelsContainer .hf-model-item");
  const candidates = [
    rows.filter({ has: page.locator(`input[value*="${name}"], input[id*="${name}"]`) }),
    rows.filter({ hasText: new RegExp(name, "i") }),
    rows.filter({ hasText: /mito/i }),
    rows,
  ];
  for (const candidate of candidates) {
    if ((await candidate.count()) > 0) return candidate.first().locator("input.hf-model-checkbox");
  }
  throw new Error("the Hugging Face accordion lists no models");
}

test.describe("browser mode", () => {
  test.skip(process.env.E2E !== "1", "set E2E=1 to run the browser smoke test (needs the network; minutes on WASM)");

  test("boots, opens a public dataset, submits a model and serves one chunk", async ({ page }, testInfo) => {
    const consoleLines: string[] = [];
    page.on("console", (message) => consoleLines.push(`[${message.type()}] ${message.text()}`));
    page.on("pageerror", (error) => consoleLines.push(`[pageerror] ${error.message}`));
    const attachLogs = async () => {
      await testInfo.attach("browser console", { body: consoleLines.join("\n"), contentType: "text/plain" });
    };

    try {
      // Boot: the first visit registers the service worker and reloads once
      // (src/main.ts); the dashboard's tabs are the sign that main.js loaded.
      await page.goto("./");
      await expect(page.locator("#myTab")).toBeVisible({ timeout: 60_000 });
      // The dashboard's own script attaches the controls' listeners; main.ts
      // marks the page once it has loaded.
      await expect(page.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });

      // The connect panel takes a URL in browser mode; connect.js posts
      // /api/set-data and reloads, and the reloaded page has the viewer.
      await page.locator("#datasetPathInput").fill(DATASET);
      // The viewer mounts its iframe as soon as the route answers, but
      // connect.js, unchanged from server mode, reloads the page a second
      // later; anything clicked before that reload is lost with the page.
      // So wait for the reload itself (the new document's load event; a
      // framenavigated event would also fire for the viewer's replaceState of
      // the page's hash), then for the dashboard's script again.
      const reloaded = page.waitForEvent("load", { timeout: 120_000 });
      await page.locator("#setDataBtn").click();
      await reloaded;
      await expect(page.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });
      await expect(page.locator("#my_iframe")).toBeAttached({ timeout: 120_000 });
      await expect(page.locator("#myTab")).toBeVisible({ timeout: 60_000 });

      // Models tab, Hugging Face accordion: opening it loads /api/huggingface-models.
      await page.locator("#model-tab").click();
      const accordion = page.locator("#heading_hf button");
      if ((await accordion.getAttribute("aria-expanded")) !== "true") await accordion.click();
      await expect(page.locator("#hfModelsContainer input.hf-model-checkbox").first()).toBeAttached({ timeout: 60_000 });
      const checkbox = await modelCheckbox(page, MODEL);
      await checkbox.check();
      consoleLines.push(`[test] ticked ${await checkbox.getAttribute("value")}`);

      // Armed before Submit: the viewer starts asking for chunks as soon as
      // the layer is added. Requests from the Neuroglancer iframe and its
      // worker are the page's requests, answered by the service worker.
      const chunk = page.waitForResponse(
        async (response) => {
          if (response.status() !== 200 || !isChunkUrl(response.url())) return false;
          try {
            return (await response.body()).byteLength > 0;
          } catch {
            return false;
          }
        },
        { timeout: CHUNK_TIMEOUT_MS },
      );
      await page.locator("#submitModelsBtn").click();

      const response = await chunk;
      consoleLines.push(`[test] chunk ${response.url()} (${(await response.body()).byteLength} bytes)`);
      expect(response.status()).toBe(200);
    } finally {
      await attachLogs();
    }
  });
});
