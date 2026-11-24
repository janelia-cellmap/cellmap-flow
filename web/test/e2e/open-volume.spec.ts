// The header's Open Volume (Open Dataset in the server mode; connect.js
// initOpenDataset): with a volume open, the connect panel in the viewer's
// place, Cancel putting the viewer back; and another volume connected from
// it, the models carried over: still listed, and their layers on the new
// volume's viewer.
//
// Opt-in with E2E=1 (see playwright.config.ts): it reads two public
// datasets' metadata from S3 and a model from Hugging Face. It reads the
// layers off the viewer rather than wait for a chunk, which takes minutes on
// WASM (smoke.spec.ts waits for one).
import { expect, type Page, test } from "@playwright/test";

// Two OpenOrganelle volumes with CORS (smoke.spec.ts's, and the connect
// field's example, viewer/index.ts EXAMPLE_DATASET_URL).
const FIRST = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
const SECOND = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8";
const MODEL = process.env.E2E_MODEL ?? "setup_53";

/** Connect `url` from the panel and wait for the reloaded page's viewer:
 * connect.js reloads the page a second after the route answers. */
async function connect(page: Page, url: string): Promise<void> {
  await page.locator("#datasetPathInput").fill(url);
  const reloaded = page.waitForEvent("load", { timeout: 120_000 });
  await page.locator("#setDataBtn").click();
  await reloaded;
  await expect(page.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });
  await expect(page.locator("#my_iframe")).toBeVisible({ timeout: 120_000 });
}

/** The layers of the viewer in the iframe, each its name and its source as
 * text: Neuroglancer's page is of this page's origin in the browser mode. */
async function layers(page: Page): Promise<{ name: string; source: string }[]> {
  return page.evaluate(() => {
    type Viewer = { state: { toJSON(): { layers?: { name: string; source: unknown }[] } } };
    const frame = document.getElementById("my_iframe") as HTMLIFrameElement | null;
    const viewer = (frame?.contentWindow as unknown as { viewer?: Viewer } | null)?.viewer;
    return (viewer?.state.toJSON().layers ?? []).map((layer) => ({ name: layer.name, source: JSON.stringify(layer.source) }));
  });
}

/** Whether the viewer shows a model's layer (served by this page, under
 * /virtual/; its name is the model's own, mito_aff_setup53_16nm for
 * setup_53) and the raw of `volume` (a part of its URL). */
async function shows(page: Page, volume: string): Promise<boolean> {
  const now = await layers(page);
  return now.some((layer) => layer.source.includes("/virtual/")) && now.some((layer) => layer.source.includes(volume));
}

test.describe("open another volume", () => {
  test.skip(process.env.E2E !== "1", "set E2E=1 to run the end-to-end tests (needs the network)");

  test("Open Volume shows the panel over the viewer, and the models carry over to the volume it opens", async ({ page }) => {
    await page.goto("./");
    await expect(page.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });
    await connect(page, FIRST);
    const panel = page.locator("#noNeuroglancerPanel");
    await expect(panel).toBeHidden();

    // A model served on the first volume.
    await page.locator("#model-tab").click();
    const accordion = page.locator("#heading_hf button");
    if ((await accordion.getAttribute("aria-expanded")) !== "true") await accordion.click();
    const row = page.locator("#hfModelsContainer .hf-model-item").filter({ has: page.locator(`input[value*="${MODEL}"]`) }).first();
    await row.locator("input.hf-model-checkbox").check();
    await page.locator("#submitModelsBtn").click();
    const loaded = page.locator("#loadedModelsList .loaded-model-item");
    await expect(loaded.filter({ hasText: MODEL })).toHaveCount(1, { timeout: 120_000 });
    await expect.poll(() => shows(page, "jrc_mus-liver"), { timeout: 120_000 }).toBe(true);

    // Open Volume, then Cancel: the viewer as it was.
    await page.locator("#openDatasetBtn").click();
    await expect(panel).toBeVisible();
    await expect(page.locator("#my_iframe")).toBeHidden();
    await expect(page.locator("#datasetPathInput")).toBeFocused();
    await expect(panel.getByText("Open another volume")).toBeVisible();
    await expect(panel.getByText("No Viewer Connected")).toBeHidden();
    await page.locator("#cancelOpenDatasetBtn").click();
    await expect(panel).toBeHidden();
    await expect(page.locator("#my_iframe")).toBeVisible();

    // Open Volume, and the second volume connected from it.
    await page.locator("#openDatasetBtn").click();
    await connect(page, SECOND);
    await expect.poll(() => shows(page, "jrc_hela-2"), { timeout: 120_000 }).toBe(true);
    expect((await layers(page)).some((layer) => layer.source.includes("jrc_mus-liver")), "nothing of the first volume is left").toBe(false);
    await page.locator("#model-tab").click();
    await expect(loaded.filter({ hasText: MODEL })).toHaveCount(1, { timeout: 60_000 });
  });
});
