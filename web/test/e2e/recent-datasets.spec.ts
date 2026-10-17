// The connect panel offers the datasets connected to before (connect.js,
// the path field's datalist): a dataset that opened is first in the list of
// the next tab's panel, which starts fresh.
//
// Opt-in with E2E=1 (see playwright.config.ts): it reads a public dataset's
// metadata from S3.
import { expect, test } from "@playwright/test";

const DATASET =
  process.env.E2E_DATASET ??
  "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";

test.describe("recent datasets", () => {
  test.skip(process.env.E2E !== "1", "set E2E=1 to run the end-to-end tests (needs the network)");

  test("a dataset connected to is offered first in a new tab's path field", async ({ page, context }) => {
    await page.goto("./");
    await expect(page.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });
    await page.locator("#datasetPathInput").fill(DATASET);
    // connect.js reloads the page once the dataset is set.
    const reloaded = page.waitForEvent("load", { timeout: 120_000 });
    await page.locator("#setDataBtn").click();
    await reloaded;

    const fresh = await context.newPage();
    await fresh.goto("./");
    await expect(fresh.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });
    await expect(fresh.locator("#datasetPathInput")).toHaveAttribute("list", "recentDatasets");
    const offered = await fresh.locator("#recentDatasets option").evaluateAll((options) => options.map((o) => (o as HTMLOptionElement).value));
    expect(offered[0]).toBe(DATASET);
  });
});
