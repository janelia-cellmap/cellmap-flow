// The dashboard's download links, clicked as a user clicks them. Chrome sends
// a download link's request (`<a download>`) to the network, past the service
// worker that answers /api/, so the page fetches and saves these itself
// (src/rebase.ts, installApiDownloads); before it did, the click saved Vite's
// index.html here and failed on the deployed site. Export Config needs no
// dataset or model, so it is the link clicked; the Finetune tab's volume and
// model downloads are the same kind of link, taken by the same handler, and
// finetune.spec.ts checks what their routes answer.
//
// Opt-in with E2E=1, as the other end-to-end tests are, though this one needs
// no network.
import { readFile } from "node:fs/promises";

import { expect, test } from "@playwright/test";

test.describe("download links", () => {
  test.skip(process.env.E2E !== "1", "set E2E=1 to run the end-to-end tests");

  test("Export Config saves the YAML the in-page backend writes", async ({ page }) => {
    await page.goto("./");
    await expect(page.locator('html[data-dashboard="loaded"]')).toBeAttached({ timeout: 60_000 });
    const [download] = await Promise.all([page.waitForEvent("download"), page.locator('a[href*="api/export-config"]').click()]);
    expect(download.suggestedFilename()).toMatch(/\.yaml$/);
    // The config itself, not the index.html saved before the page took these links.
    const saved = await readFile(await download.path(), "utf8");
    expect(saved).toMatch(/^models:/m);
  });
});
