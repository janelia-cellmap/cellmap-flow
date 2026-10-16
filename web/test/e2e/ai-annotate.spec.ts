// AI-assisted annotation in the browser, end to end, with the fake provider
// (nothing leaves the machine): a volume with the 2D mito student running, a
// new annotation volume, the AI panel on the fake provider with the
// destination acknowledged, Annotate at View Centre, the review opening with
// its three pictures, Accept writing the plane into the paint store (read
// back through the worker's paint route), Undo taking it back, the audit log
// in the volume, then Shift+G over the viewer starting a run from the
// pointer, rejected. The dashboard's own controls are driven here, since the
// panel's script is what the port had to fit; the routes it calls are
// checked by the unit tests (backend-ai-annotate.test.ts).
//
// Opt-in with E2E=1 (see playwright.config.ts). The model's own chunks are
// not waited for: the AI annotation reads the raw and writes the paint, and
// needs the served model only for the volume's grid.
import { expect, test } from "@playwright/test";

import { api } from "./helpers";

const DATASET =
  process.env.E2E_DATASET ??
  "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
/** The 2D mito student: one slice in, one out, so a plane is one chunk thick. */
const MODEL = process.env.E2E_FINETUNE_MODEL ?? "setup_53_2d";
/** The in-page backend answers slowly while WASM inference holds the main thread. */
const SLOW_MS = Number(process.env.E2E_SLOW_MS ?? 120_000);

interface Status {
  success: boolean;
  status: string;
  annotate_id: string | null;
  plane: string | null;
  write_box: { lo: number[]; hi: number[] } | null;
  mask_fraction: number | null;
  error: string | null;
  provider: string | null;
  model: string | null;
}

test.describe("AI-assisted annotation in the browser", () => {
  test.skip(process.env.E2E !== "1", "set E2E=1 to run the browser AI-annotate test (needs the network for the raw volume and the model manifest)");

  test("annotates a plane with the fake provider, accepts it, undoes it, and runs from Shift+G", async ({ page }, testInfo) => {
    const consoleLines: string[] = [];
    page.on("console", (message) => consoleLines.push(`[${message.type()}] ${message.text()}`));
    page.on("pageerror", (error) => consoleLines.push(`[pageerror] ${error.message}`));
    try {
      // A link opens the volume with the model running; the first visit
      // registers the service worker and reloads once.
      await page.goto(`./?dataset=${encodeURIComponent(DATASET)}&model=${MODEL}`);
      await expect(page.locator('html[data-session="restored"]')).toBeAttached({ timeout: 180_000 });

      // The Finetune tab and a new volume, as the user makes one.
      await page.evaluate(() => document.getElementById("finetune-tab")!.click());
      await expect(page.locator("#modelSelect option")).toHaveCount(1, { timeout: SLOW_MS });
      await page.evaluate(() => document.getElementById("createVolumeBtn")!.click());
      await expect(page.locator("#finetuneLog")).toHaveValue(/Created annotation volume/, { timeout: SLOW_MS });
      const current = await api<{ volume_id: string | null; volume: { chunk_size: number[]; dataset_shape_voxels: number[] } }>(page, "/api/finetune/session-volume");
      expect(current.volume_id).toMatch(/^vol-/);
      const volume = current.volume_id!;
      const chunk = current.volume.chunk_size;
      expect(chunk[0]).toBe(1);

      // The AI panel: the configuration lists the three providers (the panel
      // is a <details>: opened so its controls are visible), the fake one is
      // picked and its destination acknowledged for this dataset.
      await page.evaluate(() => ((document.getElementById("aiAnnotatePanel") as HTMLDetailsElement).open = true));
      await expect(page.locator("#aiAnnotateControls")).toBeVisible({ timeout: SLOW_MS });
      await expect(page.locator("#aiAnnotateProvider option")).toHaveCount(3);
      await page.selectOption("#aiAnnotateProvider", "fake");
      await expect(page.locator("#aiAnnotateModel")).toHaveValue("fake-threshold");
      await expect(page.locator("#aiAnnotateDestination")).toHaveText(/fake provider, runs in this browser/);
      await expect(page.locator("#aiAnnotateCredentials")).toBeHidden(); // the fake provider needs none
      await page.check("#aiAnnotateAck");
      await expect(page.locator("#aiAnnotateAckHint")).toHaveText(/Acknowledged for this dataset/, { timeout: SLOW_MS });

      // Annotate at View Centre: the job runs in the page and the review
      // opens over it when the result is ready, with the plane sent, the
      // "model's" image and the mask over the plane.
      await page.click("#aiAnnotateRunBtn");
      await expect(page.locator("#aiAnnotateStatus")).toHaveText(/^Ready/, { timeout: SLOW_MS });
      await expect(page.locator("#aiAnnotateReviewModal")).toHaveClass(/\bshow\b/, { timeout: 30_000 });
      for (const id of ["aiAnnotateInputImg", "aiAnnotateModelImg", "aiAnnotateOverlayImg"]) {
        await expect(page.locator(`#${id}`)).toHaveAttribute("src", /^data:image\/png;base64,[A-Za-z0-9+/]+=*$/);
      }
      const ready = await api<Status>(page, "/api/finetune/ai-annotate/status");
      consoleLines.push(`[test] ready: ${JSON.stringify({ ...ready, preview: undefined })}`);
      expect(ready.status).toBe("ready");
      expect(ready.provider).toBe("fake");
      expect(ready.model).toBe("fake-threshold");
      // The viewer opens on Neuroglancer's "xy" panel, which with z, y, x
      // dimensions is the YZ plane (docs/ai_annotate.md, "Choosing the plane").
      expect(ready.plane).toBe("YZ");
      expect(ready.write_box).not.toBeNull();
      expect(ready.mask_fraction).toBeGreaterThan(0);
      await expect(page.locator("#aiAnnotateReviewSummary")).toHaveText(/YZ plane, mitochondria/);

      // Accept: the plane's objects get ids, the rest background; the chunk
      // at the box's centre, where the fake provider paints (a disc around
      // the point), holds both when read back through the paint route.
      const { lo, hi } = ready.write_box!;
      const chunkKey = lo.map((v, a) => Math.floor(Math.floor((v + hi[a] - 1) / 2) / chunk[a])).join(".");
      const readChunk = () =>
        page.evaluate(
          async ([volume, key]) => {
            const res = await fetch(`paint/${volume}/annotation/s0/${key}`);
            if (res.status !== 200) return { status: res.status, background: 0, objects: 0, ids: [] as number[] };
            const bytes = new Uint8Array(await res.arrayBuffer());
            const zarray = await (await fetch(`paint/${volume}/annotation/s0/.zarray`)).json();
            const labels = zarray.dtype === "<u2" ? new Uint16Array(bytes.buffer) : bytes;
            let background = 0;
            const ids = new Set<number>();
            for (const v of labels) {
              if (v === 1) background++;
              else if (v >= 2) ids.add(v);
            }
            return { status: 200, background, objects: labels.filter((v) => v >= 2).length, ids: [...ids].sort((a, b) => a - b) };
          },
          [volume, chunkKey] as const,
        );
      await page.click("#aiAnnotateAcceptBtn");
      await expect(page.locator("#finetuneLog")).toHaveValue(/Accepted the AI annotation: \d+ foreground and \d+ background voxels/, { timeout: SLOW_MS });
      const written = await readChunk();
      consoleLines.push(`[test] chunk ${chunkKey} after accept: ${JSON.stringify(written)}`);
      expect(written.status).toBe(200);
      expect(written.objects).toBeGreaterThan(0);
      expect(written.background).toBeGreaterThan(0);
      expect(written.ids[0]).toBe(2); // ids from 2 up, one per object
      await expect(page.locator("#aiAnnotateReviewModal")).not.toHaveClass(/\bshow\b/, { timeout: 30_000 });
      expect((await api<Status>(page, "/api/finetune/ai-annotate/status")).status).toBe("idle");

      // Label it's Undo puts the plane back: nothing painted in that chunk any more.
      await expect(page.locator("#undoViewLabelsBtn")).toBeEnabled();
      await page.click("#undoViewLabelsBtn");
      await expect(page.locator("#finetuneLog")).toHaveValue(/Undid the last label action: \d+ voxels restored/, { timeout: SLOW_MS });
      const restored = await readChunk();
      expect(restored.objects).toBe(0);
      expect(restored.background).toBe(0);
      await expect(page.locator("#undoViewLabelsBtn")).toBeDisabled();

      // The audit log travels with the volume.
      const audit = await page.evaluate(async (volume) => {
        const res = await fetch(`paint/${volume}/ai_annotate_log.jsonl`);
        return { status: res.status, events: (await res.text()).trim().split("\n").map((line) => JSON.parse(line).event) };
      }, volume);
      expect(audit.status).toBe(200);
      expect(audit.events).toEqual(["requested", "staged", "accepted"]);

      // Shift+G over the viewer: the key is bound in Neuroglancer's page, the
      // pointer's position is the point, and the job runs as from the button.
      const frame = page.frameLocator("#my_iframe");
      await frame.locator("canvas").first().hover();
      await page.keyboard.press("Shift+G");
      await expect(page.locator("#aiAnnotateStatus")).toHaveText(/^(Running|Ready)/, { timeout: 30_000 });
      await expect(page.locator("#aiAnnotateStatus")).toHaveText(/^Ready/, { timeout: SLOW_MS });
      const fromKey = await api<Status>(page, "/api/finetune/ai-annotate/status");
      expect(fromKey.annotate_id).not.toBe(ready.annotate_id);
      expect(fromKey.plane).toBe("YZ");
      await page.click("#aiAnnotateRejectBtn");
      await expect(page.locator("#finetuneLog")).toHaveValue(/Rejected the AI annotation: nothing was written/, { timeout: SLOW_MS });
      expect((await api<Status>(page, "/api/finetune/ai-annotate/status")).status).toBe("idle");
    } finally {
      await testInfo.attach("browser console", { body: consoleLines.join("\n"), contentType: "text/plain" });
    }
  });
});
