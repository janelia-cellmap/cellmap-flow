// Finetuning in the browser, end to end: a volume with the 2D mito student
// running, a new annotation volume, one chunk painted the way the fork's
// brush paints it (a PUT to the worker's paint route), the head trained in
// the tab, and the finetuned model served in place of the base: a chunk of
// its layer comes back, the Models tab lists it, the paint and the model
// download. The dashboard's own buttons are not clicked here: they call
// these routes, which is what this test drives, through the page's fetch
// (relayed by the service worker to the in-page backend, as the dashboard's
// requests are); the smoke test covers the page's controls.
//
// Opt-in with E2E=1 (see playwright.config.ts). On the WASM provider the
// model's passes over the painted chunk and the finetuned layer's first
// chunk take minutes; the smallest 2D block is the quickest.
import { expect, type Page, test } from "@playwright/test";

import { isChunkUrl } from "./smoke.spec";

const DATASET =
  process.env.E2E_DATASET ??
  "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8";
/** The 2D mito student: one slice in, one out, so a painted chunk is one slice. */
const MODEL = process.env.E2E_FINETUNE_MODEL ?? "setup_53_2d";
const CHUNK_TIMEOUT_MS = Number(process.env.E2E_CHUNK_TIMEOUT_MS ?? 8 * 60_000);
const TRAIN_TIMEOUT_MS = Number(process.env.E2E_TRAIN_TIMEOUT_MS ?? 8 * 60_000);

/** The view a chunk URL belongs to: the segment after /virtual/<page>/. */
function viewOf(url: string): string {
  const path = new URL(url).pathname;
  return decodeURIComponent(path.split("/virtual/")[1]?.split("/")[1] ?? "");
}

/** A JSON route of the in-page backend, called from the page. */
async function api<T = Record<string, unknown>>(page: Page, path: string, body?: unknown): Promise<T> {
  return page.evaluate(
    async ([path, body]) => {
      const init: RequestInit = body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
      return (await fetch(path, init)).json();
    },
    [path, body] as const,
  );
}

test.describe("finetuning in the browser", () => {
  test.skip(process.env.E2E !== "1", "set E2E=1 to run the browser finetune test (needs the network; minutes on WASM)");

  test("paints a chunk, trains the head and serves the finetuned model", async ({ page }, testInfo) => {
    const consoleLines: string[] = [];
    page.on("console", (message) => consoleLines.push(`[${message.type()}] ${message.text()}`));
    page.on("pageerror", (error) => consoleLines.push(`[pageerror] ${error.message}`));
    try {
      // A link opens the volume with the model running (src/link.ts); the
      // first visit registers the service worker and reloads once.
      const baseChunk = page.waitForResponse((r) => r.status() === 200 && isChunkUrl(r.url()), { timeout: CHUNK_TIMEOUT_MS });
      await page.goto(`./?dataset=${encodeURIComponent(DATASET)}&model=${MODEL}`);
      await expect(page.locator('html[data-session="restored"]')).toBeAttached({ timeout: 180_000 });
      const baseView = viewOf((await baseChunk).url());
      consoleLines.push(`[test] base chunk of ${baseView}`);

      // The Finetune tab's model list names the served model; a new volume for it.
      const models = await api<{ models: { name: string; finetune_modes: string[] }[] }>(page, "/api/finetune/models");
      expect(models.models).toHaveLength(1);
      expect(models.models[0].finetune_modes).toEqual(["head"]);
      const modelName = models.models[0].name;
      const created = await api<{ success: boolean; volume_id: string; zarr_path: string; neuroglancer_url: string; metadata: { chunk_size: number[]; dataset_shape_voxels: number[] } }>(
        page, "/api/finetune/create-volume", { model_name: modelName },
      );
      expect(created.success).toBe(true);
      expect(created.metadata.chunk_size[0]).toBe(1); // one slice: a 2D model's block
      const added = await api<{ success: boolean; layer_name: string }>(page, "/api/finetune/add-to-viewer", { crop_id: created.volume_id, minio_url: created.neuroglancer_url });
      expect(added.success).toBe(true);
      expect(added.layer_name).toBe(`annotation_${created.volume_id}`);

      // Paint the centre chunk as the brush would: the whole chunk, uncompressed, PUT to the paint route.
      const painted = await page.evaluate(async (c) => {
        const chunk = c.metadata.chunk_size;
        const n = chunk.reduce((a, b) => a * b, 1);
        const zarray = await (await fetch(`${c.zarr_path}/annotation/s0/.zarray`)).json();
        const labels = zarray.dtype === "<u2" ? new Uint16Array(n) : new Uint8Array(n);
        const [cz, cy, cx] = chunk;
        for (let z = 0; z < cz; z++) for (let y = 0; y < cy; y++) for (let x = 0; x < cx; x++) {
          const i = (z * cy + y) * cx + x;
          if (x < cx / 4 && y % 3 === 0) labels[i] = 1; // background
          else if (x > cx / 2 && x < cx / 2 + cx / 8 && y > cy / 2 && y < cy / 2 + cy / 8) labels[i] = 2; // one object
        }
        const index = c.metadata.dataset_shape_voxels.map((s, a) => Math.floor(s / chunk[a] / 2));
        const url = `${c.zarr_path}/annotation/s0/${index.join(".")}`;
        const put = await fetch(url, { method: "PUT", body: labels.buffer });
        const back = await fetch(url);
        return { put: put.status, get: back.status, bytes: (await back.arrayBuffer()).byteLength, expected: labels.byteLength };
      }, created);
      expect(painted.put).toBe(200);
      expect(painted.get).toBe(200);
      expect(painted.bytes).toBe(painted.expected);
      expect(await api(page, "/api/finetune/sync-annotations", {})).toMatchObject({ success: true, synced_count: 1 });

      // Train: the job runs in the page; its status says when the finetuned model is served.
      const submitted = await api<{ success: boolean; job_id: string; output_type: string; error?: string }>(page, "/api/finetune/submit", {
        model_name: modelName, num_epochs: 3, learning_rate: 0.01, loss_type: "bce", balance_classes: true, label_smoothing: 0, auto_serve: true,
      });
      expect(submitted.success, submitted.error).toBe(true);
      expect(submitted.output_type).toBe("affinities");
      const finetunedChunk = page.waitForResponse((r) => r.status() === 200 && isChunkUrl(r.url()) && viewOf(r.url()).includes("-ft1"), { timeout: CHUNK_TIMEOUT_MS });
      await expect
        .poll(async () => (await api<{ status: string }>(page, `/api/finetune/job/${submitted.job_id}/status`)).status, { timeout: TRAIN_TIMEOUT_MS, intervals: [2000] })
        .toMatch(/^(WAITING_FOR_RESTART|COMPLETED|FAILED|CANCELLED)$/);
      const status = await api<{ status: string; current_epoch: number; total_epochs: number; finetuned_model_name: string | null; inference_server_ready: boolean }>(page, `/api/finetune/job/${submitted.job_id}/status`);
      const log = await api<{ logs: string }>(page, `/api/finetune/job/${submitted.job_id}/logs`);
      consoleLines.push(`[test] job log:\n${log.logs}`);
      expect(status.status, log.logs).toBe("WAITING_FOR_RESTART");
      expect(status.inference_server_ready).toBe(true);
      expect(status.finetuned_model_name).toBe(`${modelName}-ft1`);
      expect(log.logs).toMatch(/Epoch 3\/3 - Loss: \d+\.\d{6}/);

      // Served in place of the base: the session's model is the finetuned one, the Models tab lists it, and its layer gets a chunk.
      const sessionModels = await page.evaluate(() => (JSON.parse(sessionStorage.getItem("cellmap-flow:session") ?? "{}").models ?? []).map((m: { name: string; repo: string }) => [m.name, m.repo]));
      expect(sessionModels).toEqual([[`${modelName}-ft1`, "finetuned"]]);
      const listing = await api<Record<string, { selected?: boolean }>>(page, "/api/huggingface-models");
      expect(listing[`finetuned/${modelName}-ft1`]?.selected).toBe(true);
      const served = await finetunedChunk;
      consoleLines.push(`[test] finetuned chunk ${served.url()}`);

      // The paint as a zip of the zarr, the model as .onnx.
      const downloads = await page.evaluate(async ([volume, name]) => {
        const zip = await fetch(`/api/finetune/volume/${volume}/download`);
        const onnx = await fetch(`/api/finetune/model/${name}/download`);
        return { zip: [zip.status, zip.headers.get("content-type"), (await zip.arrayBuffer()).byteLength], onnx: [onnx.status, onnx.headers.get("content-type"), (await onnx.arrayBuffer()).byteLength] };
      }, [created.volume_id, `${modelName}-ft1`] as const);
      expect(downloads.zip[0]).toBe(200);
      expect(downloads.zip[1]).toBe("application/zip");
      expect(downloads.zip[2] as number).toBeGreaterThan(painted.expected);
      expect(downloads.onnx[0]).toBe(200);
      expect(downloads.onnx[2] as number).toBeGreaterThan(1_000_000);
    } finally {
      await testInfo.attach("browser console", { body: consoleLines.join("\n"), contentType: "text/plain" });
    }
  });
});
