// Minimal entry for the simplified `index.html` (kept around for direct
// users; the dashboard.html flow is the main one). The "Open in NG"
// button points Neuroglancer at an external cellmap-flow server. The
// "In-browser BMZ" panel runs a bioimage.io model fully client-side
// via ONNX Runtime Web (no server, no Neuroglancer involvement).

import { mountNg } from "./ng-entry";
import { runBmzOnTestInput, sliceToCanvas } from "./bmz-inference";
import { registerVirtualZarrSW } from "./sw-register";
import { activateVz } from "./vz-handler";

const $ = <T extends HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const serverInput = $<HTMLInputElement>("server-url");
const datasetInput = $<HTMLInputElement>("dataset-path");
const activateBtn = $<HTMLButtonElement>("activate");
const status = $<HTMLSpanElement>("status");

function log(msg: string): void {
  status.textContent = msg;
}

activateBtn.addEventListener("click", () => {
  const server = serverInput.value.trim().replace(/\/$/, "");
  const dataset = datasetInput.value.trim().replace(/^\/+/, "").replace(/\/$/, "");
  if (!server) {
    log("Please enter the inference server URL.");
    return;
  }
  if (!dataset) {
    log("Please enter a dataset slug.");
    return;
  }
  const NM = 1e-9;
  mountNg({
    dimensions: { z: [NM, "m"], y: [NM, "m"], x: [NM, "m"] },
    layers: [
      { type: "image", source: `zarr://${server}/${dataset}/`, name: "inference", visible: true },
    ],
    selectedLayer: { visible: true, layer: "inference" },
    layout: "4panel",
  });
  log(`opened NG: zarr://${server}/${dataset}/`);
});

// ---- In-browser BMZ panel -------------------------------------------------

const bmzToggle = $<HTMLButtonElement>("bmz-toggle");
const bmzPanel = $<HTMLElement>("bmz-panel");
const bmzModel = $<HTMLSelectElement>("bmz-model");
const bmzRun = $<HTMLButtonElement>("bmz-run");
const bmzStatus = $<HTMLSpanElement>("bmz-status");
const bmzInputCanvas = $<HTMLCanvasElement>("bmz-input");
const bmzOutputCanvas = $<HTMLCanvasElement>("bmz-output");
const bmzMeta = $<HTMLDivElement>("bmz-meta");

bmzToggle.addEventListener("click", () => {
  const open = bmzPanel.classList.toggle("open");
  bmzPanel.setAttribute("aria-hidden", open ? "false" : "true");
  bmzToggle.textContent = open ? "In-browser BMZ ◂" : "In-browser BMZ ▸";
});

function setBmzStatus(msg: string, color = "#6a6"): void {
  bmzStatus.textContent = msg;
  bmzStatus.style.color = color;
}

function replaceCanvas(holder: HTMLCanvasElement, fresh: HTMLCanvasElement): void {
  // Preserve the id + element identity so the layout class targets still apply.
  const ctx = holder.getContext("2d")!;
  holder.width = fresh.width;
  holder.height = fresh.height;
  ctx.drawImage(fresh, 0, 0);
}

bmzRun.addEventListener("click", async () => {
  const modelId = bmzModel.value;
  bmzRun.disabled = true;
  setBmzStatus("loading model …", "#cc7");
  try {
    const result = await runBmzOnTestInput(modelId);
    const { manifest, input, output, reference, provider, loadMs, runMs } = result;
    replaceCanvas(bmzInputCanvas, sliceToCanvas(input, manifest.shape_in, 0, "gray"));
    // hiding-blowfish has 2 output channels; channel 0 is the foreground prob map.
    replaceCanvas(bmzOutputCanvas, sliceToCanvas(output, manifest.shape_out, 0, "magma"));

    const lines: string[] = [
      `model       : ${manifest.name} (${manifest.id})`,
      `input shape : ${manifest.shape_in.join(" × ")}`,
      `output shape: ${manifest.shape_out.join(" × ")}`,
      `provider    : ${provider}`,
      `load        : ${loadMs.toFixed(0)} ms`,
      `inference   : ${runMs.toFixed(0)} ms`,
    ];
    if (reference) {
      let maxErr = 0;
      const n = Math.min(reference.length, output.length);
      for (let i = 0; i < n; i++) {
        const d = Math.abs(reference[i] - output[i]);
        if (d > maxErr) maxErr = d;
      }
      lines.push(`vs reference: max abs err ${maxErr.toExponential(2)}`);
    }
    bmzMeta.textContent = lines.join("\n");
    setBmzStatus(`done (${provider}, ${runMs.toFixed(0)} ms)`, "#6a6");
  } catch (err) {
    console.error(err);
    setBmzStatus((err as Error).message, "#f87171");
  } finally {
    bmzRun.disabled = false;
  }
});

// ---- Stream-from-zarr (virtual zarr + service worker) -------------------

const vzSrc = $<HTMLInputElement>("vz-src");
const vzGo = $<HTMLButtonElement>("vz-go");
const vzStatus = $<HTMLDivElement>("vz-status");

const LS_VZ_SRC = "cmf-vz-src";
vzSrc.value = localStorage.getItem(LS_VZ_SRC) ?? "";

function setVzStatus(msg: string, color = "#888"): void {
  vzStatus.textContent = msg;
  vzStatus.style.color = color;
}

function normalizeZarrUrl(raw: string): string {
  // Accepts forms like:
  //   https://host/path        → unchanged
  //   zarr://https://host/p    → strip zarr://
  //   zarr://s3://bucket/p     → strip zarr://, then expand s3://
  //   s3://bucket/path         → https://bucket.s3.amazonaws.com/path
  //   zarr://bucket-host/path  → https://bucket-host/path
  let s = raw.trim().replace(/\/+$/, "");
  while (s.startsWith("zarr://")) s = s.slice("zarr://".length);
  if (s.startsWith("s3://")) {
    const rest = s.slice("s3://".length);
    const slash = rest.indexOf("/");
    const bucket = slash === -1 ? rest : rest.slice(0, slash);
    const path = slash === -1 ? "" : rest.slice(slash);
    s = `https://${bucket}.s3.amazonaws.com${path}`;
  } else if (!/^https?:\/\//.test(s)) {
    s = "https://" + s;
  }
  return s;
}

vzGo.addEventListener("click", async () => {
  const raw = vzSrc.value.trim();
  if (!raw) {
    setVzStatus("Enter a source zarr URL (https:// or zarr:// or s3://).", "#facc15");
    return;
  }
  const modelId = bmzModel.value;
  vzGo.disabled = true;
  const zarrUrl = normalizeZarrUrl(raw);
  setVzStatus(`resolved → ${zarrUrl}\nregistering service worker …`, "#cc7");
  try {
    await registerVirtualZarrSW();
    setVzStatus(`resolved → ${zarrUrl}\nloading model + opening source zarr …`, "#cc7");
    const { vzUrl, sourceLayerUrl, chosenLevel, chosenVoxelSizeNm, centerWorld, halfExtentWorld, unit } =
      await activateVz({ modelId, zarrUrl });
    localStorage.setItem(LS_VZ_SRC, raw);
    const chosenLine = chosenLevel
      ? `\npicked level: ${chosenLevel} (${chosenVoxelSizeNm?.toFixed(2)} ${unit})`
      : "";
    setVzStatus(`resolved → ${zarrUrl}${chosenLine}\nactive: ${vzUrl} — opening Neuroglancer …`, "#6a6");

    // Convert source-units (nanometer/micrometer/...) to meters for NG's
    // canonical coordinate space, and pick a cross-section scale that fits
    // the widest in-plane half-extent across the viewport.
    const toMeters: Record<string, number> = {
      nanometer: 1e-9, micrometer: 1e-6, millimeter: 1e-3, meter: 1,
      angstrom: 1e-10, picometer: 1e-12,
    };
    const m = toMeters[unit] ?? 1e-9;
    const widestXY = Math.max(halfExtentWorld[1], halfExtentWorld[2]);
    mountNg({
      dimensions: { z: [m, "m"], y: [m, "m"], x: [m, "m"] },
      position: [centerWorld[0], centerWorld[1], centerWorld[2]],
      crossSectionScale: (widestXY / 256) * m,
      projectionScale: widestXY * 4 * m,
      layers: [
        { type: "image", source: sourceLayerUrl, name: "source", visible: true },
        { type: "image", source: vzUrl, name: modelId, visible: true },
      ],
      selectedLayer: { visible: true, layer: modelId },
      // Default to a single XY panel. A per-Z 2D model is pathological for
      // orthogonal views (one ORT run per Z), so we don't auto-open them.
      // User can switch to 4panel in NG's layout menu after the volume loads.
      layout: "xy",
    });
    setVzStatus(
      `resolved → ${zarrUrl}${chosenLine}\nstreaming ${modelId}`,
      "#6a6",
    );
  } catch (err) {
    console.error(err);
    setVzStatus(`resolved → ${zarrUrl}\nerror: ${(err as Error).message}`, "#f87171");
  } finally {
    vzGo.disabled = false;
  }
});
