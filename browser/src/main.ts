// Minimal entry for the simplified `index.html` (kept around for direct
// users; the dashboard.html flow is the main one). Both paths now point
// Neuroglancer at an external cellmap-flow inference server (HF Space or
// Colab + ngrok), no in-browser ORT.

import { mountNg } from "./ng-entry";

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
