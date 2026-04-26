// Browser-side glue between cellmap-flow's rendered dashboard HTML and an
// EXTERNAL cellmap-flow inference server (Hugging Face Space or Colab +
// ngrok). Replaces the dashboard's Flask-bound JS:
//   - "Connect" → opens NG pointed at zarr://<server>/<dataset>/
//   - HF accordion → real list of cellmap org models from HF Hub
//
// All inference happens on the inference server; the browser is just NG +
// the dashboard chrome. No service worker, no ORT.

import { mountNg } from "./ng-entry";
import { loadHfModel } from "./hf";

document.addEventListener("DOMContentLoaded", () => {
  rewireConnectPanel();
  wireHfAccordion();
  interceptApiFetches();
  restoreLastInputs();
});

const LS_KEY = "cmf-dashboard-state";

interface PersistedState {
  serverUrl: string;
  datasetPath: string;
  hfRepo: string;
}

function readLs(): PersistedState {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) ?? "{}");
  } catch {
    return { serverUrl: "", datasetPath: "", hfRepo: "" };
  }
}
function writeLs(s: PersistedState): void {
  localStorage.setItem(LS_KEY, JSON.stringify(s));
}

function restoreLastInputs(): void {
  const s = readLs();
  const server = document.getElementById("serverUrlInput") as HTMLInputElement | null;
  const dataset = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const hf = document.getElementById("hfRepoInput") as HTMLInputElement | null;
  if (server && s.serverUrl) server.value = s.serverUrl;
  if (dataset && s.datasetPath) dataset.value = s.datasetPath;
  if (hf && s.hfRepo) hf.value = s.hfRepo;
}

function rewireConnectPanel(): void {
  const btn = document.getElementById("setDataBtn") as HTMLButtonElement | null;
  const datasetInput = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const status = document.getElementById("setDataStatus");
  const panel = document.getElementById("noNeuroglancerPanel");
  if (!btn || !datasetInput || !status || !panel) return;

  // The rendered dashboard only ships a dataset-path input. Inject a
  // server-URL input above it (where the user pastes their HF Space or
  // Colab+ngrok URL), and an HF repo input below for choosing the model.
  // The UI flow is: "where's the inference server" + "what model" +
  // "what dataset" → "Open in NG".
  const inputGroup = datasetInput.closest(".no-data-input-group") as HTMLElement | null;
  if (inputGroup && !document.getElementById("serverUrlInput")) {
    const wrap = document.createElement("div");
    wrap.className = "no-data-input-group";
    wrap.style.marginBottom = "8px";
    wrap.innerHTML =
      '<input type="text" id="serverUrlInput" class="form-control" ' +
      'placeholder="cellmap-flow server URL (HF Space, Colab+ngrok, etc.)" ' +
      'value="https://your-name-cellmapflow.hf.space" style="flex:1;" />';
    inputGroup.parentElement?.insertBefore(wrap, inputGroup);
  }
  if (inputGroup && !document.getElementById("hfRepoInput")) {
    const wrap = document.createElement("div");
    wrap.className = "no-data-input-group";
    wrap.style.marginBottom = "8px";
    wrap.innerHTML =
      '<input type="text" id="hfRepoInput" class="form-control" ' +
      'placeholder="HF model repo (informational; configure model on the server)" ' +
      'style="flex:1;" />';
    inputGroup.parentElement?.insertBefore(wrap, inputGroup);
  }

  // Replace the original click handler with ours.
  const clone = btn.cloneNode(true) as HTMLButtonElement;
  clone.textContent = "Open in NG";
  btn.parentElement?.replaceChild(clone, btn);

  clone.addEventListener("click", async () => {
    const serverInput = document.getElementById("serverUrlInput") as HTMLInputElement;
    const hfInput = document.getElementById("hfRepoInput") as HTMLInputElement;
    const serverUrl = serverInput.value.trim().replace(/\/$/, "");
    const datasetPath = datasetInput.value.trim().replace(/^\/+/, "").replace(/\/$/, "");
    const hfRepo = hfInput.value.trim();

    if (!serverUrl) {
      status.textContent = "Please enter the inference server URL.";
      return;
    }
    if (!datasetPath) {
      status.textContent = "Please enter a dataset slug.";
      return;
    }

    writeLs({ serverUrl, datasetPath, hfRepo });

    clone.disabled = true;
    clone.textContent = "Opening...";
    status.style.color = "var(--text-muted)";

    try {
      // If the user pasted an HF repo, fetch metadata for display only —
      // the actual model choice has to be configured on the server side.
      let voxelSizeNm = 8;
      if (hfRepo) {
        try {
          status.textContent = `loading HF model metadata for ${hfRepo} ...`;
          const hf = await loadHfModel(hfRepo);
          voxelSizeNm = hf.spec.outputVoxelSize[0] || 8;
          status.textContent = `metadata ok (${hf.metadata.out_channels ?? "?"}c, voxel ${voxelSizeNm} nm). Make sure the inference server is configured for this model.`;
        } catch (err) {
          status.style.color = "#facc15";
          status.textContent = `metadata: ${(err as Error).message}. Continuing anyway.`;
        }
      }

      replaceNgPanel();
      const NM = 1e-9;
      const sourceUrl = `zarr://${serverUrl}/${datasetPath}/`;
      mountNg({
        dimensions: {
          z: [voxelSizeNm * NM, "m"],
          y: [voxelSizeNm * NM, "m"],
          x: [voxelSizeNm * NM, "m"],
        },
        layers: [
          { type: "image", source: sourceUrl, name: "inference", visible: true },
        ],
        selectedLayer: { visible: true, layer: "inference" },
        layout: "4panel",
      });
      status.style.color = "#4ade80";
      status.textContent = `Open. Source: ${sourceUrl}`;
    } catch (err) {
      const e = err as Error;
      status.style.color = "#f87171";
      status.textContent = "Error: " + e.message;
      console.error(e);
      clone.disabled = false;
      clone.textContent = "Open in NG";
    }
  });
}

function replaceNgPanel(): void {
  const panel = document.getElementById("noNeuroglancerPanel");
  const slot = panel?.parentElement;
  if (!slot) throw new Error("missing dashboard NG slot");
  slot.innerHTML = '<div id="ng-host" style="width:100%;height:100%;position:relative;background:#000;"></div>';
}

// Patch fetch so the rendered dashboard's /api/* calls hit our handlers
// instead of the missing Flask backend.
function interceptApiFetches(): void {
  const orig = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const urlStr = typeof input === "string" ? input
      : input instanceof URL ? input.toString()
      : (input as Request).url;
    const u = new URL(urlStr, location.href);
    if (u.origin !== location.origin || !u.pathname.startsWith("/api/")) {
      return orig(input, init);
    }
    if (u.pathname === "/api/huggingface-models" || u.pathname === "/api/huggingface-models/refresh") {
      try {
        const list = await fetch("https://huggingface.co/api/models?author=cellmap&limit=100").then(r => r.json());
        const out: Record<string, Record<string, unknown>> = {};
        for (const m of list as { modelId: string }[]) out[m.modelId] = {};
        await Promise.all(
          Object.keys(out).map(async (repo) => {
            try {
              const r = await fetch(`https://huggingface.co/${repo}/resolve/main/metadata.json`);
              if (r.ok) out[repo] = await r.json();
            } catch { /* ignore */ }
          }),
        );
        return new Response(JSON.stringify(out), { status: 200, headers: { "content-type": "application/json" } });
      } catch (err) {
        return new Response(JSON.stringify({ error: (err as Error).message }), { status: 502, headers: { "content-type": "application/json" } });
      }
    }
    return new Response(
      JSON.stringify({ error: `${u.pathname} is not implemented in the browser-only mode` }),
      { status: 501, headers: { "content-type": "application/json" } },
    );
  };
}

function wireHfAccordion(): void {
  document.addEventListener("change", (e) => {
    const t = e.target as HTMLInputElement;
    if (!t || !t.classList?.contains("hf-model-checkbox")) return;
    if (!t.checked) return;
    const hfInput = document.getElementById("hfRepoInput") as HTMLInputElement | null;
    if (hfInput) {
      hfInput.value = t.value;
      hfInput.scrollIntoView({ behavior: "smooth", block: "center" });
      hfInput.focus();
    }
  });
}
