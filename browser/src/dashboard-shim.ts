// Browser-side glue between cellmap-flow's rendered dashboard HTML and the
// two inference backends the dashboard supports:
//
//   1. EXTERNAL cellmap-flow server (HF Space / Colab + cloudflared).
//      "Connect" → NG pointed at zarr://<server>/<dataset>/.
//   2. IN-BROWSER BMZ via ORT Web (no server).
//      "Connect" with a BMZ model checked → activates the /vz/ service
//      worker, applies the dashboard's checked normalizers / postprocessors,
//      mounts NG against zarr://<origin>/vz/.
//
// The dashboard's HF accordion still queries the cellmap HF org (display
// only). The new BMZ accordion lists locally exported BMZ-ONNX models.

import { mountNg } from "./ng-entry";
import { loadHfModel } from "./hf";
import { registerVirtualZarrSW } from "./sw-register";
import { activateVz } from "./vz-handler";
import type { NormSpec, PostSpec } from "./cellmap-pipeline";

// Hardcoded catalog of BMZ models we've exported to ONNX. Mirrors the
// MODELS dict in scripts/export-bmz-onnx.py; adding to that script + this
// list keeps both surfaces in sync.
const BMZ_CATALOG: Array<{ id: string; label: string; description: string }> = [
  {
    id: "hiding-blowfish",
    label: "hiding-blowfish (EnhancerMitochondriaEM2D)",
    description: "2D mitochondria EM, runs on WebGPU. Output channel 0 = foreground.",
  },
];

document.addEventListener("DOMContentLoaded", () => {
  injectBmzAccordion();
  rewireConnectPanel();
  wireHfAccordion();
  wireBmzAccordion();
  interceptApiFetches();
  restoreLastInputs();
});

const LS_KEY = "cmf-dashboard-state";

interface PersistedState {
  serverUrl: string;
  datasetPath: string;
  hfRepo: string;
  bmzModelId: string;
  bmzSourceUrl: string;
}

function readLs(): PersistedState {
  try {
    const v = JSON.parse(localStorage.getItem(LS_KEY) ?? "{}");
    return {
      serverUrl: v.serverUrl ?? "",
      datasetPath: v.datasetPath ?? "",
      hfRepo: v.hfRepo ?? "",
      bmzModelId: v.bmzModelId ?? "",
      bmzSourceUrl: v.bmzSourceUrl ?? "",
    };
  } catch {
    return { serverUrl: "", datasetPath: "", hfRepo: "", bmzModelId: "", bmzSourceUrl: "" };
  }
}
function writeLs(s: Partial<PersistedState>): void {
  const cur = readLs();
  localStorage.setItem(LS_KEY, JSON.stringify({ ...cur, ...s }));
}

function injectBmzAccordion(): void {
  if (document.getElementById("collapse_bmz")) return;
  // Insert above the HuggingFace accordion. The HF accordion lives in
  // #hfModelAccordion; we slot ours into the same modelSelectionForm above.
  const hfAccordion = document.getElementById("hfModelAccordion");
  if (!hfAccordion) return;
  const wrap = document.createElement("div");
  wrap.innerHTML = `
    <h5>BMZ (in-browser)</h5>
    <div class="accordion" id="bmzModelAccordion">
      <div class="accordion-item">
        <h2 class="accordion-header" id="heading_bmz">
          <button class="accordion-button collapsed" type="button"
                  data-bs-toggle="collapse" data-bs-target="#collapse_bmz"
                  aria-expanded="false" aria-controls="collapse_bmz">
            bioimage.io (browser-only)
          </button>
        </h2>
        <div id="collapse_bmz" class="accordion-collapse collapse"
             aria-labelledby="heading_bmz">
          <div class="accordion-body" id="bmzModelsContainer">
            <p class="text-muted" style="font-size:0.8rem;">
              Runs locally via ORT Web on WebGPU. No server needed.
              Selecting one switches Connect to the in-browser path.
            </p>
            ${BMZ_CATALOG.map((m) => `
              <div class="form-check mb-2 bmz-model-item" data-search="${(m.id + " " + m.description).toLowerCase()}">
                <input class="form-check-input bmz-model-checkbox" type="radio"
                       name="bmz-model" id="bmz_${m.id}" value="${m.id}" />
                <label class="form-check-label" for="bmz_${m.id}">
                  ${m.label}
                  <small style="color:#aaa;display:block;">${m.description}</small>
                </label>
              </div>`).join("")}
          </div>
        </div>
      </div>
    </div>
  `;
  hfAccordion.parentElement?.insertBefore(wrap, hfAccordion.previousElementSibling);
}

function wireBmzAccordion(): void {
  // On BMZ model selection: persist + repurpose the existing dataset path
  // input for the BMZ flow (change placeholder so user knows what to paste).
  document.addEventListener("change", (e) => {
    const t = e.target as HTMLInputElement;
    if (!t || !t.classList?.contains("bmz-model-checkbox")) return;
    if (!t.checked) return;
    writeLs({ bmzModelId: t.value });
    updateConnectPanelMode();
  });
}

function selectedBmzModelId(): string | null {
  const checked = document.querySelector<HTMLInputElement>(
    "input.bmz-model-checkbox:checked",
  );
  return checked ? checked.value : null;
}

const DATASET_PLACEHOLDER_DEFAULT = "/path/to/dataset.zarr";
const DATASET_PLACEHOLDER_BMZ =
  "Source zarr URL (s3:// or https://, group or array level)";

function updateConnectPanelMode(): void {
  const bmzActive = !!selectedBmzModelId();
  const datasetInput = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const serverInput = document.getElementById("serverUrlInput") as HTMLInputElement | null;
  const hfInput = document.getElementById("hfRepoInput") as HTMLInputElement | null;
  // The dataset input + Connect button live in the same wrap; we always
  // keep that wrap visible. Just repurpose the placeholder when BMZ active.
  if (datasetInput) {
    datasetInput.placeholder = bmzActive
      ? DATASET_PLACEHOLDER_BMZ
      : DATASET_PLACEHOLDER_DEFAULT;
  }
  // Server URL + HF repo inputs only matter for the server-backed flow.
  // Hide their wraps in BMZ mode to reduce clutter.
  if (serverInput) {
    (serverInput.closest(".no-data-input-group") as HTMLElement | null)
      ?.style.setProperty("display", bmzActive ? "none" : "flex");
  }
  if (hfInput) {
    (hfInput.closest(".no-data-input-group") as HTMLElement | null)
      ?.style.setProperty("display", bmzActive ? "none" : "flex");
  }
}

function restoreLastInputs(): void {
  // URL query params override localStorage. Lets a hosted demo embed the
  // backend URL in a link like:
  //   https://demo.example/dashboard.html?backend=https://x-cellmapflow.hf.space&dataset=jrc_hela-2
  // ?model=<bmz-id> additionally pre-selects an in-browser BMZ model.
  const qp = new URLSearchParams(window.location.search);
  const s = readLs();
  const server = document.getElementById("serverUrlInput") as HTMLInputElement | null;
  const dataset = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const hf = document.getElementById("hfRepoInput") as HTMLInputElement | null;

  const qpBackend = qp.get("backend") ?? qp.get("server") ?? "";
  const qpDataset = qp.get("dataset") ?? qp.get("data") ?? "";
  const qpModel = qp.get("model") ?? "";
  const qpHfRepo = qp.get("hf") ?? "";

  if (server) server.value = qpBackend || s.serverUrl || server.value;
  if (hf) hf.value = qpHfRepo || s.hfRepo || hf.value;

  // BMZ override beats everything else if ?model= names a BMZ id.
  if (qpModel) {
    const cb = document.getElementById(`bmz_${qpModel}`) as HTMLInputElement | null;
    if (cb) {
      cb.checked = true;
      writeLs({ bmzModelId: qpModel });
    }
  } else if (s.bmzModelId) {
    const cb = document.getElementById(`bmz_${s.bmzModelId}`) as HTMLInputElement | null;
    if (cb) cb.checked = true;
  }

  // Dataset input: BMZ-mode source URL has priority when a BMZ model is set,
  // otherwise the server-backed dataset path.
  const bmzActive = !!selectedBmzModelId();
  if (dataset) {
    if (bmzActive) {
      dataset.value = qpDataset || s.bmzSourceUrl || dataset.value;
    } else {
      dataset.value = qpDataset || s.datasetPath || dataset.value;
    }
  }
  updateConnectPanelMode();
}

function normalizeZarrUrl(raw: string): string {
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

// Walk the dashboard's existing input/postprocess form state. The Jinja
// template registers gatherInputNormData / gatherPostProcessData on window
// for the dashboard's own Submit-All flow; we reuse those helpers verbatim.
function gatherPipeline(): { normalizers: NormSpec[]; postprocessors: PostSpec[] } {
  type Gather = () => Array<Record<string, unknown>>;
  const w = window as unknown as {
    gatherInputNormData?: Gather;
    gatherPostProcessData?: Gather;
  };
  const norms = (w.gatherInputNormData ? w.gatherInputNormData() : []) as NormSpec[];
  const posts = (w.gatherPostProcessData ? w.gatherPostProcessData() : []) as PostSpec[];
  return { normalizers: norms, postprocessors: posts };
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
    // Branch on whether a BMZ model is selected in the new accordion.
    const bmzModelId = selectedBmzModelId();
    if (bmzModelId) {
      await openBmzPath(bmzModelId, clone, status);
      return;
    }
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
      const inferenceUrl = `zarr://${serverUrl}/${datasetPath}/`;
      // Optional ?raw=<https-zarr-url> query param lets a hosted demo
      // link include the source EM alongside the inference layer. Plain
      // form is just an http(s) URL; we wrap it in zarr:// for NG.
      const qp = new URLSearchParams(window.location.search);
      const rawZarr = (qp.get("raw") ?? "").trim();
      const layers: Array<Record<string, unknown>> = [];
      if (rawZarr) {
        const rawSource = rawZarr.startsWith("zarr://")
          ? rawZarr
          : `zarr://${rawZarr.replace(/\/$/, "")}/`;
        layers.push({ type: "image", source: rawSource, name: "raw", visible: true });
      }
      layers.push({ type: "image", source: inferenceUrl, name: "inference", visible: true });
      mountNg({
        dimensions: {
          z: [voxelSizeNm * NM, "m"],
          y: [voxelSizeNm * NM, "m"],
          x: [voxelSizeNm * NM, "m"],
        },
        layers,
        selectedLayer: { visible: true, layer: "inference" },
        layout: "4panel",
      });
      status.style.color = "#4ade80";
      status.textContent = rawZarr
        ? `Open. Raw: ${rawZarr}, Inference: ${inferenceUrl}`
        : `Open. Source: ${inferenceUrl}`;
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

async function openBmzPath(
  bmzModelId: string,
  btn: HTMLButtonElement,
  status: HTMLElement,
): Promise<void> {
  // BMZ flow reuses the existing dataset-path input (placeholder is set to
  // a BMZ-flavored prompt by updateConnectPanelMode).
  const dataset = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const rawSrc = (dataset?.value ?? "").trim();
  if (!rawSrc) {
    status.style.color = "#facc15";
    status.textContent = "Please paste a source zarr URL for the BMZ path.";
    return;
  }
  const zarrUrl = normalizeZarrUrl(rawSrc);
  writeLs({ bmzModelId, bmzSourceUrl: rawSrc });
  const pipeline = gatherPipeline();

  btn.disabled = true;
  btn.textContent = "Activating in-browser…";
  status.style.color = "var(--text-muted)";
  try {
    status.textContent = `resolved → ${zarrUrl}\nregistering service worker…`;
    await registerVirtualZarrSW();
    status.textContent = `resolved → ${zarrUrl}\nloading model + opening source zarr…`;
    const act = await activateVz({
      modelId: bmzModelId,
      zarrUrl,
      normalizers: pipeline.normalizers,
      postprocessors: pipeline.postprocessors,
    });
    const pickedLine = act.chosenLevel
      ? `\npicked level: ${act.chosenLevel} (${act.chosenVoxelSizeNm?.toFixed(2)} ${act.unit})`
      : "";
    status.textContent =
      `resolved → ${zarrUrl}${pickedLine}\nactive: ${act.vzUrl} — opening Neuroglancer…`;

    replaceNgPanel();
    const toMeters: Record<string, number> = {
      nanometer: 1e-9, micrometer: 1e-6, millimeter: 1e-3, meter: 1,
      angstrom: 1e-10, picometer: 1e-12,
    };
    const m = toMeters[act.unit] ?? 1e-9;
    const widestXY = Math.max(act.halfExtentWorld[1], act.halfExtentWorld[2]);
    mountNg({
      dimensions: { z: [m, "m"], y: [m, "m"], x: [m, "m"] },
      position: act.centerWorld,
      crossSectionScale: (widestXY / 256) * m,
      projectionScale: widestXY * 4 * m,
      layers: [
        { type: "image", source: act.sourceLayerUrl, name: "source", visible: true },
        { type: "image", source: act.vzUrl, name: bmzModelId, visible: true },
      ],
      selectedLayer: { visible: true, layer: bmzModelId },
      layout: "xy",
    });
    status.style.color = "#4ade80";
    status.textContent =
      `resolved → ${zarrUrl}${pickedLine}\nstreaming ${bmzModelId} ` +
      `(norms=${pipeline.normalizers.length}, post=${pipeline.postprocessors.length})`;
  } catch (err) {
    const e = err as Error;
    status.style.color = "#f87171";
    status.textContent = `error: ${e.message}`;
    console.error(e);
    btn.disabled = false;
    btn.textContent = "Open in NG";
  }
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
