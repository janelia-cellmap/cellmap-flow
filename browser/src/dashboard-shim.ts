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

  // Backend + BMZ are mutually exclusive — server-backed inference vs
  // in-browser WebGPU. ?backend= explicitly opts into server-backed,
  // so clear any stale BMZ selection from localStorage so users who
  // previously played with the WebGPU demo aren't accidentally
  // sticky-stuck in BMZ mode.
  if (qpBackend) {
    writeLs({ bmzModelId: "" });
    document
      .querySelectorAll<HTMLInputElement>("[id^='bmz_']")
      .forEach((cb) => { cb.checked = false; });
  } else if (qpModel) {
    // BMZ override beats everything else if ?model= names a BMZ id.
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
      // Optional ?raw=<https-zarr-url> query param.
      const qp = new URLSearchParams(window.location.search);
      const rawZarr = (qp.get("raw") ?? "").trim();
      // Optional ?voxelSize=Z,Y,X (in nm). When set we LIE — declare both
      // the raw layer and the NG output dims to be this voxel size, so the
      // raw layer and the inference layer (which already lies via cellmap_flow_server
      // --voxel-size) share the same pseudo-isotropic NG world coords.
      // Without this, raw uses the zarr's actual anisotropic scale (e.g.
      // 5.24×4×4 nm for jrc_hela-2 s0) and the two layers float ~2× apart.
      const voxelSizeOverride = (qp.get("voxelSize") ?? "").trim();
      let vz = voxelSizeNm, vy = voxelSizeNm, vx = voxelSizeNm;
      if (voxelSizeOverride) {
        const parts = voxelSizeOverride.split(",").map((s) => parseFloat(s.trim()));
        if (parts.length === 3 && parts.every((p) => Number.isFinite(p) && p > 0)) {
          [vz, vy, vx] = parts;
        }
      }
      // Per-layer transform that forces both raw + inference layers to
      // claim the same (vz, vy, vx) voxel size, regardless of what their
      // zarr metadata reports. Identity matrix → each source voxel maps
      // 1:1 to one (vz, vy, vx) output voxel.
      const isoTransform = voxelSizeOverride ? {
        outputDimensions: {
          z: [vz * NM, "m"],
          y: [vy * NM, "m"],
          x: [vx * NM, "m"],
        },
        inputDimensions: {
          z: [vz * NM, "m"],
          y: [vy * NM, "m"],
          x: [vx * NM, "m"],
        },
        matrix: [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
        ],
      } : null;

      const layers: Array<Record<string, unknown>> = [];
      if (rawZarr) {
        const rawSource = rawZarr.startsWith("zarr://")
          ? rawZarr
          : `zarr://${rawZarr.replace(/\/$/, "")}/`;
        layers.push({
          type: "image",
          name: "raw",
          visible: true,
          source: isoTransform
            ? { url: rawSource, transform: isoTransform }
            : rawSource,
        });
      }
      // Inference layer keeps its own zarr-reported scale (cellmap-flow's
      // --voxel-size, typically 8×8×8). NG places it in world coords
      // using that scale. If the raw layer's lie (vz, vy, vx) matches
      // its finest level's xy, NG aligns them in world coords because
      // their physical extents are derived consistently.
      layers.push({ type: "image", source: inferenceUrl, name: "inference", visible: true });

      // Try to center the view on the inference dataset. We need:
      //   shape    from /<model>/s0/.zarray            (voxels)
      //   voxelNm  from /<model>/.zattrs (OME-NGFF)    (nm per voxel)
      // NG `position` is in output-dim units, NOT inference voxels, so
      // the center is shape × voxelNm / (2 × outputDimNm). Forgetting the
      // voxelNm/outputDim factor was an earlier bug that landed you at
      // 1/4 of the extent when cellmap-flow's voxel size and our world
      // unit differed (e.g. inference at 8nm, NG world at 4nm).
      let position: number[] | null = null;
      try {
        const [arrRes, attrsRes] = await Promise.all([
          fetch(`${serverUrl}/${datasetPath}/s0/.zarray`),
          fetch(`${serverUrl}/${datasetPath}/.zattrs`),
        ]);
        if (arrRes.ok && attrsRes.ok) {
          const arr = await arrRes.json() as { shape: number[] };
          const attrs = await attrsRes.json() as {
            multiscales?: Array<{
              datasets?: Array<{ coordinateTransformations?: Array<{ type: string; scale?: number[] }> }>;
            }>;
          };
          const s = arr.shape.slice(0, 3);  // [z, y, x] in voxels
          const xform = attrs.multiscales?.[0]?.datasets?.[0]?.coordinateTransformations?.find(t => t.type === "scale");
          // Scale array may have trailing channel; take leading 3.
          const infVoxNm = xform?.scale?.slice(0, 3) ?? [vz, vy, vx];
          const outNm = [vz, vy, vx];
          if (s.length === 3 && infVoxNm.length === 3) {
            position = [
              s[0] * infVoxNm[0] / (2 * outNm[0]),
              s[1] * infVoxNm[1] / (2 * outNm[1]),
              s[2] * infVoxNm[2] / (2 * outNm[2]),
            ];
          }
        }
      } catch { /* fall through to default position 0,0,0 */ }

      const ngState: Record<string, unknown> = {
        dimensions: {
          z: [vz * NM, "m"],
          y: [vy * NM, "m"],
          x: [vx * NM, "m"],
        },
        layers,
        selectedLayer: { visible: true, layer: "inference" },
        layout: "4panel",
      };
      if (position) ngState.position = position;
      const viewer = mountNg(ngState);

      // Stash for /api/process Submit-All handler so it can bump the
      // inference layer URL (cache-bust) after the server applies new
      // normalizers / postprocessors.
      activeServerBacked = {
        viewer: viewer as unknown as { state: unknown; [k: string]: unknown },
        inferenceBaseUrl: inferenceUrl,
        isoTransform,
      };
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

// Track the active BMZ in-browser activation so Submit-All POSTs to
// /api/process can re-apply the pipeline locally (no backend exists).
let activeBmz: { modelId: string; zarrUrl: string } | null = null;

// Track the active server-backed activation so Submit-All can refresh
// the inference layer (NG won't re-fetch chunks unless the layer's
// source URL changes, but cellmap_flow_server returns 200/success on
// /api/process without bumping the URL).
let activeServerBacked: {
  viewer: { state: unknown; [k: string]: unknown };  // structural — NG viewer
  inferenceBaseUrl: string;
  isoTransform: Record<string, unknown> | null;
} | null = null;

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
    // Don't set crossSectionScale / projectionScale — let NG pick its
    // own defaults based on the layer bbox. Forcing a value here zoomed
    // way in past the data scale.
    mountNg({
      dimensions: { z: [m, "m"], y: [m, "m"], x: [m, "m"] },
      position: act.centerWorld,
      layers: [
        { type: "image", source: act.sourceLayerUrl, name: "source", visible: true },
        { type: "image", source: act.vzUrl, name: bmzModelId, visible: true },
      ],
      selectedLayer: { visible: true, layer: bmzModelId },
      layout: "xy",
    });
    activeBmz = { modelId: bmzModelId, zarrUrl };
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

// Re-apply the dashboard's current Input/Postprocess pipeline to an
// already-active in-browser BMZ session. Clears the chunk cache so NG
// re-fetches with the new pipeline (user may need to nudge the view).
async function reapplyBmzPipeline(): Promise<void> {
  if (!activeBmz) return;
  const pipeline = gatherPipeline();
  await activateVz({
    modelId: activeBmz.modelId,
    zarrUrl: activeBmz.zarrUrl,
    normalizers: pipeline.normalizers,
    postprocessors: pipeline.postprocessors,
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
    // /api/process — pipeline configuration. Forward to the configured
    // inference-server URL so the cellmap-flow server (Colab / HF Space /
    // workstation) can apply the new normalizers + postprocessors via its
    // minimal /api/process endpoint. Subsequent chunk requests honor the
    // updated g.input_norms / g.postprocess globals.
    if (u.pathname === "/api/process" && init?.method?.toUpperCase() === "POST") {
      // BMZ in-browser mode: there's no backend; re-apply the pipeline
      // to the local vz handler so subsequent chunk fetches use the new
      // normalizers / postprocessors. NG cache may keep stale tiles —
      // a small pan/zoom triggers re-fetch.
      if (activeBmz) {
        try {
          await reapplyBmzPipeline();
          return new Response(
            JSON.stringify({
              success: true,
              note: "in-browser BMZ: pipeline re-applied; pan/zoom NG slightly to trigger re-fetch.",
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          );
        } catch (err) {
          return new Response(
            JSON.stringify({ error: `reapply pipeline: ${(err as Error).message}` }),
            { status: 500, headers: { "content-type": "application/json" } },
          );
        }
      }
      // Server-backed mode: forward to the configured inference URL.
      // Fall back to localStorage / ?backend= since serverUrlInput is
      // removed once Connect swaps the panel for the NG host.
      const serverInput = document.getElementById("serverUrlInput") as HTMLInputElement | null;
      const fromInput = (serverInput?.value ?? "").trim();
      const fromLs = readLs().serverUrl;
      const fromQp = new URLSearchParams(window.location.search).get("backend")
        ?? new URLSearchParams(window.location.search).get("server")
        ?? "";
      const serverUrl = (fromInput || fromLs || fromQp).trim().replace(/\/$/, "");
      if (!serverUrl) {
        return new Response(
          JSON.stringify({ error: "no inference server URL configured (form, localStorage, or ?backend=)" }),
          { status: 400, headers: { "content-type": "application/json" } },
        );
      }
      try {
        const r = await orig(`${serverUrl}/api/process`, init);
        // If submit succeeded, bump the inference layer URL with a
        // cache-busting query so NG re-fetches chunks under the new
        // server-side g.input_norms / g.postprocess. cellmap_flow_server
        // ignores unknown query params, but NG sees a new URL and
        // invalidates its tile cache.
        if (r.ok && activeServerBacked) {
          try {
            const v = activeServerBacked.viewer as { state: { layers: Array<{ name: string; source: unknown }>; toJSON: () => unknown }; setState?: (s: unknown) => void };
            const versioned = `${activeServerBacked.inferenceBaseUrl}?v=${Date.now()}`;
            const layersList = v.state.layers;
            const idx = layersList.findIndex((l) => l.name === "inference");
            if (idx >= 0) {
              layersList[idx].source = activeServerBacked.isoTransform
                ? { url: versioned, transform: activeServerBacked.isoTransform }
                : versioned;
            }
          } catch (e) {
            console.warn("[shim] could not refresh inference layer:", e);
          }
        }
        return r;
      } catch (err) {
        return new Response(
          JSON.stringify({ error: `forwarding to ${serverUrl}/api/process failed: ${(err as Error).message}` }),
          { status: 502, headers: { "content-type": "application/json" } },
        );
      }
    }
    // The remaining dashboard config endpoints (models / set-data /
    // server-config) don't have a cellmap-flow-server equivalent — the
    // browser flow handles those at Connect time. Return a success no-op
    // so the dashboard's inline JS doesn't toast errors.
    const STUBBED_ENDPOINTS = [
      "/api/models",
      "/api/set-data",
      "/api/server-config",
    ];
    if (STUBBED_ENDPOINTS.includes(u.pathname)) {
      return new Response(
        JSON.stringify({
          success: true,
          note: "browser-only mode: dataset is applied at Connect time, no separate submit needed.",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
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
