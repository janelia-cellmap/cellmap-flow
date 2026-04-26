// Browser-side glue between cellmap-flow's rendered dashboard HTML and our
// in-browser /vz/ pipeline. The dashboard expects to talk to a Flask backend
// for things like /api/set-data; we replace those handlers with calls into
// our SW + ORT pipeline so the same UI works without a server.

import { activate as activateVz } from "./virtual-zarr";
import type { activate as activateVzT } from "./virtual-zarr";
import { registerVirtualZarrSW } from "./sw-register";
import { mountNg } from "./ng-entry";
import { DEMO_SPEC, loadSpec, type ModelSpec } from "./model-spec";
import { loadHfModel } from "./hf";

type VzState = Awaited<ReturnType<typeof activateVzT>>;

const DEFAULT_MODEL_URL = "/demo-model.onnx";
const DEFAULT_SPEC_URL = "/demo-model.json";

interceptApiFetches();

document.addEventListener("DOMContentLoaded", () => {
  rewireConnectPanel();
  wireHfAccordion();
});

// Patch window.fetch so the dashboard's existing /api/* calls hit our
// in-browser handlers instead of the missing Flask backend. Today this only
// returns real data for /api/huggingface-models; everything else returns a
// friendly 501 so the dashboard's UI doesn't hang.
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
        for (const m of list as { modelId: string }[]) {
          // Try to fetch metadata.json per repo (may 404 for some). Run in
          // parallel; failures yield empty metadata.
          out[m.modelId] = {};
        }
        // Fetch metadata in parallel (best-effort).
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
      JSON.stringify({ error: `${u.pathname} is not implemented in the browser version` }),
      { status: 501, headers: { "content-type": "application/json" } },
    );
  };
}

// When the user checks a model in the dashboard's HF accordion, mirror the
// repo into the Connect panel's HF input so they can activate it with one
// click. Multiple checkboxes are allowed in the original UI; we use the
// most-recently-checked one.
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

function rewireConnectPanel(): void {
  const btn = document.getElementById("setDataBtn") as HTMLButtonElement | null;
  const datasetInput = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const status = document.getElementById("setDataStatus");
  const panel = document.getElementById("noNeuroglancerPanel");
  if (!btn || !datasetInput || !status || !panel) return;

  // Inject an HF repo input above the existing dataset path input.
  const inputGroup = datasetInput.closest(".no-data-input-group") as HTMLElement | null;
  if (inputGroup && !document.getElementById("hfRepoInput")) {
    const hfWrap = document.createElement("div");
    hfWrap.className = "no-data-input-group";
    hfWrap.style.marginBottom = "8px";
    hfWrap.innerHTML =
      '<input type="text" id="hfRepoInput" class="form-control" placeholder="HF repo (e.g. cellmap/jrc_mus-livers_16nm_to_8nm_mito) — leave blank for demo model" style="flex:1;" />';
    inputGroup.parentElement?.insertBefore(hfWrap, inputGroup);
  }

  // Strip the original click handler by cloning, then bind ours.
  const clone = btn.cloneNode(true) as HTMLButtonElement;
  btn.parentElement?.replaceChild(clone, btn);

  clone.addEventListener("click", async () => {
    const zarrUrl = datasetInput.value.trim();
    const hfRepo = (document.getElementById("hfRepoInput") as HTMLInputElement | null)?.value.trim() ?? "";
    if (!zarrUrl) {
      status.textContent = "Please enter a zarr URL.";
      return;
    }
    clone.disabled = true;
    clone.textContent = "Activating...";
    status.style.color = "var(--text-muted)";
    try {
      status.textContent = "registering service worker ...";
      await registerVirtualZarrSW();

      let spec: ModelSpec = DEMO_SPEC;
      let modelUrl = DEFAULT_MODEL_URL;
      if (hfRepo) {
        status.textContent = `loading HF model ${hfRepo} ...`;
        const hf = await loadHfModel(hfRepo);
        spec = hf.spec;
        modelUrl = hf.modelUrl;
        status.textContent =
          `HF metadata ok (${hf.metadata.out_channels ?? "?"}c, ` +
          `voxel ${spec.outputVoxelSize.join(",")} nm, ` +
          `block ${spec.blockShape.join("x")}). downloading model.onnx (this may take a while) ...`;
      } else {
        try { spec = await loadSpec(DEFAULT_SPEC_URL); } catch { /* fall back to DEMO_SPEC */ }
      }

      const downloadStart = performance.now();
      const sourceZarrUrl = zarrUrl;
      const st = await activateVz({ zarrUrl, modelUrl, spec }, (loaded, total) => {
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const pct = total ? ((loaded / total) * 100).toFixed(1) : "?";
        const totalMb = total ? (total / 1024 / 1024).toFixed(0) + " MB" : "size unknown";
        const elapsedSec = (performance.now() - downloadStart) / 1000;
        const mbps = elapsedSec > 0.1 ? ((loaded / 1024 / 1024) / elapsedSec).toFixed(1) : "?";
        status.textContent =
          `downloading model.onnx: ${mb} / ${totalMb} (${pct}%, ${mbps} MB/s) ` +
          `— cached in browser memory + Chrome HTTP cache, no file written to disk.`;
      });

      status.textContent = "mounting Neuroglancer ...";
      mountNgIntoDashboard(st, spec, sourceZarrUrl);

      status.style.color = "#4ade80";
      status.textContent = `Active. vol=${st.outShape.join("x")} (vox ${spec.outputVoxelSize.join(",")} nm) channels=${spec.outputChannels}`;
    } catch (err) {
      const e = err as Error;
      status.style.color = "#f87171";
      status.textContent = "Error: " + e.message;
      console.error(e);
      clone.disabled = false;
      clone.textContent = "Connect";
    }
  });
}

function mountNgIntoDashboard(st: VzState, spec: ModelSpec, sourceZarrUrl: string): void {
  const panel = document.getElementById("noNeuroglancerPanel");
  const slot = panel?.parentElement;
  if (!slot) throw new Error("missing dashboard NG slot (noNeuroglancerPanel parent)");

  slot.innerHTML = '<div id="ng-host" style="width:100%;height:100%;position:relative;background:#000;"></div>';

  const NM = 1e-9;
  const ovs = spec.outputVoxelSize;
  const extent: [number, number, number] = [
    st.outShape[0] * ovs[0],
    st.outShape[1] * ovs[1],
    st.outShape[2] * ovs[2],
  ];

  // Translate the user-typed source URL into something NG's zarr datasource
  // can read directly (it understands s3:// natively, but the user may have
  // pasted https or zarr://). For raw layer NG fetches the source zarr
  // independently of our /vz/ pipeline, so we bypass the SW.
  const rawSource = normalizeForNg(sourceZarrUrl);

  mountNg({
    dimensions: { x: [NM, "m"], y: [NM, "m"], z: [NM, "m"] },
    position: [extent[2] / 2, extent[1] / 2, extent[0] / 2],
    crossSectionScale: NM * Math.max(...ovs),
    projectionScale: Math.max(...extent) * NM * 2,
    layers: [
      { type: "image", source: rawSource, name: "raw" },
      { type: "image", source: `zarr://${location.origin}/vz/`, name: "inference" },
    ],
    selectedLayer: { visible: true, layer: "inference" },
    layout: "4panel",
  });
}

function normalizeForNg(input: string): string {
  let s = input.trim();
  // Strip our own user-facing prefixes that NG also accepts.
  if (s.startsWith("zarr://") || s.startsWith("zarr2://") || s.startsWith("zarr3://")) return s;
  if (s.startsWith("s3://") || s.startsWith("gs://")) return `zarr://${s}`;
  return `zarr://${s}`;
}
