// Browser-side glue between cellmap-flow's rendered dashboard HTML and an
// EXTERNAL cellmap-flow inference server (HF Space, Colab + cloudflared,
// your workstation, etc).
//
// What this does:
//   1. Reads URL params (?backend=, ?dataset=, ?raw=, ?voxelSize=)
//   2. Injects a "server URL" input alongside the dashboard's dataset path
//   3. On "Open in NG" click, mounts Neuroglancer with raw + inference layers
//   4. Forwards the dashboard's Submit-All POSTs to <backend>/api/process
//      and cache-busts the inference layer URL so NG re-fetches under the
//      new normalizers/postprocessors

import { mountNg } from "./ng-entry";

document.addEventListener("DOMContentLoaded", () => {
  rewireConnectPanel();
  interceptApiFetches();
  restoreLastInputs();
});

const LS_KEY = "cmf-dashboard-state";

interface PersistedState {
  serverUrl: string;
  datasetPath: string;
}

function readLs(): PersistedState {
  try {
    const v = JSON.parse(localStorage.getItem(LS_KEY) ?? "{}");
    return {
      serverUrl: v.serverUrl ?? "",
      datasetPath: v.datasetPath ?? "",
    };
  } catch {
    return { serverUrl: "", datasetPath: "" };
  }
}

function writeLs(s: Partial<PersistedState>): void {
  const cur = readLs();
  localStorage.setItem(LS_KEY, JSON.stringify({ ...cur, ...s }));
}

function restoreLastInputs(): void {
  // URL query params override localStorage. Lets a hosted demo embed the
  // backend URL in a link like:
  //   https://demo.example/dashboard.html?backend=https://x.trycloudflare.com&dataset=my-model
  const qp = new URLSearchParams(window.location.search);
  const s = readLs();
  const server = document.getElementById("serverUrlInput") as HTMLInputElement | null;
  const dataset = document.getElementById("datasetPathInput") as HTMLInputElement | null;

  const qpBackend = qp.get("backend") ?? qp.get("server") ?? "";
  const qpDataset = qp.get("dataset") ?? qp.get("data") ?? "";

  if (server) server.value = qpBackend || s.serverUrl || server.value;
  if (dataset) dataset.value = qpDataset || s.datasetPath || dataset.value;
}

function rewireConnectPanel(): void {
  const btn = document.getElementById("setDataBtn") as HTMLButtonElement | null;
  const datasetInput = document.getElementById("datasetPathInput") as HTMLInputElement | null;
  const status = document.getElementById("setDataStatus");
  const panel = document.getElementById("noNeuroglancerPanel");
  if (!btn || !datasetInput || !status || !panel) return;

  // The rendered dashboard only ships a dataset-path input. Inject a
  // server-URL input above it.
  const inputGroup = datasetInput.closest(".no-data-input-group") as HTMLElement | null;
  if (inputGroup && !document.getElementById("serverUrlInput")) {
    const wrap = document.createElement("div");
    wrap.className = "no-data-input-group";
    wrap.style.marginBottom = "8px";
    wrap.innerHTML =
      '<input type="text" id="serverUrlInput" class="form-control" ' +
      'placeholder="cellmap-flow server URL (HF Space, Colab + cloudflared, etc.)" ' +
      'style="flex:1;" />';
    inputGroup.parentElement?.insertBefore(wrap, inputGroup);
  }

  // Replace the dashboard's stock "Connect" handler with ours.
  const clone = btn.cloneNode(true) as HTMLButtonElement;
  clone.textContent = "Open in NG";
  btn.parentElement?.replaceChild(clone, btn);

  clone.addEventListener("click", async () => {
    const serverInput = document.getElementById("serverUrlInput") as HTMLInputElement;
    const serverUrl = serverInput.value.trim().replace(/\/$/, "");
    const datasetPath = datasetInput.value.trim().replace(/^\/+/, "").replace(/\/$/, "");

    if (!serverUrl) {
      status.textContent = "Please enter the inference server URL.";
      return;
    }
    if (!datasetPath) {
      status.textContent = "Please enter a dataset slug.";
      return;
    }

    writeLs({ serverUrl, datasetPath });

    clone.disabled = true;
    clone.textContent = "Opening...";
    status.style.color = "var(--text-muted)";

    try {
      replaceNgPanel();
      const NM = 1e-9;
      // The inference URL encodes the dashboard's current
      // input-normalizer + postprocessor pipeline in the path between
      // two __CFLOW_ARGS__ delimiters. cellmap_flow_server parses that
      // per-request, so each unique config produces a unique URL → NG
      // re-fetches under the new pipeline naturally on Submit All. No
      // global server state needed.
      const inferenceUrl = buildInferenceUrl(serverUrl, datasetPath, encodeArgs());
      const qp = new URLSearchParams(window.location.search);
      const rawZarr = (qp.get("raw") ?? "").trim();

      // ?voxelSize=Z,Y,X (in nm). When set, force the raw layer's input
      // dimensions to this size so it appears in the same
      // pseudo-isotropic NG world coords as the inference. Without it,
      // NG uses the raw zarr's actual (possibly anisotropic) scale and
      // the two layers misalign.
      const voxelSizeOverride = (qp.get("voxelSize") ?? "").trim();
      let vz = 8, vy = 8, vx = 8;
      if (voxelSizeOverride) {
        const parts = voxelSizeOverride.split(",").map((p) => parseFloat(p.trim()));
        if (parts.length === 3 && parts.every((p) => Number.isFinite(p) && p > 0)) {
          [vz, vy, vx] = parts;
        }
      }

      // Per-layer transform that forces the raw layer to claim (vz, vy, vx)
      // voxel size regardless of zarr metadata. Identity matrix → each
      // source voxel maps 1:1 to one (vz, vy, vx) output voxel. NG world
      // dims are also (vz, vy, vx). Inference keeps its honest zarr scale.
      const isoTransform = voxelSizeOverride
        ? {
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
          }
        : null;

      const layers: Array<Record<string, unknown>> = [];
      if (rawZarr) {
        // Pick the right NG scheme for the source URL. Most cases are
        // zarr but precomputed/n5 also flow through here, so detect.
        let rawSource: string;
        if (
          rawZarr.startsWith("precomputed://") ||
          rawZarr.startsWith("zarr://") ||
          rawZarr.startsWith("n5://")
        ) {
          rawSource = rawZarr.replace(/\/$/, "");
        } else if (rawZarr.includes(".zarr") || rawZarr.includes(".n5")) {
          rawSource = `zarr://${rawZarr.replace(/\/$/, "")}/`;
        } else {
          // Bare URL with no .zarr/.n5 — likely a precomputed dir on gs://.
          rawSource = `precomputed://${rawZarr.replace(/\/$/, "")}`;
        }
        layers.push({
          type: "image",
          name: "raw",
          visible: true,
          source: isoTransform
            ? { url: rawSource, transform: isoTransform }
            : rawSource,
        });
      }
      layers.push({
        type: "image",
        source: inferenceUrl,
        name: "inference",
        visible: true,
      });

      // Center the view on the inference dataset. We need the inference
      // voxel size AND the NG output dim to compute the center correctly
      // — `position` is in output-dim units, not inference voxels.
      let position: number[] | null = null;
      try {
        const [arrRes, attrsRes] = await Promise.all([
          fetch(`${serverUrl}/${datasetPath}/s0/.zarray`),
          fetch(`${serverUrl}/${datasetPath}/.zattrs`),
        ]);
        if (arrRes.ok && attrsRes.ok) {
          const arr = (await arrRes.json()) as { shape: number[] };
          const attrs = (await attrsRes.json()) as {
            multiscales?: Array<{
              datasets?: Array<{
                coordinateTransformations?: Array<{ type: string; scale?: number[] }>;
              }>;
            }>;
          };
          const s = arr.shape.slice(0, 3); // [z, y, x] in voxels
          const xform = attrs.multiscales?.[0]?.datasets?.[0]?.coordinateTransformations?.find(
            (t) => t.type === "scale",
          );
          const infVoxNm = xform?.scale?.slice(0, 3) ?? [vz, vy, vx];
          const outNm = [vz, vy, vx];
          if (s.length === 3 && infVoxNm.length === 3) {
            position = [
              (s[0] * infVoxNm[0]) / (2 * outNm[0]),
              (s[1] * infVoxNm[1]) / (2 * outNm[1]),
              (s[2] * infVoxNm[2]) / (2 * outNm[2]),
            ];
          }
        }
      } catch {
        /* default to NG's position 0,0,0 */
      }

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

      activeServerBacked = {
        viewer: viewer as unknown as { state: unknown; [k: string]: unknown },
        serverUrl,
        datasetPath,
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

// Track the active server-backed session so the /api/process intercept
// can refresh the inference layer (NG won't re-fetch chunks unless the
// layer source URL changes; cellmap_flow_server returns 200 on
// /api/process without bumping the URL).
let activeServerBacked: {
  viewer: { state: unknown; [k: string]: unknown };
  serverUrl: string;
  datasetPath: string;
  isoTransform: Record<string, unknown> | null;
} | null = null;

// Mirrors cellmap_flow.utils.web_utils.ARGS_KEY — the delimiter
// cellmap_flow_server's URL parser uses to extract a per-request
// normalizer/postprocessor config from the dataset path component.
const CFLOW_ARGS_KEY = "__CFLOW_ARGS__";

interface FormItem {
  name?: string;
  [k: string]: unknown;
}

// Read the dashboard's checked normalizers + postprocessors from the
// rendered forms. The dashboard JS exposes gatherInputNormData /
// gatherPostProcessData on window when those forms exist. Returns
// either the base64-encoded args string ready for the URL, or "" if
// nothing is configured.
function encodeArgs(): string {
  const w = window as unknown as {
    gatherInputNormData?: () => FormItem[];
    gatherPostProcessData?: () => FormItem[];
  };
  const norms = w.gatherInputNormData ? w.gatherInputNormData() : [];
  const posts = w.gatherPostProcessData ? w.gatherPostProcessData() : [];
  if (norms.length === 0 && posts.length === 0) return "";

  // Mirror cellmap_flow.utils.web_utils.list_cls_to_dict shape:
  // [{name, ...params}] → {name: {param: str(value)}}
  const toDictOfDicts = (arr: FormItem[]): Record<string, Record<string, string>> => {
    const out: Record<string, Record<string, string>> = {};
    for (const item of arr) {
      const name = item.name;
      if (!name) continue;
      const params: Record<string, string> = {};
      for (const [k, v] of Object.entries(item)) {
        if (k === "name") continue;
        params[k] = String(v);
      }
      out[String(name)] = params;
    }
    return out;
  };
  const args = {
    input_norm: toDictOfDicts(norms),
    postprocess: toDictOfDicts(posts),
  };
  const json = JSON.stringify(args);
  // URL-safe base64, padding stripped — exactly what
  // cellmap_flow.utils.web_utils.encode_to_str produces.
  const b64 = btoa(json)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  return b64;
}

function buildInferenceUrl(serverUrl: string, modelName: string, encoded: string): string {
  return encoded
    ? `zarr://${serverUrl}/${modelName}${CFLOW_ARGS_KEY}${encoded}${CFLOW_ARGS_KEY}/`
    : `zarr://${serverUrl}/${modelName}/`;
}

function replaceNgPanel(): void {
  // Swap the dashboard's "no neuroglancer URL configured" panel for an
  // empty NG host div. mountNg() will fill it.
  const panel = document.getElementById("noNeuroglancerPanel");
  if (!panel) return;
  const host = document.createElement("div");
  host.id = "ng-host";
  host.style.width = "100%";
  host.style.height = "100%";
  panel.replaceWith(host);
}

// Patch fetch so the dashboard's /api/process POST (Submit-All) gets
// forwarded to the configured backend, then bumps the inference layer
// URL so NG re-fetches chunks under the new normalizers/postprocessors.
function interceptApiFetches(): void {
  const orig = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const urlStr =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : (input as Request).url;
    const u = new URL(urlStr, location.href);
    if (u.origin !== location.origin || !u.pathname.startsWith("/api/")) {
      return orig(input, init);
    }
    if (u.pathname === "/api/process" && init?.method?.toUpperCase() === "POST") {
      const serverInput = document.getElementById("serverUrlInput") as HTMLInputElement | null;
      const fromInput = (serverInput?.value ?? "").trim();
      const fromLs = readLs().serverUrl;
      const fromQp =
        new URLSearchParams(window.location.search).get("backend") ??
        new URLSearchParams(window.location.search).get("server") ??
        "";
      const serverUrl = (fromInput || fromLs || fromQp).trim().replace(/\/$/, "");
      if (!serverUrl) {
        return new Response(
          JSON.stringify({
            error: "no inference server URL configured (form, localStorage, or ?backend=)",
          }),
          { status: 400, headers: { "content-type": "application/json" } },
        );
      }
      try {
        const r = await orig(`${serverUrl}/api/process`, init);
        // On success, rebuild the inference layer URL with the new
        // pipeline encoded between __CFLOW_ARGS__ delimiters. NG sees
        // a different URL → re-fetches all chunks → cellmap_flow_server
        // parses the args per-request and applies them. Each unique
        // pipeline config produces a unique URL, so cache invalidation
        // and per-config isolation come for free.
        if (r.ok && activeServerBacked) {
          try {
            const v = activeServerBacked.viewer as {
              state: { layers: Array<{ name: string; source: unknown }> };
            };
            const newUrl = buildInferenceUrl(
              activeServerBacked.serverUrl,
              activeServerBacked.datasetPath,
              encodeArgs(),
            );
            const layersList = v.state.layers;
            const idx = layersList.findIndex((l) => l.name === "inference");
            if (idx >= 0) {
              layersList[idx].source = activeServerBacked.isoTransform
                ? { url: newUrl, transform: activeServerBacked.isoTransform }
                : newUrl;
            }
          } catch (e) {
            console.warn("[shim] could not refresh inference layer:", e);
          }
        }
        return r;
      } catch (err) {
        return new Response(
          JSON.stringify({
            error: `forwarding to ${serverUrl}/api/process failed: ${(err as Error).message}`,
          }),
          { status: 502, headers: { "content-type": "application/json" } },
        );
      }
    }
    // Other /api/* calls — try original. Most won't resolve (no Flask
    // backend on this static page), but let them error naturally.
    return orig(input, init);
  };
}
