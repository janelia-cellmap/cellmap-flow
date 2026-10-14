// compute-worker.js — module Worker that runs the inference pipeline.
//
// The service worker can't do this itself (dynamic import() is forbidden there),
// so the page spawns this worker and bridges messages between the SW and here.
// Each message: { id, segment, tail }. Reply: { id, ok, status, contentType, body }
// plus { provider, ms } for chunk results (used by the page's stats HUD).
// Load the pipeline with this worker's cache-busting query (?v=…) so a normal
// reload picks up new worker code (no SW unregister / hard-reload needed).
// Not awaited at top level: messages that arrive before onmessage is assigned are
// dropped, and a page opened from a shared link sends its first chunk requests
// while compute.js (and ORT from the CDN) is still loading.
const computeReady = import("./compute.js" + self.location.search);

// Report device info once so the page can show which GPU/CPU is in use.
(async () => {
  try {
    await computeReady;
    self.postMessage({ type: "status", text: "compute worker loaded" });
  } catch (err) {
    self.postMessage({ type: "status", text: "ERROR: compute pipeline failed to load: " + String((err && err.message) || err) });
    return;
  }
  const isolated = self.crossOriginIsolated === true;
  const info = {
    cores: navigator.hardwareConcurrency || null,
    gpu: null,
    isolated,
    threads: isolated && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 1,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      const ai = adapter && (adapter.info || (adapter.requestAdapterInfo && (await adapter.requestAdapterInfo())));
      if (ai) info.gpu = [ai.vendor, ai.architecture, ai.device, ai.description].filter(Boolean).join(" ").trim();
      else if (adapter) info.gpu = "WebGPU adapter";
    } catch (_) {}
  }
  self.postMessage({ type: "device", info });
})();

self.onmessage = async (e) => {
  const { id, segment, tail } = e.data;
  try {
    const { handleCf } = await computeReady;
    const r = await handleCf(segment, tail);
    const transfer = r.body instanceof ArrayBuffer ? [r.body] : [];
    self.postMessage(
      {
        id,
        ok: true,
        status: r.status || 200,
        contentType: r.contentType,
        body: r.body,
        provider: r.provider, // "webgpu" | "wasm" (chunk results only)
        ms: r.ms,
      },
      transfer
    );
  } catch (err) {
    const msg = String((err && err.message) || err);
    self.postMessage({ type: "status", text: "ERROR: " + msg });
    self.postMessage({ id, ok: false, error: String((err && err.stack) || err) });
  }
};
