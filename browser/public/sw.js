// cellmap-flow virtual-zarr service worker.
// Intercepts same-origin fetches under /vz/ and forwards them to an active
// page client via MessageChannel. The page holds the ORT session + raw
// zarr state and replies with bytes.
//
// Cancellation: each fetch is assigned a unique request id. If NG aborts
// the fetch (e.g. user pans/zooms before the chunk is ready), we post a
// `vz-cancel` message to the page so it can drop the work from its queue
// or skip postprocessing of an in-flight inference. Mirrors how the
// production server architecture frees a browser HTTP slot on abort.

const PREFIX = "/vz/";
let _nextId = 0;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(PREFIX)) return;
  event.respondWith(handle(url, event.request.signal));
});

async function handle(url, signal) {
  const path = url.pathname.slice(PREFIX.length);
  const id = `r${++_nextId}`;

  const clients = await self.clients.matchAll({
    type: "window",
    includeUncontrolled: true,
  });
  const client = clients[0];
  if (!client) {
    return new Response("no active page client for /vz/", {
      status: 503,
      headers: { "content-type": "text/plain" },
    });
  }

  const channel = new MessageChannel();
  const reply = new Promise((resolve) => {
    channel.port1.onmessage = (ev) => resolve(ev.data);
    setTimeout(
      () => resolve({ status: 504, body: "page timeout", headers: {} }),
      120_000,
    );
  });

  const onAbort = () => {
    try { client.postMessage({ type: "vz-cancel", id }); } catch (_) {}
  };
  if (signal && !signal.aborted) {
    signal.addEventListener("abort", onAbort, { once: true });
  } else if (signal && signal.aborted) {
    onAbort();
  }

  client.postMessage({ type: "vz-request", path, id }, [channel.port2]);

  try {
    const res = await reply;
    const headers = {
      "access-control-allow-origin": "*",
      ...(res.headers || {}),
    };
    return new Response(res.body ?? null, { status: res.status ?? 500, headers });
  } finally {
    if (signal) signal.removeEventListener("abort", onAbort);
  }
}
